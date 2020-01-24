from datetime import datetime
import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from arguments import get_args
from configure_data import configure_data
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from learning_rates import AnnealingLR
from model import BertModel
from model import get_params_for_weight_decay_optimization
from model import gpt2_get_params_for_weight_decay_optimization
from model import DistributedDataParallel as LocalDDP
import mpu
from apex.optimizers import FusedAdam as Adam
from utils import Timers
from utils import save_checkpoint
from utils import load_checkpoint
from utils import report_memory
from utils import print_args
from utils import print_params_min_max_norm
from utils import print_rank_0
from utils import enable_adlr_autoresume
from utils import check_adlr_autoresume_termination

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

def get_model(args):
    """Build the model."""

    print_rank_0('building XLNet model ...')
    model = BertModel(args)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)
        if args.fp32_embedding:
            model.module.model.bert.embeddings.word_embeddings.float()
            model.module.model.bert.embeddings.position_embeddings.float()
            model.module.model.bert.embeddings.token_type_embeddings.float()
        if args.fp32_tokentypes:
            model.module.model.bert.embeddings.token_type_embeddings.float()
        if args.fp32_layernorm:
            for name, _module in model.named_modules():
                if 'LayerNorm' in name:
                    _module.float()

    # Wrap model for distributed training.
    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        args.DDP_type = torch.nn.parallel.distributed.DistributedDataParallel
        model = args.DDP_type(model, device_ids=[i], output_device=i,
                              process_group=mpu.get_data_parallel_group())
    elif args.DDP_impl == 'local':
        args.DDP_type = LocalDDP
        model = args.DDP_type(model)
    else:
        print_rank_0('Unknown DDP implementation specified: {}. '
                     'Exiting.'.format(args.DDP_impl))
        exit()

    return model


def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (args.DDP_type, FP16_Module)):
        model = model.module
    layers = model.model.bert.encoder.layer
    pooler = model.model.bert.pooler
    lmheads = model.model.cls.predictions
    nspheads = model.model.cls.seq_relationship
    embeddings = model.model.bert.embeddings
    param_groups = []
    param_groups += list(get_params_for_weight_decay_optimization(layers))
    param_groups += list(get_params_for_weight_decay_optimization(pooler))
    param_groups += list(get_params_for_weight_decay_optimization(nspheads))
    param_groups += list(get_params_for_weight_decay_optimization(embeddings))
    param_groups += list(get_params_for_weight_decay_optimization(
        lmheads.transform))
    param_groups[1]['params'].append(lmheads.bias)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    # Use Adam.
    betas = (0.9, 0.999)
    optimizer = Adam(param_groups, betas=betas,
                     lr=args.lr, weight_decay=args.weight_decay)

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale':args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               min_lr=args.min_lr,
                               use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
                               override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


def setup_model_and_optimizer(args):
    """Setup model and optimizer."""

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
    else:
        args.iteration = 0

    return model, optimizer, lr_scheduler

def get_batch(data_iterator, timers):
    ''' get_batch subdivides the source data into chunks of
    length args.seq_length. If source is equal to the example
    output of the data loading example, with a seq_length limit
    of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the data loader. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM. A Variable representing an appropriate
    shard reset mask of the same dimensions is also returned.
    '''
    # Items and their type.
    keys = ['input_k', 'seg_id', 'target', 'perm_mask', 'target_mapping', 'input_q', 'target_mask']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
#     print("serialdata", data)
    if data is not None:
        for key in data.keys():
            print(key,"type is")
            print(data[key][0].dtype)
        timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    print("this is data pll",data_b)
    # Unpack.
    input_k = data_b['input_k'].long() #int64
    print("input_k: ", input_k.dtype)
    
#     seg_id = data_b['seg_id'].long() #int64, required-->int32
    seg_id = data_b['seg_id'].type(torch.int32) #int32
    print("seg_id: ", seg_id.dtype)
    
    target = data_b['target'].long()
    print("target: ", target.dtype)
    
    perm_mask = data_b['perm_mask'].float() #float32
    print("perm_mask: ", perm_mask.dtype)
    
#     target_mapping = data_b['target_mapping'].long() #int64, required-->float32
    target_mapping = data_b['target_mapping'].type(torch.float32)
    print("target_mapping: ", target_mapping.dtype)
    
#     input_q = data_b['input_q'].byte() #unit8, required-->float32
    input_q = data_b['input_q'].type(torch.float32)
    print("input_q: ", input_q.dtype)
    
    target_mask = data_b['target_mask'].type(torch.float32)
    print("target_mask: ", target_mask.dtype)

    return input_k, seg_id, target, perm_mask, target_mapping, input_q, target_mask


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, types, next_sentence, loss_mask, lm_labels, \
        padding_mask = get_batch(data_iterator, timers)
    timers('batch generator').stop()
    # Forward model.
    output, nsp = model(tokens, types, 1-padding_mask,
                        checkpoint_activations=args.checkpoint_activations)

    nsp_loss = F.cross_entropy(nsp.view(-1, 2).contiguous().float(),
                               next_sentence.view(-1).contiguous(),
                               ignore_index=-1)

    losses = mpu.vocab_parallel_cross_entropy(
        output.contiguous().float(), lm_labels.contiguous())
    loss_mask = loss_mask.contiguous()
    lm_loss = torch.sum(
        losses.view(-1) * loss_mask.view(-1).float()) / loss_mask.sum()

    return lm_loss, nsp_loss


def backward_step(optimizer, model, lm_loss, nsp_loss, args, timers):
    """Backward step."""

    # Total loss.
    loss = lm_loss + nsp_loss

    # Backward pass.
    optimizer.zero_grad()
    if args.fp16:
        optimizer.backward(loss, update_master_grads=False)
    else:
        loss.backward()

    # Reduce across processes.
    lm_loss_reduced = lm_loss
    nsp_loss_reduced = nsp_loss

    reduced_losses = torch.cat((lm_loss.view(1), nsp_loss.view(1)))
    torch.distributed.all_reduce(reduced_losses.data)
    reduced_losses.data = reduced_losses.data / args.world_size
    if args.DDP_impl == 'local':
        timers('allreduce').start()
        model.allreduce_params(reduce_after=False,
                               fp32_allreduce=args.fp32_allreduce)
        timers('allreduce').stop()
    lm_loss_reduced = reduced_losses[0]
    nsp_loss_reduced = reduced_losses[1]

    # Update master gradients.
    if args.fp16:
        optimizer.update_master_grads()

    # Clipping gradients helps prevent the exploding gradient.
    if args.clip_grad > 0:
        if not args.fp16:
            mpu.clip_grad_norm(model.parameters(), args.clip_grad)
        else:
            optimizer.clip_master_grads(args.clip_grad)

    return lm_loss_reduced, nsp_loss_reduced


def train_step(data_iterator, model, optimizer, lr_scheduler,
               args, timers):
    """Single training step."""

    # Forward model for one step.
    timers('forward').start()
    lm_loss, nsp_loss = forward_step(data_iterator, model,
                                     args, timers)
    timers('forward').stop()

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    lm_loss_reduced, nsp_loss_reduced = backward_step(optimizer, model, lm_loss,
                                                      nsp_loss, args, timers)
    timers('backward').stop()

    # Update parameters.
    timers('optimizer').start()
    optimizer.step()
    timers('optimizer').stop()

    # Update learning rate.
    skipped_iter = 0
    if not (args.fp16 and optimizer.overflow):
        lr_scheduler.step()
    else:
        skipped_iter = 1

    return lm_loss_reduced, nsp_loss_reduced, skipped_iter


def train(model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator, timers, args, writer):
    """Train the model."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0
    total_nsp_loss = 0.0

    # Iterations.
    iteration = args.iteration
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    while iteration < args.train_iters:

        lm_loss, nsp_loss, skipped_iter = train_step(train_data_iterator,
                                                     model,
                                                     optimizer,
                                                     lr_scheduler,
                                                     args, timers)
        skipped_iters += skipped_iter
        iteration += 1

        # Update losses.
        current_lm_loss = lm_loss.data.detach().float()
        current_nsp_loss = nsp_loss.data.detach().float()
        total_lm_loss += current_lm_loss
        total_nsp_loss += current_nsp_loss

        # Logging.

        if args.DDP_impl == 'torch':
            timers_to_log = ['forward', 'backward', 'optimizer',
                            'batch generator', 'data loader']
        else:
            timers_to_log = ['forward', 'backward', 'allreduce', 'optimizer',
                             'batch generator', 'data loader']

        learning_rate = optimizer.param_groups[0]['lr']

        if writer and args.rank == 0:
            writer.add_scalar('learning_rate', learning_rate, iteration)
            writer.add_scalar('lm_loss', current_lm_loss, iteration)
            writer.add_scalar('nsp_loss', current_nsp_loss, iteration)
            if args.fp16:
                writer.add_scalar('loss_scale', optimizer.loss_scale, iteration)
            normalizer = iteration % args.log_interval
            if normalizer == 0:
                normalizer = args.log_interval
            timers.write(timers_to_log, writer, iteration,
                         normalizer=normalizer)

        if iteration % args.log_interval == 0:
            avg_nsp_loss = total_nsp_loss.item() / args.log_interval
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            elapsed_time = timers('interval time').elapsed()
            if writer and args.rank == 0:
                writer.add_scalar('iteration_time',
                                  elapsed_time / args.log_interval, iteration)
            log_string = ' iteration {:8d}/{:8d} |'.format(iteration,
                                                            args.train_iters)
            log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
                elapsed_time * 1000.0 / args.log_interval)
            log_string += ' learning rate {:.3E} |'.format(learning_rate)
            log_string += ' lm loss {:.6E} |'.format(avg_lm_loss)
            log_string += ' nsp loss {:.6E} |'.format(avg_nsp_loss)
            if args.fp16:
                log_string += ' loss scale {:.1f} |'.format(
                    optimizer.loss_scale)
            print_rank_0(log_string)
            total_nsp_loss = 0.0
            total_lm_loss = 0.0
            if report_memory_flag:
                report_memory('after {} iterations'.format(iteration))
                report_memory_flag = False
            timers.log(timers_to_log, normalizer=args.log_interval)

        # Autoresume
        if (iteration % args.adlr_autoresume_interval == 0) and args.adlr_autoresume:
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              lr_scheduler, args)

        # Checkpointing
        if args.save and args.save_interval and iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, val_data_iterator, model, args,
                                       writer, iteration, timers, False)

        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print('rank: {} | time: {} | exiting the program at iteration {}'.
                  format(rank, time_str, iteration), flush=True)
            exit()

    return iteration, skipped_iters

def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)

        
def get_train_val_test_data(args):
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""

    (train_data, val_data, test_data) = (None, None, None)

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        data_config = configure_data()
        data_config.set_defaults(data_set_type='XLNET', transpose=False)
        (train_data, val_data, test_data), tokenizer = data_config.apply(args)
        before = tokenizer.num_tokens
        after = before
        multiple = args.make_vocab_size_divisible_by * \
                   mpu.get_model_parallel_world_size()
        while (after % multiple) != 0:
            after += 1
        print_rank_0('> padded vocab (size: {}) with {} dummy '
                     'tokens (new size: {})'.format(
                         before, after - before, after))
        # Need to broadcast num_tokens and num_type_tokens.
        token_counts = torch.cuda.LongTensor([after,
                                              tokenizer.num_type_tokens,
                                              int(args.do_train), int(args.do_valid), int(args.do_test)])
    else:
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0])
    
    # Broadcast num tokens.
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    num_tokens = token_counts[0].item()
    num_type_tokens = token_counts[1].item()
    args.do_train = token_counts[2].item()
    args.do_valid = token_counts[3].item()
    args.do_test = token_counts[4].item()

    return train_data, val_data, test_data, num_tokens, num_type_tokens

def main():
    """Main training program."""
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False
    
    # Timer.
    timers = Timers()
    
    # Arguments.
    args = get_args()
    writer = None
    # Pytorch distributed.
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        print('Pretrain XLNet model')
        print_args(args)
    
    # Random seeds for reproducability.
    set_random_seed(args.seed)
    
    train_data, val_data, test_data, args.tokenizer_num_tokens, \
        args.tokenizer_num_type_tokens = get_train_val_test_data(args)
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)

    if args.resume_dataloader:
        if train_data is not None:
            train_data.batch_sampler.start_iter = args.iteration % \
                                                  len(train_data)
            print_rank_0('setting training data start iteration to {}'.
                         format(train_data.batch_sampler.start_iter))
#         if val_data is not None:
#             start_iter_val = (args.iteration // args.eval_interval) * \
#                              args.eval_iters
#             val_data.batch_sampler.start_iter = start_iter_val % \
#                                                 len(val_data)
#             print_rank_0('setting validation data start iteration to {}'.
#                          format(val_data.batch_sampler.start_iter))

    if train_data is not None:
        train_data_iterator = iter(train_data)
#         print("----pretrain_bert.py/main>ln(617..)----")
#         print("train_data_iterator: ", train_data_iterator)
    else:
        train_data_iterator = None
    if val_data is not None:
        val_data_iterator = iter(val_data)
    else:
        val_data_iterator = None

    iteration = 0
    if args.train_iters > 0:
        if args.do_train:
            iteration, skipped = train(model, optimizer,
                                       lr_scheduler,
                                       train_data_iterator,
                                       val_data_iterator,
                                       timers, args, writer)#removing argument writer
        if args.do_valid:
            prefix = 'the end of training for val data'
            val_loss = evaluate_and_print_results(prefix, val_data_iterator,
                                                  model, args, writer, iteration,
                                                  timers, False)
    
    
if __name__ == "__main__":
    main()
    