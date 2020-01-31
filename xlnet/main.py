# -*- coding: utf-8 -*-

"""
Copyright 2019 Tae Hwan Jung

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import Timers
import time

from utils import print_args
from utils import print_params_min_max_norm
from utils import print_rank_0
from utils import enable_adlr_autoresume
from utils import check_adlr_autoresume_termination

from arguments import get_args
from configure_data import configure_data
import random
import torch.nn.functional as F

from model import get_params_for_weight_decay_optimization#
from model import gpt2_get_params_for_weight_decay_optimization#
from model import DistributedDataParallel as LocalDDP

import os
import data_utils
import argparse
import xlnet
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer
from model import DistributedDataParallel as LocalDDP
import mpu 
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'


def get_model(args):
    """Build the model."""

    print_rank_0('building XLNet model ...')
    model = xlnet.XLNet(n_token=args.vocab_len, n_layer=6, n_head=4, d_head=8,
                        d_inner=32, d_model=32,
                        dropout=0.1, dropatt=0.1,
                        attn_type="bi", bi_data=args.bi_data,
                        clamp_len=-1, same_length=False,
                        reuse_len=args.reuse_len, mem_len=args.mem_len)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

#     Fp16 conversion.
#     if args.fp16:
#         model = FP16_Module(model)
#         if args.fp32_embedding:
#             model.module.model.bert.embeddings.word_embeddings.float()
#             model.module.model.bert.embeddings.position_embeddings.float()
#             model.module.model.bert.embeddings.token_type_embeddings.float()
#         if args.fp32_tokentypes:
#             model.module.model.bert.embeddings.token_type_embeddings.float()
#         if args.fp32_layernorm:
#             for name, _module in model.named_modules():
#                 if 'LayerNorm' in name:
#                     _module.float()

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
#     timers('data loader').start()
    if data_iterator is not None:
        print("data_iterator: ", data_iterator)
        data = next(data_iterator)
    else:
        data = None
#     print("serialdata", data)
    if data is not None:
        for key in data.keys():
            print(key,"type is")
            print(data[key][0].dtype)
#         timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    
#     print("data_b: ", data_b)

    # Unpack.
#     print("this is data pll",data_b)
    # Unpack.
    input_k = data_b['input_k'].long() #int64
#     print("input_k: ", input_k.dtype)
    
#     seg_id = data_b['seg_id'].long() #int64, required-->int32
    seg_id = data_b['seg_id'].type(torch.int32) #int32
#     print("seg_id: ", seg_id.dtype)
    
    target = data_b['target'].long()
#     print("target: ", target.dtype)
    
    perm_mask = data_b['perm_mask'].float() #float32
#     print("perm_mask: ", perm_mask.dtype)
    
#     target_mapping = data_b['target_mapping'].long() #int64, required-->float32
    target_mapping = data_b['target_mapping'].type(torch.float32)
#     print("target_mapping: ", target_mapping.dtype)
    
#     input_q = data_b['input_q'].byte() #unit8, required-->float32
    input_q = data_b['input_q'].type(torch.float32)
#     print("input_q: ", input_q.dtype)
    
    target_mask = data_b['target_mask'].type(torch.float32)
#     print("target_mask: ", target_mask.dtype)

    return input_k, seg_id, target, perm_mask, target_mapping, input_q, target_mask

def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    print("device=",device)
    if args.local_rank is not None:
        device = args.local_rank
        print("local",args.local_rank)
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
        

# def print_args(args, writer=None):
#     """Print arguments."""

#     print('arguments:', flush=True)
#     for arg in vars(args):
#         dots = '.' * (29 - len(arg))
#         print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)

#         if writer:
#             writer.add_text(arg, str(getattr(args, arg)))

# def print_rank_0(message):
#     if torch.distributed.is_initialized():
#         if torch.distributed.get_rank() == 0:
#             print(message, flush=True)
#     else:
#         print(message, flush=True)

def get_train_val_test_data(args):
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""

    (train_data, val_data, test_data) = (None, None, None)

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        data_config = configure_data()
        data_config.set_defaults(data_set_type='XLNET', transpose=False)
        (train_data, val_data, test_data), tokenizer = data_config.apply(args)
        print(tokenizer)
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
        
        token_counts = torch.cuda.LongTensor([after, before,
                                              tokenizer.num_type_tokens,
                                              int(args.do_train), int(args.do_valid), int(args.do_test)])
    else:
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0, 0])
    
    # Broadcast num tokens.
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    num_tokens = token_counts[0].item()
    vocab_len = token_counts[1].item()
    num_type_tokens = token_counts[2].item()
    args.do_train = token_counts[3].item()
    args.do_valid = token_counts[4].item()
    args.do_test = token_counts[5].item()

    return train_data, val_data, test_data, num_tokens, num_type_tokens ,vocab_len

def main():
    
     # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()
    
    # Arguments.
    args = get_args()
    
    writer = None
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        print('Pretrain GPT2 model')
        print_args(args, writer)
        
    # Random seeds for reproducability.
    set_random_seed(args.seed)
    
    train_data, val_data, test_data, args.tokenizer_num_tokens, \
        args.tokenizer_num_type_tokens, args.vocab_len = get_train_val_test_data(args)

    # Autoresume.
    print("barrier ahead")
    torch.distributed.barrier()
    print("barrier done")
    
#     sp = BertTokenizer.from_pretrained(args.tokenizer)
    model = get_model(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    print(torch.cuda.current_device())
    
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
        
#     iteration = 0
#     if args.train_iters > 0:
#         if args.do_train:
#             iteration, skipped = train(model, optimizer,
#                                        lr_scheduler,
#                                        train_data_iterator,
#                                        val_data_iterator,
#                                        timers, args, writer)
#         if args.do_valid:
#             prefix = 'the end of training for val data'
#             val_loss = evaluate_and_print_results(prefix, val_data_iterator,
#                                                   model, args, writer, iteration,
#                                                   timers, False)

    for num_step in range(args.num_epoch):
        mems = None
        
        timers('batch generator').start()
        input_k, seg_id, target, perm_mask, target_mapping, input_q, target_mask = get_batch(train_data_iterator, timers)
        timers('batch generator').stop()
        start = time.time()

        inp_k = input_k.permute(1,0)
#         print("input_k: ", input_k.size())#[1, 512]
         # [seq_len, 1(=bsz)]
        seg_id =  seg_id.permute(1,0)# [seq_len, 1(=bsz)]
        target =  target.permute(1,0)# [num_predict, 1(=bsz)]
#         print("########",perm_mask.shape,"perm_mask",target_mapping.shape,"targetmapp",target_mask.shape,"tgt_mask",sep="\n")
        
        perm_mask =perm_mask.permute(2,1,0) # [seq_len, seq_len, 1(=bsz)]
        target_mapping = target_mapping.permute(1, 2, 0)#[num_predict, seq_len, 1(=bsz)], 
        inp_q = input_q.permute(1,0) # [seq_len, 1(=bsz)]
        tgt_mask = target_mask.permute(1,0)# [num_predict, 1(=bsz)]
#         print("########",perm_mask.shape,"perm_mask",target_mapping.shape,"targetmapp",tgt_mask.shape,"tgt_mask",sep="\n")

#         features = data_utils._create_data(sp=sp,
#                                            input_paths=args.data,
#                                            seq_len=args.seq_len,
#                                            reuse_len=args.reuse_len,
#                                            bi_data=args.bi_data,
#                                            num_predict=args.num_predict,
#                                            mask_alpha=args.mask_alpha,
#                                            mask_beta=args.mask_beta)

#         num_step = 0
#         for feature in features:

#             permutation = data_utils.make_permute(feature,
#                                                   reuse_len=args.reuse_len,
#                                                   seq_len=args.seq_len,
#                                                   perm_size=args.perm_size,
#                                                   num_predict=args.num_predict)
#             keys = ['seg_id']
#             datatype = torch.int32
#             permutations = mpu.broadcast_data(keys,permutation,datatype)
#             key = ['input_k', 'target']
#             segu = mpu.broadcast_data(key,permutation,torch.int64)
#             keyf32 = ['perm_mask','target_mapping', 'input_q','target_mask']
#             f32u = mpu.broadcast_data(keyf32,permutation,torch.float32)
            
#             #########################################################################3
#             # batch size is 1
#             inp_k = segu['input_k'].unsqueeze(-1) # [seq_len, 1(=bsz)]
# #             print("inp_k, inp_k.type: ", inp_k, type(inp_k), inp_k.dtype)
#             seg_id = permutations['seg_id'].unsqueeze(-1) # [seq_len, 1(=bsz)]
# #             print("seg_id, seg_id.type: ", seg_id, type(seg_id), seg_id.dtype)
#             target = segu['target'].unsqueeze(-1) # [num_predict, 1(=bsz)]
#             perm_mask = f32u['perm_mask'].unsqueeze(-1) # [seq_len, seq_len, 1(=bsz)]
# #             print("perm_mask: ", perm_mask, type(perm_mask), perm_mask.dtype)
#             target_mapping = \
#                 f32u['target_mapping'].unsqueeze(-1) # [num_predict, seq_len, 1(=bsz)]
# #             print("target_mapping: ", target_mapping, type(target_mapping), target_mapping.dtype)
#             inp_q = f32u['input_q'].unsqueeze(-1) # [seq_len, 1(=bsz)]
# #             print("inp_q: ", inp_q, type(inp_q), inp_q.dtype)
#             tgt_mask = f32u['target_mask'].unsqueeze(-1) # [num_predict, 1(=bsz)]
#             ###############################################################################

        logits, new_mems = model(inp_k=inp_k, seg_id=seg_id, input_mask=None,
              mems=mems, perm_mask=perm_mask,
              target_mapping=target_mapping, inp_q=inp_q)

        #lm_loss = criterion(logits.transpose(1, 2), target).type(torch.float32)
        #####changed loss
        lm_loss = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(),
                              target)
        tgt_mask_sum = tgt_mask.reshape(-1).sum()
        lm_loss_sum = (lm_loss * tgt_mask).reshape(-1).sum()

        optimizer.zero_grad()
        total_loss = lm_loss_sum / tgt_mask_sum
        print('Number of  %d Step' %  (num_step + 1),'cost =', '{:.6f}'.format(total_loss))

        total_loss.backward()

        # Reduce across processes.
#         lm_loss_reduced = total_loss

#         reduced_losses = total_loss.view(1)
#         torch.distributed.all_reduce(reduced_losses.data)
#         reduced_losses.data = reduced_losses.data / args.world_size
#         if args.DDP_impl == 'local':
#             timers('allreduce').start()
#             model.allreduce_params(reduce_after=False,
#                                    fp32_allreduce=args.fp32_allreduce)
#             timers('allreduce').stop()
#         lm_loss_reduced = reduced_losses

#         # Update master gradients.
#         if args.fp16:
#             optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
#         if args.clip_grad > 0:
#             if not args.fp16:
#                 mpu.clip_grad_norm(model.parameters(), args.clip_grad)
#             else:
#                 optimizer.clip_master_grads(args.clip_grad)

        #num_step += 1


        optimizer.step()

        mems = new_mems
        now=time.time()
        print("#################")
        print("@@@@@@@@@@@@@@@@")
        print("this num_step",num_step+1,"took",now-start)
        print("#################")
        print("@@@@@@@@@@@@@@@@")
            
main()