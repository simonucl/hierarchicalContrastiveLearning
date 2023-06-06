'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import os
import sys
import argparse
import torch
# from torch.utils.data import DataLoader
# from data_processing.generate_dataset import DatasetCategory
from data_processing.hierarchical_dataset import BertDataset_rcv, HierarchicalBatchSampler
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel

import pickle
import json

from torch.optim import lr_scheduler
from util.util import adjust_learning_rate, warmup_learning_rate, TwoCropTransform
from losses.losses import HMLC
from network.resnet_modified import LinearClassifier
import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import time
import shutil
import math
import builtins
from tqdm import tqdm
import sys
import numpy as np

sys.path.append('../')
from supContrativeHMTC.utils import get_hierarchy_info

def parse_option():
    parser = argparse.ArgumentParser(description='Training/finetuning on Deep Fashion Dataset')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset, the superset of train/val')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train-listfile', default='', type=str,
                        help='training file with annotation')
    parser.add_argument('--val-listfile', default='', type=str,
                        help='validation file with annotation')
    parser.add_argument('--class-map-file', default='', type=str,
                        help='class mapping between str and int', required=True)
    parser.add_argument('--class-seen-file', default='', type=str,
                        help='seen classes text file. Used for seen/unseen split experiments.')
    parser.add_argument('--class-unseen-file', default='', type=str,
                        help='unseen classes text file. Used for seen/unseen split experiments.')
    parser.add_argument('--repeating-product-file', default='', type=str,
                        help='repeating product ids file')
    parser.add_argument('--mode', default='train', type=str,
                        help='Train or val')
    parser.add_argument('--input-size', default=224, type=int,
                        help='input size')
    parser.add_argument('--scale-size', default=256, type=int,
                        help='scale size in validation')
    parser.add_argument('--crop-size', default=224, type=int,
                        help='crop size')
    parser.add_argument('--num-classes', type=int,
                        help='number of classes')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 512)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                        help='use pre-trained model')
    parser.add_argument('--feature-extract', action='store_false',
                        help='When flase, finetune the whole model; else only update the reshaped layer para')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,60,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    #other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--loss', type=str, default='hmce',
                        help='loss type', choices=['hmc', 'hce', 'hmce'])
    parser.add_argument('--tag', type=str, default='',
                        help='tag for model name')
    parser.add_argument('--device', type=str, default='cuda',)
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    # warm-up for large-batch training,
    if args.batch_size >= 256:
        args.warm = True
    if args.warm:
        args.model_name = '{}_warm'.format(args.model)
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    return args

best_prec1 = 0

def main():
    global args, best_prec1
    args = parse_option()

    args.save_folder = './model'
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    args.tb_folder = './tensorboard'
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    args.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_loss_{}_trial_{}'.\
        format('bert-base-uncased', 'dataset', args.model, args.learning_rate,
               args.lr_decay_rate, args.batch_size, args.loss, 5)
    if args.tag:
        args.model_name = args.model_name + '_tag_' + args.tag
    args.tb_folder = os.path.join(args.tb_folder, args.model_name)
    args.save_folder = os.path.join(args.save_folder, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    # distributed training
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print(args.multiprocessing_distributed)
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        print("Adopting distributed multi processing training")
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    print("GPU in main worker is {}".format(gpu))
    args.gpu = gpu
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            print("In the process of multi processing with rank as {}".format(args.rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    # load data first 

    # TODO check the data is right or not
    data_path = args.data
    if 'nyt' in data_path:
        label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
        # label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}

        num_class = len(label_dict)
    
    # load the new label dict
    
        with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
            new_label_dict = pickle.load(f)

        # hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'nyt.taxonomy'))
        depths = [len(l.split('/')) - 1 for l in new_label_dict.values()]

    elif 'rcv' in data_path:
        # hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'rcv1.taxonomy'))
        with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
            label_dict = pickle.load(f)

        new_label_dict = label_dict

        r_hiera = {new_label_dict[_label_dict[k]]: v if (v == 'Root') else new_label_dict[_label_dict[v]] for k, v in r_hiera.items()}

        # {label_name: label_id}
        label_dict = {v: k for k, v in _label_dict.items()}
        num_class = len(label_dict)
        
        # rcv_label_map = pd.read_csv(os.path.join(data_path, 'rcv1_v2_topics_desc.csv'))
        # rcv_label_amp = dict(zip(rcv_label_map['topic_code'], rcv_label_map['topic_name']))

        # new_label_dict = {k: rcv_label_amp[v] for k, v in label_dict.items()}
        depths = [label_depth[name] for id, name in label_dict.items()]

    elif 'bgc' in data_path:
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'bgc.taxonomy'))
        label_dict = {v: k for k, v in _label_dict.items()}
        new_label_dict = label_dict
        num_class = len(label_dict)
        label_dict =_label_dict
        depths = list(label_depth.values())
    elif 'patent' in data_path:
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'patent.taxonomy'))
        with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
            label_dict = pickle.load(f)
        label_dict = {v: k for k, v in _label_dict.items()}
        new_label_dict = label_dict

        num_class = len(label_dict)
        depths = list(label_depth.values())
    elif 'aapd' in data_path:
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'aapd.taxonomy'))
        with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
            label_dict = pickle.load(f)
        # label_dict = {v: k for k, v in label_dict.items()}
        new_label_dict = {v: k for k, v in label_dict.items()}
        num_class = len(label_dict)
        depths = [label_depth[name] for id, name in new_label_dict.items()]
    elif 'wos' in data_path:
        # hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'wos.taxonomy'))
        with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
            label_dict = pickle.load(f)
        label_dict = {v: k for k, v in label_dict.items()}
        new_label_dict = label_dict
        num_class = len(label_dict)
        depths = [label_depth[name] for id, name in label_dict.items()]

    args.depths = torch.tensor(depths)

    def get_path(label):
        path = []
        # label_name = label_dict[label]
        while label != 'Root':
            path.insert(0, label)
            label = r_hiera[label]
        return path
    
    if ('nyt' in data_path):
        label_path = {k: get_path(v) for k, v in label_dict.items()}
    elif ('rcv' in data_path):
        label_path = {k: get_path(k) for k, v in _label_dict.items()}
    elif ('bgc' in data_path):
        label_path = {k: get_path(k) for k, v in _label_dict.items()}
    elif ('aapd' in data_path):
        label_path = {k: get_path(k) for k, v in label_dict.items()}
    else:
        label_path = {v: get_path(k) for k, v in label_dict.items()}
    # label_path = {k: self.get_path(v) for k, v in label_dict.items()}
    depth_label_path = {}
    for label in label_path:
        depth = len(label_path[label])
        if depth not in depth_label_path:
            depth_label_path[depth] = {}
        depth_label_path[depth][label] = label_path[label]

    args.label_path = label_path
    args.depth_label_path = depth_label_path

    # create model
    print("=> creating model '{}'".format(args.model))
    model, criterion, tokenizer = set_model_bert(ngpus_per_node, args)

    num_classess = len(label_dict)
    dataloader, sampler = load_data(args.data, args.train_listfile, args.val_listfile, label_dict, args.batch_size, tokenizer)

    args.classifier = LinearClassifier(name=args.model, num_classes=num_classess).cuda(args.gpu)
    # TODO disable or modify the function below, as it freeze some resnet parameters
    set_parameter_requires_grad(model, args.feature_extract)
    optimizer = setup_optimizer(model, args.learning_rate,
                                   args.momentum, args.weight_decay,
                                   args.feature_extract)
    cudnn.benchmark = True

    # dataloaders_dict, sampler = load_deep_fashion_hierarchical(args.data, args.train_listfile,
    #                              args.val_listfile, args.class_map_file, args.repeating_product_file,
    #                              args)

    # train_sampler, val_sampler = sampler['train'], sampler['val']
    for epoch in range(1, args.epochs + 1):
        print('Epoch {}/{}'.format(epoch, args.epochs + 1))
        print('-' * 10)
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        #     val_sampler.set_epoch(epoch)
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train(dataloader['train'], tokenizer, model, criterion, optimizer, epoch, args, logger)
        output_file = args.save_folder + '/checkpoint_{:04d}.pth.tar'.format(epoch)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False,
                filename=output_file)

def set_model_bert(ngpus_per_node, args):
    print(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BertModel.from_pretrained(args.model)

    criterion = HMLC(temperature=args.temp, loss_type=args.loss, layer_penalty=torch.exp,
                     label_depths=args.depths)

    if args.ckpt is not None:
        # load the pretrained model for the second training
        pass

    # TODO implement distributed training
    # if args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)
    #     model = model.cuda(args.gpu)

    model = model.to(args.device)
    criterion = criterion.to(args.device)

    return model, criterion, tokenizer

# def set_model(ngpus_per_node, args):
#     model = resnet_modified.MyResNet(name='resnet50')
#     criterion = HMLC(temperature=args.temp, loss_type=args.loss, layer_penalty=torch.exp)

#     # This part is to load a pretrained model
#     ckpt = torch.load(args.ckpt, map_location='cpu')
#     state_dict = ckpt['state_dict']
#     # state_dict = ckpt
#     model_dict = model.state_dict()
#     new_state_dict = {}
#     # for k, v in state_dict.items():
#     #     if not k.startswith('module.encoder_q.fc'):
#     #         k = k.replace('module.encoder_q', 'encoder')
#     #         new_state_dict[k] = v
#     for k, v in state_dict.items():
#         if not k.startswith('module.head'):
#             k = k.replace('module.encoder', 'encoder')
#             new_state_dict[k] = v
#     state_dict = new_state_dict
#     model_dict.update(state_dict)
#     model.load_state_dict(model_dict)

#     state_dict = None
#     if args.distributed:
#         # For multiprocessing distributed, DistributedDataParallel constructor
#         # should always set the single device scope, otherwise,
#         # DistributedDataParallel will use all available devices.
#         print("GPU setting", args.gpu)
#         if args.gpu is not None:
#             torch.cuda.set_device(args.gpu)
#             model.cuda(args.gpu)
#             # When using a single GPU per process and per
#             # DistributedDataParallel, we need to divide the batch size
#             # ourselves based on the total number of GPUs we have
#             args.batch_size = int(args.batch_size / ngpus_per_node)
#             print("Updated batch size is {}".format(args.batch_size))
#             args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
#             # There is memory issue in data loader
#             # args.workers = 0
#             model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
#         else:
#             model.cuda()
#             # DistributedDataParallel will divide and allocate batch_size to all
#             # available GPUs if device_ids are not set
#             model = torch.nn.parallel.DistributedDataParallel(model)
#             print('Loading state dict from ckpt')
#             model.load_state_dict(state_dict)
#     elif args.gpu is not None:
#         torch.cuda.set_device(args.gpu)
#         model = model.cuda(args.gpu)
#         # comment out the following line for debugging
#         # raise NotImplementedError("Only DistributedDataParallel is supported.")
#     else:
#         # AllGather implementation (batch shuffle, queue update, etc.) in
#         # this code only supports DistributedDataParallel.
#         raise NotImplementedError("Only DistributedDataParallel is supported.")

#     criterion = criterion.cuda(args.gpu)

#     return model, criterion

def train(train_data, tokenizer, model, criterion, optimizer, epoch, args, logger):
    """
    one epoch training
    """

    classifier = args.classifier
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    micro_f1 = AverageMeter('Acc@1', ':6.2f')
    macro_f1 = AverageMeter('Acc@5', ':6.2f')

    end = time.time()

    # Each epoch has a training and/or validation phase
    for phase in ['train']:
        if phase == 'train':
            progress = ProgressMeter(len(train_data),
                [batch_time, data_time, losses, micro_f1, macro_f1],
                prefix="Epoch: [{}]".format(epoch))
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode
        classifier.eval()

        # Iterate over data.
        for idx, data in tqdm(enumerate(train_data), total=len(train_data)):
            # images.shape [shape([1, 260, 3, 224, 224]), shape([1, 260, 3, 224, 224])]
            # labels.shape [shape([1, 260, 4])]
            input, labels, _ = data
            input = input.to(args.device)
            labels = labels.to(args.device)

            data_time.update(time.time() - end)

            labels = labels.squeeze()
            bsz = labels.shape[0] #batch size
            if phase == 'train':
                warmup_learning_rate(args, epoch, idx, len(train_data), optimizer)
            # tokens = tokenizer(input, padding=True, truncation=True, return_tensors="pt")

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                # Get model outputs and calculate loss
                bert_out = model(input) # [batch_size, hidden_size]

                features = bert_out.pooler_output # [batch_size, hidden_size]
                print(features.shape)

                features = features.unsqueeze(1)

                loss = criterion(features, labels)
                losses.update(loss.item(), bsz)

                # top1.update(accuracy(features, labels))
                # top5.update(accuracy(features, labels, topk=(5,)))
                # backward + optimize only if in training phase
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            sys.stdout.flush()
            if idx % args.print_freq == 0:
                progress.display(idx)
                print(loss)
        logger.log_value('loss', losses.avg, epoch)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_data(root_dir, train_listfile, val_listfile, label_dict, batch_size, tokenizer, device="cuda:0"):
    # Data augmentation and normalization
    # Create training and validation datasets
    train_dataset = BertDataset_rcv(data_path=os.path.join(root_dir, train_listfile),
                                    device=device, pad_idx=tokenizer.pad_token_id, label_dict=label_dict,
                                    label_path=args.label_path)
    val_dataset = BertDataset_rcv(data_path=os.path.join(root_dir, val_listfile),
                                    device=device, pad_idx=tokenizer.pad_token_id, label_dict=label_dict,
                                    label_path=args.label_path)
    # train_dataset = DatasetCategory(root_dir, 'train', train_listfile, val_listfile, '',
    #                                 class_map_file, class_seen_file,
    #                                 class_unseen_file, TwoCropTransform(data_transforms['train']))
    # val_dataset = DatasetCategory(root_dir, 'val', train_listfile, val_listfile, '',
    #                               class_map_file, class_seen_file,
    #                               class_unseen_file, TwoCropTransform(data_transforms['val']))
    image_datasets = {'train': train_dataset,
                      'val': val_dataset}
    print("Initializing Datasets and Dataloaders...")

    # if distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    # else:

    ## TODO implement this
    train_sampler = HierarchicalBatchSampler(batch_size=batch_size,
                                       drop_last=False,
                                       dataset=train_dataset)
    val_sampler = HierarchicalBatchSampler(batch_size=batch_size,
                                           drop_last=False,
                                           dataset=val_dataset)
    # train_sampler = None
    # val_sampler = None
    
    sampler = {'train': train_sampler,
                'val': val_sampler}

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                       sampler=sampler[x], drop_last=False,
                                       collate_fn=image_datasets[x].collate_fn_1)
        for x in ['train', 'val']}
    return dataloaders_dict, sampler


# def load_deep_fashion_hierarchical(root_dir, train_list_file, val_list_file, class_map_file, repeating_product_file, opt):
#     train_transform = transforms.Compose([
#             transforms.RandomResizedCrop(size=opt.input_size, scale=(0.8, 1.)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.4)
#             ], p=0.8),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ])
#     val_transform = transforms.Compose([transforms.ToTensor(),
#                                        transforms.Normalize([0.485, 0.456, 0.406], [
#                                                             0.229, 0.224, 0.225]),
#                                        ])
#     train_dataset = DeepFashionHierarchihcalDataset(os.path.join(root_dir, train_list_file),
#                                                     os.path.join(root_dir, class_map_file),
#                                                     os.path.join(root_dir, repeating_product_file),
#                                                     transform=TwoCropTransform(train_transform))
    
#     val_dataset = DeepFashionHierarchihcalDataset(os.path.join(root_dir, val_list_file),
#                                                   os.path.join(
#                                                       root_dir, class_map_file),
#                                                   os.path.join(
#                                                       root_dir, repeating_product_file),
#                                                   transform=TwoCropTransform(val_transform))
#     print('LENGTH TRAIN', len(train_dataset))
#     image_datasets = {'train': train_dataset,
#                       'val': val_dataset}
#     train_sampler = HierarchicalBatchSampler(batch_size=opt.batch_size,
#                                        drop_last=False,
#                                        dataset=train_dataset)
#     val_sampler = HierarchicalBatchSampler(batch_size=opt.batch_size,
#                                            drop_last=False,
#                                            dataset=val_dataset)
#     sampler = {'train': train_sampler,
#                'val': val_sampler}
#     print(opt.workers, "workers")
#     dataloaders_dict = {
#         x: torch.utils.data.DataLoader(image_datasets[x], sampler=sampler[x],
#                                        num_workers=opt.workers, batch_size=1,
#                                        pin_memory=True)
#         for x in ['train', 'val']}
#     return dataloaders_dict, sampler


def setup_optimizer(model_ft, lr, momentum, weight_decay, feature_extract):
    # Send the model to GPU
    # model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer_ft

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        # Select which params to finetune
        # for param in model.parameters():
        #     param.requires_grad = True
        # Freeze all the parameters in the model
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the last 4 layers
        for param in model.encoder.layer[-4:].parameters():
            param.requires_grad = True

        # Unfreeze the pooler layer
        for param in model.pooler.parameters():
            param.requires_grad = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f


def evaluate(epoch_predicts, epoch_labels, id2label, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    # label2id = vocab.v2i['label']
    # id2label = vocab.i2v['label']
    # epoch_gold_label = list()
    # # get id label name of ground truth
    # for sample_labels in epoch_labels:
    #     sample_gold = []
    #     for label in sample_labels:
    #         assert label in id2label.keys(), print(label)
    #         sample_gold.append(id2label[label])
    #     epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(id2label))] for _ in range(len(id2label))]
    right_count_list = [0 for _ in range(len(id2label))]
    gold_count_list = [0 for _ in range(len(id2label))]
    predicted_count_list = [0 for _ in range(len(id2label))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        label = label + '_' + str(i)
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'full': [precision_dict, recall_dict, fscore_dict, right_count_list, predicted_count_list, gold_count_list]}


if __name__ == '__main__':
    main()
    pass
