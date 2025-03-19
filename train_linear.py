'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''

from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import pickle
import csv
import numpy as np
from util import *

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch Self-Disitllation')

# Datasets
parser.add_argument('-d', '--dataset', default='caltech101', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--train-batch', default=128, type=int, metavar='N', help='train batchsize')
parser.add_argument('--test-batch', default=200, type=int, metavar='N', help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', type=float, default=0.98, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')

# Architecture
parser.add_argument('--depth', type=int, default=34, help='model depth')

# Miscs
parser.add_argument('--manualSeed', default=0, type=int, help='manual seed')

# Device options
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

# Distillation
parser.add_argument('--dist', default=0, type=int, metavar='DIST', help='distillation count')
parser.add_argument('--last', default=0, type=int, metavar='LAST', help='use best/last checkpoint of teacher model')
parser.add_argument('--ctype', default='cc', type=str, help='corruption type')
parser.add_argument('--corruption', default=0.5, type=float, metavar='CORR', help='label corruption ratio (default=0)')
parser.add_argument('--partial', default=False, type=bool, metavar='PARTIAL', help='partial student model')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
    
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
np.random.seed(args.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LinearSoftmaxModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearSoftmaxModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x):
        x = self.linear(x)
        return x

class FeatureDataset(data.Dataset):
    def __init__(self, features, targets, corrupted_target):
        super(FeatureDataset, self).__init__()
        
        self.features = features
        self.targets = targets
        self.corrupted_target = corrupted_target
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.corrupted_target[idx]
    

def main():
    
    if args.partial:
        args.checkpoint = os.path.join(args.checkpoint, args.dataset, str(args.lr), str(args.weight_decay), args.ctype, str(int(100*args.corruption)), 'p')
    else:
        args.checkpoint = os.path.join(args.checkpoint, args.dataset, str(args.lr), str(args.weight_decay), args.ctype, str(int(100*args.corruption)), str(args.dist))
        
    print(f'CKPT PATH: {args.checkpoint}')
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    print(f'==> Preparing feature {args.dataset} with resnet-{args.depth}')
    
    train_feature, train_label, test_feature, test_label = get_feature(args.dataset, args.depth)
    feature_mean = torch.mean(train_feature, dim=0)
    
    num_classes = len(list(set(train_label)))
    model = LinearSoftmaxModel(train_feature.size(1), num_classes)
    model.cuda()

    target_dir = os.path.join('./data/Targets/', args.dataset, args.ctype)
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, str(int(args.corruption*100))+ '.pkl')
    
    corrupted_label = get_corrupted_target(train_label, args.dataset, args.ctype, args.corruption)
    
    cnt = 0
    for idx in range(len(corrupted_label)):
        if corrupted_label[idx] != train_label[idx]:
            cnt += 1
    print(f'corruption ratio: {cnt/len(corrupted_label):.2f}')
    
    trainset = FeatureDataset(train_feature, train_label, corrupted_label)
    testset = FeatureDataset(test_feature, test_label, test_label)
    
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
    if args.dist == 0 and args.partial is False:
        teacher = None
    
    elif args.partial is True:
        if args.last:
            teacher_path = os.path.join(args.checkpoint[:-1]+str(0), 'last.ckpt')
        else:
            teacher_path = os.path.join(args.checkpoint[:-1]+str(0), 'best.ckpt')

        print(f'Teacher PATH: {teacher_path}')
        teacher = LinearSoftmaxModel(train_feature.size(1), num_classes)
        teacher = teacher.cuda()
        teacher_checkpoint = torch.load(teacher_path)
        teacher.load_state_dict(teacher_checkpoint['state_dict'])
        model.load_state_dict(teacher_checkpoint['state_dict'])
    else:
        if args.last:
            teacher_path = os.path.join(args.checkpoint[:-1]+str(args.dist-1), 'last.ckpt')
            
        else:
            teacher_path = os.path.join(args.checkpoint[:-1]+str(args.dist-1), 'best.ckpt')
        
        teacher = LinearSoftmaxModel(train_feature.size(1), num_classes)
        teacher = teacher.cuda()
        teacher_checkpoint = torch.load(teacher_path)
        teacher.load_state_dict(teacher_checkpoint['state_dict'])
        
    if args.partial:
        criterion = partial_loss
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        
    # Train and val
    best_test_acc = 0  
    best_train_acc = 0 
    
    for epoch in range(args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, teacher, criterion, optimizer, epoch, use_cuda, args.partial)
        test_loss, test_acc = test(testloader, model, teacher, criterion, epoch, use_cuda, args.partial)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_test_acc
        best_test_acc = max(test_acc, best_test_acc)
        best_train_acc = max(train_acc, best_train_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_test_acc': best_test_acc,
                'optimizer' : optimizer.state_dict(),
            }, False, is_best, checkpoint=args.checkpoint)
        
        adjust_learning_rate(optimizer, epoch)
        
    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_test_acc': best_test_acc,
                'optimizer' : optimizer.state_dict(),
            }, True, False, checkpoint=args.checkpoint)
    
    logger.close()
    
    fieldnames = ['Dataset', 'LR', 'WD', 'Corruption', 'Distill', 'Train Acc.', 'Test Acc.', 'Final Acc.']
    row = {'Dataset':args.dataset,
           'LR':args.lr,
           'WD':args.weight_decay,
           'Corruption':args.corruption,
           'Distill':'p' if args.partial else args.dist,
           'Train Acc.': best_train_acc,
           'Test Acc.': best_test_acc,
           'Final Acc.': test_acc}
    
    results_dir = './results' 
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, args.ctype+'2.csv')
    file_exists = os.path.exists(result_file)
    
    with open(result_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(row)

def train(trainloader, model, teacher, criterion, optimizer, epoch, use_cuda, partial):
    # switch to train mode
    model.train()
    if teacher is not None:
        teacher.eval()
        
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    # top5 = AverageMeter()
    ctop1 = AverageMeter()
    ctop2 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    
    cnt = 0
    for batch_idx, (inputs, true_targets, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets, true_targets = inputs.cuda(), targets.cuda(), true_targets.cuda()
        
        # compute output
        outputs = model(inputs)
        if teacher is None:
            loss = criterion(outputs, targets)
        
        else:
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
                
            if partial:
                loss = criterion(outputs, two_hot(teacher_outputs))
                
            else:
                loss = criterion(outputs, F.softmax(teacher_outputs, dim=1))
        
        # measure accuracy and record loss
        prec1, prec2 = accuracy(outputs.data, true_targets.data, topk=(1, 2))
        cprec1, cprec2 = accuracy(outputs.data, targets.data, topk=(1,2))
        
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top2.update(prec2.item(), inputs.size(0))
        
        ctop1.update(cprec1.item(), inputs.size(0))
        ctop2.update(cprec2.item(), inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top2: {top2: .4f} | ctop1: {ctop1: .4f} | ctop2:{ctop2: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top2=top2.avg,
                    ctop1=ctop1.avg,
                    ctop2=ctop2.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, teacher, criterion, epoch, use_cuda, partial):
    global best_test_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()    
    if teacher is not None:
        teacher.eval()
        
    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets, _) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        with torch.no_grad():
        # compute output
            outputs = model(inputs)
            if teacher is None:
                loss = criterion(outputs, targets)
                
            else:
                teacher_outputs = teacher(inputs)
                if partial:
                    loss = criterion(outputs, two_hot(teacher_outputs))
                    
                else:
                    loss = criterion(outputs, F.softmax(teacher_outputs, dim=1))
                
            # measure accuracy and record loss
            prec1, prec2 = accuracy(outputs.data, targets.data, topk=(1, 2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top2.update(prec2.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top2: {top2: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top2=top2.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_last, is_best, checkpoint='checkpoint'):
    
    if is_last:
        torch.save(state, os.path.join(checkpoint, 'last.ckpt'))
    
    if is_best:
        torch.save(state, os.path.join(checkpoint, 'best.ckpt'))

def two_hot(matrix):
    max_values, max_indices = torch.topk(matrix, k=2, dim=1)
    return max_indices

def adjust_learning_rate(optimizer, epoch):
    global state
    state['lr'] *= args.gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()