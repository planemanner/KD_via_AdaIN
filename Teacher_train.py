# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:48:49 2020

@author: lky
"""

from models import WideResNet
import torch
import argparse
from torchvision import transforms
from torchvision import datasets
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn
from Utils import *

def main(args):
    
    train_transform = transforms.Compose([transforms.ColorJitter(),
                                    transforms.RandomResizedCrop(32,(0.5,1.5)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))])
    
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675,0.2565,0.2761))])
    
    train_dataset = datasets.CIFAR100('./dataset',train = True, transform = train_transform, download=True)
    test_dataset = datasets.CIFAR100('./dataset',train = False, transform = test_transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers)         
    
    Teacher = WideResNet(depth=args.depth, num_classes=100, widen_factor=args.width_factor, drop_rate=0.3)
    Teacher.cuda()
    cudnn.benchmark = True
    optimizer = torch.optim.SGD(Teacher.parameters(), lr = args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [60, 120, 160], gamma=2e-1)
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_acc5 = 0
    best_flag = False
    """
    Training
    """
    for epoch in range(args.total_epochs):
        for iter_, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs, *acts = Teacher(images)
            cls_loss = criterion(outputs, labels)
            optimizer.zero_grad()
            cls_loss.backward()
            optimizer.step()
            if iter_%100 == 0:
                print("Epoch : {}/ {}, Iteration:{}/{}, Loss:{:02.5f}".format(epoch, args.total_epochs, iter_, train_loader.__len__(), cls_loss.item()))
        """
        Validation
        """
        top1, top5 = evaluator(test_loader, Teacher)
        
        if top1 > best_acc:
            best_acc = top1
            best_acc5 = top5
            best_flag = True    
        if best_flag:
            state = {'epoch':epoch+1, 'state_dict':Teacher.state_dict(), 'optimizer': optimizer.state_dict()}       
            save_ckpt(state, is_best=best_flag, root_path = args.weight_path)
            best_flag = False
        
        opt_scheduler.step()
        
    print("Best top 1 acc: {}".format(best_acc))
    print("Best top 5 acc: {}".format(best_acc5))
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type = str, default = './weights')
    parser.add_argument('--total_epochs', type = int, default = 200)
    parser.add_argument('--lr', type = float, default = 1e-1)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--depth', type = int, default = 40)
    parser.add_argument('--width_factor', type = int, default = 4)
    args = parser.parse_args()
    main(args)
    
    