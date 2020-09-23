# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:34:56 2020

@author: lky
"""
import torch
import torch.nn as nn
from torchvision import datasets
from Utils import *
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from models import WideResNet
import argparse
from os import path
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

"""
Now, I am editing this code. Updated version will be coming soon.
"""

def main(args):
    # writer = SummaryWriter('./runs/CIFAR_100_exp')
     
    train_transform = transforms.Compose([transforms.Pad(4, padding_mode='reflect'),
                                          transforms.RandomRotation(15),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675,0.2565,0.2761))])
    
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675,0.2565,0.2761))])
    
    train_dataset = datasets.CIFAR100('./dataset',train = True, transform = train_transform, download=True)
    test_dataset = datasets.CIFAR100('./dataset',train = False, transform = test_transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    Teacher = WideResNet(depth=args.teacher_depth, num_classes=100, widen_factor=args.teacher_width_factor, drop_rate=0.3)
    Teacher.cuda()
    Teacher.eval()
    
    teacher_weight_path = path.join(args.teacher_root_path, 'model_best.pth.tar')
    t_load = torch.load(teacher_weight_path)
    Teacher.load_state_dict(t_load)
    
    Student = WideResNet(depth = args.student_depth, num_classes=100, widen_factor=args.student_width_factor, drop_rate=0.0)
    Student.cuda()
    
    cudnn.benchmark = True
    
    optimizer = torch.optim.SGD(Teacher.parameters(), lr = args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [60, 120, 160], gamma=2e-1)
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_acc5 = 0
    best_flag = False
    
    for epoch in range(args.total_epochs):
        for iter_, (img, label) in enumerate(train_loader):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            t_outs, *t_acts = Teacher(images)
            s_outs, *s_acts = Student(images)
            
            cls_loss = criterion(s_outs, labels)
            
            """
            statistical matching and AdaIN losses
            """
            
            if args.aux_flag==0:
                aux_loss_1 = SM_Loss(t_acts[2], s_acts[2]) # group conv2
            else:
                aux_loss_1 = 0
                for i in range(3):
                    aux_loss_1 += SM_Loss(t_acts[i], s_acts[i])
                    
            F_hat = AdaIN(t_acts[2], s_acts[2])
            interim_out_q = Teacher.bn1(F_hat)
            interim_out_q = Teacher.relu(interim_out_q)
            interim_out_q = F.avg_pool2d(interim_out_q, 8)
            interim_out_q = interim_out_q.view(-1, Teacher.last_ch)
            q = Teacher.fc(interim_out_q)
            
            aux_loss_2 = torch.mean(torch.pow(t_outs-q, 2))
            
            total_loss = cls_loss + aux_loss_1 + aux_loss_2
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
        top1, top5 = evaluator(test_loader, Student)
        
        if top1 > best_acc:
            best_acc = top1
            best_acc5 = top5
            best_flag = True    
        if best_flag:
            state = {'epoch':epoch+1, 'state_dict':Student.state_dict(), 'optimizer': optimizer.state_dict()}       
            save_ckpt(state, is_best=best_flag, root_path = args.student_weight_path)
            best_flag = False
        
        opt_scheduler.step()
        
        # writer.add_scalar('acc/top1', top1, epoch)
        # writer.add_scalar('acc/top5', top5, epoch)
        # writer.close()
        
        
    print("Best top 1 acc: {}".format(best_acc))
    print("Best top 5 acc: {}".format(best_acc5))    
        
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_root_path', type = str, default = './weights')
    parser.add_argument('--student_weight_path', type = str, default = './student/weights')
    parser.add_argument('--total_epochs', type = int, default = 200)
    parser.add_argument('--lr', type = float, default = 1e-1)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--teacher_depth', type = int, default = 40)
    parser.add_argument('--teacher_width_factor', type = int, default = 4)
    parser.add_argument('--student_depth', type = int, default = 16)
    parser.add_argument('--student_width_factor', type = int, default = 4)
    parser.add_argument('--aux_flag', type=int, default = 0)
    
    args = parser.parse_args()
    
    main(args)
    
    
