# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:34:37 2020

@author: lky
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.relu = F.relu
        self.equal_in_out = (in_planes==out_planes)
        self.convShortcut = (not self.equal_in_out) and nn.Conv2d(in_planes, out_planes, kernel_size=1,stride=stride, bias=False) or None
    
    def forward(self, x):
        if not self.equal_in_out:
            x = self.relu(self.bn1(x))
        else:
            out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.conv1(out if self.equal_in_out else x)))
        
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        
        return torch.add(x if self.equal_in_out else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)
    
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i==0 and in_planes or out_planes, out_planes, i==0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)
    
class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0):
        super(WideResNet, self).__init__()
        
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth-4)%6==0)
        n = (depth-4)/6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        self.block1 = NetworkBlock(n, nChannels[0],nChannels[1], block, 1, drop_rate)
        self.block2 = NetworkBlock(n, nChannels[1],nChannels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, nChannels[2],nChannels[3], block, 2, drop_rate)
        
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.block1(out)
        act1 = out
        out = self.block2(out)
        act2 = out
        out = self.block3(out)
        act3 = out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out,8)
        out = out.view(-1, self.nChannels[3])
        # logit and group 2,3,4
        return self.fc(out), act1, act2, act3
                
        
        