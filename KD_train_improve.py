# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 00:53:44 2020

@author: lky
"""

"""
Following description is just an idea of mine...

I will use batchnorm statistics.
Why?-> Batch normalization gamma and beta map the feature to specific space.
So if the space statistics of student is matched to the teacher's one, then we 
can guide student to project in relatively 'good' space.


"""


def bn_statistics(T_block, S_block):
    """
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
    """
    
    T_bn = T_block.layer[-1].bn2
    S_bn = S_block.layer[-1].bn2
    
    T_gamma = T_bn.weight.data
    T_beta = T_bn.bias.data
    
    S_gamma = S_bn.weight.data
    S_beta = S_bn.bias.data
    
    
