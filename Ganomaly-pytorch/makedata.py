import os
import torchvision.datasets as datasets
import torch
import numpy as np 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def load_data(opt):

    if opt.dataroot == '': 

        opt.dataroot = './data/{}'.format(opt.dataset)  



    if opt.dataset in ['cifar10']:#这里主要是我第一次用的数据集是cifar10，但是数据集有点少。我后来换成了celeba，所以这里就没改。

        splits = ['train', 'test']  #训练集train用来训练参数，验证集test根据AUC的值来挑选参数

        drop_last_batch = {'train': True, 'test': False}

        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose([transforms.Resize(opt.isize),

                                        transforms.CenterCrop(opt.isize),

                                        transforms.ToTensor(),

                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])



        dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), transform) for x in splits}

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],

                                                     batch_size=opt.batchsize,

                                                     shuffle=shuffle[x],

                                                     num_workers=int(opt.workers),

                                                     drop_last=drop_last_batch[x]) for x in splits}

        return dataloader  #从dataloader里面进行迭代的时候 有两个内容：图片和标签。源码的这部分很棒，我觉得用字典的形式来加载数据集方便且快
    
    
