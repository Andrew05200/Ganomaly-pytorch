import os
import torchvision.datasets as datasets
import torch
import numpy as np 
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.datasets import ImageFolder

def load_data(opt):

    if opt.dataroot == '': 

        opt.dataroot = './data/{}'.format(opt.dataset)  #dataroot是./data/mnist  如果dataset的参数是mnist



    if opt.dataset in ['cifar10']:

        splits = ['train', 'testing']

        drop_last_batch = {'train': True, 'testing': False}

        shuffle = {'train': True, 'testing': True}

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

        return dataloader  #从dataloader里面进行迭代的时候 有两个内容：图片和标签
    
    