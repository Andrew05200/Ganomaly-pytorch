import torch
import torch.nn as nn
import torch.nn.parallel

#权重的初始化

def weights_init(mod):
    classname=mod.__class__.__name__
    if classname.find('Conv')!= -1:    #这里的Conv和BatchNnorm是torc.nn里的形式
        mod.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm')!= -1:
        mod.weight.data.normal_(1.0,0.02) #bn层里初始化γ，服从（1，0.02）的正态分布
        mod.bias.data.fill_(0)  #bn层里初始化β，默认为0
        
#定义一个编码器

class Encoder(nn.Module):  #输入图片的大小、噪声的维度-100、输入图片的通道、ndf=64、
    def __init__(self,isize,nz,nc,ndf,ngpu,n_exter_layers=0,add_final_conv=True):
        super(Encoder,self).__init__()
        self.ngpu=ngpu
        assert isize % 16==0,"isize has to be a multiple of 16"

        main=nn.Sequential()
        main.add_module('initial-conv-{0}-{1}'.format(nc,ndf),nn.Conv2d(nc,ndf,4,2,1,bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),nn.LeakyReLU(0.2,inplace=True))
        csize,cndf=isize/2,ndf

        for t in range(n_exter_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t,cndf),nn.Conv2d(cndf,cndf,3,1,1,bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t,cndf),nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t,cndf),nn.LeakyReLU(0.2,inplace=True))

        while csize>4:
            in_feat = cndf

            out_feat = cndf * 2

            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))

            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),nn.BatchNorm2d(out_feat))

            main.add_module('pyramid-{0}-relu'.format(out_feat),nn.LeakyReLU(0.2, inplace=True))

            cndf = cndf * 2

            csize = csize / 2
        if add_final_conv:

            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))
            self.main=main
    
    def forward(self,input):
        if self.ngpu>1:
             output=nn.parallel.data_parallel(self.main,input,range(self.ngpu)) #在多个gpu上运行模型，并行计算
        else:
            output=self.main(input)
        
        return output  #如果输入的大小是3×32×32，最后的输出是100×1×1.
# 定义一个解码器

class Decoder(nn.Module): #图片的大小、噪声的维度-100、图片的通道数、ngf=64

    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):

        super(Decoder, self).__init__()

        self.ngpu = ngpu

        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4

        while tisize != isize:

            cngf = cngf * 2

            tisize = tisize * 2

        main = nn.Sequential()

        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))

        main.add_module('initial-{0}-batchnorm'.format(cngf),nn.BatchNorm2d(cngf))

        main.add_module('initial-{0}-relu'.format(cngf),nn.ReLU(True))

        csize, _ = 4, cngf

        while csize < isize // 2:

            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))

            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),nn.BatchNorm2d(cngf // 2))

            main.add_module('pyramid-{0}-relu'.format(cngf // 2),nn.ReLU(True))

            cngf = cngf // 2

            csize = csize * 2

        for t in range(n_extra_layers):

            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))

            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),nn.BatchNorm2d(cngf))

            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))

        main.add_module('final-{0}-tanh'.format(nc),nn.Tanh())

        self.main = main

    def forward(self, input):

        if self.ngpu > 1:

            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu)) #model里的43行定义的device起始是0，这里也必须从0开始 如果ngpu=4.则用0,1,2,3块

        else:

            output = self.main(input)

        return output  #输出的结果是3×32×32

 #定义判别器D （编码器）     
class NetD(nn.Module):
    def __init__(self, opt):
        super(NetD, self).__init__()

        model = Encoder(opt.isize, 1, opt.nc, opt.ndf, opt.ngpu, opt.extralayers)#第二个参数是1 因为判别器最后输出一个数  不过编码器在生成器里的时候 
        #这个参数是100 因为它要把图片下采样成100×1×1的向量

        layers = list(model.main.children())

        '''layers的输出如下：
        [
        Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),

        LeakyReLU(negative_slope=0.2, inplace),

        Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),

        BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

        LeakyReLU(negative_slope=0.2, inplace),

        Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),

        BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

        LeakyReLU(negative_slope=0.2, inplace),

        Conv2d(256, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)] 因为132行定义的nz参数是1，所以经过这层之后输出的大小是1×1×1
        '''

        self.features = nn.Sequential(*layers[:-1])
        #self.features的内容为除了最后一层的前8层，nn.Sequential函数里面的参数一定是Module的子类，而list：list is not a Module subclass。所以不能当做参数，
        # 当然model.children()也是一样：Module.children is not a Module subclass。这里的*就起了作用，将list或者children的内容迭代的一个一个的传进去。


        self.classifier = nn.Sequential(layers[-1])
        #self.classifier的内容为Conv2d(256, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)最后一层

        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):

        features = self.features(x) #图片通过前8层之后的结果256×4×4，前提是输入的图片的大小是32

        features = features

        classifier = self.classifier(features)#此时的结果是1×1×1，值在[0，1]，因为经过了sigmoid

        classifier = classifier.view(-1, 1).squeeze(1)
        #a=torch.ones([1,1,1])  即a=tensor([[[ 1.]]])   再a.view(-1,1) 变成tensor([[ 1.]])  再加一个squeeze就是
        # a.view(-1,1).squeeze(1) 结果是tensor([ 1.])，squeeze里的值是0是1随意，只要去掉一个维度上的就可以


        return classifier, features

#定义生成器（编码器+解码器+编码器）
class NetG(nn.Module):

    def __init__(self, opt):

        super(NetG, self).__init__()

        self.encoder1 = Encoder(opt.isize, opt.nz, opt.nc, opt.ndf, opt.ngpu, opt.extralayers)

        self.decoder = Decoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)

        self.encoder2 = Encoder(opt.isize, opt.nz, opt.nc, opt.ndf, opt.ngpu, opt.extralayers)

    def forward(self, x):

        latent_i = self.encoder1(x) #100×1×1

        gen_imag = self.decoder(latent_i) #3×32×32

        latent_o = self.encoder2(gen_imag) #100×1×1

        return gen_imag, latent_i, latent_o  

#定义L1和L2
def L1_loss(input,target):

    return torch.mean(torch.abs(input - target))

def L2_loss(input,target,size_average=True):

    if size_average:

        return torch.mean(torch.pow((input-target), 2))

    else:

        return torch.pow((input-target), 2)
