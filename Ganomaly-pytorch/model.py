from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm   #tqdm是一个进度条库
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from evaluate import evaluate
from ganomaly import NetD,NetG,weights_init,L2_loss

from visualizer import Visualizer

class Ganomaly(object):  #在python3中，object加不加无所谓

    @staticmethod #所有class里定义的函数，第一个参数必须是self。但 @staticmethod 声明了静态方法，该方法不强制要求传递参数
    #类可以不用实例化就可以调用该方法  Ganomaly.name()，当然也可以实例化后调用  cobj=Ganomaly(),cobj.name() 

    def name():

        return 'Ganomaly'

    def __init__(self, opt, dataloader=None):#__init__是类的构造函数，在类的实例创建后被立即调用，可以在__init__中将变量赋值给class自己的属性变量

        super(Ganomaly, self).__init__()

        #初始化Variables

        self.opt = opt    #self.xxx是全局的，不加self写成xxx是局部的，对于该class有效，只能在该class内使用，不能全局调用
        #通过self.xxx = zzz 的操作，将数据封装在类里面，调用时直接通过类去进行调用。

        self.visualizer = Visualizer(opt)

        self.dataloader = dataloader

        self.trn_dir = '/home/lab-lu.chengdong/Pictures/output/ganomaly/cifar10/train/' #训练结果保存的路径

        self.tst_dir = '/home/lab-lu.chengdong/Pictures/output/ganomaly/cifar10/test/'  #测试结果保存的路径

        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu") #cuda:0 表示是从第0块gpu开始

        #判别器的参数
        self.out_d_real = None

        self.feat_real = None

        self.err_d_real = None

        self.fake = None

        self.latent_i = None

        self.latent_o = None

        self.out_d_fake = None

        self.feat_fake = None

        self.err_d_fake = None

        self.err_d = None

        #生成器的参数
        self.out_g = None

        self.err_g_bce = None

        self.err_g_l1l = None

        self.err_g_enc = None

        self.err_g = None
        #
        self.epoch = 0

        self.times = []

        self.total_steps = 0
        
        #初始化网络

        self.netg = NetG(self.opt).to(self.device)

        self.netd = NetD(self.opt).to(self.device)

        self.netg.apply(weights_init)

        self.netd.apply(weights_init)

        ##继续训练时调用
        if self.opt.resume != '':  #resume是上一次模型训练的结果所保存的地址

            print("\nLoading pre-trained networks.")

            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']

            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])

            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])

            print("\tDone.\n")
        
        #损失函数

        self.bce_criterion = nn.BCELoss() #交叉熵损失函数

        self.l1l_criterion = nn.L1Loss()

        self.l2l_criterion = L2_loss

        #初始化输入的张量 - torch.empty是返回一个包含未初始化数据的张量

        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)

        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)

        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)

        self.real_label = 1

        self.fake_label = 0

        #设置优化
        if self.opt.isTrain:

            self.netg.train()  #使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，eval（）时，框架会自动把BN和DropOut固定住，不会取平均，

            self.netd.train()  #而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大

            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))


    #设置输入的数据 
    def set_input(self, input):

        self.input.data.resize_(input[0].size()).copy_(input[0])  #把data的第一项：图片数据复制给self.input

        self.gt.data.resize_(input[1].size()).copy_(input[1])  #把data的第二项：图片的标签复制给sele.gt

        if self.total_steps == self.opt.batchsize:

            self.fixed_input.data.resize_(input[0].size()).copy_(input[0])  #self.fixed_input接收了第一个epoch的第一个batchsize的数据，不再改变


    #更新判别器
    def update_netd(self):

        # Feature Matching.

        self.netd.zero_grad()

        # Train with real

        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)  #1行64列   都是1

        self.out_d_real, self.feat_real = self.netd(self.input) # 1行64列-返回[0，1]之间的值 和 输入图片的特征图  self.input是训练集的真实数据

        #self.err_d_real= self.bce_criterion(self.out_d_real, self.label)

        # Train with fake

        self.label.data.resize_(self.opt.batchsize).fill_(self.fake_label)

        self.fake, self.latent_i, self.latent_o = self.netg(self.input) #经过生成器生成的假图、经过第一个编码器生成的z、经过第二个编码器生成的z'

        self.out_d_fake, self.feat_fake = self.netd(self.fake.detach()) #假图通过判别器返回[0,1]之间的值 和 假图的特征图，detach表示不更新生成器
        #如果我们有两个网络 , 两个关系是这样的 y=A(x), z=B(y) 求B中参数的梯度，不求A中参数的梯度。可以写为y = A(x)，z = B(y.detach())，z.bakward()

        #self.err_d_fake= self.bce_criterion(self.out_d_fake, self.label)    

        self.err_d = L2_loss(self.feat_real, self.feat_fake)  #算真假图片的特征图之间的距离，算出来是一个值，这是判别器的loss

        self.err_d_real = self.err_d

        self.err_d_fake = self.err_d

        self.err_d.backward()

        self.optimizer_d.step()

    #重新初始化判别器
    def reinitialize_netd(self):

        self.netd.apply(weights_init)

        print('Reloading d net')


    #更新生成器
    def update_netg(self):

        self.netg.zero_grad()

        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)# 标签是1

        self.out_g, _ = self.netd(self.fake) #假图通过判别器之后返回[0,1]间的值 和 特征图

        self.err_g_bce = self.bce_criterion(self.out_g, self.label) #交叉熵损失函数 x=torch.Tensor([0.5,0.1])，y=torch.Tensor([1,1])
        #a=nn.BCELoss(),a(x,y)=tensor(1.4979)  结果来自 (-ln0.5-ln0.1)/2

        self.err_g_l1l = self.l1l_criterion(self.fake, self.input)  # 计算输入图片与生成器生成的图片之间的距离

        self.err_g_enc = self.l2l_criterion(self.latent_o, self.latent_i) #计算两个 下采样之后的向量的距离 

        self.err_g = self.err_g_bce * self.opt.w_bce + self.err_g_l1l * self.opt.w_rec + self.err_g_enc * self.opt.w_enc

        #生成器的loss=输入图片与生成器生成的图片之间的距离+两个 下采样之后的向量的距离+ 生成器生成的图片通过判别器得到的分数与真实标签1之间的距离
        self.err_g.backward(retain_graph=True)

        self.optimizer_g.step()

#更新D G
    def optimize(self):

        self.update_netd()

        self.update_netg()

        #如果判别器的loss=0,重新初始化判别器的参数
        if self.err_d_real.item() < 1e-5 or self.err_d_fake.item() < 1e-5:

            self.reinitialize_netd()

#获得 D G的误差     使用OrderedDict会根据放入元素的先后顺序进行排序。
    def get_errors(self):   

        errors = OrderedDict([('err_d', self.err_d.item()),  #判别器的loss

                              ('err_g', self.err_g.item()),  #生成器的loss

                              ('err_d_real', self.err_d_real.item()), 

                              ('err_d_fake', self.err_d_fake.item()), 

                              ('err_g_bce', self.err_g_bce.item()), #生成器生成的图片通过判别器得到的分数与真实标签1之间的距离

                              ('err_g_l1l', self.err_g_l1l.item()), #计算输入图片与生成器生成的图片之间的距离

                              ('err_g_enc', self.err_g_enc.item())]) #计算两个 下采样之后的向量的距离



        return errors
#获得输入图片、输入图片经过生成器生成的图片、
    def get_current_images(self):

        reals = self.input.data

        fakes = self.fake.data

        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed


#保存网络训练好的参数
    def save_weights(self,epoch):

        weight_dir = '/home/lab-lu.chengdong/Pictures/output/ganomaly/cifar10/train/weights/' #权重保存的地址 

        if not os.path.exists(weight_dir):

            os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},'%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},'%s/netD.pth' % (weight_dir))

#对模型训练一次
    def train_epoch(self):

        self.netg.train()  #当网络中有dropout，bn 的时候。训练的要记得net.train(), 测试要记得 net.eval()

        epoch_iter = 0

        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):  #dataloader出来有2个参数，第一个是数据，第二个是标签

            self.total_steps += self.opt.batchsize

            epoch_iter += self.opt.batchsize

            self.set_input(data)

            self.optimize()  #优化 D G



            #if self.total_steps % self.opt.print_freq == 0: #每80的倍数次 在console展示训练结果

            #errors = self.get_errors()

                #if self.opt.display:

                    #counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)  #len(self.dataloader['train'].dataset表示训练集图片个数

                    #self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0: #每100的倍数次 保存模型生成的图片  

                reals, fakes, fixed = self.get_current_images()

                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)

                #if self.opt.display:

                    #self.visualizer.display_current_images(reals, fakes, fixed)

        print(">> Training model %s. Epoch %d/%d Loss_D: %.3f Loss_G %.3f " % (self.name(), self.epoch+1, self.opt.niter,self.err_d.item(),self.err_g.item()))

		
				

#训练模型        


    def train(self):


        self.total_steps = 0

        best_auc = 0

        print(">> Training model %s." % self.name())

        for self.epoch in range(self.opt.iter, self.opt.niter):

        


            self.train_epoch()
            
            res = self.test()

            if res['AUC'] > best_auc:

                best_auc = res['AUC']

                self.save_weights(self.epoch)

            self.visualizer.print_current_performance(res, best_auc)

        print('>> Training model %s.[Done]' % self.name())
    
#测试模型
    def test(self):
        
        with torch.no_grad():  #做异常检测的时候分数只由生成器来决定，所以不调用判别器的参数

            #with torch.no_grad()是代替了volatile=True的，因为测试的时候不用求梯度进行反向传播，该参数可以实现一定速度的提升，并节省一半的显存

            if self.opt.load_weights:

                path = "./output/{}/{}/train/weights/netG.pth".format(self.name().lower(), self.opt.dataset)

                pretrained_dict = torch.load(path)['state_dict']



                try:

                    self.netg.load_state_dict(pretrained_dict)

                except IOError:

                    raise IOError("netG weights not found")

                print('   Loaded weights.')
            ##    
            self.opt.phase = 'test'
            
            #an_scores的大小是测试集图片的个数，因为它最终要给每一张图片一个[0,1]之间的数，通过比较这个数和阈值之间的大小，判断这张图是否正常
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)

            #gt_labels的大小也是测试集图片的个数
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,    device=self.device)

            #latent_i的大小也是（测试集图片的个数，100）因为一张图片会提取一个瓶颈特征,瓶颈特征的大小是（100，1，1）
            self.latent_i  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            #latent_o的大小也是（测试集图片的个数，100）因为一张图片会提取一个瓶颈特征,瓶颈特征的大小是（100，1，1）
            self.latent_o  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            
            self.times = []

            self.total_steps = 0

            epoch_iter = 0


            for i, data in enumerate(self.dataloader['test'], 0): #因为dataloader是个迭代器，所以用enumerate，且后面的那个0表示i从0开始表示，如果后面那个
                #是1，那么i就从1开始表示

                self.total_steps += self.opt.batchsize

                epoch_iter += self.opt.batchsize

                time_i = time.time()

                self.set_input(data)

                self.fake, latent_i, latent_o = self.netg(self.input)



                error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1) #error的size是（64，1，1）

                time_o = time.time()



                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0)) #(64,1,1)reshape变成了1行64列
                #在第一次迭代的时候，an_scores[0：64]的前64个位置接收这64个数

                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))


                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)

                #latent_i，latent_o从（64，100，1，1）变成了（64，100），latent_i，latent_o的第0行到第63行接受这（64，100）个数，下次循环第64行到第127行接受
                # （64，100）个数，直到测试集里的所有图片遍历结束

                self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)



                self.times.append(time_o - time_i)

                #保存图片

                if self.opt.save_test_images:

                    dst ='/home/lab-lu.chengdong/Pictures/output/ganomaly/cifar10/test/images/'

                    if not os.path.isdir(dst):

                        os.makedirs(dst)

                    real, fake, _ = self.get_current_images()

                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)

                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)

            self.times = np.array(self.times)

            self.times = np.mean(self.times[:100] * 1000)



            # Scale error vector between [0, 1]

            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))

            # auc, eer = roc(self.gt_labels, self.an_scores)

            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)

            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])

            #if self.opt.display_id > 0 and self.opt.phase == 'test':

                #counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)

                #self.visualizer.plot_performance(self.epoch, counter_ratio, performance)

            return performance
    def test1(self):
        
        with torch.no_grad():  #做异常检测的时候分数只由生成器来决定，所以不调用判别器的参数

            #with torch.no_grad()是代替了volatile=True的，因为测试的时候不用求梯度进行反向传播，该参数可以实现一定速度的提升，并节省一半的显存

            
            path = "/home/lab-lu.chengdong/Pictures/output/ganomaly/cifar10/train/weights/netG.pth"

            pretrained_dict = torch.load(path)['state_dict']



            try:

                self.netg.load_state_dict(pretrained_dict)

            except IOError:

                raise IOError("netG weights not found")

            print('   Loaded weights.')

            self.netg.eval()

            for e in np.arange(0.01,0.80,0.02):

                correct=0

                total=0

                #an_scores的大小是测试集图片的个数，因为它最终要给每一张图片一个[0,1]之间的数，通过比较这个数和阈值之间的大小，判断这张图是否正常
                self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)

                #gt_labels的大小也是测试集图片的个数
                self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,    device=self.device)

                #latent_i的大小也是（测试集图片的个数，100）因为一张图片会提取一个瓶颈特征,瓶颈特征的大小是（100，1，1）
                self.latent_i  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

                #latent_o的大小也是（测试集图片的个数，100）因为一张图片会提取一个瓶颈特征,瓶颈特征的大小是（100，1，1）
                self.latent_o  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

                


                for i, data in enumerate(self.dataloader['test'], 0): #因为dataloader是个迭代器，所以用enumerate，且后面的那个0表示i从0开始表示，如果后面那个
                    #是1，那么i就从1开始表示


                    self.set_input(data)

                    self.fake, latent_i, latent_o = self.netg(self.input)



                    error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1) #error的size是（64，1，1）


                    self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0)) #(64,1,1)reshape变成了1行64列
                    #在第一次迭代的时候，an_scores[0：64]的前64个位置接收这64个数

                    self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))


                    self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)

                    #latent_i，latent_o从（64，100，1，1）变成了（64，100），latent_i，latent_o的第0行到第63行接受这（64，100）个数，下次循环第64行到第127行接受
                    # （64，100）个数，直到测试集里的所有图片遍历结束

                    self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)

                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))

                self.an_scores[self.an_scores >= e] = 1

                self.an_scores[self.an_scores <  e] = 0

                correct += (self.an_scores.float() == self.gt_labels.float()).sum()

                total += self.gt_labels.size(0)

                print('Test Accuracy of the model on the {} test images: {} %, e:'.format(total, 100 * correct / total), e)

                print('correct:%d; total:%d' % (correct , total))


    def test2(self):
        
        with torch.no_grad():  #做异常检测的时候分数只由生成器来决定，所以不调用判别器的参数

            #with torch.no_grad()是代替了volatile=True的，因为测试的时候不用求梯度进行反向传播，该参数可以实现一定速度的提升，并节省一半的显存

            
            path = "/home/lab-lu.chengdong/Pictures/output/ganomaly/cifar10/train/weights/netG.pth"

            pretrained_dict = torch.load(path)['state_dict']



            try:

                self.netg.load_state_dict(pretrained_dict)

            except IOError:

                raise IOError("netG weights not found")

            print('   Loaded weights.')

            self.netg.eval()

            

            correct=0

            total=0

            #an_scores的大小是测试集图片的个数，因为它最终要给每一张图片一个[0,1]之间的数，通过比较这个数和阈值之间的大小，判断这张图是否正常
            self.an_scores = torch.zeros(size=(len(self.dataloader['testing'].dataset),), dtype=torch.float32, device=self.device)

            #gt_labels的大小也是测试集图片的个数
            self.gt_labels = torch.zeros(size=(len(self.dataloader['testing'].dataset),), dtype=torch.long,    device=self.device)

            #latent_i的大小也是（测试集图片的个数，100）因为一张图片会提取一个瓶颈特征,瓶颈特征的大小是（100，1，1）
            self.latent_i  = torch.zeros(size=(len(self.dataloader['testing'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            #latent_o的大小也是（测试集图片的个数，100）因为一张图片会提取一个瓶颈特征,瓶颈特征的大小是（100，1，1）
            self.latent_o  = torch.zeros(size=(len(self.dataloader['testing'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            


            for i, data in enumerate(self.dataloader['testing'], 0): #因为dataloader是个迭代器，所以用enumerate，且后面的那个0表示i从0开始表示，如果后面那个
                #是1，那么i就从1开始表示


                self.set_input(data)

                self.fake, latent_i, latent_o = self.netg(self.input)



                error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1) #error的size是（64，1，1）


                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0)) #(64,1,1)reshape变成了1行64列
                #在第一次迭代的时候，an_scores[0：64]的前64个位置接收这64个数

                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))


                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)

                #latent_i，latent_o从（64，100，1，1）变成了（64，100），latent_i，latent_o的第0行到第63行接受这（64，100）个数，下次循环第64行到第127行接受
                # （64，100）个数，直到测试集里的所有图片遍历结束

                self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)

            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))

            self.an_scores[self.an_scores >= 0.07] = 1

            self.an_scores[self.an_scores <  0.07] = 0

            correct += (self.an_scores.float() == self.gt_labels.float()).sum()

            total += self.gt_labels.size(0)

            print('Test Accuracy of the model on the {} testing images: {}% '.format(total, 100 * correct / total))

            print('correct:%d; total:%d' % (correct , total))




