from __future__ import print_function

from options import Options

from makedata import load_data

from model import Ganomaly

opt = Options().parse()

dataloader = load_data(opt)

model = Ganomaly(opt, dataloader)

model.train()  #训练的时候注释掉17 19 行

model.test1()  #在验证集上选取阈值的时候，注释掉15 19行

model.test2()   #在测试集测试的时候，注释掉15 17行，反正这三个只留一个。


 #代码的意思是训练时每一个epoch的参数在测试集上算一次auc的值，我只保存auc最大的那次epoch的参数。所以训练完成之后，我只有一对参数，netg.pth和
  #netg.pth   然后我用netg.pth在验证集上确定一个阈值 最后把选取好的参数和阈值 放到测试集上测试
