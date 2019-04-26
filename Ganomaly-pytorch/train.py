from __future__ import print_function

from options import Options

from makedata import load_data

from model import Ganomaly

opt = Options().parse()

dataloader = load_data(opt)

model = Ganomaly(opt, dataloader)

#model.train()

#model.test1()

model.test2()