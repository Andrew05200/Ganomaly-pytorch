# Ganomaly-pytorch
1.关于杜伦大学的GANomaly的代码，加了一点自己的注释。并且没有进行可视化，所以比源码少一点。
代码直接复制的论文的源码，我在最后加了测试的部分代码，源码只有训练的内容。这是我自己写来总结的，所以不是很正规。

实验用的是python3，gpu，pytorch，训练集中的正常图片用的是celeba，12w张。验证集里的正常数据是celeba，异常数据用的猫狗图片。测试集和验证集一样。
这是数据集的格式： 其中train是训练集，test是验证集，testing是测试集。




![数据集的格式](https://github.com/lcd111/Ganomaly-pytorch/blob/master/Ganomaly-pytorch/数据集的格式.png)

2.代码里在训练的时候进行可视化的代码我都注销了，没有进行可视化。


