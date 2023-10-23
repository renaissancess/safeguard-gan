import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from random import randint
from model import Generator, Discriminator, Generator_UseLeakyReLU, Discriminator_UseReLU, Generator_UseIN, Discriminator_UseIN
import os
from torchvision.utils import save_image

netG = Generator(500, 64)
state_dict = torch.load("C:/Users/hp/Desktop/safe/datazong/data/netG_025.pth")
netG.load_state_dict(state_dict)

noise = torch.randn(64, 500, 1, 1)  # 生成100维的随机噪声 [64,100,1,1]
fake_img = netG(noise)


for i in range(64):
    save_image(fake_img.data[i],
               '%s/fake_samples_03d.png' % ("result/", i),
               normalize=True)
