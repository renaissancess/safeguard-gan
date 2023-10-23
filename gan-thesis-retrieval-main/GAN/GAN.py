import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import os

# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

# 随机生成数据，大小为10 * 20列
source_data = np.random.rand(10, 20)
# 随机生成标签，大小为10 * 1列
source_label = np.random.randint(0,2,(10, 1))
# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
torch_data = GetLoader(source_data, source_label)
 

# 读取数据
datas =torch.utils.data.DataLoader(torch_data, batch_size=6, shuffle=True, drop_last=False)
for i, data in enumerate(datas):
	# i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
    print("第 {} 个Batch \n{}".format(i,data))


# if not os.path.exists('./img_gan'): # 报错中间结果
#     os.mkdir('./img_gan')
 
# def to_img(x):# 将结果的-0.5~0.5变为0~1保存图片
#     out = 0.5 * (x + 1)
#     out = out.clamp(0, 1)
#     out = out.view(-1, 1, 28, 28)
#     return out
 
 
# batch_size = 128
# num_epoch = 20
# z_dimension = 100
 
# # 数据预处理
# img_transform = transforms.Compose([
#     transforms.ToTensor(),# 图像数据转换成了张量，并且归一化到了[0,1]。
#     transforms.Normalize([0.5],[0.5])#这一句的实际结果是将[0，1]的张量归一化到[-1, 1]上。前面的（0.5）均值， 后面(0.5)标准差，
# ])
# # MNIST数据集
# mnist = datasets.MNIST(
#     root='./data/', train=True, transform=img_transform, download=True)
# # 数据集加载器
# dataloader = torch.utils.data.DataLoader(
#     dataset=mnist, batch_size=batch_size, shuffle=True)
 
# for (i,img) in enumerate(dataloader):
#     print(i,img[0].size(),img[1].size())
#     break


# # 判别器 判别图片是不是来自MNIST数据集
# class discriminator(nn.Module):
#     def __init__(self):
#         super(discriminator, self).__init__()
#         self.dis = nn.Sequential(
#             nn.Linear(784, 256), # 784=28*28
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 1), 
#             nn.Sigmoid()
#         )
 
#     def forward(self, x):
#         x = self.dis(x)
#         return x
 
 
# # 生成器 生成伪造的MNIST数据集
# class generator(nn.Module):
#     def __init__(self):
#         super(generator, self).__init__()
#         self.gen = nn.Sequential(
#             nn.Linear(100, 256), # 输入为100维的随机噪声
#             nn.ReLU(),
#             nn.Linear(256, 256), 
#             nn.ReLU(),
#             nn.Linear(256, 784),
#             nn.Tanh()
#         )
 
#     def forward(self, x):
#         x = self.gen(x)
#         return x

# D = discriminator() # 创建生成器
# G = generator() # 创建判别器
# # if torch.cuda.is_available():# 放入GPU
# #     D = D.cuda()
# #     G = G.cuda()
# # Binary cross entropy loss and optimizer
# criterion = nn.BCELoss() # BCELoss 因为可以当成是一个分类任务，如果后面不加Sigmod就用BCEWithLogitsLoss
# d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003) # 优化器
# g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003) # 优化器

# # 开始训练
# for epoch in range(num_epoch):
#     for i, (img, _) in enumerate(dataloader):# img[128,1,28,28]
#         num_img = img.size(0) # num_img=batchsize
#         # =================train discriminator
#         real_img = img.view(num_img, -1) # 把图片拉平,为了输入判别器 [128,784]
#        # real_img = img.cuda() # 装进cuda
#         real_label = torch.ones(num_img).reshape(num_img,1)#.cuda() # 希望判别器对real_img输出为1 [128,]
#         fake_label = torch.zeros(num_img).reshape(num_img,1)#.cuda() # 希望判别器对fake_img输出为0  [128,]

#         # 先训练鉴别器
#         # 计算真实图片的loss
#         real_out = D(real_img) # 将真实图片输入鉴别器 [128,1]
#         d_loss_real = criterion(real_out, real_label) # 希望real_out越接近1越好 [1]
#         real_scores = real_out  # 后面print用的
 
#         # 计算生成图片的loss
#         z = torch.randn(num_img, z_dimension)#.cuda() # 创建一个100维度的随机噪声作为生成器的输入 [128,1]
#         # 避免计算G的梯度
#         fake_img = G(z).detach() # 生成伪造图片 [128,748]  #detach返回不含梯度的张量
#         fake_out = D(fake_img) # 给判别器判断生成的好不好 [128,1]
#         d_loss_fake = criterion(fake_out, fake_label) # 希望判别器给fake_out越接近0越好 [1]
#         fake_scores = fake_out   # 后面print用的

#         d_loss = d_loss_real + d_loss_fake
#         d_optimizer.zero_grad()
#         d_loss.backward()
#         d_optimizer.step()
 
#         # 训练生成器
#         # 计算生成图片的loss
#         z = torch.randn(num_img, z_dimension)#.cuda()  #生成随机噪声 [128,100]
#         fake_img = G(z) # 生成器伪造图像 [128,784]
#         output = D(fake_img) # 将伪造图像给判别器判断真伪 [128,1]
#         g_loss = criterion(output, real_label) # 生成器希望判别器给的值越接近1越好 [1]
 
#         # 更新生成器
#         g_optimizer.zero_grad()
#         g_loss.backward()
#         g_optimizer.step()
 
#         if (i + 1) % 100 == 0:
#             print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
#                   'D real: {:.6f}, D fake: {:.6f}'.format(
#                       epoch, num_epoch, d_loss.cpu().detach(), g_loss.cpu().detach(),
#                       real_scores.cpu().detach().mean(), fake_scores.cpu().detach().mean()))
#     if epoch == 0: # 保存图片
#         real_images = to_img(real_img.cpu().detach())
#         save_image(real_images, './img_gan/real_images.png')
 
#     fake_images = to_img(fake_img.cpu().detach())
#     save_image(fake_images, './img_gan/fake_images-{}.png'.format(epoch + 1))
# # 保存模型
# torch.save(G.state_dict(), './generator.pth')
# torch.save(D.state_dict(), './discriminator.pth')