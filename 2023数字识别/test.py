"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
torchvision
matplotlib
"""
# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False
train_transform = transforms.Compose([
                                transforms.RandomAffine(degrees = 0,translate=(0.1, 0.1)),#对照片进行随机平移
                                transforms.RandomRotation(90),        #随机旋转
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])

test_transform = transforms.Compose([transforms.RandomAffine(degrees = 0,translate=(0.1, 0.1)),#对照片进行随机平移
                                    transforms.ToTensor(),
                                    transforms.RandomRotation((-90,180)),        #随机旋转
                                    transforms.Normalize((0.1307,),(0.3081,))])

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=transforms.ToTensor(),    # o
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)
# plot one example
print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
k=1000
fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all')
ax = ax.flatten()
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# transforms.RandomRotation(90)(train_data.train_data)
# print(train_data.train_data[0])
for step, (x, y) in enumerate(train_loader):
    print(x)
    print(y)
    if step == 1:
        break

for i in range(25):
    img=train_data.train_data[i+k].numpy()
    ax[i].set_title(train_data.train_labels[i+k])
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()