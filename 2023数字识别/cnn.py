import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x
    



class ResidualBlock(nn.Module):
    # Residual Block需要保证输出和输入通道数x一样
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        # 3*3卷积核，保证图像大小不变将padding设为1
        # 第一个卷积
        self.conv1 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)
        # 第二个卷积
        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)
        #第三个卷积，1*1
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # 激活
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        # 先求和 后激活
        z = self.conv3(x)
        return F.relu(z + y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #卷积层
        self.conv1 = nn.Conv2d(1  ,32, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(32 ,64, kernel_size=5,padding=2)
        self.conv3 = nn.Conv2d(64 ,128,kernel_size=5,padding=2)
        self.conv4 = nn.Conv2d(128,192,kernel_size=5,padding=2)

        #残差神经网络层，其中已经包含了relu
        self.rblock1 = ResidualBlock(32)
        self.rblock2 = ResidualBlock(64)
        self.rblock3 = ResidualBlock(128)
        self.rblock4 = ResidualBlock(192)
        
        #BN层，归一化，使数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(192)
        
        #最大池化，一般最大池化效果都比平均池化好些
        self.mp = nn.MaxPool2d(2)
        
        #fully connectected全连接层
        self.fc1 = nn.Linear(192*7*7, 256)  # 线性
        self.fc6 = nn.Linear(256, 10)  # 线性

    def forward(self, x):
        in_size = x.size(0)
        
        x = self.conv1(x)   #channels:1-32  w*h:28*28  
        x = self.bn1(x)
        x = F.relu(x) 
        x = self.rblock1(x)
        
        x = self.conv2(x)   #channels:32-64 w*h:28*28
        x = F.relu(x)   
        x = self.bn2(x)
        x = self.rblock2(x)
        
        x = self.mp(x)      #最大池化,channels:64-64    w*h:28*28->14*14
        

        x = self.conv3(x)   #channels:64-128    w*h:14*14
        x = self.bn3(x)
        x = F.relu(x)
        x = self.rblock3(x)

        x = self.conv4(x)   #channels:128-192   w*h:14*14
        x = self.bn4(x)
        x = F.relu(x)
        x = self.rblock4(x)
        
        x = self.mp(x)      #最大池化,channels:192-192  w*h:14*14->7*7

        x = x.view(in_size, -1)     #展开成向量
        x = F.relu(self.fc1(x))  # 使用relu函数来激活
        
        return self.fc6(x)