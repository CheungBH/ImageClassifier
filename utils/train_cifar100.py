#-*-coding:utf-8-*-
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

class LeNet(nn.Module):
    # 一般在__init__中定义网络需要的操作算子，比如卷积、全连接算子等等
    def __init__(self):
        super(LeNet, self).__init__()
        # Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是kernel size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 由于上一层有16个channel输出，每个feature map大小为5*5，所以全连接层的输入是16*5*5
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        # 最终有10类，所以最后一个全连接层输出数量是10
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)
    # forward这个函数定义了前向传播的运算，只需要像写普通的python算数运算那样就可以了
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 下面这步把二维特征图变为一维，这样全连接层才能处理
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 在构建数据集的时候指定transform，就会应用我们定义好的transform
# root是存储数据的文件夹，download=True指定如果数据不存在先下载数据
class ImageClassify(object):
    def __init__(self):
        self.cifar_train = 0
        self.cifar_test = 0
        self.trainloader =0
        self.testloader  =0
        self.criterion = 0
        self.optimizer = 0
        self.net = LeNet()
    def GetImgData(self):
        # cifar-10官方提供的数据集是用numpy array存储的
        # 下面这个transform会把numpy array变成torch tensor，然后把rgb值归一到[0, 1]这个区间
        print('获取数据集....')
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.cifar_train = torchvision.datasets.CIFAR10(root='/media/hkuit155/Windows1/research/ViT-cifar10-pruning/data', train=True,download=True, transform=transform)
        self.cifar_test  = torchvision.datasets.CIFAR10(root='/media/hkuit155/Windows1/research/ViT-cifar10-pruning/data', train=False,download=True, transform=transform)
        print(self.cifar_train)
        self.x = print(self.cifar_test)

    def LoadData(self):
        # 加载数据集
        print("加载数据集...")
        batch_size = 32
        self.trainloader = torch.utils.data.DataLoader(self.cifar_train, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.cifar_test, batch_size=batch_size, shuffle=True)
    def LossFun(self):
        print("lossing...")
        # CrossEntropyLoss就是我们需要的损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
    def TrainingData(self):
        self.LoadData()
        self.LossFun()
        print("Start Training...")
        for epoch in range(30):
            # 我们用一个变量来记录每100个batch的平均loss
            loss100 = 0.0
            # 我们的dataloader派上了用场
            for i, data in enumerate(self.trainloader):
                inputs, labels = data
                # inputs, labels = inputs.to(device), labels.to(device)  # 注意需要复制到GPU
                self.optimizer.zero_grad()
                # print(len(inputs))
                outputs = self.net.forward(inputs)
                # print(len(outputs))
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                loss100 += loss.item()
                if i % 2 == 0:
                    print('[Epoch %d, Batch %5d] loss: %.3f' %
                          (epoch + 1, i + 1, loss100 / 2))
                    loss100 = 0.0
        # 保存网络模型 保存整个模型
        torch.save(self.net, 'model_shanbu_128.pkl')
        print("Done Training!")
    def TestingData(self):
        model_net = torch.load('model_shanbu_128.pkl')
        self.LoadData()
        # 构造测试的dataloader
        dataiter = iter(self.testloader)
        # 预测正确的数量和总数量
        correct = 0
        total = 0
        # 使用torch.no_grad的话在前向传播中不记录梯度，节省内存
        # cv2.namedWindow('predictPic', cv2.WINDOW_NORMAL)
        to_pil_image  = transforms.ToPILImage()
        with torch.no_grad():
            for images, labels in dataiter:
                # images, labels = data
                # print(images)
                print(len(images.data))
                # images, labels = images.to(device), labels.to(device)
                # 预测
                # outputs = self.net(images)
                outputs = model_net(images)

                # 我们的网络输出的实际上是个概率分布，去最大概率的哪一项作为预测分类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # print(images.data[0])
                # print(len(images.data[0]))
                # input_flag = input()
                # if input_flag == 'p':
                #     break
                # elif input_flag == 'c':
                #     continue
                # cv2.imshow('predictPic', images)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        print('Accuracy of the self.network on the 10000 test images: %d %%' % (
                100 * correct / total))
def main():
    ImgCla = ImageClassify()
    ImgCla.GetImgData()
    ImgCla.TrainingData()
    ImgCla.TestingData()
    pass

if __name__ == '__main__':
    main()
