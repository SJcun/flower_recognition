

'''
测试
'''

#这个是读取训练好的模型
#再测试
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms

#import VGG16_model

# 定义一些超参数
batch_size = 100  #批大小

#数据预处理
data_transform = transforms.Compose([
    transforms.Resize((224,224), 2),                           #对图像大小统一
    transforms.RandomHorizontalFlip(),                        #图像翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[    #图像归一化
                             0.229, 0.224, 0.225])
         ])

test_dataset = torchvision.datasets.ImageFolder(root='work/data/test/', transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True, num_workers=0)

data_classes = test_dataset.classes

#选择CPU还是GPU的操作
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#选择模型
net = models.vgg16()
net.classifier = nn.Sequential(nn.Linear(25088, 4096),      #vgg16
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 20))
#net = VGG16_model.VGG16()
net.load_state_dict(torch.load("VGG16_flower_200.pkl"))
net.to(device)
#net.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        images, labels = Variable(images), Variable(labels)
        
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

