'''
改成用现成的模型
'''
import torch
from torchvision import models
from torch import nn, optim
from torch.utils.data import DataLoader
#import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

#import model

# 定义一些超参数
batch_size = 100  #批大小
learning_rate = 0.001  #学习率
num_epoches = 15000    #期望训练次数

#数据预处理
data_transform = transforms.Compose([
    #transforms.Scale((224,224), 2),                           #对图像大小统一
    transforms.Resize([224, 224], 2),
    transforms.RandomHorizontalFlip(),                        #图像翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[    #图像归一化
                             0.229, 0.224, 0.225])
         ])

#获取数据集
#训练集
train_dataset = ImageFolder(root='work/data/train/',transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
#验证集
val_dataset = ImageFolder(root='work/data/val/', transform=data_transform)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True)                                               

#类别
data_classes = train_dataset.classes

#选择模型
net = models.alexnet()
net.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 20),
        )


#net.load_state_dict(torch.load('alexnet_flower_450.pkl'))

#损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

#选择CPU还是GPU的操作
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)

#开始训练
for epoch in range(num_epoches):
 
    running_loss = 0.
    
    for i, data in enumerate(train_loader):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = Variable(inputs), Variable(labels)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        print('[%d, %5d] loss: %.4f' %(epoch + 1, (i+1)*batch_size, loss.item()))

    '''
    #可以每隔一段时间保存一次模型参数
    if((epoch+1)%100 == 0) :
        torch.save(net.state_dict(), 'alexnet_flower_'+str(epoch+1)+'.pkl')
    if((epoch+1)%20 == 0) :
        print('是否继续训练？')
        a=input()
        if (a=='y'):
            continue
        else:
            break
    '''
 
print('Finished Training')

#保存模型参数
torch.save(net.state_dict(), 'flower.pkl')
