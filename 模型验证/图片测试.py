

from PIL import Image
import torchvision.transforms as transforms
from torchvision import models  #人家的模型
from torch.autograd import Variable
import torch
#from torchvision.datasets import ImageFolder
from torch import nn
#import VGG16_model


#数据预处理
data_transform = transforms.Compose([
    transforms.Resize((224,224), 2),                           #对图像大小统一
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[    #图像归一化
                             0.229, 0.224, 0.225])
         ])

#类别
#这个类别是我在训练的过程输出的训练集的类别，是按照训练的顺序排列的
data_classes = ['Cerasus', 'Dianthus', 'Digitalis_purpurea', 'Eschscholtzia', 
                'Gazania', 'Jasminum', 'Matthiola', 'Narcissus', 'Nymphaea', 
                'Pharbitis', 'Rhododendron', 'Rosa', 'Tithonia', 'Tropaeolum_majus', 
                'daisy', 'dandelion', 'peach_blossom', 'roses', 'sunflowers', 'tulips']

#读取数据
img = Image.open('./图片/向日葵.jpg') 
img=data_transform(img)#这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
img = img.unsqueeze(0)#增加一维，输出的img格式为[1,C,H,W]

#类别
#train_dataset = ImageFolder(root='work/data/train/',transform=data_transform)
#data_classes = train_dataset.classes

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


#读取参数
net.load_state_dict(torch.load("VGG16_flower_200.pkl",map_location=torch.device('cpu')))
net.eval()
net.to(device)

img = Variable(img)
score = net(img)#将图片输入网络得到输出
probability = nn.functional.softmax(score,dim=1)#计算softmax，即该图片属于各类的概率
max_value,index = torch.max(probability,1)#找到最大概率对应的索引号，该图片即为该索引号对应的类别
print()
print("识别为'{}'的概率为{}".format(data_classes[index.item()],max_value.item()))


#pytorch网络输入图片的格式是[B,C,H,W],分别为batch（每批送入网络的图片数量），图片通道数，图片高，图片宽。
#通过PIL的Image读取的图片是一个图片对象，可以进行裁剪翻转等torchvision.transforms变换。



