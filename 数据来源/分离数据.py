import scipy.io
import numpy as np
import os
from PIL import Image
import shutil

########取出 imagelabels 文件的值############

imagelabels_path='imagelabels.mat'
labels = scipy.io.loadmat(imagelabels_path)
labels = np.array(labels['labels'][0])-1

######## 取出 flower dataset: train test valid 数据id标识 ########
setid_path='setid.mat'
setid = scipy.io.loadmat(setid_path)

validation = np.array(setid['valid'][0]) - 1
np.random.shuffle(validation)

train = np.array(setid['trnid'][0]) - 1
np.random.shuffle(train)

test=np.array(setid['tstid'][0]) -1
np.random.shuffle(test)
######## flower data path 数据保存路径 ########
flower_dir = list()

######## flower data dirs 生成保存数据的绝对路径和名称 ########
for img in os.listdir("jpg"):
    
    ######## flower data ########
    flower_dir.append(os.path.join("jpg", img))

######## flower data dirs sort 数据的绝对路径和名称排序 从小到大 ########
flower_dir.sort()

#print(flower_dir)

#####生成flower data train的分类数据 #######
des_folder_train="prepare_pic\\train"
for tid in train:
    ######## open image and get label ########
    img=Image.open(flower_dir[tid])
    #print(flower_dir[tid])
    ######## resize img #######
    img = img.resize((256, 256),Image.ANTIALIAS)
    lable=labels[tid]
    #print(lable)
    
    path=flower_dir[tid]
    #print("path:",path)
    
    base_path=os.path.basename(path)
    #print("base_path:",base_path) 
    ######类别目录路径
    classes="c"+str(lable)
    class_path=os.path.join(des_folder_train,classes)
    # 没有这个文件夹，就创造这个文件夹
    if not os.path.exists(class_path):
        os.makedirs(class_path) 
    
    #print("class_path:",class_path) 
    despath=os.path.join(class_path,base_path)
    #print("despath:",despath)
    img.save(despath)


#####生成flower data validation的分类数据 #######   
des_folder_validation="prepare_pic\\validation"

for tid in validation:
    ######## open image and get label ########
    img=Image.open(flower_dir[tid])
    #print(flower_dir[tid])
    img = img.resize((256, 256),Image.ANTIALIAS)
    lable=labels[tid]
    #print(lable)
    path=flower_dir[tid]
    print("path:",path)
    base_path=os.path.basename(path)
    print("base_path:",base_path) 
    classes="c"+str(lable)
    class_path=os.path.join(des_folder_validation,classes)
    # 没有这个文件夹，就创造这个文件夹
    if not os.path.exists(class_path):

        os.makedirs(class_path) 
    print("class_path:",class_path) 
    despath=os.path.join(class_path,base_path)
    print("despath:",despath)
    img.save(despath)


#####生成flower data test的分类数据 #######     
des_folder_test="prepare_pic\\test"
for tid in test:
    ######## open image and get label ########
    img=Image.open(flower_dir[tid])
    #print(flower_dir[tid])
    img = img.resize((256, 256),Image.ANTIALIAS)
    lable=labels[tid]
    #print(lable)
    path=flower_dir[tid]
    print("path:",path)
    base_path=os.path.basename(path)
    print("base_path:",base_path) 
    classes="c"+str(lable)
    class_path=os.path.join(des_folder_test,classes)
    # 没有这个文件夹，就创造这个文件夹
    if not os.path.exists(class_path):
        os.makedirs(class_path) 
    print("class_path:",class_path) 
    despath=os.path.join(class_path,base_path)
    print("despath:",despath)
    img.save(despath)