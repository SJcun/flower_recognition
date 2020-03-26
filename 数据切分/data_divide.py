'''
将原有数据集划分为训练集、验证集和测试集
'''


import os
import random
#import shutil
from shutil import copy2

#比例
scale = [0.6, 0.2, 0.2]

#类别
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips','Nymphaea','Tropaeolum_majus','Digitalis_purpurea','peach_blossom',
'Jasminum','Matthiola','Rosa','Rhododendron','Dianthus','Cerasus','Narcissus','Pharbitis','Gazania','Eschscholtzia','Tithonia']


for each in classes:
    datadir_normal = "work/data/"+each+"/"  #原文件夹
    
    all_data = os.listdir(datadir_normal)#（图片文件夹）
    num_all_data = len(all_data)
    print(each+ "类图片数量: " + str(num_all_data) )
    index_list = list(range(num_all_data))
    #print(index_list)
    random.shuffle(index_list)
    num = 0
    
    trainDir = "work/new_data/train/"+each#（将训练集放在这个文件夹下）
    if not os.path.exists(trainDir):  #如果不存在这个文件夹，就创造一个
        os.makedirs(trainDir)
            
    validDir = "work/new_data/val/"+each#（将验证集放在这个文件夹下）
    if not os.path.exists(validDir):
        os.makedirs(validDir)
            
    testDir = "work/new_data/test/"+each#（将测试集放在这个文件夹下）        
    if not os.path.exists(testDir):
        os.makedirs(testDir)
            
    for i in index_list:
        fileName = os.path.join(datadir_normal, all_data[i])
        if num < num_all_data*scale[0]:
            #print(str(fileName))
            copy2(fileName, trainDir)
        elif num>num_all_data*scale[0] and num < num_all_data*(scale[0]+scale[1]):
            #print(str(fileName))
            copy2(fileName, validDir)
        else:
            copy2(fileName, testDir)
        num += 1