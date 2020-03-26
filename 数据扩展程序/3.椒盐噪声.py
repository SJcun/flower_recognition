
#coding=utf-8
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 


flowe_classes = ['peach_blossom','Jasminum','Matthiola',
                 'Rosa','Rhododendron','Dianthus','Cerasus','Narcissus','Pharbitis','Gazania',
                 'Eschscholtzia','Tithonia'] 

for name in flowe_classes:
    a=os.listdir('work/data/'+name) 
    count=1
    print(name)
    for x in a:
        oldname='work/data/'+name+'/'+x
        img=np.array(Image.open(oldname)) 
        #随机生成5000个椒盐
        rows,cols,dims=img.shape
        for i in range(5000):
            x=np.random.randint(0,rows)
            y=np.random.randint(0,cols)
            img[x,y,:]=255
        img.flags.writeable = True  # 将数组改为读写模式
        dst=Image.fromarray(np.uint8(img))
        newname='work/data/'+name+'/'+name+'2_'+str(count)+'.jpg'
        dst=dst.convert('RGB')
        dst.save(newname)
        count+=1