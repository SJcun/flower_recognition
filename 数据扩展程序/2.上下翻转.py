
import os
from PIL import Image
#import matplotlib.pyplot as plt

flowe_classes = ['peach_blossom','Jasminum','Matthiola',
                 'Rosa','Rhododendron','Dianthus','Cerasus','Narcissus','Pharbitis','Gazania',
                 'Eschscholtzia','Tithonia'] 

for name in flowe_classes:
    a = os.listdir('work/data/'+name)
    count = 1
    print(name)
    for x in a:
        
        img=Image.open('work/data/'+name+'/'+x)
        dst=img.transpose(Image.FLIP_TOP_BOTTOM)#上下互换
        newname='work/data/'+name+'/'+name+'1_'+str(count)+'.jpg'
        dst=dst.convert('RGB')
        dst.save(newname)
        count += 1
