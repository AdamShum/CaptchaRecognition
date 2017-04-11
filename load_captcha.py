
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

ROW   = 27
COL   = 72
depth = 1

def read_images(train_dir):
    images = os.listdir(train_dir)
    # volume of data_set (train_dir)
    data_volume = len(images)
#    data_volume = 5

    # return to the main Call function
    # data.shape  = (volume,row,col,depth)
    data  = np.zeros((data_volume,ROW*COL))
    # label.shape = (volume,row,col,depth)
    #label = np.zeros((data_volume,1))
    #label = np.zeros((data_volume,1))
    #当label长度不是5的时候
    label = np.zeros((data_volume,4))
    for loop in range(1):
        image_name  =  images[loop]
        image       =  os.path.join(train_dir,image_name)
       # img         =  cv2.imread(image,cv2.IMREAD_GRAYSCALE)
        img        =  Image.open(image).convert('L')
        img        =  np.array(img)
        #XXX: I want to feed data by COL
        #XXX: 按列来组织图片数据

        img         =  img.transpose(1,0)
        img         =  img.reshape((ROW*COL))
        #img         =  img.reshape((ROW,COL,depth))
        data[loop]  =  img

        image_name  =  image_name.strip()
        image_name  =  image_name.split('.')[0]
        image_name  =  image_name.split('_')[1]
        image_label =  image_name[:]
        #当label长度不是5的时候
        print(image_label)
        print(type(image_label[0]))
        image_label = [0,0,0,0]
        #这里直接设置为4个，否则会为变值
        length_name = len(image_name)
        for i in range(length_name):
            image_label[i] = image_name[i]
        if  length_name != 4:
            image_label[3] = 10
        #    for j in xrange(length_name,5):
        #        image_label[i] = 10
        #label.append(image_label)
        
        label[loop] = image_label 
    return data,label


def read_data(train_dir,one_hot=False,reshape=True):
    train_images,train_labels = read_images(train_dir) 
    return train_images,train_labels
    
read_data('./img_train')
   