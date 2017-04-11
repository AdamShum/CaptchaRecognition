# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:22:06 2017

@author: cilab_2
"""
import matplotlib.pyplot as plt  
import numpy as np
from tqdm import tqdm
import imageProcess as IP

import downloadCaptcha as downPic
#dataNum = 300000
dataNum = 1
dataPredictNum = 100

print ("Loading train data...")




length_label  = 4
ROW   = 27
COL   = 72

def read_images(train_dir):
    global dataNum
    downPic.prepare_data('http://jwxt.njupt.edu.cn/CheckCode.aspx')
#fImgPridict = open('realImgTest.data', 'rb')
    if(train_dir == "2"):       
        dataNum = 1
    else:
        dataNum = 0
    data  = np.zeros((dataNum,ROW*COL))
    retlabel = np.zeros((dataNum,4))
    img = []
    label = []

    for index in tqdm(range(dataNum)):
        
        img = []
        label = []
        if(train_dir == "1"):           
            pass
                
        else:
            _, img = IP.loadImageFromFile('./1.gif')
            a = IP.preProcess(img)
            img = np.array(a).reshape(ROW,COL)
            plt.imshow(img)
            
        label = [0,0,0,0]
        

        label = str(label)
        img =  np.array(img)
        img =  img.transpose(1,0)
        img =  img.reshape((ROW*COL))
        data[index]  =img

        image_label = [0,0,0,0]      
       
        retlabel[index] = image_label 
#        print(retlabel)

    return data,retlabel
    
    

def read_data(train_dir,one_hot=False,reshape=True):
    train_images,train_labels = read_images(train_dir) 
    return train_images,train_labels

  
  
    