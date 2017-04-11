# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 19:13:34 2017

@author: Administrator
"""
import urllib
from PIL import Image  

from numpy import matrix
from numpy import loadtxt
def prepare_data(url):
    print("downloading ....")
    response = urllib.request.urlopen(url)
    html = response.read()
    fp = open("todo.png","wb")
    fp.write(html)
    fp.close()
    im = Image.open("todo.png")    
                     
    im.save("1.gif","gif")   
   
if(__name__ =='__main__'):
    prepare_data('http://jwxt.njupt.edu.cn/CheckCode.aspx')
