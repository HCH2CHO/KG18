# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 22:55:13 2018

@author: HCHO
"""

#import os
read=open('test.nt','r')
write=open('new.nt','w',encoding='utf-8')
re=read.readlines()
num=len(re)
for i in range(0,num):
    line=re[i].encode('utf-8').decode('unicode_escape')
    write.write(line)
    if (i%10000==0):
        print(line)
read.close()
write.close()