#!/usr/bin/env python
import csv
import numpy as np
import random
from numpy.linalg import inv
import time


def basic(x,y,u1,u2):
    return np.exp(-(x-u1)**2/(2*(5**2))-(y-u2)**2/(2*(5**2)))
if __name__ == '__main__':
    
    
    np.set_printoptions(threshold='nan')
    validation=[]
    training=[]
    K_fold=1
    for i in reversed(range(5)):
        validation_temp = [x for x in range(i*8000,(i+1)*8000)]
        #print nums
        validation.append(validation_temp)
        training_temp=np.arange(40000)[np.concatenate(( np.where(np.arange(40000)<i*8000),np.where(np.arange(40000)>((i+1)*8000-1))),axis=1)]
        training.append(training_temp[0])
    
    #print training[0]
    with open('X_train.csv', 'rb') as f:
        reader = csv.reader(f)
        pos = list(reader)
    with open('T_train.csv', 'rb') as f:
        reader = csv.reader(f)
        height = list(reader)
        
    for K_fold in range(5):
        tStart = time.time()
        Matrix=np.zeros((32000, 4900))
       
        train_height=np.zeros(32000,dtype=np.float64)
        
        for i in range(32000):
            train_height[i]=int(height[training[K_fold][i]][0])
        #print train_height
        
        for num in range(32000):
            #print num
            for i in range(70):
                for j in range(70):
                    Matrix[num][i*70+j]=basic(int(pos[training[K_fold][num]][0]),int(pos[training[K_fold][num]][1]),15*(j+1),15*(i+1))
                    #print (i,j),(pos[num][1],pos[num][0]),Matrix[num][i*70+j]
                    
        WML=np.dot(np.dot(inv(np.dot(Matrix.transpose(), Matrix)),Matrix.transpose()),train_height)
        #print WML
        
        
        val=np.zeros(4900,dtype=np.float64)
        MSE=0.0
        for num in range(8000):
            #print num
            for i in range(70):
                for j in range(70):
                    val[i*70+j]=basic(int(pos[validation[K_fold][num]][0]),int(pos[validation[K_fold][num]][1]),15*(j+1),15*(i+1))
            MSE=MSE+np.power(np.dot(WML,val)-float(height[validation[K_fold][num]][0]),2)
            
        print MSE/8000
        tEnd = time.time()
        print "It cost %f sec" % (tEnd - tStart)