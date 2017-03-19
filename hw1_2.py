#!/usr/bin/env python
import csv
import numpy as np
import random
from numpy.linalg import inv
import time


def basic(x,y,u1,u2):
    return np.exp(-(x-u1)**2/(2*(15**2))-(y-u2)**2/(2*(15**2)))
if __name__ == '__main__':
    
    
    np.set_printoptions(threshold='nan')
    validation=[]
    training=[]
    K_fold=2
    min_MSE=float('Inf')
    all_data_index = [x for x in range(40000)]
    for i in range(K_fold):
        
        random.shuffle(all_data_index)
        #print all_data_index
        validation_temp=[x for index,x in enumerate(all_data_index) if index > 40000*(K_fold-1)/K_fold-1]
        validation.append(validation_temp)
        training_temp=[x for index,x in enumerate(all_data_index) if index < 40000*(K_fold-1)/K_fold]
        training.append(training_temp)
    
    #print training[0]
    
    with open('X_train.csv', 'rb') as f:
        reader = csv.reader(f)
        pos = list(reader)
    with open('T_train.csv', 'rb') as f:
        reader = csv.reader(f)
        height = list(reader)
    with open('X_test.csv', 'rb') as f:
        reader = csv.reader(f)
        pos_test = list(reader)
    print("start training")     
    for K_fold_num in range(K_fold):
        tStart = time.time()
        num_traing_data=(40000*(K_fold-1)/K_fold)
        Matrix=np.zeros((num_traing_data, 4900))
       
        train_height=np.zeros(num_traing_data,dtype=np.float64)
        
        for i in range(num_traing_data):
            train_height[i]=int(height[training[K_fold_num][i]][0])
        #print train_height
        
        for num in range(num_traing_data):
            #print num
            for i in range(70):
                for j in range(70):
                    Matrix[num][i*70+j]=basic(int(pos[training[K_fold_num][num]][0]),int(pos[training[K_fold_num][num]][1]),15*j+7,15*i+7)
                    #print (i,j),(pos[num][1],pos[num][0]),Matrix[num][i*70+j]
        landa=5
        I_matrix=np.diag(np.diag(np.ones((4900,4900))))
        WML=np.dot(np.dot(inv(landa*I_matrix+np.dot(Matrix.transpose(), Matrix)),Matrix.transpose()),train_height)
        #print WML

        val=np.zeros(4900,dtype=np.float64)
        MSE=0.0
        num_validatation_data=(40000/K_fold)
        for num in range(num_validatation_data):
            #print num
            for i in range(70):
                for j in range(70):
                    val[i*70+j]=basic(int(pos[validation[K_fold_num][num]][0]),int(pos[validation[K_fold_num][num]][1]),15*j+7,15*i+7)
            MSE=MSE+np.power(np.dot(WML,val)-float(height[validation[K_fold_num][num]][0]),2)
            
        MSE=MSE/2+landa*np.dot(WML,WML.transpose())/2 
        print MSE
        if min_MSE > MSE:
            WML_min=WML
            min_MSE=MSE
            
        tEnd = time.time()
        print "It cost %f sec" % (tEnd - tStart)
    
    with open('test2.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        val=np.zeros(4900,dtype=np.float64)
        predict_h=np.zeros(len(pos_test),dtype=np.float64)
        for num in range(len(pos_test)):
            for i in range(70):
                for j in range(70):
                    val[i*70+j]=basic(int(pos_test[num][0]),int(pos_test[num][1]),15*j+7,15*i+7)
            predict_h[num]=np.dot(WML_min,val)
            
        predict_h.shape=(len(pos_test),1)    
        writer.writerows(predict_h)