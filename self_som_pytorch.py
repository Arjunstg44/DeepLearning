import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch.nn as nn
import time
import sys
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn import datasets
import seaborn as sns
import math
#iris=datasets.load_breast_cancer()
iris=datasets.load_iris()
X = iris.data
y = iris.target


from sklearn.preprocessing import MinMaxScaler 
ms=MinMaxScaler((-1,1))
ms.fit(X=X)
x_scaled=ms.transform(X)
#winners=[]
#x_scaled=x_scaled[:100,11:13]

#differences=[]
#weights=np.random.uniform(0,1,(dims[0]*dims[1],x_scaled.shape[1]))*(-1)
#old_weights=np.copy(weights)
#from sklearn.metrics import mean_squared_error
#print(time.asctime())
#nbhood=[]
#gondu1=0


#gondu2=0

class SOM(nn.Module):
    def __init__(self, m, n, dim, epochs, alpha, radius):
        super(SOM, self).__init__()
        self.m = m
        self.n = n
        self.dim = dim
        self.epochs = epochs
        self.alpha = float(alpha)
        self.radius = float(radius)
        self.weights = torch.randn(m*n, dim)
    
    def forward(self,x,it):
        
        gondu2=torch.pow(torch.stack([x for i in range(self.m*self.n)])-self.weights,2)
        gondu2=torch.sum(gondu2,1)
        
        _,bmu=torch.min(gondu2,0)
        
        learning_rate_op = 1.0 - it/self.epochs
        alpha_op = self.alpha * learning_rate_op
        radius_op = self.radius * learning_rate_op
        #bmu_distance_squares=torch.sum(torch.pow(  (  torch.stack([self.weights[i,:] for i in range(self.m*self.n)])-torch.stack([self.weights[bmu_index,:] for i in range(self.m*self.n)])  )  ,2),1)
        #print(bmu_index)
        bmu=bmu.numpy()
        #print(bmu_index)
        #bmu_index=bmu_index[0]
        stack1=torch.stack([torch.Tensor([bmu//self.m,bmu%self.n]) for i in range(self.m*self.n)])
        stack2=torch.stack([torch.Tensor([i//self.m,i%self.n]) for i in range(self.m*self.n)])
        bmu_distance_squares=torch.sum(torch.pow(stack1-stack2,2),1)
        neighbourhood_func=torch.exp(torch.neg(torch.div(bmu_distance_squares,2*radius_op**2)))
        learning_rate_op= alpha_op*neighbourhood_func
        learning_rate_multiplier = torch.stack([learning_rate_op[i:i+1].repeat(self.dim) for i in range(self.m*self.n)])
        delta=torch.mul(learning_rate_multiplier,(torch.stack([x for i in range(self.m*self.n)]) - self.weights  ))
        self.weights=torch.add(self.weights,delta)

    def map_vects(self,input_vect):
        dist = self.pdist(torch.stack([input_vect for i in (self.m*self.n)]),self.weights)
        _,bmu_index=torch.argmin(dist)
        

dat = list()
for i in range(x_scaled.shape[0]):
    dat.append(torch.FloatTensor(x_scaled[i,:]))
old=time.time()
som = SOM(20,20,x_scaled.shape[1],10,0.5,15)
for itera in range(10):
    print("epoch :"+str(itera),end="  ")
    
    print(time.time()-old)
    old=time.time()
    for data_x in dat:
        som.forward(data_x,itera)
weights=som.weights.numpy()
print(weights)
#print(time.asctime())

totns=som.m*som.n
som_map=np.ndarray((totns,totns))
for i in range(totns):
    for j in range(totns):
        som_map[i,j]=distance.euclidean(weights[i],weights[j])

som_viz=np.ndarray((som.m,som.n))
for i in range(som.m):
    for j in range(som.n):
        som_viz[i,j]=som_map[i//som.m,i+j%som.n]
        
som_viz2=np.ndarray((som.m,som.n))
loxx=[]
import cv2
ctrlr=0
img=np.zeros((som.m,som.n,3))
colors=[[0,0,255],[0,255,0],[255,0,0]]
for i in range(len(y)):
     gondu2=torch.pow(torch.stack([torch.FloatTensor(x_scaled[i,:]) for k in range(som.m*som.n)])-som.weights,2)
     gondu2=torch.sum(gondu2,1)
     _,bmu=torch.min(gondu2,0)
     bmu=bmu.numpy()
     loxx.append(bmu)
     #if y[i]==0:
      #   ctrlr+=1
     img[bmu//som.m,bmu%som.n,:]=colors[y[i]]
img=cv2.resize(img,(400,400))
cv2.imshow('img',img)

sns.heatmap(som_viz)

import collections
#print(collections.Counter(list(loxx)))