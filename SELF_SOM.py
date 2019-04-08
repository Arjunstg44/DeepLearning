# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:04:17 2019

@author: arjun
"""
import time
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn import datasets
import seaborn as sns
import math
iris=datasets.load_breast_cancer()
X = iris.data
y = iris.target
dims=[10,10]
alpha=0.3
radius=1.0

epochs=2

from sklearn.preprocessing import MinMaxScaler 
ms=MinMaxScaler((-1,1))
ms.fit(X=X)
x_scaled=ms.transform(X)
winners=[]
x_scaled=x_scaled[:20,2:4]
ms2=MinMaxScaler((0,1))
differences=[]
weights=np.random.uniform(0,1,(dims[0]*dims[1],x_scaled.shape[1]))*(-1)
old_weights=np.copy(weights)
from sklearn.metrics import mean_squared_error
print(time.asctime())
nbhood=[]
plt.plot(x_scaled[:,0],x_scaled[:,1],'k*')
#plt.scatter(weights[:,0],weights[:,1])
plt.pause(5)
for i in range(epochs):
    print("epoch"+str(i))
    summ=0
    for j,val in enumerate(x_scaled):#for each input vector val
        
        dist_list=[distance.euclidean(weight,val) for weight in weights]
        BMU=dist_list.index(min(dist_list))
        #we have found closest node: Now move it and move its neighbours
        #moving it(learning rate alpha)
        #move members within its radius
        #distance.
        #plt.scatter(x_scaled[:,0],x_scaled[:,1])
        #plt.scatter(weights[:,0],weights[:,1],s=8)
        #plt.pause(2)
        
        dist=np.array([distance.euclidean([i//dims[0],i%dims[1]],[BMU//dims[0],BMU//dims[1]]) for i in range(dims[0]*dims[1])])
        #print(dist)
        #dist=dist[dist<=radius]
        #weights=[]
        
        ct=0
        for iter_wt in range(dims[0]*dims[1]):
            if True:
                #dist=np.sqrt((weights[iter_wt,:])+(x_scaled[j,:]))
                dist = distance.euclidean(weights[iter_wt,:],val)#change neighbourhood func
                #dist=ms2.fit_transform(dist)
                if dist<=radius:#neighbourhood function filtering out 
                  ct+=1
                  #bring neighbours closer to datapoint, neighbours farther moving less
                  theta=math.exp(-dist**2/(2*radius**2))
                  weights[iter_wt,:]=weights[iter_wt,:]+theta*alpha*(val-weights[iter_wt,:])
                  summ+=theta*alpha*(val-weights[iter_wt,:])
     
        #print("nodes found in nbhood:"+str(ct))
        #if(ct==0):
         #   print("nothing hppaned: inspect")
        #nbhood.append(ct)
    #print(summ/569)
    radius=radius*(1-i/epochs)#radius*math.exp(-i*10/epochs)
    #print(nbhood[-1])
    #print(radius)
    alpha=alpha*(1-i/epochs)#math.exp(-i*10/epochs)
    #sns.heatmap(weights-old_weights)
    #plt.show(block=True)
    
    old_weights=np.copy(weights)
    #if i==1:
     #   print(weights-old_weights)
plt.scatter(weights[:,0],weights[:,1],s=8)
print(time.asctime())
winners=[]
'''
import collections
for x in enumerate(x_scaled):
    units=[distance.euclidean(x,wt) for wt in weights ]
    winners.append(units.index(min(units)))
print(collections.Counter(winners))
   ''' 
'''
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(np.ones((5,1)))#later change to internode distance
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(x_scaled):
    w=-1
    min_diff=100
    for iter_wt in range(25):
        try:
            min_here=mean_squared_error(weights[iter_wt,:],val)
        except:
            print(weights[iter_wt,:])
            print(val)
            break
        if min_here<min_diff:
            min_diff=min_here
            w=iter_wt
    w=[w//5,w%5]
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)

show()
'''

#sns.heatmap(weights-old_weights)