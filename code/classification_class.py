# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 12:05:13 2018

@author: francesco
"""


import random
import numpy as np
import kmeans as km
import pandas as pd



# %% Decision region
  
class classify:
    
    def __init__(self,train,cent):
        
        self.train = train
        self.cent = cent
        
    def regione_dec(self,I):
        
        self.I = I
        self.k = len(self.cent)
        self.raggio = []
        for i in range(self.k):
            dist = 0
            index = [n for n,item in enumerate(self.I) if item==i]
            for n in index:
                dist = dist + np.linalg.norm(self.train[n,:]-self.cent[i,:])
            nc = len(index)
            self.raggio.append(dist/nc)
        
# %% Validation
    
    def validation(self,target):
    
        self.I_cluster = []
        self.Dist_set = []
        self.I_class = []
        
        dim = target.shape[1] - 1   #Il -1 deriva dal fatto che l'ultima colonna è la classe
                                    # di apparteneza
                                    
        for x in target.iloc[:,0:dim].values:
            min_ind = 0
            min_norm = np.linalg.norm(x - self.cent[0,:])
            for j in range(self.k):
                norm = np.linalg.norm(x - self.cent[j,:])
                if norm < min_norm:
                    min_norm = norm
                    min_ind = j
            self.I_cluster.append(min_ind)
            self.Dist_set.append(min_norm)
            
        for i in range(len(target)):
            if self.Dist_set[i] < self.raggio[self.I_cluster[i]]:
                self.I_class.append(1)
            else:
                self.I_class.append(0)
        
#    Id = pd.DataFrame(I_set,columns=['predict'])
#    X_val.index = range(0,len(X_val))
#    X_val=pd.concat([X_val,Id],axis = 1)
#    X_val['predict'] =  