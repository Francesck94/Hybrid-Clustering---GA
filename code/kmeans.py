# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:01:21 2018

@author: francesco
"""
# %% Import library and clear
#for name in dir():
#    if not name.startswith('_'):
#        del globals()[name]

import matplotlib.pyplot as plt
import random
import numpy as np
import statistics as st
import distanza_custom as d
import math
import sys

# %% class definition

class k_means:
    
    def __init__(self,k,iters,soglia,init,start_cent,W):
        self.k = k
        self.iters = iters
        self.soglia = soglia
        self.init = init
        self.start_cent = start_cent
        self.W = W
    
    def train(self,data):
        dim = data.shape[1]
        n_pat = data.shape[0]        
        idx = random.sample(range(len(data)),self.k)
        
        if str(self.init) == 'random':
            self.cent = np.array([data[i] for i in idx],dtype=float)
        elif str(self.init) == 'predef':
            self.cent = np.array([data[i] for i in self.start_cent],dtype = float)
        else:
            self.cent = np.array([data[i] for i in idx],dtype=float)
            
        self.I = []
        
        for i in range(self.iters):
            old_cent = np.array(self.cent)
#    print ('old cent=',old_cent,'\n')
#    I = []
            for n in range(n_pat):
                minIdx = 0
#                minVal = np.linalg.norm(data[n,:]-self.cent[minIdx,:])
                minVal = d.dweight(data[n,:],self.cent[minIdx,:],self.W)
                for j in range(0,self.k):
#                    dist = np.linalg.norm(data[n,:]-self.cent[j,:])
                    dist = d.dweight(data[n,:],self.cent[j,:],self.W)
                    if dist < minVal:
                        minVal = dist
                        minIdx = j #considero label 1,2,3.. (saltando 0)
        
                if i == 0:
                    self.I.append(minIdx)
                else:
                    self.I[n] = minIdx
     
#    #centroide
                indici = [x for x,item in enumerate(self.I) if item==minIdx]
       
                self.cent[minIdx] = np.median(data[indici,:],axis=0)
#    print('new cent=',cent,'\n')
   
    #soglia
            
            delta = sum(np.linalg.norm(self.cent - old_cent,axis=0))/self.k
            if delta < self.soglia:
#                print (i)
                break
        
        self.i = i
        self.delta = delta
# %% Main
if __name__=='__main__':
    W = np.matrix([[1,0],[0,1]])
    start = [10,20,30,40]
    x = k_means(4,10,0.001,'predef',start,W)
    x.train(X)
    C = x.cent
    M = x.I
    print (C)
