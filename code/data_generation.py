# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:17:53 2018

@author: francesco
"""
import matplotlib.pyplot as plt
import math
import sys
import numpy as np
import pandas as pd
import config
from normalize import normz


n_dim = 2
#mu1 = [0,0]
#mu2 = [3.5, 0]
#mu3 = [0.5, 3]
#mu4 = [3.5, 3]
mu1 = [0]*n_dim
mu2 = [3.5]+[0]*(n_dim-1)
mu3 = [0.5]+[3]*(n_dim-1)
mu4 = [3.5]+[3]*(n_dim-1)

#cov = [[0.5,0.05],[0.05,0.5]]
sigma_ii = 0.5
sigma_ij = 0.05


cov = np.matrix((np.ones((n_dim,n_dim))*sigma_ij - np.identity(n_dim)*sigma_ij)+
                np.identity(n_dim)*sigma_ii)

np.random.seed(0)
n_start = 1000
n1 = (n_start/10)*7
n2 = n_start - n1
d1 = np.random.multivariate_normal(mu1,cov,int(n1/2))
d2 = np.random.multivariate_normal(mu2,cov,int(n2/2))
d3 = np.random.multivariate_normal(mu3,cov,int(n2/2))
d4 = np.random.multivariate_normal(mu4,cov,int(n1/2))
X = np.concatenate((d1,d2,d3,d4),axis=0)
#labels = np.random.randint(2,size=(1000,1))
W = np.matrix([[1,0],[0,1]])

class1 = np.ones((int(n1/2),1))
class0 = np.zeros((int(n2/2),1))

labels = np.concatenate((class1,class0,class0,class1),axis=0)

X = normz(X,'affine')

dataset = pd.DataFrame(X)
dataset['classe'] = labels

config.dataset = dataset
config.n_dim = n_dim
