# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:21:53 2018

@author: francesco
"""

from sklearn import preprocessing
import sys
import numpy as np
import pandas as pd



def normz(data,mode):
    
    if mode == 'affine':
        min_max_scaler = preprocessing.MinMaxScaler()
        norm_data = min_max_scaler.fit_transform(data)
        
    elif mode == 'statistica':
        
        norm_data = preprocessing.scale(data)
    
    else:
        
        min_max_scaler = preprocessing.MinMaxScaler()
        norm_data = min_max_scaler.fit_transform(data)


    return norm_data

