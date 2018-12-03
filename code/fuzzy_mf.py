# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:12:58 2018

@author: francesco
"""

import skfuzzy as fuzz
import numpy as np


    
def score(data,cluster,raggio,dist):
    
    mf = []

    for i in range(len(data)):
    
        idx = cluster[i]
        m = raggio[idx]
    
        FWHM = 2*m
        sigma = FWHM/2.355
    
        mf.append(fuzz.gaussmf(dist[i],0,sigma))
    
    return mf
    
    
