# -*- coding: utf-8 -*-

"""
Created on Thu Nov  8 19:37:53 2018

@author: francesco
"""

#for name in dir():
#    if not name.startswith('_'):
#        del globals()[name]

import math
import sys
import numpy as np
import pandas as pd
#import config


#test_set = config.test_set
#val_set = config.val_set

def Platt(labels,scores):
    
    
    
    # %% Input parameters:
    #out = val_set.score
    deci = scores
    #target = val_set.classe
    label = labels
    label = label[:].tolist()
    prior1 = label.count(True)
    prior0 = label.count(False)
    
    # %% Parameters setting
    maxiter = 100
    minstep = 1e-10
    sigma = 1e-12
    
    # %% Output
    A = 0
    B = math.log((prior0+1.0)/(prior1 +1.0))
    fval = 0.0
    hiTarget = (prior1 + 1.0)/(prior1 + 2.0)
    loTarget = 1/(prior0 + 2.0)
    t = []
    
    for i in range(len(labels)):
        if (label[i] > 0):
            t.append(hiTarget)
        else:
            t.append(loTarget)
    
        
    for i in range(len(labels)):
        fApB = deci[i]*A+B
        
        if (fApB >= 0):
            fval += t[i]*fApB+math.log(1+math.exp(-fApB))
        else:
            fval += (t[i]-1)*fApB+log(1+exp(fApB))
    
    
    # %%
    
    for it in range(maxiter):
        
        h11=h22=sigma
        h21=g1=g2=0.0
        
        for i in range(len(labels)):
            
           fApB = deci[i]*A+B
           
           if (fApB >= 0):
               p=math.exp(-fApB)/(1.0+math.exp(-fApB))
               q=1.0/(1.0+math.exp(-fApB))
           else:
               p=1.0/(1.0+math.exp(fApB))
               q=math.exp(fApB)/(1.0+math.exp(fApB))
               
           d2=p*q
           h11 += deci[i]*deci[i]*d2
           h22 += d2
           h21 += deci[i]*d2
           d1=t[i]-p
           g1 += deci[i]*d1
           g2 += d1
        
        
        if (abs(g1) < 1e-5 and abs(g2) < 1e-5):
#            print('first break\n')
            break
        
        det=h11*h22-h21*h21
        dA=-(h22*g1-h21*g2)/det
        dB=-(-h21*g1+h11*g2)/det
        gd=g1*dA+g2*dB
        stepsize=1
        
        while (stepsize >= minstep):
            newA=A+stepsize*dA
            newB=B+stepsize*dB
            newf=0.0
            
            for i in range(len(labels)):
                fApB=deci[i]*newA+newB
                if (fApB >= 0):
                    newf += t[i]*fApB+math.log(1+math.exp(-fApB))
                else:
                    newf += (t[i]-1)*fApB+math.log(1+math.exp(fApB))
    
            if (newf < fval+0.0001*stepsize*gd):
                    A=newA
                    B=newB
                    fval=newf
#                    print('second break\n')
                    break 
            else:
                stepsize /= 2.0
        
        if (stepsize < minstep):
                print ('Line search fails')
#                print ('third break')
                break
    
    if (it >= maxiter):
        print ('Reaching maximum iterations')
    
    return A,B
    # %% test
#            
#    dati_test = config.dati_test
#    mf_test = dati_test['mf']
#    pt = [1/(1+math.exp(mf_test[i]*A + B)) for i in range(len(dati_test))]
#    dati_test['prob'] = pt
