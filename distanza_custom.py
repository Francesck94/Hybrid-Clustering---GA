# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 15:27:30 2018

@author: francesco
"""

#distanza
import numpy as np
import math
import sys


#dw = float((x-y)*W*np.transpose(x-y))
#
#dw = math.sqrt(dw)

def dweight(x,y,W):
    r1 = x.size
    r2 = y.size
    
    rw,cw = W.shape
    
    if r1!=r2:
        print('error: x e y devono avere la stessa dimensione')
        sys.exit(1)
    
    if r1!=rw or r1!=cw:
        print('error: x e y devono avere dimensione pari alle righe e colonne di W')
        
    dw = float((x-y)*W*np.transpose([x-y]))
#    dw = math.sqrt(dw)
    
    return dw

if __name__=='__main__':
    
    x = np.array([1,2])
    y = np.array([5,6])
    W = np.matrix([[1,0],[0,1]])
    print (dweight(x,y,W))
    print (dweight(X[0,:],X[1,:],W))