# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:29:57 2018

@author: francesco
"""


import random
import numpy as np
import statistics as st
import kmeans as km

import pandas as pd
import classification as cl
from sklearn.metrics import accuracy_score,confusion_matrix
import math


from deap import base
from deap import creator
from deap import tools

from deap import algorithms
from scoop import futures
from multiprocessing import Pool
from operator import attrgetter
from multiprocessing.dummy import Pool as ThreadPool


#Funzione per la valutazione della fitness    
def evalscore(individual,train_set,val_set,cluster,tipo,cod,kpar):
   
    
   n_dim = train_set.shape[1]
   
   if str(tipo) == 'tria':
    
       M = cod(individual,n_dim)
   
   elif str(tipo) == 'diag':
        
        codice = np.array(individual)
        M = np.matrix(np.diag(codice))
    
   else:
        
        codice = np.array(individual)
        M = np.matrix(np.diag(codice))
        
    
#   start = [0,19,29,59]
#   x = cluster.k_means(4,20,0.001,'predef',start,M)
   x = cluster.k_means(kpar.k,kpar.iter,kpar.soglia,kpar.init,kpar.start,M)
   
   x.train(train_set)
   
   # validation
   obj = cl.classify(train_set,x.cent)
   obj.regione_dec(x.I)
   obj.validation(val_set)

   Id = pd.DataFrame(obj.I_class,columns=['predict'])

#   val_set=pd.concat([val_set,Id],axis = 1)
#   
#   score = accuracy_score(val_set['classe'], val_set['predict'])
   
   score = accuracy_score(val_set['classe'],Id)
   
   return score,

#funzione per il controllo dei valori dei pesi
def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrapper
    return decorator
