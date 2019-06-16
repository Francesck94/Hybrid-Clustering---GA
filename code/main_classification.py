# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:49:23 2018

@author: francesco
"""

# %% clear
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
        

# alternativa for clear all
#from IPython import get_ipython
#get_ipython().magic('reset -sf')



# %% Import library
import matplotlib.pyplot as plt
import random
import numpy as np
import kmeans as km
from sklearn.model_selection import train_test_split
import pandas as pd
import classification_class as cl
from sklearn.metrics import accuracy_score,confusion_matrix
import time

from deap import base
from deap import creator
from deap import tools

from deap import algorithms
from scoop import futures
from multiprocessing import Pool
from operator import attrgetter
from multiprocessing.dummy import Pool as ThreadPool
from drawnow import drawnow

import plot_confusion_matrix as pcm
import skfuzzy as fuzz
from fuzzy import score as sc

import config
import gen_data
from genetic_functions import evalscore,checkBounds

import pickle
# %% Close open windows

plt.close('all')
# %% Take data
dataset = config.dataset
dim = config.n_dim

if dim ==2:
    plt.figure()
    plt.scatter(dataset[0],dataset[1],s=10)
else:
    pass
# %% train set, validation set, test set

#indici per dataset1
idx1 = [i for i,x in enumerate(dataset.classe == 1) if x]

#indici per daataset0
idx0 = [i for i,x in enumerate(dataset.classe == 0) if x]

dataset1 = dataset.loc[idx1]
dataset0 = dataset.loc[idx0]

# Suddivido dataset1 in train_set e test_val
X_train, X_test_val, y_train, y_test_val = train_test_split(dataset1,dataset1.classe,test_size = 0.4)

# Setto correttamente gli indici dei due insiemi
X_train.index = range(0,len(X_train))
X_test_val.index = range(0,len(X_test_val))

# Unisco dataset0 con test_val
data = pd.concat([dataset0, X_test_val])

# Ricavo Test set e Validation Set suddividendo in maniera equa al 50%
X_val, X_test, y_val, y_test = train_test_split(data, data.classe, test_size=0.5)

X_val.index = range(0,len(X_val))
X_test.index = range(0,len(X_test))

#Ricavo i valori dei pattern del training set
train_set = X_train.iloc[:,0:dim].values

    
# %% k-means parameters

class kmeans_par:
    
    def __init__(self):
        self.k = 4
        self.init = 'predef'
        self.start = [0,19,29,59]
        self.iter = 20
        self.soglia = 0.001


kpar = kmeans_par()

# %% Setup genetic algorithm

# dimensione pattern

#dim = 2

# Tipo codifica matrice: piena(tria) o diagonale(diag)
tipo = 'tria'

# In base al tipo di codifica si sceglie la dimensione del cromosoma
if str(tipo) == 'tria':
    IND_SIZE = int(dim*(dim+1)/2)

elif str(tipo) == 'diag':
    IND_SIZE = dim

else:
    IND_SIZE = dim


creator.create("FitnessMax", base.Fitness, weights=(1.0,)*IND_SIZE)
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)

#ind1 = toolbox.individual()   

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


#Funzione per costruire la matrice piena a partire dal cromosoma
def codifica(codice,dim):
    
    n = dim
    M = np.ones((n,n))
    M = np.tril(M)
    
    r = 0
    c= 0
    for i in codice:
        if M[r,c]!=0:
            M[r,c] = i
            c = c+1
        else:
            c = 0
            r = r+1
            M[r,c] = i
            c = c+1
    
    M = np.matrix(M + np.transpose(np.tril(M,k=-1)))
    
    return M



#Uso una lambda function perchè la funzione di valutazione deve prendere un solo argomento
#la funzione evalscore è importata dal modulo genetic_functions
f = lambda ind: evalscore(ind,train_set,X_val,km,tipo,codifica,kpar)


# Toolbox per le funzioni di evoluzione
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", f)

toolbox.decorate("mate", checkBounds(0, 1))
toolbox.decorate("mutate", checkBounds(0, 1))



# %% genetic algorithm
def ga(verbose=True):
    pop = toolbox.population(n=100)
#    print(pop)
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
 
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2
     # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
       # Variable keeping track of the number of generations
    g = 0
    
    avg_fit = []
    stall_f = 1
    
    plt.figure()
    # Begin the evolution
    while g < 100 and stall_f > 0.0001:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
         
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        
        avg_fit.append(mean)
        
        if g > 1:
            stall_f = abs(avg_fit[g-1] - avg_fit[g-2])
        
#        print(avg_fit)
#        print(stall_f)
        
        best = max(pop, key=attrgetter("fitness"))
        
        plt.scatter(g,max(fits),s=10,c='r')
#        drawnow(plt.scatter(g,max(fits)))
    plt.show()
    
    #ritorna il miglior individuo
    return best
# %% execute main
   
#Multithreading
pool = ThreadPool(4)
toolbox.register("map",pool.map)
    
start_time = time.time()  
best = ga()
elapsed_time = time.time() - start_time
print ('elapsed time=',elapsed_time)
    
    
    
    
# %% codifica matrice
    
if str(tipo) == 'tria':
    
    M = codifica(best,dim)
   
elif str(tipo) == 'diag':
    codice = np.array(best)
    M = np.matrix(np.diag(codice))

else:
    codice = np.array(best)
    M = np.matrix(np.diag(codice))
    

# %% best k-means

kt = km.k_means(kpar.k,kpar.iter,kpar.soglia,'predef',kpar.init,M)
kt.train(train_set)

if dim ==2:

    fig,ax = plt.subplots()
    for i in range(kt.k):
        indici = [n for n,item in enumerate(kt.I) if item==i]
        ax.scatter(train_set[indici,0],train_set[indici,1],label='c'+str(i))
    
    ax.scatter(kt.cent[:,0],kt.cent[:,1],color='k',marker='*')
    ax.legend()
    plt.title('kmeans output')
    plt.show

else:
    pass

# %% validation using best individual

 # validation
obj_val= cl.classify(train_set,kt.cent)
obj_val.regione_dec(kt.I)
obj_val.validation(X_val)

Val_cluster = pd.DataFrame(obj_val.I_cluster,columns=['cluster'])
Val_class = pd.DataFrame(obj_val.I_class,columns=['predict'])


val_set = pd.concat([X_val,Val_class],axis = 1)
val_set = pd.concat([val_set,Val_cluster],axis = 1)


score = accuracy_score(val_set['classe'], val_set['predict'])

# %% confusion matrix validation set

cm_val = confusion_matrix(val_set['classe'],val_set['predict'])

plt.figure()

pcm.plot_confusion_matrix(cm_val, classes=['1','0'],
                      title='Confusion matrix Val Set')
plt.show()

# %% plot validation set

if dim == 2:
    fig = plt.figure()
    plt.scatter(X_val[0],X_val[1],s=10)
    
    fig = plt.gcf()
    ax = fig.gca()
    r = 0
    for a,b in kt.cent:
        circ = plt.Circle((a,b), radius=obj_val.raggio[r], edgecolor='r', facecolor='None')
        r = r+1
        ax.add_artist(circ)
    
    plt.title('validation set')    
    #    ax.set_xlim((-4,8))
    #    ax.set_ylim((-4,8))    
    plt.show()
else:
    pass

# %% test on test_set

obj_test= cl.classify(train_set,kt.cent)
obj_test.regione_dec(kt.I)
obj_test.validation(X_test)

Test_cluster = pd.DataFrame(obj_test.I_cluster,columns=['cluster'])
Test_class = pd.DataFrame(obj_test.I_class,columns=['predict'])


test_set = pd.concat([X_test,Test_class],axis = 1)
test_set = pd.concat([test_set,Test_cluster],axis = 1)


test_score = accuracy_score(test_set['classe'], test_set['predict'])

# %% confusion matrix test set

cm_test = confusion_matrix(test_set['classe'],test_set['predict'])

plt.figure()

pcm.plot_confusion_matrix(cm_test, classes=['1','0'],
                      title='Confusion matrix Test Set')
plt.show()

# %%
print('Accuracy val set %1.4f' %(score))
print('Accuracy test set %1.4f' %(test_score))

# %% fuzzy score validation set


mf = sc(val_set,val_set.cluster,obj_val.raggio,obj_val.Dist_set)  
mf = pd.DataFrame(mf,columns=['score'])
val_set = pd.concat([val_set,mf],axis = 1)


# %% fuzzy score test set

mf_1 = sc(test_set,test_set.cluster,obj_test.raggio,obj_test.Dist_set)
mf_1 = pd.DataFrame(mf_1,columns=['score'])
test_set = pd.concat([test_set,mf_1],axis=1)

# %% config
config.val_set = val_set
config.test_set = test_set

# %%

with open('data_pre_calibration.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([val_set, test_set], f)


