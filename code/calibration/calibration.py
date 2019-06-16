# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:32:09 2019

@author: francesco
"""

import math
import sys
import numpy as np
from get_diagram_data import get_diagram_data
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, brier_score_loss
from Platt_scaling import Platt
from sklearn.isotonic import IsotonicRegression
import ml_insights as mli

import pickle

# %% Input data for calibration

#val_set = config.val_set
#test_set = config.test_set
with open('..\data_pre_calibration.pkl','rb') as f:  # Python 3: open(..., 'rb')
    val_set,test_set = pickle.load(f)
# %% Reliability Diagram validation set

m_train,p_train = get_diagram_data(val_set['classe'],val_set['score'],10)

# %% plot reliability diagram val set
plt.figure()
plt.scatter(m_train,p_train,c='r')
plt.xlabel('score')
plt.ylabel('empirical probabilities')
plt.title('Reliability Diagram Val Set')

# %% Reliability diagtam test set
m_test,p_test = get_diagram_data(test_set['classe'],test_set['score'],10)

# %% plot reliability diagram test set

plt.figure()
plt.scatter(m_test,p_test,c='b')
plt.xlabel('score')
plt.ylabel('empirical probabilities')
plt.title('Reliability Diagram Test Set')

# %% Platt scaling calibration

A,B = Platt(val_set['classe'],val_set['score'])

# %% Platt probabilities validation set

prob_platt_val = []

for i in range(len(val_set)):
    fApB = val_set.score[i]*A + B
    
    if (fApB >= 0 ):
        prob_platt_val.append(math.exp(-fApB)/(1.0 + math.exp(-fApB)))
    else:
        prob_platt_val.append(1.0/(1.0 + math.exp(fApB)))
        
# %% Platt probabilities test set

prob_platt_test = []

for i in range(len(test_set)):
    fApB = test_set.score[i]*A + B
    
    if (fApB >= 0 ):
        prob_platt_test.append(math.exp(-fApB)/(1.0 + math.exp(-fApB)))
    else:
        prob_platt_test.append(1.0/(1.0 + math.exp(fApB)))
        
# %% Isotonic Regression calibration
IR = IsotonicRegression()
IR.set_params(out_of_bounds='clip')
IR.fit(np.array(val_set['score']),np.array(val_set['classe']))

# %% Isotonic probabilities val set & test set

prob_iso_val = IR.transform(np.array(val_set['score']))
prob_iso_test = IR.transform(np.array(test_set['score']))
    
# %% SplineCalib calibration

calibrate_occ = mli.prob_calibration_function(val_set['classe'],val_set['score'])

# %% Spline probabilities val set & test set

prob_spline_val = calibrate_occ(val_set['score'])
prob_spline_test = calibrate_occ(test_set['score'])
        

# %% Reliability Diagram after calibration

#plt.figure()
m_test_platt,p_test_platt = get_diagram_data(test_set['classe'],np.array(prob_platt_test),8);
m_test_iso,p_test_iso = get_diagram_data(test_set['classe'],np.array(prob_iso_test),8);
m_test_spline,p_test_spline = get_diagram_data(test_set['classe'],prob_spline_test,8);

# %%
plt.figure()
plt.plot(m_test_platt,p_test_platt,marker='*')
plt.plot(m_test_iso,p_test_iso,marker='*')
plt.plot(m_test_spline,p_test_spline,marker='*')
plt.plot(np.array([0,1]),np.array([0,1]),'--')
plt.show()
plt.xlabel('Mean predicted value(score)')
plt.ylabel('Empirical probability')
plt.legend(('Platt','Isotonic','Spline','Perfect Calibrated'))

# %% Brier score

mse_val = brier_score_loss(val_set['classe'],val_set['score'])
mse_test = brier_score_loss(test_set['classe'],test_set['score'])

mse_platt_val = brier_score_loss(val_set['classe'],prob_platt_val)
mse_platt_test = brier_score_loss(test_set['classe'],prob_platt_test)

mse_iso_val = brier_score_loss(val_set['classe'],prob_iso_val)
mse_iso_test = brier_score_loss(test_set['classe'],prob_iso_test)

mse_spline_val = brier_score_loss(val_set['classe'],prob_spline_val)
mse_spline_test = brier_score_loss(test_set['classe'],prob_spline_test)

# %%
print('MSE val: %1.4f' % (mse_val))
print('MSE test: %1.4f' % (mse_test))

print('MSE platt val: %1.4f' % (mse_platt_val))
print('MSE platt test: %1.4f' % (mse_platt_test))

print('MSE iso val: %1.4f' % (mse_iso_val))
print('MSE iso test: %1.4f' % (mse_iso_test))

print('MSE spline val: %1.4f' % (mse_spline_val))
print('MSE spline test: %1.4f' % (mse_spline_test))

# %% Log - loss score
loss_val = log_loss(val_set['classe'],val_set['score'])
loss_test = log_loss(test_set['classe'],test_set['score'])

loss_platt_val = log_loss(val_set['classe'],prob_platt_val)
loss_platt_test = log_loss(test_set['classe'],prob_platt_test)

loss_iso_val = log_loss(val_set['classe'],prob_iso_val)
loss_iso_test = log_loss(test_set['classe'],prob_iso_test)

loss_spline_val = log_loss(val_set['classe'],prob_spline_val)
loss_spline_test = log_loss(test_set['classe'],prob_spline_test)

# %%
print('loss val %1.4f' % (loss_val))
print('loss test: %1.4f' % (loss_test))

print('loss platt val: %1.4f' % (loss_platt_val))
print('loss platt test: %1.4f' % (loss_platt_test))

print('loss iso val: %1.4f' % (loss_iso_val))
print('loss iso test: %1.4f' % (loss_iso_test))

print('loss spline val: %1.4f' % (loss_spline_val))
print('loss spline test: %1.4f' % (loss_spline_test))

# %% bar plot mse
mse_vec = [mse_test,mse_platt_test,mse_iso_test,mse_spline_test]
x = np.arange(4)
plt.figure()
plt.bar(x,mse_vec)
plt.xticks(x, ('Uncalibrated', 'Platt', 'Isotonic', 'SplineCalib'))
plt.ylabel('Brier score')
plt.title('Test Set Brier score comparison')
plt.show()

# %% bar plot log loss
loss_vec = [loss_test,loss_platt_test,loss_iso_test,loss_spline_test]
#x = np.arange(4)
plt.figure()
plt.bar(x,loss_vec,color='r')
plt.xticks(x, ('Uncalibrated', 'Platt', 'Isotonic', 'SplineCalib'))
plt.ylabel('Brier score')
plt.title('Test Set Log - Loss score comparison')
plt.show()