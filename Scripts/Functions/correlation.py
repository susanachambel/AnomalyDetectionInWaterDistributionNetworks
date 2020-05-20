# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:56:59 2020

@author: susan
"""

import pandas as pd
import numpy as np
from scipy import stats, signal
from math import log2
from numpy.matlib import repmat

from configuration import *
from data_selection import *


## Aux Functions ##

"""
    Finds the common datapoints between two dataframes and transforms them
    into numpy arrays.
"""
def get_common_datapoints(df1, df2):
    xconcat = pd.concat([df1, df2], axis=1, sort=False)
    xconcat = xconcat.dropna()        
    x1 = xconcat.iloc[:,0].to_numpy()
    x2 = xconcat.iloc[:,1].to_numpy()   
    return x1, x2

def laplace_smoothing(k, cnts, bins):
    return (cnts + k) / (sum(cnts) + (len(bins)*k))

"""
    Function to generate boxes given dataset(xx) and box size (k)
"""
def sliding_window(xx,k):
    idx = np.arange(k)[None, :]+np.arange(len(xx)-k+1)[:, None]
    return xx[idx],idx

## Correlation Functions ##  

"""
    The Pearson correlation does not support time series that don't change
    its change in time (straight line). Therefore, 999999999 is returned instead.
    
    Note: When this value is passed to the website, it will be interpreted as NULL
"""
def calculate_pearson(df1, df2):
    x1, x2 = get_common_datapoints(df1, df2)   
    value = stats.pearsonr(x1, x2)    
    if np.isnan(value[0]):
        return 999999999
    else:
        return value[0]
    
def calculate_pearson_v2(x1, x2):
    value = stats.pearsonr(x1, x2)    
    if np.isnan(value[0]):
        return 999999999
    else:
        return value[0]
    
    
"""
    The Kullback–Leibler divergence does not support probability distributions with zeros.
    Therefore, we decided to apply the Laplase Smoothing to 'eliminate' them.
    
    Note: The Kullback–Leibler divergence does not support time series that don't change
    its change in time (straight line). Therefore, 999999999 is returned instead.
"""
def calculate_kl_divergence(x1, x2):             
    x3 = np.append(x1, x2)      
    cnts, bins = np.histogram(x3, bins='auto')         
    cnts1, bins1 = np.histogram(x1, bins=bins)
    cnts2, bins2 = np.histogram(x2, bins=bins)         
    k = 1
    distprob1 = laplace_smoothing(k, cnts1, bins)
    distprob2 = laplace_smoothing(k, cnts2, bins)   
    return sum(distprob1[i] * log2(distprob1[i]/distprob2[i]) for i in range(len(distprob1)))


def calculate_dcca(x1, x2, k):
    cdata = np.array([x1,x2]).T
    nsamples,nvars = cdata.shape
    #cdata = signal.detrend(cdata,axis=0)
    cdata = cdata-cdata.mean(axis=0)
    xx = np.cumsum(cdata,axis=0)  
    F2_dfa_x = np.zeros(nvars)
    allxdif = []
    for ivar in range(nvars):
        xx_swin , idx = sliding_window(xx[:,ivar],k)
        nwin = xx_swin.shape[0]
        b1, b0 = np.polyfit(np.arange(k),xx_swin.T,deg=1)
        x_hatx = repmat(b1,k,1).T*repmat(range(k),nwin,1) + repmat(b0,k,1).T
        xdif = xx_swin-x_hatx
        allxdif.append(xdif)
        F2_dfa_x[ivar] = (xdif**2).mean()
    dcca = np.zeros([nvars,nvars])
    for i in range(nvars):
        for j in range(nvars):
            F2_dcca = (allxdif[i]*allxdif[j]).mean()
            dcca[i,j] = F2_dcca / np.sqrt(F2_dfa_x[i] * F2_dfa_x[j]) 
       
    if np.isnan(dcca[0][1]):
        return 999999999
    else:
        return dcca[0][1]
        
"""
    The corr_array should be something like: ["pearson", "kl"]
    
    Pearson correlation -> pearson
    Kullback-Leibler divergence -> kl 
"""
def calculate_correlations(df1, df2, corr_array):    
    x1, x2 = get_common_datapoints(df1, df2)
        
    # TODO Não esquecer que tratar o caso em que não existem pontos em comum
        
    result = {} 
    k = 2
          
    for corr in corr_array:
        if (corr == "pearson"):      
            result["pearson"] = round(calculate_pearson_v2(x1, x2),3)
        elif (corr == "kullback-leibler"):
            result["kullback-leibler"] = round(calculate_kl_divergence(x1, x2),3) 
        elif (corr == "dcca"):
            result["dcca"] = round(calculate_dcca(x1, x2, k),3)
            
    return result

def test_correlation():
    
    config = Configuration() 
    path_init = config.path
     
    df1 = select_data(path_init, "infraquinta", "interpolated", 1, '2017-01-01 00:00:00', '2017-12-30 23:59:59')
    df2 = select_data(path_init, "infraquinta", "interpolated", 4, '2017-01-01 00:00:00', '2017-12-30 23:59:59')
    
    result = calculate_correlations(df1, df2, ["pearson","kl", "dcca"])
    
    kl_pq = result["kl"]
    print('KL(P || Q): %.4f bits' % kl_pq)
    
    pearson = result["pearson"]
    print('Pearson: %.4f' % pearson)
    
    dcca = result["dcca"]
    print('DCCA: %.4f'% dcca)
    
        
#test_correlation()