# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:56:59 2020

@author: susan
"""

import pandas as pd
import numpy as np
from scipy import stats

def calculate_pearson(df1, df2):
    array1 = df1['value'].to_numpy()
    array2 = df2['value'].to_numpy()
    len_array1 = len(array1)
    len_array2 = len(array2)
    if(len_array1 != len_array2):
        min_len = min(len_array1, len_array2)     
        array1 = array1[:min_len]
        array2 = array2[:min_len]        
    value = stats.pearsonr(array1, array2)    
    if np.isnan(value[0]):
        return 999999999
    else:
        return value[0]

