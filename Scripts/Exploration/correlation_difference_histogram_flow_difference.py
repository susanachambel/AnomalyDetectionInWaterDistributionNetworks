# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan


"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *
from data_selection import *


config = Configuration() 
path_init = config.path
    
date_min = '2017-01-01 00:00:00'
date_max = '2017-12-30 23:59:59'

df1 = select_data(path_init, "infraquinta", "interpolated", 6, date_min, date_max)
df2 = select_data(path_init, "infraquinta", "interpolated", 2, date_min, date_max)
   


df = pd.concat([df1, df2], axis=1, sort=False)
df = df.dropna()
df['diff_value'] = df.iloc[:,0] - df.iloc[:,1]

del df['value']
#x1 = xconcat.iloc[:,0].to_numpy()
#x2 = xconcat.iloc[:,1].to_numpy()

df = df.rename(columns={"diff_value": "value"}) 

print(len(df[df['value']<0]))

print(df)        
        
        
        
        
        
        
       
                        