# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:42:49 2020

@author: susan
"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *
import json
import numpy as np

config = Configuration()
path_init = config.path
wmes = config.wmes

dic = {}

index_start = 0

for wme in wmes:
    df = pd.read_csv(path_init + "\\Data\\" + wme + "_sensors_list.csv", delimiter=";" )
    df = df.astype('str')
    df.index = np.arange(index_start, index_start+len(df))   
    df_json = df.to_json(orient='index')
    df_json = json.loads(df_json) 
    dic[wme] = df_json
    
    index_start += len(df)
    
with open(path_init + "\\Data\\sensors_list.json", 'w') as outfile:
    json.dump(dic, outfile)    