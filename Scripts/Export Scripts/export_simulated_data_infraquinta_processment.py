# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan
"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *

config = Configuration()
path_init = config.path

files = ['node_pressure_summer', 'node_pressure_winter', 'link_flow_summer', 'link_flow_winter']

print("Export initiated")

threshold = 0.0001

for file in files:
    
    print("  " + file)
    
    path_import = path_init + "\\Data\\infraquinta\\simulated\\" + file + ".csv"
        
    df = pd.read_csv(path_import, sep=",")

    df[df < threshold] = 0
        
    path_export = path_init + "\\Data\\infraquinta\\simulated\\processed\\" + file + ".csv"
    
    df.to_csv(index=False, path_or_buf=path_export)

print("Export completed")