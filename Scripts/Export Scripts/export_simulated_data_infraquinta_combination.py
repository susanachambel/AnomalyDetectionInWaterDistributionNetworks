# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan
"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *

root = read_config()
path_init = get_path(root)

files = ['node_pressure_summer', 'node_pressure_winter', 'link_flow_summer', 'link_flow_winter']

print("Export initiated")

for file in files:
    
    print("  " + file)
    
    path_import = path_init + "\\Data\\infraquinta\\simulated\\" + file + ".csv"
    
    df_combined = pd.DataFrame()
    
    df = pd.read_csv(path_import, sep=",")

    for column in df:
        df_combined = pd.concat([df_combined,df[column]], ignore_index=True)
        
    path_export = path_init + "\\Data\\infraquinta\\simulated\\combined\\" + file + ".csv"
    
    df_combined.to_csv(index=False, path_or_buf=path_export)

print("Export completed")