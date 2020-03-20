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

sensors = [1,2,3,6,7,8,9,10,11,12,13,14,15]
seasons = ['summer', 'winter']

df = pd.DataFrame()

print("\nReport initiated\n")

for sensor_id in sensors:
    
    print("Sensor " + str(sensor_id))
    
    for season in seasons:
               
        path_import = path_init + "\\Data\\infraquinta\\simulated\\sensor_" + str(sensor_id) + "_" + season + ".csv"
        df_sensor = pd.read_csv(path_import, sep=",", header=None)
        
        
        print(df_sensor)
        
        
        #df = df.append({'wme': wme, 'id': i, 'name': name, 'type': type, 'n_rows': n_rows,
                        'min_date': min_date, 'max_date': max_date}, ignore_index=True)
        

path_export = path_init + "\\Reports\\simulated_data_report.csv"
df.to_csv(index=False, path_or_buf=path_export)

print(df.info())


print("\nReport completed")  
