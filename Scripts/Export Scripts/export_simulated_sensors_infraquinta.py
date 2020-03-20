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

files = ['node_pressure','link_flow']
seasons = ['summer', 'winter']

links = {'1': '6', '2': 'aTU1096150205',
         '6': 'aTU1093150205', '9': 'aTU455150205',
         '10': 'aTU4981150302', '12': '2', '14':'aTU1477150205'}

nodes = {'3':'aMI817150114','7':'aMI817150114',
        '8':'aMC402150114', '11':'aMC404150114',
        '13':'aMC401150114', '15':'aMC406150114',}

print("Export initiated")

for file in files:
    
    sensors = {}
    
    if 'node' in file:
        sensors = nodes
    else:
        sensors = links

    for season in seasons:
        path_import = path_init + "\\Data\\infraquinta\\simulated\\" + file + "_" + season + ".csv"
        df = pd.read_csv(path_import, sep=",")
              
        for sensor in sensors:
            
            path_export = path_init + "\\Data\\infraquinta\\simulated\\sensor_" + sensor + "_" + season + ".csv"
            value = sensors[sensor]
            df_sensor = df[[value]]  
            df_sensor.to_csv(index=False, path_or_buf=path_export)
            print("  Sensor " + sensor + " --> " + value + " (" + season +  ")")
            
print("Export completed")


