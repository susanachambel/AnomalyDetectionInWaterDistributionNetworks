# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:54:48 2020

@author: susan

Processes the original simulated event data and:
    1) Turns negative values into positive ones
    2) Adds time column with the correct time point
    3) Adds the names of the sensors (columns)

"""

import sys
sys.path.append('../Functions')
from configuration import *
import matplotlib.pyplot as plt
import pandas as pd

def plot_sensors(event_id, df, sensors):
    fig = plt.figure(figsize=(10,6*len(sensors)))
    gs0 = fig.add_gridspec(ncols=1, nrows=len(sensors))
    i = 0
    for sensor in sensors:
        ax = fig.add_subplot(gs0[i])
        ax.plot(df.index,df.loc[:,sensor], label="smt")
        title = "Event " + str(event_id) + " | Sensor " + str(sensor)
        ax.set(xlabel='', ylabel='', title=title)
        i += 1
    plt.show()

def export_event(path_imp, path_exp, sensors):
    
    event_range_min = 1
    event_range_max = 18696
    
    print(2*'\x1b[2K\r' + "Progress " + str(event_range_min-1) + "/" + str(event_range_max), flush=True, end="\r")
    
    for event_id in range(event_range_min, event_range_max+1):
        
        path = path_init + path_imp + str(event_id) + '.txt'
        df = pd.read_csv(path, delimiter='\s+', header=None, names=sensors)  
        df = df.apply(abs)
        df.index = list(range(0,145*600, 600))
        df.index.name = 'time'
        
        #plot_sensors(event_id, df, sensors)
           
        path_export = path_init + path_exp + str(event_id) + '.csv'
        #df.to_csv(index=True, path_or_buf=path_export)
        
        print(2*'\x1b[2K\r' + "Progress " + str(event_id) + "/" + str(event_range_max), flush=True, end="\r")
    
    print(2*'\x1b[2K\r' + "Completed " + str(event_range_max) + "/" + str(event_range_max), flush=True, end="\r")
    

config = Configuration()
path_init = config.path

path_imp = '\\Data\\infraquinta\\events\\Original\\Rotura_Q\\Rotura_Q_Medidores'
path_exp = '\\Data\\infraquinta\\events\\Event_Q\\event_'
sensors = [9, 6, 12, 1, 10, 14, 2]

#export_event(path_imp, path_exp, sensors)

path_imp = '\\Data\\infraquinta\\events\\Original\\Rotura_P_SensoresEscolhidos\\Rotura_Pnova'
path_exp = '\\Data\\infraquinta\\events\\Event_P\\event_'
sensors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

#export_event(path_imp, path_exp, sensors)


def export_event_all(path_init, path_imp_1, path_imp_22, path_exp, sensors_1, sensors_2):
    
    event_range_min = 1
    event_range_max = 18696
    
    print(2*'\x1b[2K\r' + "Progress " + str(event_range_min-1) + "/" + str(event_range_max), flush=True, end="\r")
    
    for event_id in range(event_range_min, event_range_max+1):
        
        path_1 = path_init + path_imp_1 + str(event_id) + '.txt'
        df_1 = pd.read_csv(path_1, delimiter='\s+', header=None, names=sensors_1)  
        df_1 = df_1.apply(abs)
        df_1.index = list(range(0,145*600, 600))
        df_1.index.name = 'time'
        
        path_2 = path_init + path_imp_2 + str(event_id) + '.txt'
        df_2 = pd.read_csv(path_2, delimiter='\s+', header=None, names=sensors_2)  
        df_2 = df_2.apply(abs)
        df_2.index = list(range(0,145*600, 600))
        df_2.index.name = 'time'
        df_2 = df_2[[22, 23, 24, 25, 26, 27, 28]]
        
        df_1 = pd.concat([df_1, df_2], axis=1)
        
        #plot_sensors(event_id, df, sensors)
           
        path_export = path_init + path_exp + str(event_id) + '.csv'
        df_1.to_csv(index=True, path_or_buf=path_export)
        
        print(2*'\x1b[2K\r' + "Progress " + str(event_id) + "/" + str(event_range_max), flush=True, end="\r")
    
    print(2*'\x1b[2K\r' + "Completed " + str(event_range_max) + "/" + str(event_range_max), flush=True, end="\r")
 

path_imp_1 = '\\Data\\infraquinta\\events\\Original\\Rotura_P_SensoresEscolhidos\\Rotura_Pnova'
path_imp_2 = '\\Data\\infraquinta\\events\\Original\\Rotura_Q\\Rotura_Q_Medidores'
path_exp = '\\Data\\infraquinta\\events\\Event_All\event_'
sensors_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
sensors_2 = [25, 24, 27, 22, 26, 28, 23] #[9, 6, 12, 1, 10, 14, 2]

export_event_all(path_init, path_imp_1, path_imp_2, path_exp, sensors_1, sensors_2)



