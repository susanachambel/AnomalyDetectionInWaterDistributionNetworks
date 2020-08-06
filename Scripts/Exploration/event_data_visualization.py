# -*- coding: utf-8 -*-
"""
Created on Fri Aug 5 20:54:48 2020

@author: susan
"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *
import matplotlib.pyplot as plt


config = Configuration()
path_init = config.path


event_range_min = 40
event_range_max = 41

events_id = list(range(event_range_min,event_range_max+1))


fig, ax = plt.subplots(1, 1, figsize=(8,4))

for event_id in events_id:
    
    path = path_init + '\\Data\\infraquinta\\events\\Event_Q\\event_' + str(event_id) + '.csv'


    df = pd.read_csv(path, index_col=0)

    ax.plot(df.index,df.loc[:,'12'])


plt.show()


