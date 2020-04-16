# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:12:48 2020

@author: susan
"""

import pandas as pd
import numpy as np
import time
import datetime
from configuration import *
from data_selection import *
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


config = Configuration()
path_init = config.path

# quando for a meter isto numa função, tenho que garantir que os segundos são 00
df_min = '2017-01-01 00:00:00'
df_max = '2017-12-30 23:59:59'

df = select_data(path_init, 'infraquinta', 'interpolated', '2',df_min ,df_max )

ylabel = "pressure/flow"
title = "test"

df1 = df.resample('2m').mean()

print(df1)


fig, ax = plt.subplots()
ax.plot(df1.index,df1['value'])
ax.set(xlabel='', ylabel=ylabel, title=title)  

date = mdates.MonthLocator(interval=1)
ax.xaxis.set_major_locator(date)
date_fmt = mdates.DateFormatter('%d-%m-%y')
ax.xaxis.set_major_formatter(date_fmt)
    
#date_min = mdates.dayLocator(interval=10)
#ax.xaxis.set_minor_locator(date_min)

fig.autofmt_xdate()

plt.show()





