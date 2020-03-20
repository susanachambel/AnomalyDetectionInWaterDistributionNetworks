# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 16:12:29 2020

@author: susan
"""

# Tirar sensor 2 e sensor 3 da primeira semana de junho

# Fazer o plot

# Pierson correlation


import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *
from data_selection import *
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy import stats

def show_month_plot(df_day, title, ylabel, color):
    fig, ax = plt.subplots()
    ax.plot(df_day.index,df_day['value'], color=color)
    ax.set(xlabel='', ylabel=ylabel, title=title)
    
    date = mdates.DayLocator(interval=5)
    ax.xaxis.set_major_locator(date)
    date_fmt = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_fmt)
    
    date_min = mdates.DayLocator(interval=1)
    ax.xaxis.set_minor_locator(date_min)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def show_2week_plot(df_day, title, ylabel, color):
    fig, ax = plt.subplots()
    ax.plot(df_day.index,df_day['value'], color=color)
    ax.set(xlabel='', ylabel=ylabel, title=title)
    
    date = mdates.DayLocator(interval=2)
    ax.xaxis.set_major_locator(date)
    date_fmt = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_fmt)
    
    date_min = mdates.HourLocator(interval=12)
    ax.xaxis.set_minor_locator(date_min)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
    
def show_day_plot(df_day, title, ylabel, color):
    fig, ax = plt.subplots()
    ax.plot(df_day.index,df_day['value'], color=color)
    ax.set(xlabel='', ylabel=ylabel, title=title)
    
    date = mdates.HourLocator(interval=2)
    ax.xaxis.set_major_locator(date)
    date_fmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_fmt)
    
    date_min = mdates.MinuteLocator(interval=30)
    ax.xaxis.set_minor_locator(date_min)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

config = Configuration()
path_init = config.path

df_2 = select_data(path_init, "infraquinta", "interpolated", 1, "2017-04-01 00:00:00", "2017-12-31 23:59:59")
df_3 = select_data(path_init, "infraquinta", "interpolated", 3, "2017-04-01 00:00:00", "2017-12-31 23:59:59")






df_2 = df_2.resample('1d').mean()
#show_month_plot(df_2, "sensor 2", "label", "peru")
df_3 = df_3.resample('1d').mean()
#show_month_plot(df_3, "sensor 3", "label", "peru")

array_2 = df_2['value'].to_numpy()
array_3 = df_3['value'].to_numpy()

print(stats.pearsonr(array_2, array_3))

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(array_2, model='multiplicative', freq=1)
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)

result.plot()
plt.show()







