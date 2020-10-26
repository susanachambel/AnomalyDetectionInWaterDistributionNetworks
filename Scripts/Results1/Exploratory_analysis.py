# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan

@about: 
    
"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *
from data_selection import *
from event_archive_2 import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL


def plot_real_week(path_init):
    color = 'LIGHTSLATEGRAY'
    locator = mdates.HourLocator(interval=24)
    formatter = mdates.ConciseDateFormatter(locator)
    locator_min = mdates.HourLocator(interval=12)
    sensors_id = [3,14]
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(11.4,4.8), sharex=True)
    i=0
    for ax in axs.flat:
        sensor_id = sensors_id[i]
        df = select_data(path_init, "infraquinta", "interpolated", sensor_id, '2017-05-15 00:00:00', '2017-05-28 23:59:59')
        df = df.resample('1h').mean()
        ax.plot(df.index,df['value'], color=color)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, axis='y', alpha=0.3)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(locator_min)
        title = "Sensor " + str(sensor_id)    
        if i == 1:
            ylabel = "Volumetric Flowrate [m3/h]"
        else:
            ylabel = "Pressure [bar]"
        ax.set(xlabel='', ylabel=ylabel, title=title)
        i+=1
    plt.xlabel("Time")
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results1\\Exploratory Analysis\\real_week.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
def plot_real_day(path_init):    
    color = 'LIGHTSLATEGRAY'
    dates = [['2017-05-15 00:00:00','2017-05-15 23:59:59'],['2017-05-16 00:00:00','2017-05-16 23:59:59']]
    formatter = mdates.DateFormatter('%H:%M')#mdates.ConciseDateFormatter(locator)
    for sensor_id in [14, 3]:
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(11.4,4.8), sharey=True)
        i=0
        for ax in axs.flat:
            date = dates[i]
            df = select_data(path_init, "infraquinta", "interpolated", sensor_id, date[0], date[1])
            df = df.resample('5min').mean()
            ax.plot(df.index,df['value'], color=color)
            ax.xaxis.set_major_formatter(formatter)
            ax.grid(True, axis='y', alpha=0.3)
            if i == 0:
                ax.axes.xaxis.set_ticklabels([])
                ax.set_title("Sensor " + str(sensor_id) + " on May 15, 2017")
            else:
                ax.set_title("Sensor " + str(sensor_id) + " on May 16, 2017")
            i+=1
            if (sensor_id == 14):
                title = "Volumetric Flowrate [m3/h]"
            else:
                title = "Pressure [bar]"
            
            ax.set(xlabel='', ylabel=title)
        
        plt.xlabel("Time")
        #fig.text(0.001, 0.50, title, va='center', ha='center', rotation='vertical')
        fig.tight_layout()
        plt.savefig(path_init + '\\Images\\Results1\\Exploratory Analysis\\real_day_' + str(sensor_id) + '.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
def plot_synthetic(path_init):
    color1 = 'LIGHTSLATEGRAY'
    color2 = 'tab:red'
    event_id = 697 #192 #216
    data_type = 'all'
    ea = EventArchive(path_init, data_type)
    sensors_id = ['4','25'] #['9','23']
    event_info = ea.get_event_info(event_id)
    df = ea.get_event(event_id)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(11.4,4.8), sharex=True)
    i=0
    for ax in axs.flat:
        sensor_id = sensors_id[i]
        ax.plot(df.index,df.loc[:,sensor_id],color=color1)
        ax.grid(True, axis='y', alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(12*600))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(6*600))
        #ax.axvspan(event_info['time_init'], event_info['time_final'], color=color2, alpha=0.1, label="Leakage")
        ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        if i == 1:
            ylabel = "Volumetric Flowrate [m3/h]"
            #ax.legend(loc="upper left")
        else:
            ylabel = "Pressure [bar]"
        title = "Sensor " + str(sensor_id)    
        ax.set(xlabel='', ylabel=ylabel, title=title)
        i+=1
    plt.xlabel("Time Point")
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results1\\Exploratory Analysis\\synthetic_' + str(event_id) + '.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_windows(path_init):
    
    color1 = 'LIGHTSLATEGRAY'
    color2 = 'tab:red'
    color3 ='#AAB5BF'
    sensor_id = '25'
    event_id = 192
    width_aux = 30/2
    
    data_type = 'all'
    ea = EventArchive(path_init, data_type)
    event_info = ea.get_event_info(event_id)
    df = ea.get_event(event_id)
    
    time_init = event_info.time_init
    time_final = event_info.time_final
    
    df = df.iloc[int((time_init)/600-40):int((time_final)/600+40,):]
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11.4,4.8), sharex=True, sharey=True)
    i=0
    for ax in axs.flat:
    
        ax.plot(df.index,df.loc[:,sensor_id], color=color1)
    
        if i==0:
            init_point = ((time_init/600) - width_aux + 1)*600
            final_point = ((time_init/600) + width_aux)*600
            ax.axvline(x=time_init, color=color2, linestyle='--', linewidth=1.25, label="Start of leakage")
            ax.set_ylabel("Volumetric Flowrate [m3/h]")
        else:
            init_point = ((time_final/600) - width_aux + 1)*600
            final_point = ((time_final/600) + width_aux)*600
            ax.axvline(x=time_final, color=color2, linestyle='--', linewidth=1.25, label="End of leakage")
         
        ax.set_xlabel("Time Point")
        ax.axvspan(init_point, final_point, color=color3, alpha=0.2, label="Time Window")
        ax.legend()       
        ax.xaxis.set_major_locator(ticker.MultipleLocator(12*600))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(6*600))
        ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        i+=1
    
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Solution\\window_w_leakage.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11.4,4.8), sharex=True, sharey=True)
    i=0
    for ax in axs.flat:
    
        ax.plot(df.index,df.loc[:,sensor_id], color=color1)
    
        if i==0:
            final_point = (time_init/600)
            init_point = (final_point - width_aux*2)*600
            final_point = (final_point - 1)*600
            ax.axvline(x=time_init, color=color2, linestyle='--', linewidth=1.25, label="Start of leakage")
            ax.set_ylabel("Volumetric Flowrate [m3/h]")
        else:
            init_point = (time_final/600) + 1
            final_point = (init_point + width_aux*2 - 1)*600
            init_point = (init_point)*600
            ax.axvline(x=time_final, color=color2, linestyle='--', linewidth=1.25, label="End of leakage")
         
        ax.set_xlabel("Time Point")
        ax.axvspan(init_point, final_point, color=color3, alpha=0.2, label="Time Window")
        ax.legend()       
        ax.xaxis.set_major_locator(ticker.MultipleLocator(12*600))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(6*600))
        ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        i+=1
    
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Solution\\window_wo_leakage.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_real_window(path_init):

    locator = mdates.HourLocator(interval=1)
    formatter = mdates.ConciseDateFormatter(locator)
    df = select_data(path_init, "infraquinta", "interpolated", 12, '2017-06-21 00:00:00', '2017-06-21 12:59:59')
    plt.figure(figsize=[7.4, 4.8])
    ax = plt.gca()
    df = df.resample('2min').mean()
    ax.plot(df.index,df['value'], color='LIGHTSLATEGRAY')
    ylabel = "Volumetric Flowrate [m3/h]"
    ax.set(xlabel='', ylabel=ylabel, title="")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    init_point = df.index[0]
    time_end = df.index[-1]
    i = 1
    color = 'LIGHTSLATEGRAY'
    
    final_point = init_point + timedelta(hours=1)
    pt2 = plt.axvspan(init_point, final_point, color=color, alpha=0.2, label="Windows")
    init_point = final_point
    while init_point < time_end:
        final_point = init_point + timedelta(hours=1)
        if i%2 == 0:
            color = 'LIGHTSLATEGRAY'
        else:
            color = '#AAB5BF'
        plt.axvspan(init_point, final_point, color=color, alpha=0.2)
        init_point = final_point
        i += 1
    
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(loc="upper left")
    ax.grid(True, axis='y', alpha=0.3)
    plt.savefig(path_init + '\\Images\\window_real.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def synthetic_stats(path_init):
    
    event_id = 697
    data_type = 'all'
    ea = EventArchive(path_init, data_type)
    sensors_id = ['4','25'] #['9','23']
    event_info = ea.get_event_info(event_id)
    df = ea.get_event(event_id)
    
    sensors_id = []
    for sensor_id in range(1,28,1):
        sensors_id.append(str(sensor_id))
        
    print(df.describe())
    path_export = path_init + '\\Images\\Results1\\Exploratory Analysis\\synthetic_stats.csv'
    df.describe().to_csv(index=True, path_or_buf=path_export, sep=';',decimal=',')
    
def real_stats(path_init):
    
    date_init1 = '2017-05-15 00:00:00'
    date_final1 = '2017-05-28 23:59:59'
    date_init2 = '2017-01-01 00:00:00'
    date_final2 = '2017-12-30 23:59:59'
    
    flow = [1,2,6,9,10,12,14]
    pressure = [3,7,8,11,13,15]
    sensors_id = flow 
    sensors_id.extend(pressure)
    
    i = 0
    for date in [[date_init1,date_final1],[date_init2,date_final2]]:
        date_init = date[0]
        date_final = date[1]
        df = pd.DataFrame()
        for sensor_id in sensors_id:
            df = pd.concat([df,select_data(path_init, "infraquinta", "interpolated", str(sensor_id), date_init, date_final).rename(columns={'value':sensor_id})], axis=1)
        print(df.describe())
        if i == 0:
            path_export = path_init + '\\Images\\Results1\\Exploratory Analysis\\stats_real_week.csv'
        else:
            path_export = path_init + '\\Images\\Results1\\Exploratory Analysis\\stats_real_year.csv'
        
        df.describe().to_csv(index=True, path_or_buf=path_export, sep=';',decimal=',')
        i+=1


def decomposition_week(path_init):
    
    color = 'LIGHTSLATEGRAY'
    sensor_id = 14
    df = select_data(path_init, "infraquinta", "interpolated", sensor_id, '2017-05-15 00:00:00', '2017-05-28 23:59:59')
        
    #df = df.asfreq('h')
    
    for model in ['additive', 'multiplicative', 'stl']:
        
        if model == 'stl':
            stl = STL(df, seasonal=60*24+1, period=60*24)
            res = stl.fit()
        else:
            res = seasonal_decompose(df, model=model, period=60*24) # additive multiplicative
        
        #result.plot()
        residual = res.resid
        seasonal = res.seasonal 
        trend = res.trend
        
        locator = mdates.HourLocator(interval=24)
        formatter = mdates.ConciseDateFormatter(locator)
        locator_min = mdates.HourLocator(interval=12)
        
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(13.4,7.8), sharex=True)
    
        resample = '30min'
        """
        if resample is not None:
            df = df.resample(resample).mean()
            trend = trend.resample(resample).mean()
            seasonal = seasonal.resample(resample).mean()
            residual = residual.resample(resample).mean()
        """
        
        axs[0].plot(df.resample(resample).mean(), color=color)
        axs[1].plot(trend.resample(resample).mean(), color=color)
        axs[2].plot(seasonal.resample(resample).mean(), color=color)
        axs[3].plot(residual.resample(resample).mean(), color=color)
        
        axs[0].set_ylabel('Observed')
        axs[1].set_ylabel('Trend')
        axs[2].set_ylabel('Seasonal')
        axs[3].set_ylabel('Residual')
        
        axs[3].set_xlabel('Time')
        
        for ax in axs.flat:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_minor_locator(locator_min)
            ax.grid(True, axis='y', alpha=0.3)
            #plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        fig.tight_layout()
        plt.savefig(path_init + '\\Images\\Results1\\Exploratory Analysis\\decomposition_' + model + '.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


config = Configuration()
path_init = config.path

#plot_real_week(path_init)
#plot_real_day(path_init)
#plot_synthetic(path_init)
#plot_windows(path_init) 
    
#real_stats(path_init) 
#synthetic_stats(path_init)

decomposition_week(path_init)
