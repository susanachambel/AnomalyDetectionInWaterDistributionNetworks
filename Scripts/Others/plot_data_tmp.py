# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:12:48 2020

@author: susan
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd 

def process_hour_data(df):
    df_hour = df.copy()
    df_hour.index = df_hour['date']
    del df_hour['date']
    df_hour = df_hour.resample('1min').mean()
    df_hour['value'] = df_hour['value'].interpolate()
    return df_hour

def process_day_data(df):
    df_day = df.copy()
    df_day.index = df_day['date']
    del df_day['date']    
    df_day = df_day.resample('10min').mean()
    return df_day

def show_hour_plots(df_wf_hour,df_wp_hour,title,ylabel_wf,ylabel_wp,color_wf,color_wp):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('')
    ax1.set_ylabel(ylabel_wf, color=color_wf)
    ax1.plot(df_wf_hour.index,df_wf_hour['value'], color=color_wf)
    ax1.tick_params(axis='y', labelcolor=color_wf)
    ax1.set_title(title)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel(ylabel_wp, color=color_wp)
    ax2.plot(df_wp_hour.index,df_wp_hour['value'], color=color_wp)
    ax2.tick_params(axis='y', labelcolor=color_wp)
    
    date = mdates.MinuteLocator(interval=5)
    ax1.xaxis.set_major_locator(date)
    date_fmt = mdates.DateFormatter('%H:%M')
    ax1.xaxis.set_major_formatter(date_fmt)
    date_min = mdates.MinuteLocator(interval=1)
    ax1.xaxis.set_minor_locator(date_min)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def show_hour_plot(df_hour, title, ylabel, color):
    fig, ax = plt.subplots()
    ax.plot(df_hour.index,df_hour['value'], color=color)
    ax.set(xlabel='', ylabel=ylabel, title=title)
    
    date = mdates.MinuteLocator(interval=5)
    ax.xaxis.set_major_locator(date)
    date_fmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_fmt)
    
    date_min = mdates.MinuteLocator(interval=1)
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


path_s2 = "C:\\Users\\susan\\Documents\\IST\\Tese\\Python Files\\sensor2.csv"
path_s3 = "C:\\Users\\susan\\Documents\\IST\\Tese\\Python Files\\sensor3.csv"

df_s2 = pd.read_csv(path_s2)
df_s2['date'] = pd.to_datetime(df_s2['date'], format='%Y-%m-%d %H:%M:%S')
df_s3 = pd.read_csv(path_s3)
df_s3['date'] = pd.to_datetime(df_s2['date'], format='%Y-%m-%d %H:%M:%S')

"""
print(df_s2)
print(df_s2.info())
print(df_s3)
print(df_s3.info())
"""

color_wf = 'steelblue'
color_wp = 'peru'

df_s2_hour = df_s2.loc[df_s2['date'].dt.hour < 1]
df_s2_hour = process_hour_data(df_s2_hour)
df_s3_hour = df_s3.loc[df_s3['date'].dt.hour < 1]
df_s3_hour = process_hour_data(df_s3_hour)

show_hour_plots(df_s2_hour,df_s3_hour,
                  'Sensor 2 & 3 @ 29-05-2017 (12pm-1am)',
                  'Water flow [m3/h]\n(sensor 2)',
                  'Water pressure [bar]\n(sensor 3)',
                  color_wf,color_wp)

"""
show_hour_plot(df_s2_hour,'Sensor 2 @ 29-05-2017 (12pm-1am)',
                 'Water flow [m3/h]',color_wf)
show_hour_plot(df_s3_hour,'Sensor 3 @ 29-05-2017 (12pm-1am)',
                 'Water pressure [bar]',color_wp)
"""

df_s2_day = process_day_data(df_s2)
show_day_plot(df_s2_day,'Sensor 2 @ 29-05-2017',
                 'Water flow [m3/h]',color_wf)



# a week
# a month
# a year






