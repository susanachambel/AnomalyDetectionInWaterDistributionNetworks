# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan

@about: For each event, we save an image with the plots of each sensor from 2 
days prior to the event until the water is opened. It also shows when the event 
was detected and when the water was closed (if applied).
    
"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *
from data_selection import *
from event_archive_2 import *
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np

def plot_real_volumetric_flowrate(path_init):
    locator = mdates.HourLocator(interval=2)
    formatter = mdates.ConciseDateFormatter(locator)
    locator_min = mdates.HourLocator(interval=1)
    df = select_data(path_init, "infraquinta", "interpolated", 12, '2017-06-21 00:00:00', '2017-06-21 23:59:59')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    df = df.resample('5min').mean()
    ax.plot(df.index,df['value'])
    ylabel = "Volumetric Flowrate [m3/h]"
    ax.set(xlabel='', ylabel=ylabel, title="")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(locator_min)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.savefig(path_init + '\\Images\\real_volumetric_flow.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_real_pressure(path_init): 
    locator = mdates.HourLocator(interval=2)
    formatter = mdates.ConciseDateFormatter(locator)
    locator_min = mdates.HourLocator(interval=1)
    df = select_data(path_init, "infraquinta", "interpolated", 7, '2017-06-21 00:00:00', '2017-06-21 23:59:59')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    df = df.resample('5min').mean()
    ax.plot(df.index,df['value'])
    ylabel = "Pressure [bar]"
    ax.set(xlabel='', ylabel=ylabel, title="")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(locator_min)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.savefig(path_init + '\\Images\\real_pressure.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_synthetic_volumetric_flowrate(path_init):
    event_id = 216
    data_type = 'q'
    ea = EventArchive(path_init, data_type)
    event_info = ea.get_event_info(event_id)
    df = ea.get_event(event_id)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ylabel = "Volumetric Flowrate [m3/h]"
    ax.set(xlabel='', ylabel=ylabel, title="")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12*600))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(6*600))
    ax.plot(df.index,df.loc[:,'2'])
    plt.axvspan(event_info['time_init']-600, event_info['time_final'], color='salmon', alpha=0.2, label="Leakage")
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(loc="upper right")
    plt.savefig(path_init + '\\Images\\synthetic_volumetric_flowrate.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_synthetic_pressure(path_init):
    event_id = 216
    data_type = 'p'
    ea = EventArchive(path_init, data_type)
    event_info = ea.get_event_info(event_id)
    df = ea.get_event(event_id)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ylabel = "Pressure [bar]"
    ax.set(xlabel='', ylabel=ylabel, title="")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12*600))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(6*600))
    ax.plot(df.index,df.loc[:,'9'])
    plt.axvspan(event_info['time_init']-600, event_info['time_final'], color='salmon', alpha=0.2, label="Leakage")
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(loc="lower right")
    plt.savefig(path_init + '\\Images\\synthetic_pressure.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_windows(path_init):

    event_id = 216
    data_type = 'q'
    ea = EventArchive(path_init, data_type)
    event_info = ea.get_event_info(event_id)
    df = ea.get_event(event_id)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ylabel = "Volumetric Flowrate [m3/h]"
    ax.set(xlabel='', ylabel=ylabel, title="")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12*600))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(6*600))
    ax.plot(df.index,df.loc[:,'2'])
    
    time_init = event_info.time_init
    width_aux = 30/2
    middle_point = (time_init/600) - width_aux + 1
    final_point = (time_init/600) + width_aux + 1
    init_point = middle_point - width_aux*2
    init_point = init_point*600
    middle_point = middle_point*600
    final_point = final_point*600
    plt.axvspan(init_point, middle_point, color='sandybrown', alpha=0.2, label="Window 1")
    plt.axvspan(middle_point, final_point, color='saddlebrown', alpha=0.2, label="Window 2")
    ax.axvline(x=time_init, color='#d62728', linestyle='--', linewidth=1.25, label="Start of leakage")
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(loc="upper right")
    
    plt.savefig(path_init + '\\Images\\window_leakage_start.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    event_id = 6
    data_type = 'q'
    ea = EventArchive(path_init, data_type)
    event_info = ea.get_event_info(event_id)
    df = ea.get_event(event_id)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ylabel = "Volumetric Flowrate [m3/h]"
    ax.set(xlabel='', ylabel=ylabel, title="")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12*600))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(6*600))
    ax.plot(df.index,df.loc[:,'2'])
    
    width_aux = 30/2
    time_final = event_info.time_final
    init_point = (time_final/600) - width_aux + 1
    middle_point = (time_final/600) + width_aux + 1
    final_point = middle_point + width_aux*2
    init_point = init_point*600
    middle_point = middle_point*600
    final_point = final_point*600
    plt.axvspan(init_point, middle_point, color='sandybrown', alpha=0.2, label="Window 1")
    plt.axvspan(middle_point, final_point, color='saddlebrown', alpha=0.2, label="Window 2")
    ax.axvline(x=time_final, color='#d62728', linestyle='--', linewidth=1.25, label="End of leakage")
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(loc="upper right")
    
    plt.savefig(path_init + '\\Images\\window_leakage_end.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    event_id = 216
    data_type = 'q'
    ea = EventArchive(path_init, data_type)
    event_info = ea.get_event_info(event_id)
    df = ea.get_event(event_id)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ylabel = "Volumetric Flowrate [m3/h]"
    ax.set(xlabel='', ylabel=ylabel, title="")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12*600))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(6*600))
    ax.plot(df.index,df.loc[:,'2'])
    
    width_aux = 30/2
    time_final = event_info.time_final
    init_point = (time_final/600) + 1
    middle_point = init_point + width_aux*2
    final_point = middle_point + width_aux*2
    init_point = init_point*600
    middle_point = middle_point*600
    final_point = final_point*600
    plt.axvspan(init_point, middle_point, color='#17becf', alpha=0.2, label="Window 1")
    plt.axvspan(middle_point, final_point, color='#1f77b4', alpha=0.2, label="Window 2")
    
    ax.axvline(x=time_final+600, color='#d62728', linestyle='--', linewidth=1.25, label="End of leakage")
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(loc="upper left")
    plt.savefig(path_init + '\\Images\\window_normal_end.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    event_id = 16422
    data_type = 'q'
    ea = EventArchive(path_init, data_type)
    event_info = ea.get_event_info(event_id)
    df = ea.get_event(event_id)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ylabel = "Volumetric Flowrate [m3/h]"
    ax.set(xlabel='', ylabel=ylabel, title="")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12*600))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(6*600))
    ax.plot(df.index,df.loc[:,'2'])
    
    width_aux = 30/2
    time_init = event_info.time_init
    final_point = (time_init/600)
    middle_point = final_point - width_aux*2
    init_point = middle_point - width_aux*2
    init_point = init_point*600
    middle_point = middle_point*600
    final_point = final_point*600
    plt.axvspan(init_point, middle_point, color='#17becf', alpha=0.2, label="Window 1")
    plt.axvspan(middle_point, final_point, color='#1f77b4', alpha=0.2, label="Window 2")
    ax.axvline(x=time_init, color='#d62728', linestyle='--', linewidth=1.25, label="Start of leakage")
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(loc="upper left")
    plt.savefig(path_init + '\\Images\\window_normal_start.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()



config = Configuration()
path_init = config.path


plot_synthetic_volumetric_flowrate(path_init)
plot_synthetic_pressure(path_init)


#plot_windows(path_init)

