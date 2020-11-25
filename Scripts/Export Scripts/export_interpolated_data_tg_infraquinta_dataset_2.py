# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:54:48 2020

@author: susan

Creates the dataset and files we are going to use to perform event detection
in the real data.

"""

import sys
sys.path.append('../Functions')
from configuration import *
from event_archive_2 import *
from correlation import *
from itertools import combinations
from datetime import datetime, timedelta 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from scipy import stats
import pandas as pd
import numpy as np
import calendar


def calculate_correlation(df, sensors, correlation_type, dcca_k):  
    x1 = df.loc[:,sensors[0]].to_numpy()
    x2 = df.loc[:,sensors[1]].to_numpy()
    if correlation_type == "pearson":
        return stats.pearsonr(x1, x2)[0]
    elif correlation_type == "dcca":
        return calculate_dcca_2(x1, x2, dcca_k)

def get_combo_name(combo):
    return str(combo[0]) + "-" + str(combo[1])    

def get_df_events(path_init):
    df_events = pd.read_csv(path_init + "\\Data\\events_ruturas_infraquinta_2017.csv", sep=';')
    df_events['date_executed'] = pd.to_datetime(df_events['date_executed'], format='%Y/%m/%d %H:%M:%S')
    df_events['date_detected'] = pd.to_datetime(df_events['date_detected'], format='%Y/%m/%d %H:%M:%S')
    df_events['date_water_closed'] = pd.to_datetime(df_events['date_water_closed'], format='%Y/%m/%d %H:%M:%S')
    df_events['date_water_opened'] = pd.to_datetime(df_events['date_water_opened'], format='%Y/%m/%d %H:%M:%S')
    df_events['date_possible'] = pd.to_datetime(df_events['date_possible'], format='%Y/%m/%d %H:%M:%S')
    df_events['date_start'] = pd.to_datetime(df_events['date_start'], format='%Y/%m/%d %H:%M:%S')
    df_events['date_end'] = pd.to_datetime(df_events['date_end'], format='%Y/%m/%d %H:%M:%S')
    df_events = df_events[df_events['read'] == 'y']
    return df_events

def get_df_sensors(path_init):
    df_sensors = pd.read_csv(path_init + "\\Data\\events_ruturas_infraquinta_2017_sensors.csv", sep=';')
    df_sensors = df_sensors[df_sensors['read'] == 'y']
    return df_sensors

def export_dataset(path_init):
    
    d1 = ['2017-01-01 00:00:00', '2017-01-30 23:59:59']
    d2 = ['2017-02-01 00:00:00', '2017-02-27 23:59:59']
    d3 = ['2017-03-01 00:00:00', '2017-03-30 23:59:59']
    d4 = ['2017-04-01 00:00:00', '2017-04-29 23:59:59']
    d5 = ['2017-05-01 00:00:00', '2017-05-30 23:59:59']
    d6 = ['2017-06-01 00:00:00', '2017-06-29 23:59:59']
    d7 = ['2017-07-01 00:00:00', '2017-07-30 23:59:59']
    d8 = ['2017-08-01 00:00:00', '2017-08-30 23:59:59']
    d9 = ['2017-09-01 00:00:00', '2017-09-29 23:59:59']
    d10 = ['2017-10-01 00:00:00', '2017-10-30 23:59:59']
    d11 = ['2017-11-01 00:00:00', '2017-11-29 23:59:59']
    d12 = ['2017-12-01 00:00:00', '2017-12-30 23:59:59']
    dates = [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12]
    
    
    #d13 = ['2017-02-07 00:00:00', '2017-02-07 23:59:59']
    #d13 = ['2017-02-06 20:00:00', '2017-02-07 19:59:59']
    #dates = [d13]
    
    flow = [1, 2, 6, 9, 10, 12, 14]
    pressure = [3, 7] #, 8, 11, 13, 15
    sensors = flow
    sensors.extend(pressure)
    correlation_type = 'dcca'
    combos = list(combinations(sensors, 2))
    widths = [60,120,180,240]
    
    for width in widths:
        
        for dcca_k_aux in range(4,5,1):
            
            dcca_k = int(width/dcca_k_aux)
            
            df_corr = pd.DataFrame()
            for date in dates:
            
                df = pd.DataFrame()
                for sensor_id in sensors:
                    df = pd.concat([df,select_data(path_init, "infraquinta", "interpolated", str(sensor_id), date[0], date[1]).rename(columns={'value':sensor_id})], axis=1)
            
                init = 0
                final = init + width
                len_df = len(df)
                while (final < len_df):
                    chunk = df.iloc[init:final, :]
                    results = {}
                    init_date = df.index[init]
                    results['init'] = init_date
                    results['final'] = df.index[final]
                    for combo in combos:
                        corr = calculate_correlation(chunk, combo, correlation_type, dcca_k)
                        results[get_combo_name(combo)] = corr
                    
                    df_corr = df_corr.append(results, ignore_index=True)
                    print(2*'\x1b[2K\r' + "Progress " + str(init_date), flush=True, end="\r")
                    init += 15
                    final = init + width
        
            print(df_corr)
            
            path_export = path_init + '\\Data\\infraquinta\\events\\Organized_Data_3\\dataset_r_' + correlation_type + '_' + str(width) + '_' + str(dcca_k) +  '.csv'
            df_corr.to_csv(index=True, path_or_buf=path_export)
            
            
            include_y(path_export, correlation_type, width, dcca_k)


def include_y(path_import, correlation_type, width, dcca_k):
        
    df = pd.read_csv(path_import, index_col=0)
        
    df['init'] = pd.to_datetime(df['init'], format='%Y-%m-%d %H:%M:%S')
    df['final'] = pd.to_datetime(df['final'], format='%Y-%m-%d %H:%M:%S')
    df['y'] = 0

    for index, event in df_events.iterrows():
        
        date_detected = event['date_detected'] - timedelta(hours=2)
        date_closed = event['date_water_closed']
        date_opened = event['date_water_opened']

        mask1 =  ((date_detected >= df['init']) & (date_detected <= df['final']))
        df.loc[mask1, 'y'] = index+1
        
        mask2 =  ((df['init'] >= date_detected) & (df['final'] < date_closed))
        df.loc[mask2, 'y'] = index+1
        
        mask3 = ((date_closed >= df['init']) & (date_closed <= df['final']))
        df.loc[mask3, 'y'] = -1
        
        mask4 = ((df['init'] >= date_closed) & (df['final'] < date_opened))
        df.loc[mask4, 'y'] = -1
        
    df.to_csv(index=True, path_or_buf=path_import)
        
def plot_dcca_k(path_init):
    
    correlation_type = 'dcca'
    width = 120  # [60, 120, 180, 240]
    
    combos = ['6-12', '3-7', '6-3']
    
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(11.4,6.4), sharex=True, sharey=True)
    
    for j, dcca_k_aux in enumerate(range(2, 6, 1)):
        
        dcca_k = int(width/dcca_k_aux)
        
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_4\\dataset_r_' + correlation_type + '_' + str(width) + '_' + str(dcca_k) + '.csv'
        df = pd.read_csv(path, index_col=0)
        
        df['init'] = pd.to_datetime(df['init'], format='%Y-%m-%d %H:%M:%S')
        df['final'] = pd.to_datetime(df['final'], format='%Y-%m-%d %H:%M:%S')
        df.index = df['init']
        #del df['init']
        df = df.sort_index()
        
        df = df[(df['init']>=datetime(2017,2,7,0,0,0)) & (df['init']<=datetime(2017,2,7,10,0,0))]
        
        for i, ax in enumerate(axs.flat):
            
            combo = combos[i]
            ax.plot(df.index,df.loc[:,combo], label='n = ' + str(dcca_k-1))
            
            locator = mdates.HourLocator(interval=1)
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            
            #plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            
            title_split = combo.split('-')
            title = 'Sensors ' + title_split[0] + ' & ' + title_split[1]        
            if i == 1:
                ax.set_ylabel('DCCA')
                title += ' (two flowrate sensors)'
                ax.legend(ncol=6)
            elif i == 2:
                ax.set_xlabel('Time')
                title += ' (one sensor of each type)'
            else:
                title += ' (two pressure sensors)'
            
            if j == 0:
                df_aux = df[df['y']==1]
                ax.axvspan(df_aux.iloc[0,:]['init'], df_aux.iloc[-1,:]['init'], color='tab:red', alpha=0.1, label="Positive Instances")
            
            ax.set_title(title)
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            ax.grid(True, axis='y', alpha=0.3,which='both')
    
    plt.ylim(-1,1)        
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results1\\DCCA K\\r_dcca_k_lc.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
def plot_time_windows(path_init):
    
    correlation_type = 'dcca'
    dcca_k = 40
    
    combos = ['6-12', '3-7', '6-3']
    widths = [60,120,180,240]
    
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(11.4,9), sharex=True, sharey=True)
    
    for i, ax in enumerate(axs.flat):
            
        width = widths[i]
        dcca_k = int(width/4)
            
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_4\\dataset_r_' + correlation_type + '_' + str(width) + '_' + str(dcca_k) + '.csv'
        df = pd.read_csv(path, index_col=0)
            
        df['init'] = pd.to_datetime(df['init'], format='%Y-%m-%d %H:%M:%S')
        df['final'] = pd.to_datetime(df['final'], format='%Y-%m-%d %H:%M:%S')
        df.index = df['init']
        #del df['init']
        df = df.sort_index()
        
        if(width == 60):
            df = df[(df['init']>=datetime(2017,2,7,0,0,0)) & (df['init']<=datetime(2017,2,7,11,0,0))]
            df = df.dropna()
        elif (width == 120):
            df = df[(df['init']>=datetime(2017,2,7,0,0,0)) & (df['init']<=datetime(2017,2,7,10,0,0))]
        elif (width == 180):
            df = df[(df['init']>=datetime(2017,2,7,0,0,0)) & (df['init']<=datetime(2017,2,7,9,0,0))]
        else:
            df = df[(df['init']>=datetime(2017,2,7,0,0,0)) & (df['init']<=datetime(2017,2,7,8,0,0))]
        
        for combo in combos:
            title_split = combo.split('-')
            title = 'Sensors ' + title_split[0] + ' & ' + title_split[1] 
            ax.plot(df.index,df.loc[:,combo], label=title)
            
        locator = mdates.HourLocator(interval=1)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
            
        #plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        ax.set_title('Time Window Size = ' + str(width))
        ax.set_ylabel('DCCA (n=' + str(dcca_k-1) + ')')
        
        df_aux = df[df['y']==1]
        ax.axvspan(df_aux.iloc[0,:]['init'], df_aux.iloc[-1,:]['init'], color='tab:red', alpha=0.1, label="Positive Instances")
        #df_aux = df[df['y']==-1]
        #ax.axvspan(df_aux.iloc[0,:]['init'], df_aux.iloc[-1,:]['init'], color='tab:blue', alpha=0.1, label="Positive instances")           
                
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.grid(True, axis='y', alpha=0.3,which='both')
        
        if i == 3:
            ax.set_xlabel('Time')
            ax.legend(loc='lower right')

    
    plt.ylim(-1,1.1)       
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results1\\Window Sizes\\r_window_sizes_lc.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()    

def plot_correlation_evolution(path_init):
    
    correlation_type = 'dcca'
    width = 120  # [60, 120, 180, 240]
    
    combos = ['6-12', '3-7', '6-3']
    
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(11.4,6.4), sharex=True, sharey=True)
    
    dcca_k = int(width/4)
    
    path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_4\\dataset_r_' + correlation_type + '_' + str(width) + '_' + str(dcca_k) + '.csv'
    df = pd.read_csv(path, index_col=0)
    df['init'] = pd.to_datetime(df['init'], format='%Y-%m-%d %H:%M:%S')
    df['final'] = pd.to_datetime(df['final'], format='%Y-%m-%d %H:%M:%S')
    df.index = df['init']
    #del df['init']
    df = df.sort_index()
    
    path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_4\\dataset_r_' + 'pearson' + '_' + str(width) + '_' + str(width) + '.csv'
    df1 = pd.read_csv(path, index_col=0)
    df1['init'] = pd.to_datetime(df1['init'], format='%Y-%m-%d %H:%M:%S')
    df1['final'] = pd.to_datetime(df1['final'], format='%Y-%m-%d %H:%M:%S')
    df1.index = df1['init']
    #del df['init']
    df1 = df1.sort_index()
        
    df = df[(df['init']>=datetime(2017,2,7,0,0,0)) & (df['init']<=datetime(2017,2,7,10,0,0))]
    df1 = df1[(df1['init']>=datetime(2017,2,7,0,0,0)) & (df1['init']<=datetime(2017,2,7,10,0,0))]    
    
    for i, ax in enumerate(axs.flat):
            
        combo = combos[i]
        
        ax.plot(df.index,df.loc[:,combo], label=('DCCA (n=' + str(dcca_k-1) + ')'))
        ax.plot(df1.index,df1.loc[:,combo], label='PCC')
        
        locator = mdates.HourLocator(interval=1)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
            
        #plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        df_aux = df[df['y']==1]
        ax.axvspan(df_aux.iloc[0,:]['init'], df_aux.iloc[-1,:]['init'], color='tab:red', alpha=0.1, label="Positive Instances")
            
        title_split = combo.split('-')
        title = 'Sensors ' + title_split[0] + ' & ' + title_split[1]        
        if i == 1:
            ax.set_ylabel('Correlation')
            title += ' (two flowrate sensors)'
            ax.legend(ncol=3, loc='lower left')
        elif i == 2:
            ax.set_xlabel('Time')
            title += ' (one sensor of each type)'
        else:
            title += ' (two pressure sensors)'
        
            
        ax.set_title(title)
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.grid(True, axis='y', alpha=0.3,which='both')
    
    plt.ylim(-1,1.1)        
    fig.tight_layout()
    #plt.savefig(path_init + '\\Images\\Results1\\Correlation Evolution\\r_correlation_evolution_lc.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

config = Configuration()
path_init = config.path

df_events = get_df_events(path_init)
df_sensors = get_df_sensors(path_init)

#export_dataset(path_init)

#plot_dcca_k(path_init)
plot_time_windows(path_init)
#plot_correlation_evolution(path_init)

"""
width = 120
correlation_type = "dcca"
dcca_k = int(width/4)

path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_3\\dataset_r_' + correlation_type + '_' + str(width) + '_' + str(dcca_k) + '.csv'
df = pd.read_csv(path, index_col=0)

df['init'] = pd.to_datetime(df['init'], format='%Y-%m-%d %H:%M:%S')
df['final'] = pd.to_datetime(df['final'], format='%Y-%m-%d %H:%M:%S')
df.index = df['init']
df = df.sort_index()

for index, row in df[df['y']>=1].iterrows():
    print(row['init'], row['y'])


flow = [1, 2, 6, 9, 10, 12, 14]
pressure = [3, 7] # , 8, 11, 13, 15
sensors = flow
sensors.extend(pressure)
correlation_type = 'dcca'
combos = list(combinations(sensors, 2))

columns = ['final']
sensors_exclude = [8,11,13,15]
for combo in combos:
    if((combo[0] not in sensors_exclude) and (combo[1] not in sensors_exclude)):
        columns.append(str(combo[0]) + '-' + str(combo[1]))


#print(df)
df_aux = df.isna().sum(axis = 0).sort_values(ascending=False)
print(df_aux)
columns = df_aux[df_aux < 10000].index
df = df.loc[:,columns]

df_aux = df.isna().sum(axis = 0).sort_values(ascending=False)
#print(df_aux)

df_nulls = df[pd.isnull(df).any(axis=1)]
#print(df_nulls)

df_totals1 = df.groupby(pd.Grouper(freq='M')).count()
df_totals2 = df_nulls.groupby(pd.Grouper(freq='M')).count()

df_totals2['total'] = df_totals1['final']

df_total = df_totals2.loc[:,['total','final']]

df_total['%'] = round((df_total['final']*100)/df_total['total'],2)
print(df_total)

"""


#print(df_events)
#mask = (df.index >= str(event['date_start'])) & (df_corr.index < event['date_end'])
#df_corr[mask] = np.NaN










