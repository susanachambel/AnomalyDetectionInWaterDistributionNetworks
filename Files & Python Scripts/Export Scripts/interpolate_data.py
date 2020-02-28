# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:12:48 2020

@author: susan
"""

import pandas as pd
import numpy as np
import time
import datetime


"""
    When df_min == 1 or df_max == 1, the date range is the minimum and maximum
    dates of the dataframe
"""
def input_data(df, df_min, df_max):
    print("    Creating data ", end = '')
    start_time = time.time()
    
    if (df_min == 1 or df_max == 1):
        df_min = df.index.min()    
        df_min = df_min.replace(second=0, microsecond=0)
        df_max = df.index.max()
             
    dates = pd.date_range(df_min, df_max, freq='1min')

    elapsed_time = time.time() - start_time
    print(elapsed_time)
    
    
    start_time = time.time()
    dates_len = len(dates)
    counter = 1
    
    for date in dates:
        if (date in df.index):
            dates = dates.drop(date)

        print(2*'\x1b[2K\r' + "    Inputing data " +  str(counter) + " / " + str(dates_len), flush=True, end="\r")
        counter += 1
    
    dates_len = len(dates)
    
    df_tmp = pd.DataFrame({'date':dates, 'value':[np.nan]*dates_len})
    df_tmp.index = df_tmp['date']
    del df_tmp['date']
       
    elapsed_time = time.time() - start_time
    print(2*'\x1b[2K\r' + "    Inputing data " + str(datetime.timedelta(seconds=elapsed_time)))
    
    start_time = time.time()
    
    df = pd.concat([df,df_tmp])
    
    elapsed_time = time.time() - start_time
    print(2*'\x1b[2K\r' + "    Concat " + str(datetime.timedelta(seconds=elapsed_time)))
    
    print("    Sorting data ", end = '')
    start_time = time.time()
    df = df.sort_index() 
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    
    return df

def calculate_value(min_date, date, max_date, min_value, max_value):  
    dist = max_date - min_date  
    min_dist = abs(date-min_date)
    max_dist = abs(date-max_date)  
    result = ((1-(min_dist/dist))*min_value)+((1-(max_dist/dist))*max_value)  
    return round(result,3)
  
def find_next_row(df, index):
    row = df.iloc[index+1]
    if pd.isnull(row['value']):
        return find_next_row(df,index+1)
    else:
        return row         

def interpolate(df):
      
    df.reset_index(level=0, inplace=True)
    
    last_row_flag = 0
    last_row = np.NaN
    
    counter = 1
    df_len = df.shape[0]
     
    for index, row in df.iterrows():
        value = row['value']
        
        if not(pd.isnull(value)):
            last_row = row
            last_row_flag = 1
        
        elif last_row_flag == 0:
            try:
                next_row = find_next_row(df, index)
                df.iloc[index] = [row['date'], next_row['value']]
                       
            except IndexError:
                df.iloc[index] = [row['date'], last_row['value']]
        else:
            try:
                next_row = find_next_row(df, index)
                value = calculate_value(last_row['date'],row['date'], next_row['date'],
                                        last_row['value'], next_row['value'])
                df.iloc[index] = [row['date'], value]
            
            except IndexError:
                df.iloc[index] = [row['date'], last_row['value']]
                     
        print(2*'\x1b[2K\r' + "  Interpolation " +  str(counter) + " / " + str(df_len), flush=True, end="\r")
        counter += 1
        
    df.index = df['date']
    del df['date']
    
    return df
 
def delete_flagged_elements(df):
    flagged = df[df.index.second != 0].index
    df = df.drop(flagged)
    return df

def create_interpolated_df(df, df_min, df_max):
    
    print("  Inputation")
    df = input_data(df, df_min, df_max)
              
    start_time = time.time()
    df = interpolate(df)
    elapsed_time = time.time() - start_time
    print(2*'\x1b[2K\r' + "  Interpolation " + str(datetime.timedelta(seconds=elapsed_time)))
                     
    print("  Delete flagged elements ", end="")
    start_time = time.time()
    df = delete_flagged_elements(df)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    
    return df


