# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:12:48 2020

@author: susan
"""

import pandas as pd
from datetime import datetime

"""
    1. Converts the dates from string to datatime
    2. Transforms the date column as index
    Returns: df ready to use 
"""
def process_df(df):   
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    df.index = df['date']
    del df['date']
    df = df.sort_index()
    return df

def process_simulated_df(df, date_min, date_max):
    dates = pd.date_range(date_min, date_max, freq='1min')
    len_dates = len(dates)
    df = df[:len_dates]
    df.index = dates
    return df   

"""
    Date format: 2017-04-21 23:59:59
    Returns: df with data from files
"""  
def select_data(path_init, wme, data_type, sensor_id, date_min, date_max):
        
    path = path_init + "\\Data\\" + wme + "\\" + data_type + "\\sensor_" + str(sensor_id) + ".csv"
    df = pd.read_csv(path, delimiter='')
    df = process_df(df)
      
    if ((date_min == 1) or (date_max == 1)):
        return df
    else:
        df = df[(df.index >= date_min) & (df.index <= date_max)]
        return df

"""
    Date format: 2017-04-21 23:59:59
    Returns: df with data from the database
"""    
def select_data_db(mydb, wme, data_type, sensor_id, date_min, date_max):
    
    measure_table = ""
    sensorid_row = ""
     
    if data_type == "telemanagement":
        measure_table = "sensortgmeasure"
        sensorid_row = "sensortgId"
    elif data_type == "telemetry":
        measure_table = "sensortmmeasure"
        sensorid_row = "sensortmId"
    elif data_type == "simulated":
        measure_table = "sensorsimmeasure"
        sensorid_row = "sensorsimId"
        
    query = ""
        
    if data_type == "simulated":
        
        date_min_aux = datetime.strptime(date_min, '%Y-%m-%d %H:%M:%S')
        summer_min = datetime.strptime('2017-05-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        summer_max = datetime.strptime('2017-10-31 23:59:59', '%Y-%m-%d %H:%M:%S')
        
        season = ""
        
        if (date_min_aux > summer_min) and (date_min_aux < summer_max):
            season = "summer"
        else:
            season = "winter"
        
        query = ("SELECT value" + " FROM " +  wme + "." + measure_table +
        " where " + sensorid_row + " = " + str(sensor_id) + " and season='" + season + "';") 
        
        df = pd.read_sql(query, con=mydb)
        df = process_simulated_df(df, date_min, date_max)
        return df
    
    else:
     
        query = ("SELECT date, value" + " FROM " +  wme + "." + measure_table +
        " where date > cast('" + date_min + "' AS datetime) and date < cast('" + 
        date_max + "' AS datetime) and " + sensorid_row + " = " + str(sensor_id))
        
        df = pd.read_sql(query, con=mydb)
        df = process_df(df)
        return df

"""
    Returns: array of sensor ids
"""
def select_sensors_db(mydb, wme, data_type):
    
    sensor_table = ""
    
    if data_type == "telemanagement":
        sensor_table = "sensortg"
    elif data_type == "telemetry":
        sensor_table = "sensortm"
    elif data_type == "simulated":
        sensor_table == "sensorsim"
    
    query = ("SELECT id FROM " + wme + "." + sensor_table) 
    df = pd.read_sql(query, con=mydb)['id'].to_numpy() 
    return df
