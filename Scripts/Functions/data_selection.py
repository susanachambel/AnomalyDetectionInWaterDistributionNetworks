# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:12:48 2020

@author: susan
"""

import pandas as pd



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

"""
    Date format: 2017-04-21 23:59:59
    Returns: df with data from files
"""  
def select_data(path_init, wme, data_type, sensor_id, date_min, date_max):
        
    path = path_init + "\\Data\\" + wme + "\\" + data_type + "\\sensor_" + str(sensor_id) + ".csv"
    df = pd.read_csv(path)
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
        measure_table= "sensortmmeasure"
        sensorid_row = "sensortmId"
    else:
        measure_table == "sensorsimmeasure"
        sensorid_row == "sensorsimId"
        
    # o query para o simulado tem de ser diferente
     
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
    else:
        sensor_table == "sensorsm"
    
    query = ("SELECT id FROM " + wme + "." + sensor_table) 
    df = pd.read_sql(query, con=mydb)['id'].to_numpy() 
    return df