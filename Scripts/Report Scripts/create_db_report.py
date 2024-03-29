# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:48:49 2020

@author: susan

Exports telemanagement (telegestao) data to a csv -> one sensor per file

"""

import mysql.connector
import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *

config = Configuration()
path_init = config.path
wmes= config.wmes
mydb = config.create_db_connection()

print(mydb)

print("\nReport initiated\n")

cursor = mydb.cursor(buffered=True)

path = path_init + "\\Reports\\db_report.csv"


df = pd.DataFrame()
df["n_pressure_sensors"] = pd.Series([], dtype=int)
df["n_flow_sensors"] = pd.Series([], dtype=int)
df["wme"] = pd.Series([], dtype=str)

for wme in wmes:
    
    print(wme)
    
    query = "SELECT count(*) FROM " + wme + ".sensortg where unit = 'm3/h'"  
    cursor.execute(query)
    n_flow_sensors = cursor.fetchall()[0][0]
    
    query = "SELECT count(*) FROM " + wme + ".sensortg where unit = 'bar'"  
    cursor.execute(query)
    n_pressure_sensors = cursor.fetchall()[0][0]
    
    query = "SELECT min(date) FROM " + wme + ".sensortgmeasure"
    cursor.execute(query)
    min_date = cursor.fetchall()[0][0]
    
    query = "SELECT max(date) FROM " + wme + ".sensortgmeasure"
    cursor.execute(query)
    max_date = cursor.fetchall()[0][0]
           
    df = df.append({'wme': wme, 'n_pressure_sensors': n_pressure_sensors, 
                    'n_flow_sensors': n_flow_sensors, 
                    'min_date': min_date, 'max_date': max_date}, 
                    ignore_index=True)
        
        
df['max_date'] = pd.to_datetime(df['max_date'], format='%Y-%m-%d %H:%M:%S')
df['min_date'] = pd.to_datetime(df['min_date'], format='%Y-%m-%d %H:%M:%S')         
df.to_csv(index=False, path_or_buf=path)
print("\nReport completed")
cursor.close()
mydb.close()