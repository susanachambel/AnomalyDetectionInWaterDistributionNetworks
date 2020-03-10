# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:48:49 2020

@author: susan

"""

import mysql.connector
import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *

root = read_config()
path_init = get_path(root)
db_config = get_db(root)
wmes = get_wmes(root)

mydb = mysql.connector.connect(
  host=db_config['host'],
  user=db_config['user'],
  passwd=db_config['pw']
)

print(mydb)

print("\nReport initiated\n")

cursor = mydb.cursor(buffered=True)

path = path_init + "\\Reports\\sensor_report.csv"

df = pd.DataFrame()
df["id"] = pd.Series([], dtype=int)
df["n_rows"] = pd.Series([], dtype=int)
df["name"] = pd.Series([], dtype=str)
df["type"] = pd.Series([], dtype=str)
df["wme"] = pd.Series([], dtype=str)

for wme in wmes:
    query = "SELECT count(*) FROM " + wme + ".sensortg"
      
    cursor.execute(query)
    sensor_count = cursor.fetchall()[0][0]
    
    print(wme + ": " + str(sensor_count) + " sensors found")
    
    sensor_id = 1
    
    for i in range(1,sensor_count+1):
            
        query = "SELECT count(*) FROM " + wme + ".sensortgmeasure where sensortgId = " + str(sensor_id)
        cursor.execute(query)
        n_rows = cursor.fetchall()[0][0]
        
        query = "SELECT name, unit FROM " + wme + ".sensortg where id = " + str(sensor_id)
        cursor.execute(query)
        result = cursor.fetchall()[0]
        name = result[0]
        unit = result[1]
        
        query = "SELECT min(date) FROM " + wme + ".sensortgmeasure where sensortgId = " + str(sensor_id)
        cursor.execute(query)
        min_date = cursor.fetchall()[0][0]
        
        query = "SELECT max(date) FROM " + wme + ".sensortgmeasure where sensortgId = " + str(sensor_id)
        cursor.execute(query)
        max_date = cursor.fetchall()[0][0]
        
        type = ""
        
        if unit == "bar":
            type = "pressure"
        else:
            type = "flow"
        
        df = df.append({'wme': wme, 'id': i, 'name': name, 'type': type, 'n_rows': n_rows, 'min_date': min_date, 'max_date': max_date}, ignore_index=True)
        
        sensor_id += 1
        
df['max_date'] = pd.to_datetime(df['max_date'], format='%Y-%m-%d %H:%M:%S')
df['min_date'] = pd.to_datetime(df['min_date'], format='%Y-%m-%d %H:%M:%S')
       
df.to_csv(index=False, path_or_buf=path)
print("\nReport completed")  

cursor.close()
mydb.close()