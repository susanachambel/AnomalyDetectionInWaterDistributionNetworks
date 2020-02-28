# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:48:49 2020

@author: susan
"""

import mysql.connector
import pandas as pd

path = "C:\\Users\\susan\\Documents\\IST\\Tese\\Python Files\\Data\\Infraquinta\\Real\\"

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="@GatitO26@"
)

print(mydb)

sensor_id = 1

for i in range(1,16):
    
    path_tmp = path
    
    query = """SELECT date, value 
    FROM infraquinta.sensortgmeasure
    where sensortgmeasure.date > cast('2017-05-01 00:00:00' AS datetime)
    and sensortgmeasure.date < cast('2017-05-01 23:59:59' AS datetime)
    and sensortgId = """
    
    query = """SELECT date, value 
    FROM infraquinta.sensortgmeasure
    where sensortgId = """
    
    query = query + str(sensor_id)
    path_tmp += "sensor_" + str(sensor_id) + ".csv"
    
    df = pd.read_sql(query, con=mydb)
    df.to_csv(index=False, path_or_buf=path_tmp)
    print(df)
    
    sensor_id += 1


