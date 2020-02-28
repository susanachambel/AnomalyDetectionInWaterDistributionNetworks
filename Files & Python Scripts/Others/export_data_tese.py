# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:48:49 2020

@author: susan
"""

import mysql.connector
import pandas as pd

path_1 = "C:\\Users\\susan\\Documents\\IST\\Tese\\Python Files\\sensor2_month.csv"
path_2 = "C:\\Users\\susan\\Documents\\IST\\Tese\\Python Files\\sensor3_month.csv"

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="@GatitO26@"
)

print(mydb)

query = """SELECT date, value 
FROM infraquinta.sensortgmeasure
where sensortgmeasure.date > cast('2017-05-01 00:00:00' AS datetime)
and sensortgmeasure.date < cast('2017-05-31 23:59:59' AS datetime)
and sensortgId = 2"""

df = pd.read_sql(query, con=mydb)
df.to_csv(index=False, path_or_buf=path_1)
print(df)

query = """SELECT date, value 
FROM infraquinta.sensortgmeasure
where sensortgmeasure.date > cast('2017-05-01 00:00:00' AS datetime)
and sensortgmeasure.date < cast('2017-05-31 23:59:59' AS datetime)
and sensortgId = 3"""

df = pd.read_sql(query, con=mydb)
df.to_csv(index=False, path_or_buf=path_2)
print(df)