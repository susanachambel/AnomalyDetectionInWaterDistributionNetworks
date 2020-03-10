# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:12:48 2020

@author: susan
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path_1 = "C:\\Users\\susan\\Documents\\IST\\Tese\\Python Files\\sensor2.csv"
path_2 = "C:\\Users\\susan\\Documents\\IST\\Tese\\Python Files\\sensor3.csv"
path_3 = "C:\\Users\\susan\\Documents\\IST\\Tese\\Python Files\\sensor2_month.csv"
path_4 = "C:\\Users\\susan\\Documents\\IST\\Tese\\Python Files\\sensor3_month.csv"


df = pd.read_csv(path_1)
#df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
df['date'] = df['date'].map(lambda x:x.split(' ')[1])

print(df)
print(df.info())

fig, ax = plt.subplots()
ax.plot(df['date'],df['value'])
ax.set(xlabel='', ylabel='Water flow [m3/h]',
       title='Sensor 2 during 29-05-2017')
plt.xticks(np.arange(0, 1124, step=100),rotation=45)
plt.tight_layout()
plt.savefig("C:\\Users\\susan\\Desktop\\sensor2.png", dpi=100)

df = pd.read_csv(path_2)
split_tmp = df['date'].map(lambda x:x.split(' ')[1])
df['date'] = split_tmp

print(df)
print(df.info())

fig, ax = plt.subplots()
ax.plot(df['date'],df['value'])
ax.set(xlabel='', ylabel='Water pressure [bar]',
       title='Sensor 3 during 29-05-2017')
plt.xticks(np.arange(0, 1124, step=100),rotation=45)
plt.tight_layout()
plt.savefig("C:\\Users\\susan\\Desktop\\sensor3.png", dpi=100)


##########################fazer a media do mês##########################################


"""
df = pd.read_csv(path_3)
#df['date'] = df['date'].map(lambda x:x.split(' ')[0])

print(df)
print(df.info())

fig, ax = plt.subplots()
ax.plot(df['date'],df['value'])
ax.set(xlabel='', ylabel='flow [m3/h]',
       title='Sensor 2 during 29-05-2017')
plt.tight_layout()
plt.xticks(np.arange(0, 34389, step=5000),rotation=45)

df = pd.read_csv(path_4)
#df['date'] = df['date'].map(lambda x:x.split(' ')[0])

print(df)
print(df.info())

fig, ax = plt.subplots()
ax.plot(df['date'],df['value'])
ax.set(xlabel='', ylabel='flow [m3/h]',
       title='Sensor 3 during 29-05-2017')
plt.tight_layout()
plt.xticks(np.arange(0, 34389, step=5000),rotation=45)
"""












