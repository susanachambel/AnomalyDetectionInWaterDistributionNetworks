# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan
"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *
from epanettools.epanettools import EPANetSimulation, Node, Link, Network, Nodes, Links, Patterns, Pattern, Controls, Control # import all elements needed
from epanettools.examples import simple # this is just to get the path of standard examples
import matplotlib.pyplot as plt
import numpy as np

config = Configuration()
path_init = config.path

file = 'link_flow_summer'

path_import = path_init + "\\Data\\infraquinta\\simulated\\" + file + "_combined.csv"

df = pd.read_csv(path_import, sep=",")

#df_description = df.describe().T[["mean", "std","min","max"]]

#print(df_description)


#df = df.iloc[:,1]

df = df['0']
df = df[(df>0.0001) & (df < 10)]

df[df < 0.0001] = 0


#df = df[(df!=0) & (df < 0.0001)]

print(df)

array = df.values
print(array)

_ = plt.hist(array, bins='auto')  # arguments are passed to np.histogram


plt.title("Histogram with 'auto' bins")

plt.show()


