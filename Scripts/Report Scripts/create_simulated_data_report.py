# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan

"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *

root = read_config()
path_init = get_path(root)
wmes = get_wmes(root)


print("\nReport initiated\n")

path = path_init + "\\Data\\infraquinta\\simulated\\NodePressure.txt"

df = pd.read_csv(path, sep="  ", header=None)

print(df.info())


print("\nReport completed")  
