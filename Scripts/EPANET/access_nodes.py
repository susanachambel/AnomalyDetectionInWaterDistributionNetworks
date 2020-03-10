# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan
"""

import numpy as np
import os, pprint
from epanettools.epanettools import EPANetSimulation, Node, Link, Network, Nodes, Links, Patterns, Pattern, Controls, Control # import all elements needed
from epanettools.examples import simple # this is just to get the path of standard examples
import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *


path = "C:\\Users\\susan\\Documents\\IST\\Tese\\EPANET\\Simulation Models\\Verao_LM.net"
path = "C:\\Users\\susan\\Documents\\IST\\Tese\\EPANET\\Files\\StatusQuoInverno2018.inp"

pp=pprint.PrettyPrinter() # we'll use this later.
#file = os.path.join(os.path.dirname(simple.__file__),'Net3.inp') # open an example
es=EPANetSimulation(path)

nodes = [es.network.nodes[x].id for x in list(es.network.nodes)[:]]

links = [es.network.links[x].id for x in list(es.network.links)[:]]

print(len(es.network.nodes))

print(len(es.network.links))

root = read_config()
path_init = get_path(root)
wmes = get_wmes(root)


path = path_init + "\\Data\\infraquinta\\simulated\\NodePressureVerao.txt"
path = path_init + "\\Data\\infraquinta\\simulated\\NodePressure.txt"

df_pressure = pd.read_csv(path, sep="  ", header=None)
df_pressure.columns = nodes

df_flow = pd.read_csv(path, sep="  ", header=None)



print(df)









