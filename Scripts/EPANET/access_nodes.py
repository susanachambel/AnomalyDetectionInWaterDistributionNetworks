# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan
"""

import pandas as pd
import numpy as np
import os, pprint
import sys
sys.path.append('../Functions')
from configuration import *
from epanettools.epanettools import EPANetSimulation, Node, Link, Network, Nodes, Links, Patterns, Pattern, Controls, Control # import all elements needed
from epanettools.examples import simple # this is just to get the path of standard examples

config = Configuration()
path_init = config.path


path = path_init + "\\Data\\infraquinta\\EPANET Models\\INP Files\\StatusQuoInverno2018.inp"
es=EPANetSimulation(path)

nodes = [es.network.nodes[x].id for x in list(es.network.nodes)[:]]
n=es.network.nodes
links = [es.network.links[x].id for x in list(es.network.links)[:]]
m=es.network.links

print("Nodes: " + str(len(es.network.nodes)))
print("Links: " + str(len(es.network.links)))


node_types = []
for node_type in Node.node_types:
    node_types.append(node_type)

link_types = []
for link_type in Link.link_types:
    link_types.append(link_type)
       
node_types_count = {}
for node_type in node_types:
    node_types_count[node_type] = len([y.id for x,y in n.items() if y.node_type==Node.node_types[node_type]])
    #print([y.id for x,y in n.items() if y.node_type==Node.node_types[node_type]][:10])
print(node_types_count)

link_types_count = {}
for link_type in link_types:
    link_types_count[link_type] = len([y.id for x,y in m.items() if y.link_type==Link.link_types[link_type]])
    #print([y.id for x,y in m.items() if y.link_type==Link.link_types[link_type]][:5])
print(link_types_count)

print([x for x in links if not(("aVS" in x) or ("aRM" in x) or ("aTU" in x))])
print(len([x for x in nodes if "NODE" in x]))









