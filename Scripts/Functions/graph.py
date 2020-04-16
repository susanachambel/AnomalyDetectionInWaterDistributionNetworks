# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:12:48 2020

@author: susan
"""

import pandas as pd
from configuration import *
import networkx as nx

class Graph:
    def __init__(self):
        self.df= read_network()
        self.G = create_graph(self.df)
        self.df_telemanagement = read_map_telemagament()
        self.df_telemetry = read_map_telemetry()
            
    def find_node(self,link):
        return self.df.loc[link,:]['Node1']
    
    def find_correspondent_sensor(self, group, sensor_id):       
        if (group == "telemanagement"):
            return self.df_telemanagement.loc[int(sensor_id),:]['value']
        else:
            return self.df_telemetry.loc[int(sensor_id),:]['value']

        
    def find_distance(self, source, target):
        try:
            return round(nx.shortest_path_length(self.G,source=str(source),target=str(target), weight="length"),3)
        except nx.NetworkXNoPath:
            return 999999999
    
def create_network_file():
    
    config = Configuration()
    path_init = config.path
    
    df_p = pd.read_csv(path_init + "\\Data\\infraquinta\\EPANET Models\\Network\\pipes.txt", delimiter=r"\s+", header=None)
    df_p = df_p.drop(columns=[4,5,6,7])    
    df_p.columns = ['ID', 'Node1', 'Node2', 'Length']
    
    df_v = pd.read_csv(path_init + "\\Data\\infraquinta\\EPANET Models\\Network\\valves.txt", delimiter=r"\s+", header=None)
    df_v = df_v.drop(columns=[3,4,5,6])
    df_v.columns = ['ID', 'Node1', 'Node2']
    df_v['Length'] = 0
  
    df = pd.concat([df_p,df_v])  
    df.to_csv(path_init + "\\Data\\infraquinta\\EPANET Models\\Network\\network.csv", index = False, header=True)

def read_network():
    config = Configuration()
    path_init = config.path
    df = pd.read_csv(path_init + "\\Data\\infraquinta\\EPANET Models\\Network\\network.csv")
    df.index = df['ID']
    del df['ID']
    return df

def create_graph(df):    
    G = nx.Graph()
    for row in df.itertuples(index=True, name='Pandas'):
        G.add_edge(getattr(row, "Node1"),getattr(row, "Node2"), id=getattr(row, "Index"), length=getattr(row, "Length"))
    return G

def read_map_telemagament():
    config = Configuration()
    path_init = config.path 
    df_aux =  pd.read_csv(path_init + "\\Data\\infraquinta\\EPANET Models\\Network\\map_telemanagement.csv", delimiter=";")
    df_aux.index = df_aux['id']
    del df_aux['id']
    return df_aux


def read_map_telemetry():
    config = Configuration()
    path_init = config.path 
    df_aux = pd.read_csv(path_init + "\\Data\\infraquinta\\EPANET Models\\Network\\map_telemetry.csv", delimiter=";")
    df_aux.index = df_aux['id']
    del df_aux['id']
    return df_aux

  
#myGraph = Graph()
#print(myGraph.df)
#print(myGraph.G.number_of_nodes())
#print(myGraph.G.number_of_edges())
#print(myGraph.G['NODE903']['NODE2082'])
#print(myGraph.find_node('aRM1887150127'))
#print(myGraph.find_distance("NODE903","NODE1716"))




