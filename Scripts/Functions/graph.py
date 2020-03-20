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
            
    def find_node(self,link):
        return self.df.loc[link,:]['Node1']
    
    def find_distance(self, source, target):
        return nx.shortest_path_length(self.G,source=source,target=target, weight="length")
    
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
    print(df)

def read_network():
    config = Configuration()
    path_init = config.path
    df = pd.read_csv(path_init + "\\Data\\infraquinta\\EPANET Models\\Network\\network.csv")
    df.index = df['ID']
    del df['ID']
    print(df)
    return df

def create_graph(df):    
    G = nx.Graph()
    for row in df.itertuples(index=True, name='Pandas'):
        G.add_edge(getattr(row, "Node1"),getattr(row, "Node2"), id=getattr(row, "Index"), length=getattr(row, "Length"))
    return G

    
myGraph = Graph()

print(myGraph.G.number_of_nodes())
print(myGraph.G.number_of_edges())
print(myGraph.G['NODE903']['NODE2082'])
print(myGraph.find_distance("aMI817150114","aMC402150114"))





