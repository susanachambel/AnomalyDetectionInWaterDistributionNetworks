# -*- coding: utf-8 -*-
"""
Created on Fri Mar 6 20:54:48 2020

@author: susan
"""

import sys
sys.path.append('../Functions')
from configuration import *

from shapely.geometry import Point
from geopandas import GeoSeries
import matplotlib.pyplot as plt
import geopandas

def open_event_coordinates(path_init):
    points = []
    f = open(path_init + "\\Data\\infraquinta\\events\\Event_Location\\CoordXY.txt", "r")
    for line in f:      
      line_split = line.split()
      points.append(Point(float(line_split[0]), float(line_split[1])))
    f.close()
    gs = GeoSeries(points)
    return gs

config = Configuration() 
path_init = config.path
path_qgis = 'zip://C:/Users/susan/Documents/IST/Tese/QGIS/'

gs = open_event_coordinates(path_init)

tubagens = geopandas.read_file(path_qgis + 'shp.zip!shp/tubagens.shp')
medidores_caudal = geopandas.read_file(path_qgis + 'shp.zip!shp/medidores_caudal.shp')
intervencoes = geopandas.read_file(path_qgis + 'enderecos.zip')

medidores_caudal_target = ['aMC404150114','aMC409150114','aMC406150114']
medidores_caudal = medidores_caudal[medidores_caudal.identidade.isin(medidores_caudal_target)]

fig, ax = plt.subplots(1, 1, figsize=(15,15))

tubagens.plot(ax=ax)
medidores_caudal.plot(ax=ax, label="Medidores de Caudal", color='purple')
gs.iloc[:2].plot(ax=ax, marker='*',label="Events", color='red' )
ax.set(xlabel='Location x', ylabel='Location y', title="Infraquinta's Map")
ax.legend(loc='upper left')

plt.show()
