# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan
"""

import geopandas
import matplotlib.pyplot as plt
import pandas as pd


path = 'zip://C:/Users/susan/Documents/IST/Tese/QGIS/'

tubagens = geopandas.read_file(path + 'shp.zip!shp/tubagens.shp')
medidores_caudal = geopandas.read_file(path + 'shp.zip!shp/medidores_caudal.shp')
intervencoes = geopandas.read_file(path + 'enderecos.zip')

print(intervencoes.columns)

medidores_caudal_target = ['aMC404150114','aMC409150114','aMC406150114']

medidores_caudal = medidores_caudal[medidores_caudal.identidade.isin(medidores_caudal_target)]


print(tubagens['geometry'])

"""
fig, ax = plt.subplots()

tubagens.plot(ax=ax)
medidores_caudal.plot(ax=ax, label="Medidores de Caudal", color='purple')
ax.set(xlabel='Location y', ylabel='Location x', title="Infraquinta's Map")
ax.legend(loc='upper center')


plt.show()
"""


fig = plt.figure(figsize=(8*3,4*3))

widths = [8,8,8]
heights = [4*3]



gs0 = fig.add_gridspec(ncols=3, nrows=1, width_ratios=widths, height_ratios=heights)

gs1 = gs0[0].subgridspec(1, 1)
ax1 = fig.add_subplot(gs1[0])
tubagens.plot(ax=ax1)
medidores_caudal.plot(ax=ax1, label="Medidores de Caudal", color='purple')
ax1.set(xlabel='Location y', ylabel='Location x', title="Infraquinta's Map")
#ax1.legend(loc='upper center')

gs2 = gs0[1].subgridspec(3, 1)
i = 0
for ss in gs2:
    ax = fig.add_subplot(ss)
    ax.set_title(i)
    ax.set_xlabel("")
    i += 1
   
    
gs3 = gs0[2].subgridspec(3, 1)
i = 0
for ss in gs3:
    ax = fig.add_subplot(ss)
    ax.set_title(i)
    ax.set_xlabel("")
    i += 1

