# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:12:48 2020

@author: susan
"""

import pandas as pd
from configuration import *
from data_selection import *
import json

def process_data_request(request):
    
    wme = request.form.get('wme')
    sensors_id = json.loads(request.form.get('sensors_id'))
    date_range_min = request.form['date_range_min']
    date_range_max = request.form['date_range_max']
    calendar = json.loads(request.form['calendar'])
    granularity_unit = request.form['granularity_unit']
    granularity_frequence = int(request.form['granularity_frequence'])
    mode = request.form['mode']
    pairwise_comparisons = request.form['pairwise_comparisons']
    correlations = json.loads(request.form['correlations'])
    pca = request.form['pca']
    
    dfs = {}
    
    if(wme == 'infraquinta'):
        dfs = get_data(wme, sensors_id, date_range_min, date_range_max, calendar, granularity_unit, granularity_frequence)    
        line_chart_data = dfs_to_json(dfs)
        #heat_map_data = dfs_analysis(dfs)
        #data = {'line_chart': line_chart_data}
        return line_chart_data
    else:
        return "do something"

def dfs_to_json(dfs):   
    dfs_json = {} 
    for sensor_id in dfs:   
        df = dfs[sensor_id]
        df.index = df.index.map(str)
        df_json = df.to_json(orient='index')
        df_json = json.loads(df_json)       
        dfs_json[sensor_id] = df_json
    return dfs_json    


def get_json():
    config = Configuration()
    path_init = config.path
    with open(path_init + '//Data//sensors_list.json', 'r') as myfile:
        data=myfile.read()
    json_data = json.loads(data)
    return json_data

def get_data(wme, sensors_id, date_range_min, date_range_max, calendar, granularity_unit, granularity_frequence): 
    
    config = Configuration()
    mydb = config.create_db_connection()    
    path_init = config.path
    json_data = get_json()
    
    date_range_min += " 00:00:00"
    date_range_max += " 23:59:59"
    
    dfs = {}
    
    for sensor_id in sensors_id:
        info = json_data[wme][sensor_id]

        df = pd.DataFrame()
        
        if(info['focus']=='real'):
            if(info['group']=='telemetry'):
                df = select_data_db(mydb, wme, info['group'], int(info['name']), date_range_min, date_range_max)
            else:
                df = select_data(path_init, wme, "interpolated", int(info['name']), date_range_min, date_range_max)
        else:
            df = select_data_db(mydb, wme, info['focus'], int(info['name']), date_range_min, date_range_max)
        
        dfs[sensor_id] = df
        
        #tratar da granularidade
        #tradar dos dias da semana
    
    mydb.close()    
    return dfs
        
    
#get_data('infraquinta', ['1', '2', '3'],"2017-06-01", "2017-06-08", [], 'hours', 1);  