# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:12:48 2020

@author: susan
"""

import pandas as pd
from configuration import *
from data_selection import *
from correlation import *
import json
import numpy as np
from itertools import combinations, product
from graph import *

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
        #heat_map_data = dfs_analysis(dfs, mode, pairwise_comparisons, correlations, pca)
        #heat_map_data = dfs_analysis_2(dfs, mode, pairwise_comparisons, correlations, pca)
        heat_map_data = dfs_analysis_3(wme, dfs, mode, pairwise_comparisons, correlations, pca)
        data = {'wme':wme ,'line_chart': line_chart_data, 'heat_map': heat_map_data,
                'selected_sensor_list':sensors_id, 'pairwise_comparisons':pairwise_comparisons}
        
        #data = {'wme':wme ,'line_chart': line_chart_data, 'selected_sensor_list':sensors_id}
        return data
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
              
        df = set_granularity(df, granularity_unit, granularity_frequence)
        df = set_calendar(df, calendar)      
        dfs[sensor_id] = df
    
    mydb.close()    
    return dfs

def set_granularity(df, unit, frequence):
    unit_aux = "min"
    if unit == "0":
        unit_aux = "min"
    elif unit == "1":
        unit_aux = "h"
    elif unit == "2":
        unit_aux = "d"
    elif unit == "3":
        unit_aux = "m"
    df = df.resample(str(frequence) + unit_aux).mean()
    return df     
 
def set_calendar(df, calendar):
    if (len(calendar) == 7):
        return df
    else:
        calendar = [int(day) for day in calendar]
        # df = df[df.index.weekday.isin(calendar)]
        df[df.index.weekday.isin(calendar)] = np.nan       
        return df
    
def dfs_analysis(dfs, mode, pairwise_comparisons, correlations, pca):
        
    for key in dfs:
        df = dfs[key]
        dfs[key] = df.dropna()
    
    df_keys = list(dfs.keys())
    df_keys_len = len(df_keys)
    
    dic = {}
    i = 0   
    for df_key in df_keys:
        dic[df_key] = i
        i += 1
    
    combos = combinations(df_keys, 2)
    z = np.zeros(shape=(df_keys_len,df_keys_len))
    z[z == 0] = 999999999
       
    i = 0
    for combo in combos:
        i +=1
        z[dic[combo[0]], dic[combo[1]]] = round(calculate_pearson(dfs[combo[0]], dfs[combo[1]]),3)
            
    x = df_keys
    y = df_keys

    z_list = z.tolist()
           
    result = {}
    result['pearson'] = {'x':x, 'y':y, 'z':z_list}
    
    return result

def dfs_analysis_2(dfs, mode, pairwise_comparisons, correlations, pca):
    
    for key in dfs:
        df = dfs[key]
        dfs[key] = df.dropna()
    
    df_keys = list(dfs.keys())
    
    combo_results = []
    
    combos = combinations(df_keys, 2)
    
    i = 0
    for combo in combos:
        i +=1
        combo_results.append([combo[0],combo[1],round(calculate_pearson(dfs[combo[0]], dfs[combo[1]]),3)])
        
    distances = calculate_distances(df_keys)      
    matrixes = calculate_matrix(combo_results, distances)
    
    result = {}
    result['pearson'] = matrixes
        
    return result
    
def calculate_matrix(combo_results, distances):
    
    sensors = list(distances.keys())
    sensors_len = len(sensors)
    
    dic = {}
    for sensor in sensors:
        sensors_ordered = distances[sensor]       
    
        dic_aux = {}
        i = 0   
        for sensor_ordered in sensors_ordered:
            dic_aux[sensor_ordered] = i
            i += 1
        
        z = np.zeros(shape=(sensors_len,sensors_len))
        z[z == 0] = 999999999
        
        combos = combinations(sensors_ordered, 2)
        
        i = 0
        for combo in combos:
            i +=1
            
            for combo_result in combo_results:
                if(((combo_result[0] == combo[0]) and (combo_result[1] == combo[1])) 
                or ((combo_result[0] == combo[1]) and (combo_result[1] == combo[0]))):
                    z[dic_aux[combo[0]], dic_aux[combo[1]]] = combo_result[2]
        
        dic[sensor] = {'x': sensors_ordered, 'y': sensors_ordered, 'z': z.tolist()}
        
    return dic
    

def find_node_by_sensor(sensor):
    
    myGraph = Graph()
    
    node = ""
    
    if(sensor['focus'] == 'real'):
        node_aux = myGraph.find_correspondent_sensor(sensor['group'],sensor['name'])
                    
        if(sensor['type'] == 'flow'):
            node = myGraph.find_node(node_aux)
        else:
            node = node_aux
    else:
        if(sensor['type'] == 'flow'):
            node = myGraph.find_node(sensor['name_long'])
        else:
            node = sensor['name_long']
            
    return node
    

def calculate_distances(sensors_id):
    
    myGraph = Graph()
    
    json_data = get_json()
    
    dist = {}
    
    for sensor_id in sensors_id:
        dist_aux = {}
        for sensor_id_aux in sensors_id:
            
            if sensor_id_aux == sensor_id:
                dist_aux[sensor_id_aux] = 0
            else:
                
                try:
                    node1 = find_node_by_sensor(json_data['infraquinta'][sensor_id])
                    node2 = find_node_by_sensor(json_data['infraquinta'][sensor_id_aux])
                    dist_aux[sensor_id_aux] = myGraph.find_distance(node1, node2)
                except KeyError:
                    dist_aux[sensor_id_aux] = 999999999
                    
        dist[sensor_id] = list({k: v for k, v in sorted(dist_aux.items(), key=lambda item: item[1])}.keys())
                          
    return dist         

def find_distance_sensors(sensor_id1, sensor_id2):
    myGraph = Graph()   
    json_data = get_json()
    try:
        node1 = find_node_by_sensor(json_data['infraquinta'][sensor_id1])
        node2 = find_node_by_sensor(json_data['infraquinta'][sensor_id2])
        return myGraph.find_distance(node1, node2)
    except KeyError:
        return 999999999

def dfs_analysis_3(wme, dfs, mode, pairwise_comparisons, correlations, pca):
     
    result = {}
    
    for key in dfs:
        df = dfs[key]
        dfs[key] = df.dropna()
    
    df_keys = list(dfs.keys())
    
    dic = {}
    
    if (pairwise_comparisons == "all pairs"):
    
        combos = combinations(df_keys, 2)
        
        for df_key in df_keys:
            dic[df_key] = [{'id':df_key, 'dist':0, 'corr':1}]
        
        for combo in combos:
            
            pearson_correlation = round(calculate_pearson(dfs[combo[0]], dfs[combo[1]]),3)
            distance = 1
            
            if (wme == 'infraquinta'):  
                distance = find_distance_sensors(combo[0], combo[1])
                              
            dic[combo[0]].append({'id':combo[1], 'dist':distance, 'corr':pearson_correlation})
            dic[combo[1]].append({'id':combo[0], 'dist':distance, 'corr':pearson_correlation})
        
    else:
        
        json_data = get_json()
        sensors_flow = []
        sensors_pressure = []
        
        for df_key in df_keys:          
            sensor_type = json_data[wme][df_key]['type']        
            if (sensor_type == 'flow'):
                sensors_flow.append(df_key)
                dic[df_key] = []
            else:
                sensors_pressure.append(df_key)
        
        combos = product(sensors_flow, sensors_pressure)
        
        for combo in combos:
            
            pearson_correlation = round(calculate_pearson(dfs[combo[0]], dfs[combo[1]]),3)
            distance = 1
            
            if (wme == 'infraquinta'):  
                distance = find_distance_sensors(combo[0], combo[1])
                
            dic[combo[0]].append({'id':combo[1], 'dist':distance, 'corr':pearson_correlation})
    
    result['pearson'] = dic
          
    return result

#dist = calculate_distances(list(map(str, range(0,3))))       
        
#dfs = get_data('infraquinta', ['0', '1', '2', '4', '5'], "2017-06-01", "2017-06-07", ["0","1","2","3","4","5","6"], 'hours', 1);
#data = dfs_analysis_infraquinta('infraquinta', dfs, "", "cross type", "", "")
#print(data)


