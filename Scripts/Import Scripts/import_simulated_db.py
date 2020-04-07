# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:04:32 2020

@author: susan
"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *


def import_simulated_sensors_db():
    
    config = Configuration()
    path_init = config.path
    mydb = config.create_db_connection()
    cursor = mydb.cursor()
    
    files = ['node_pressure','link_flow']
    
    links_tg = {'6':1, 'aTU1096150205':2, 'aTU1093150205':6, 
                'aTU455150205':9, 'aTU4981150302':'10', 
                '2':'12' , 'aTU1477150205':'14'}

    nodes_tg = {'aMI817150114':3, 'aMC402150114':8,
               'aMC404150114':11, 'aMC401150114':13, 
               'aMC406150114':15} 
    
    # FYI: o sensor real 7 aponta para o mesmo simulado que o 3
    
    links_tm = {'aTU4992150504':393, 'aTU4971150302':950, 
               'aTU1320150302':949, 'aTU4972150302':1186,
               'aTU4970150302':574, 'aTU1094150205': 976}
                
    for file in files:
        
        print(2*'\x1b[2K\r' + file, end="")
        
        unit = ""
        
        if "node" in file:
            unit = "bar"
        else:
            unit = "m3/h"
        
        path_import = path_init + "\\Data\\infraquinta\\simulated\\" + file + "_summer.csv"
        df = pd.read_csv(path_import, sep=",")
            
        columns = list(df.columns)
        columns_len = len(columns)
        
        counter = 0
            
        for column in columns:
            
            corr_id_tm = "NULL"
            corr_id_tg = "NULL"
                     
            if unit == 'bar':
                if column in nodes_tg:
                    corr_id_tg = nodes_tg.get(column)
            else:
                 if column in links_tg:
                     corr_id_tg = links_tg.get(column)
                 elif column in links_tm:
                     corr_id_tm = links_tm.get(column)
                     
            
            query = "INSERT INTO infraquinta.sensorsim(name, unit, corr_id_tg, corr_id_tm) VALUES ('" + str(column) + "', '" + unit + "'," + str(corr_id_tg) + "," + str(corr_id_tm) + ");"
            cursor.execute(query)
            mydb.commit()
            
            counter += 1
            
            print(2*'\x1b[2K\r' + file + " " +  str(counter) + " / " + str(columns_len), flush=True, end="\r")
        print(2*'\x1b[2K\r' + file + " " + str(columns_len) + " / " + str(columns_len))    
    cursor.close()
    mydb.close()


def import_simulated_data_db():
    
    config = Configuration()
    path_init = config.path
    mydb = config.create_db_connection()
    cursor = mydb.cursor()
    
    files = ['node_pressure','link_flow']
    seasons = ['summer', 'winter']
    
    for file in files:
        
        for season in seasons:
            path_import = path_init + "\\Data\\infraquinta\\simulated\\" + file + "_" + season + ".csv"
            df = pd.read_csv(path_import, sep=",")
            
            columns = list(df.columns)
            columns_len = len(columns)
            
            print(2*'\x1b[2K\r' + file + "_" + season, end="")
        
            counter = 0
            
            for column in columns:
                values = df[column].to_list()
                sensorsimId = 0
                
                query = "SELECT id FROM infraquinta.sensorsim where name='" + column + "';" 
                
                cursor.execute(query)
                         
                for (id) in cursor:
                    sensorsimId = id[0]
                    
                rows = []
                
                for value in values:
                    rows.append((value, sensorsimId, season))
                    
                query = "INSERT INTO infraquinta.sensorsimmeasure(value, sensorsimId, season) VALUES (%s, %s, %s);"
                cursor.executemany(query, rows)
                mydb.commit()
                
                counter += 1
                print(2*'\x1b[2K\r' + file + "_" + season + " " +  str(counter) + " / " + str(columns_len), flush=True, end="\r")
                
            print(2*'\x1b[2K\r' + file + "_" + season + " " + str(columns_len) + " / " + str(columns_len))    

    cursor.close()
    mydb.close()  

                                      
def delete_sensor_rows():
    config = Configuration()
    mydb = config.create_db_connection()
    cursor = mydb.cursor()
     
    query = "delete from infraquinta.sensorsim"
    cursor.execute(query)
    mydb.commit()
    
    query = "ALTER TABLE infraquinta.sensorsim AUTO_INCREMENT = 1"
    cursor.execute(query)
    mydb.commit()

    cursor.close()
    mydb.close()

def delete_sensor_rows_measures():
    config = Configuration()
    mydb = config.create_db_connection()
    cursor = mydb.cursor()
    
    query = "delete from infraquinta.sensorsimmeasure"
    cursor.execute(query)
    mydb.commit()
    
    query = "ALTER TABLE infraquinta.sensorsimmeasure AUTO_INCREMENT = 1"
    cursor.execute(query)
    mydb.commit()

    cursor.close()
    mydb.close()