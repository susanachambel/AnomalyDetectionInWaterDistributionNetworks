# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 20:31:44 2020

@author: susan

"""

import pandas as pd

class EventArchive:
    def __init__(self, path_init, option):
        self.option = option
        self.path_init = path_init
        if option == 0:
            self.df_archive = read_event_arquive(path_init)
        elif option == 1:
            self.df_archive = read_event_arquive_with_event(path_init)
        else:
            self.df_archive = read_event_arquive_without_event(path_init)

    def get_event_info(self, event_id):
        event_info = self.df_archive.iloc[event_id-1,:]
        return event_info
    
    def get_event(self, event_id):
        path = self.path_init
        if self.option == 0:
            path += '\\Data\\infraquinta\\events\\Event_Q\\event_' + str(event_id) + '.csv'
        elif self.option == 1:
            path += '\\Data\\infraquinta\\events\\Organized_Data\\with_event\\event_' + str(event_id) + '.csv'
        else:
            path += '\\Data\\infraquinta\\events\\Organized_Data\\without_event\\event_' + str(event_id) + '.csv'
        df = pd.read_csv(path, index_col=0) 
        return df
        
def read_event_arquive(path_init):   
    path_archive = path_init + '\\Data\\infraquinta\\events\\Notes\\event_archive.csv'
    df_archive = pd.read_csv(path_archive, index_col=0, delimiter=';', decimal=',')
    return df_archive

def read_event_arquive_with_event(path_init):   
    path_archive = path_init + '\\Data\\infraquinta\\events\\Notes\\with_event_archive.csv'
    df_archive = pd.read_csv(path_archive, index_col=0)
    return df_archive

def read_event_arquive_without_event(path_init):   
    path_archive = path_init + '\\Data\\infraquinta\\events\\Notes\\without_event_archive.csv'
    df_archive = pd.read_csv(path_archive, index_col=0)
    return df_archive



















