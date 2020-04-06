# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:12:48 2020

@author: susan
"""

import pandas as pd
from configuration import *
from data_selection import *
import json


def get_data():
    config = Configuration()
    path_init = config.path
    df_1 = select_data(path_init, "infraquinta", "interpolated", 1, "2017-06-01 00:00:00", "2017-06-01 23:59:59")
    df_2 = select_data(path_init, "infraquinta", "interpolated", 6, "2017-06-01 00:00:00", "2017-06-01 23:59:59")
    df = pd.concat([df_1, df_2], axis=1)
    df.index = df_1.index.map(str)
    df.columns = ["1 (R, TLMT, F)","2 (S, TLMG, P)"] 
    
    return df

def get_json():
    config = Configuration()
    path_init = config.path
    with open(path_init + '//Data//sensors_list.json', 'r') as myfile:
        data=myfile.read()
    json_data = json.loads(data)
    return json_data