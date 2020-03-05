# - coding: utf-8 --
'''
@info webpage for predictive analysis of metro data
@author Inês Leite and Rui Henriques
@version 1.0
'''

import pandas as pd, dash
from app import app
import gui_utils as gui
import plot_utils
from dash.dependencies import Input, Output, State

''' ================================ '''
''' ====== A: WEB PAGE PARAMS ====== '''
''' ================================ '''

pagetitle = 'WISDOM'

target_options = [
        ('source',["infraquinta","beja","barreiro"],gui.Button.unidrop,["infraquinta"]), 
        ('focus',['real','simulated'],gui.Button.checkbox,['real','simulated']),
        ('sensor_group',['telegestao','telemetria'],gui.Button.checkbox,['telegestao']),
        ('sensor_type',["all"],gui.Button.multidrop,["all"]), 
        ('sensor_name',["all"],gui.Button.multidrop,["all"]), 
        ('period',['2017-06-05','2017-06-11'],gui.Button.daterange),
        ('calendar',list(gui.calendar.keys())+list(gui.week_days.keys()),gui.Button.multidrop,['all']),
        ('granularity_(minutes)','1',gui.Button.input)]
analysis_parameters = [
        ('mode',['default','parametric'],gui.Button.radio,'default'),
        ('pairwise comparisons',['cross_type','all_pairs'],gui.Button.checkbox,['all_pairs']),
        ('correlation_type',["all",'tuples','temporal'],gui.Button.multidrop,["all"]), 
        ('correlation',['pearson','kullback_leibler','dcca','dcca_ln'],gui.Button.multidrop,['pearson']),
        ('pca',['pca_statistics'],gui.Button.checkbox,['pca_statistics'])]

parameters = [('Target time series',27,target_options),('Data analysis',27,analysis_parameters)]
charts = [('visualizacao',None,gui.Button.figure),('correlograma',None,gui.Button.graph),('network',None,gui.Button.figure)]
layout = gui.get_layout(pagetitle,parameters,charts)

def get_states():
    return gui.get_states(target_options+analysis_parameters)

def agregar(mlist):
    agregado = set()
    for entries in mlist: 
        for entry in entries: 
            agregado.add(entry) 
    return list(agregado)

''' ============================== '''
''' ====== B: CORE BEHAVIOR ====== '''
''' ============================== '''

def get_data(states, series=False):
    
    '''A: process parameters'''
    idate, fdate = pd.to_datetime(states['date.start_date']), pd.to_datetime(states['date.end_date'])    
    minutes = int(states['granularidade_em_minutos.value'])
    dias = [gui.get_calendar_days(states['calendario.value'])]
    
    '''B: retrieve data'''
    #data, name, = query_metro.retrieve_data(idate,fdate,contagem,dias,estacoes_entrada,estacoes_saida,minutes,["record_count"])'''
    return None #series_utils.fill_time_series(data,name,idate,fdate,minutes)


@app.callback([Output('correlograma', 'figure'),Output('visualizacao','figure'),Output('network', 'figure')],
              [Input('button','n_clicks')],[State('period','start_date'),State('period','end_date')])
def update_charts(inp,*args):
    print('OLA')
    if inp is None: 
        nullplot = plot_utils.get_null_plot()
        return nullplot, nullplot
    states = dash.callback_context.states
    print(states)
    series = get_data(states,series=True)

    '''A: Run preprocessing here'''
    
    '''B: Plot time series and correlogram'''
    fig = plot_utils.get_series_plot(series,'titulo aqui')
    corr = plot_utils.get_correlogram(series)
    network = plot_utils.get_series_plot(series,'net schema')

    '''C: Plot statistics here'''

    return fig, corr


''' ===================== '''
''' ====== C: MAIN ====== '''
''' ====================== '''

if __name__ == '__main__':
    app.layout = layout
    app.run_server()
