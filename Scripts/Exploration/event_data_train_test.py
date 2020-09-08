# -*- coding: utf-8 -*-
"""
Created on Fri Aug 5 20:54:48 2020

@author: susan
"""

import sys
sys.path.append('../Functions')
from configuration import *
from event_archive import *
from correlation import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, NuSVC
from itertools import combinations, product
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np


def get_complete_archive(ea1, ea2):

    df_archive1 = ea1.df_archive.loc[:,['time_middle', 'c']]
    df_archive2 = ea2.df_archive
    
    df_archive2['y'] = 0
    
    conditions = [
        (df_archive1['c'] == 0.05),
        (df_archive1['c'] == 0.1),
        (df_archive1['c'] == 0.5),
        (df_archive1['c'] == 1.0),
        (df_archive1['c'] == 1.5),
        (df_archive1['c'] == 2.0)]
    
    choices = [1, 2, 3, 4, 5, 6]
    
    df_archive1['y'] = np.select(conditions, choices, default=0)
    
    df_archive1 = df_archive1.loc[:,['time_middle', 'y']]
        
    df_archive1['event_id'] = df_archive1.index
    df_archive2['event_id'] = df_archive2.index
    
    df_archive = df_archive1.append(df_archive2, ignore_index=True)

    return df_archive

def split_df(df): 
    n = int(df.shape[0]/2)
    df1 = df.iloc[:n,:]
    df2 = df.iloc[n:,:]
    return df1, df2

def calculate_correlation_difference(df1, df2, sensors, correlation_type):   
    x11 = df1.loc[:,sensors[0]].to_numpy()
    x12 = df1.loc[:,sensors[1]].to_numpy()    
    x21 = df2.loc[:,sensors[0]].to_numpy() 
    x22 = df2.loc[:,sensors[1]].to_numpy()
    
    corr1 = 0
    corr2 = 0
    
    if correlation_type == "pearson":
        corr1 = stats.pearsonr(x11, x12)[0]
        corr2 = stats.pearsonr(x21, x22)[0]
    elif correlation_type == "dcca":
        corr1 = calculate_dcca(x11, x12, 2)
        corr2 = calculate_dcca(x21, x22, 2)
    
    return abs(corr1-corr2)

def update_df_diff(df1, df2, df_diff, combos, correlation_type):
    for combo in combos:
        sensor1 = combo[0]
        sensor2 = combo[1]
        diff = calculate_correlation_difference(df1, df2, [sensor1, sensor2], correlation_type)
        df_diff[get_combo_name(combo)].append(diff)
    return df_diff

def plot_histogram(x, combo_name):
    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    n, bins, patches = ax.hist(x, bins=bin_edges, color='darkturquoise', edgecolor='k')
    
    mean = np.mean(x)
        
    ax.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = ax.get_ylim()
    ax.text(mean+0.05, max_ylim*0.9, 'Mean: {:.3f}'.format(mean), bbox=dict(facecolor="w",alpha=0.5,boxstyle="round"))
    ax.set(xticks=bin_edges)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    title = "[ " + combo_name + " ] Correlation Difference Histogram"
    ax.set(xlabel='Correlation difference [0-2]', ylabel='Number of observations', title=title)
    
    plt.show()

def plot_histogram_df(df):
    df_len = len(list(df.columns))
    fig = plt.figure(figsize=(10,6*df_len))
    gs0 = fig.add_gridspec(ncols=1, nrows=df_len, hspace=.3)
    i = 0
    for column in df.columns:
        x = df.loc[:, column].to_numpy()
        ax = fig.add_subplot(gs0[i])
        bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        n, bins, patches = ax.hist(x, bins=bin_edges, color='darkturquoise', edgecolor='k')
        mean = np.mean(x)
        ax.axvline(mean, color='k', linestyle='dashed', linewidth=1)
        min_ylim, max_ylim = ax.get_ylim()
        ax.text(mean+0.05, max_ylim*0.9, 'Mean: {:.3f}'.format(mean), bbox=dict(facecolor="w",alpha=0.5,boxstyle="round"))
        ax.set(xticks=bin_edges)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        title = "[ " + column + " ] Correlation Difference Histogram"
        ax.set(xlabel='Correlation difference [0-2]', ylabel='Number of observations', title=title)
        i+=1
    plt.show()

def get_tn_fp_tp_fn(X_test, y_test, y_pred):
        
    df_aux = X_test.copy()
    df_aux['y'] = y_test
    df_aux['y_pred'] = y_pred
    
    df_tn = df_aux[(df_aux.y == 0) & (df_aux.y_pred == 0)]
    df_fp = df_aux[(df_aux.y == 0) & (df_aux.y_pred == 1)]
    df_tp = df_aux[(df_aux.y == 1) & (df_aux.y_pred == 1)]
    df_fn = df_aux[(df_aux.y == 1) & (df_aux.y_pred == 0)]
    
    del df_tn['y']
    del df_tn['y_pred']
    del df_fp['y']
    del df_fp['y_pred']
    del df_tp['y']
    del df_tp['y_pred']
    del df_fn['y']
    del df_fn['y_pred']
   
    return df_tn, df_fp, df_tp, df_fn 

def plot_confusion_matrix(cnf_matrix, classesNames, normalize=False,
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)

    if normalize:
        soma = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / soma
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        title = 'Confusion Matrix'

    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classesNames))
    plt.xticks(tick_marks, classesNames, rotation=45)
    plt.yticks(tick_marks, classesNames)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_sensors(X_test, y_test, y_pred):
    df_aux = X_test.copy()
    df_aux['y'] = y_test
    df_aux['y_pred'] = y_pred
    df_aux = df_aux[(df_aux.y == 1) & (df_aux.y_pred == 1)]
    true_pos = df_aux.index.to_numpy()
        
    ea1 = EventArchive(path_init, 0)
    for event_id in true_pos:    
        event = ea1.get_event(event_id)
        event_info = ea1.get_event_info(event_id)
        
        fig = plt.figure(figsize=(10,6*len(sensors)))
        gs0 = fig.add_gridspec(ncols=1, nrows=len(sensors))
        
        i = 0
        for sensor in sensors:
                
            ax = fig.add_subplot(gs0[i])
            ax.plot(event.index,event.loc[:,sensor], label="smt")
            title = "Event " + str(event_id) + " | Sensor " + sensor
            ax.set(xlabel='', ylabel='Water Flow [m3/h]', title=title)
            ax.axvline(x=event_info.time_init, color='red', linestyle='dashed', linewidth=1)
            ax.axvline(x=event_info.time_final, color='red', linestyle='dashed', linewidth=1)
            i += 1
        
        #plt.savefig(path_init + "\\Reports\\Results Simulated\\" + str(event_id) + '_event_flow.png', format='png', dpi=300, bbox_inches='tight')
        #plt.close(fig)
        plt.show()

def plot_roc_curve(fpr, tpr, points):
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', 
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    for point in points:
        plt.plot(point['fpr'],point['tpr'],'ro',label=point['label'], color=point['color'])
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
      
def get_combo_name(combo):
    return str(combo[0]) + "-" + str(combo[1])

def get_instances_confusion_matrix(cnf_matrix):
    TN = cnf_matrix[0][0]
    FP = cnf_matrix[0][1]
    FN = cnf_matrix[1][0]
    TP = cnf_matrix[1][1]
    return TN, FP, FN, TP
      
def process_set_data(df, combos, ea1, ea2, correlation_type):
    
    df_diff1 = {}
    df_diff2 = {} 
    for combo in combos:
        df_diff1[get_combo_name(combo)] = []
        df_diff2[get_combo_name(combo)] = []
        
    mask = df['y'] > 0
    
    df1 = df[mask]
    X1 = df1.loc[:,'event_id'].to_numpy()
    y1 = df1.loc[:,'y'].to_numpy()
    
    df2 = df[~mask]
    X2 = df2.loc[:,'event_id'].to_numpy()
    y2 = df2.loc[:,'y'].to_numpy()
    
    for event_id in X1:
        df = ea1.get_event(event_id)                      
        df1, df2 = split_df(df)
        df_diff1 = update_df_diff(df1, df2, df_diff1, combos, correlation_type)
        print(2*'\x1b[2K\r' + "Progress " + str(event_id), flush=True, end="\r")
        
    print("")
    
    for event_id in X2:
        df = ea2.get_event(event_id)                      
        df1, df2 = split_df(df)
        df_diff2 = update_df_diff(df1, df2, df_diff2, combos, correlation_type)
        print(2*'\x1b[2K\r' + "Progress " + str(event_id), flush=True, end="\r")
    
            
    df_diff1 = pd.DataFrame(df_diff1)
    df_diff1['y'] = y1
    
    df_diff2 = pd.DataFrame(df_diff2)
    df_diff2['y'] = y2
    
    df = df_diff1.append(df_diff2, ignore_index=True)
    
    return df

def save_set_data(path_init, df, correlation_type):
    path_export = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\dataset_' + correlation_type + '.csv'
    df.to_csv(index=True, path_or_buf=path_export)

def get_set_data(path_init, sensors, correlation_type):
    sensors_init = ['1', '2', '6', '9', '10']
    combos = list(combinations(sensors_init, 2))
    combos_str = []
    for combo in combos:
        if ((combo[0] in sensors) and (combo[1] in sensors)):
            combos_str.append(get_combo_name(combo))
    combos_str.append('y')
    path = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\dataset_' + correlation_type + '.csv'
    df = pd.read_csv(path, index_col=0)
    df = df.loc[:, combos_str]
    return df
    
def execute_create_dataset(path_init, correlation_type):
    ea1 = EventArchive(path_init, 1)
    ea2 = EventArchive(path_init, 2)
    df_archive = get_complete_archive(ea1, ea2) 
    sensors = ['1', '2', '6', '9', '10']
    combos = list(combinations(sensors, 2))
    df = process_set_data(df_archive, combos, ea1, ea2, correlation_type)
    save_set_data(path_init, df, correlation_type)
    
def execute_train_test(path_init, sensors, correlation_type, classifier_type):

    df = get_set_data(path_init, sensors, correlation_type)
    
    """
    df_pos = df[(df.y >= 1)]
    df_neg = df[(df.y == 0)]
    plot_histogram_df(df_pos.iloc[:,:-1])
    plot_histogram_df(df_neg.iloc[:,:-1])
    """
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=True)
    
    X = df.iloc[:,:-1]
    y = df.loc[:,'y']
    
    df_results = pd.DataFrame()
    
    n_fold = 1
    for train_index, test_index in skf.split(X, y):
    
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
        y_train = y_train.replace([2, 3, 4, 5, 6], 1)
        y_test = y_test.replace([2, 3, 4, 5, 6], 1)
        
        y_pred = []
        y_scores = []
        
        if classifier_type == "GaussianNB":
            gnb = GaussianNB()
            gnb_fit = gnb.fit(X_train, y_train)
            y_pred = gnb_fit.predict(X_test)
            y_scores = gnb_fit.predict_proba(X_test)[:,1]
        elif classifier_type == "LinearSVC":
            li_svc = LinearSVC(random_state=0)
            li_svc_fit = li_svc.fit(X_train, y_train)
            y_pred = li_svc_fit.predict(X_test)
            y_scores = li_svc_fit.decision_function(X_test)
        elif classifier_type == "NuSVC":
            nu_svc = NuSVC()
            nu_svc_fit = nu_svc.fit(X_train, y_train)
            y_pred = nu_svc_fit.predict(X_test)
            y_scores = nu_svc_fit.decision_function(X_test)
        
        
        # Before Optimal Threshold
        cnf_matrix = confusion_matrix(y_test, y_pred, [0,1])
        TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix)
        point1 = {}
        point1['fpr'] = 1-(TN/(TN+FP))
        point1['tpr'] = TP/(TP+FN)
        point1['label'] = "Non Optimal"
        point1['color'] = "blue"
               
        fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        y_pred = (y_scores >= optimal_threshold).astype(bool)
        
        # After Optimal Threshold
        cnf_matrix = confusion_matrix(y_test, y_pred, [0,1]) # conjunto de testes, as previsões e as labels
        TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix)
        point2 = {}
        point2['fpr'] = 1-(TN/(TN+FP))
        point2['tpr'] = TP/(TP+FN)
        point2['label'] = "Optimal"
        point2['color'] = "red"
        
        #plot_roc_curve(fpr, tpr, [point1, point2])
        
        results = {}
        
        results['TPR'] = TP/(TP+FN)
        results['TNR'] = TN/(TN+FP)
        results['PPV'] = TP/(TP+FP)
        results['NPV'] = TN/(TN+FN)
        results['ACC'] = (TP+TN)/(TP+FN+TN+FP)
        
        df_results = df_results.append(results, ignore_index=True)
        
        #plot_confusion_matrix(cnf_matrix, [0,1])
        #plot_sensors(X_test, y_test, y_pred)
        
        #df_tn, df_fp, df_tp, df_fn = get_tn_fp_tp_fn(X_test, y_test, y_pred)
        #plot_histogram_df(df_tn)
        #plot_histogram_df(df_fp)
        #plot_histogram_df(df_tp)
        #plot_histogram_df(df_fn)
      
        n_fold += 1
        
    print(round(df_results,2))  
    results = round(df_results.describe().loc['mean',:],2).to_dict()
    print(results)
    return results



config = Configuration()
path_init = config.path

#execute_create_dataset(path_init, "dcca")

sensors_init = ['1', '2', '6', '9', '10']
correlation_type = "dcca"
classifier_type = "NuSVC" #"GaussianNB" "NuSVC" "LinearSVC"
combos = []
sensors = []

if correlation_type == "pearson":
    #sensors = ['1', '2', '9']
    sensors = ['1', '6', '9']

elif correlation_type == "dcca":
    sensors = ['1', '2', '6', '9', '10'] 


results = execute_train_test(path_init, sensors, correlation_type, classifier_type)

"""
df = pd.DataFrame()

for i in range(3, 6):
    combos.append(list(combinations(sensors_init, i))) 

for combo in combos:
    for sensor_list in combo:
        results = execute_train_test(path_init, list(sensor_list), correlation_type, classifier_type)
        results['combo'] = str(list(sensor_list))
        df = df.append(results, ignore_index=True)
        
print(df)
"""













    
