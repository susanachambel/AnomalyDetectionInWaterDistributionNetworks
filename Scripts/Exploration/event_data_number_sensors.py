# -*- coding: utf-8 -*-
"""
Created on Fri Aug 5 20:54:48 2020

@author: susan
"""

import sys
sys.path.append('../Functions')
from configuration import *
from event_archive_2 import *
from correlation import *
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, NuSVC
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from itertools import combinations, product
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
import pandas as pd
import numpy as np
import statistics as stats

def get_dataset(path_init, correlation_type, data_type, width):
    sensors_init = []
    combos_str = []
    path = ''  
    if(data_type == 'r'):
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\dataset_'+ data_type + '_' + correlation_type.lower() + '.csv'
    elif(data_type == 'p'):
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\dataset_'+ data_type + '_' + correlation_type.lower() +'_' + str(width) + '_20.csv'
    else:
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\dataset_'+ data_type + '_' + correlation_type.lower() +'_' + str(width) + '.csv'
    df = pd.read_csv(path, index_col=0)
    return df

def get_dataset_corr(path_init, correlation_type, data_type, width):
    sensors_init = []
    combos_str = []
    path = ''  
    if(data_type == 'r'):
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_2\\dataset_'+ data_type + '_' + correlation_type.lower() + '.csv'
    elif(data_type == 'p'):
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_2\\dataset_'+ data_type + '_' + correlation_type.lower() +'_' + str(width) + '.csv'
    else:
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\dataset_'+ data_type + '_' + correlation_type.lower() +'_' + str(width) + '.csv'
    df = pd.read_csv(path, index_col=0)
    return df    

def get_combo_name(combo):
    return str(combo[0]) + "-" + str(combo[1])

def get_instances_confusion_matrix(cnf_matrix):
    TN = cnf_matrix[0][0]
    FP = cnf_matrix[0][1]
    FN = cnf_matrix[1][0]
    TP = cnf_matrix[1][1]
    return TN, FP, FN, TP

def update_results(df_results, TN, FP, FN, TP):
    results = get_results(TN, FP, FN, TP)
    df_results = df_results.append(results, ignore_index=True)
    return df_results

def get_results(TN, FP, FN, TP):
    results = {}
    results['TPR'] = TP/(TP+FN)
    results['TNR'] = TN/(TN+FP)
    results['PPV'] = TP/(TP+FP)
    results['NPV'] = TN/(TN+FN)
    results['ACC'] = (TP+TN)/(TP+FN+TN+FP)
    return results

def optimize_y_pred(y_scores, x_curve, y_curve, thresholds):  
    optimal_idx = np.argmax(y_curve - x_curve)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_scores >= optimal_threshold).astype(bool)
    #print(optimal_threshold)
    return y_pred, optimal_threshold

def plot_confusion_matrix(cnf_matrix, classesNames, normalize=False, cmap=plt.cm.Blues):
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

def plot_roc_curve(fpr, tnr):
    lw = 2
    label = 'ROC curve (auc=%0.2f)' % auc(fpr, tnr)
    plt.figure(figsize=[7.4, 4.8])
    plt.plot(fpr, tnr, color='#FF7F50', lw=lw, label=label)
    plt.plot([0, 1], [0, 1], color='#8F7F7A', lw=lw, linestyle='--', label='Chance (auc=0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(path_init + '\\Images\\roc_curve_example.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def get_combos(df_train):

    df_aux = df_train.copy()
    sensors_init = []
    sensors = []
    sensors_init_aux = list(range(1, 21, 1))
    for sensor in sensors_init_aux:
        sensors_init.append(str(sensor))
        sensors.append(str(sensor))
    
    dic_results = {}
    diff = []
    sensors_aux = []
    
    for sensor in sensors:
        df_aux = df.copy()
        df_aux['y'] = df_aux['y'].replace([2, 3, 4, 5, 6], 1)
        combos_str = []
        combos = list(combinations(sensors_init, 2))
        for combo in combos:
            if ((combo[0] == sensor) or (combo[1] == sensor)):
                combos_str.append(get_combo_name(combo))
        combos_str.append('y')
        df_aux = df_aux.loc[:,combos_str]
        df_aux = df_aux.groupby(['y']).mean().T
        df_aux['diff'] = df_aux[1] #abs(df_aux[1]-df_aux[0])
        df_aux = df_aux.nlargest(2, 'diff')
        diff.extend(df_aux['diff'].to_numpy().tolist())
        sensors_aux.extend(df_aux.index.to_numpy().tolist())
        
    dic_results['combo'] = sensors_aux
    dic_results['diff'] = diff   
    df_aux = pd.DataFrame.from_dict(dic_results)
    df_aux = df_aux.drop_duplicates(subset=['combo'])
    #print(df_aux.sort_values(by='diff', ascending=False))
    df_aux = df_aux.nlargest(10, 'diff')
    combos_str = list(df_aux['combo'])
    #combos_str=['4-17', '17-19','4-9','5-9', '5-17','4-20','2-20','2-18','1-4', '1-12']
    #combos_str.append('y')
    return combos_str

def get_combos_2(df_train):
    df_aux = df_train.copy()
    sensors_init = []
    sensors = []
    sensors_init_aux = list(range(1, 21, 1))
    for sensor in sensors_init_aux:
        sensors_init.append(str(sensor))
        sensors.append(str(sensor))
        
    dic_results = {}
    diff = []
    sensors_aux = []
    
    for sensor in sensors:
        df_aux = df.copy()
        df_aux['y'] = df_aux['y'].replace([2, 3, 4, 5, 6], 1)
        combos_str = []
        combos = list(combinations(sensors_init, 2))
        for combo in combos:
            if ((combo[0] == sensor) or (combo[1] == sensor)):
                combos_str.append(get_combo_name(combo))
        combos_str.append('y')
        df_aux = df_aux.loc[:,combos_str]
        df_aux = df_aux.groupby(['y']).mean().T
        df_aux['diff'] = df_aux[0]
        df_aux = df_aux.nlargest(2, 'diff')
        #print(df_aux)
        diff.extend(df_aux['diff'].to_numpy().tolist())
        sensors_aux.extend(df_aux.index.to_numpy().tolist())
        
    dic_results['combo'] = sensors_aux
    dic_results['diff'] = diff   
    df_aux = pd.DataFrame.from_dict(dic_results)
    df_aux = df_aux.drop_duplicates(subset=['combo'])
    #print(df_aux.sort_values(by='diff', ascending=False))
    df_aux = df_aux.nlargest(15, 'diff')
    combos_str = list(df_aux['combo'])
    #combos_str=['4-17', '17-19','4-9','5-9', '5-17','4-20','2-20','2-18','1-4', '1-12']
    #combos_str.append('y')
    return combos_str

def get_df_train_test(df):
    X = df.iloc[:,:-1]
    y = df.loc[:,'y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)   
    df_train = X_train.copy()
    df_train['y'] = y_train
    df_test = X_test.copy()
    df_test['y'] = y_test
    return df_train, df_test

def final_test(df_train, df_test, optimal_threshold):

    X_train = df_train.iloc[:,:-1]
    y_train = df_train.loc[:,'y']
    X_test = df_test.iloc[:,:-1]
    y_test = df_test.loc[:,'y']
    
    combos_str = get_combos(df_train)
    X_train = X_train.loc[:,combos_str]
    X_test = X_test.loc[:,combos_str]
    
    y_train = y_train.replace([2, 3, 4, 5, 6], 1)
    y_test = y_test.replace([2, 3, 4, 5, 6], 1)
    
    #clf = NuSVC(random_state=1, kernel='rbf')
    clf = GaussianNB()
    
    y_pred = []
    
    if optimal_threshold is None:
        y_pred = clf.fit(X_train, y_train).predict(X_test)
    else:
        #y_scores = clf.fit(X_train, y_train).decision_function(X_test)
        y_scores = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
        y_pred = (y_scores >= optimal_threshold).astype(bool)
        
    cnf_matrix = confusion_matrix(y_test, y_pred, [0,1])
    TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix)
        
    results = get_results(TN, FP, FN, TP)
    
    plot_confusion_matrix(cnf_matrix, [0,1])
    
    print(results)

def cross_validation(df_train):

    df_results = pd.DataFrame()
    
    X = df_train.iloc[:,:-1]
    y = df_train.loc[:,'y']
    
    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    
    optimal_thresholds = []
    
    for train_index, test_index in skf.split(X, y):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        df_train_fold = X_train.copy()
        df_train_fold['y'] = y_train
        
        combos_str = get_combos(df_train_fold)
        X_train = X_train.loc[:,combos_str]
        X_test = X_test.loc[:,combos_str]
        
        y_train = y_train.replace([2, 3, 4, 5, 6], 1)
        y_test = y_test.replace([2, 3, 4, 5, 6], 1)
        
        #clf = NuSVC(random_state=1, kernel='rbf')
        clf = GaussianNB()
        
        
        clf_fit = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    
        #y_scores = clf_fit.decision_function(X_test)
        y_scores = clf_fit.predict_proba(X_test)[:,1]
        fpr, tnr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
        
        
        y_pred, optimal_threshold = optimize_y_pred(y_scores, fpr, tnr, thresholds)
        
        optimal_thresholds.append(optimal_threshold)
        
        cnf_matrix = confusion_matrix(y_test, y_pred, [0,1])
        TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix)
        
        df_results = update_results(df_results, TN, FP, FN, TP)
        
        #plot_confusion_matrix(cnf_matrix, [0,1])
        plot_roc_curve(fpr, tnr)

    results = round(df_results.describe().loc['mean',:],2).to_dict()
    print(round(df_results,2))  
    print(results)
    optimal_threshold = stats.mean(optimal_thresholds)
    return optimal_threshold



config = Configuration()
path_init = config.path

correlation_type = "DCCA"
data_type = "p"
width = 40

"""
df = get_dataset(path_init, correlation_type, data_type, width)
df = df.sample(frac = 1, random_state=1).reset_index(drop=True)

df_train, df_test = get_df_train_test(df)

optimal_threshold = cross_validation(df_train)

final_test(df_train, df_test, None)
final_test(df_train, df_test, optimal_threshold)
"""

df = get_dataset(path_init, correlation_type, data_type, width)

print(df)

df = df.sample(frac = 1, random_state=1).reset_index(drop=True)

df_train, df_test = get_df_train_test(df)

optimal_threshold = cross_validation(df_train)

final_test(df_train, df_test, None)
final_test(df_train, df_test, optimal_threshold)




