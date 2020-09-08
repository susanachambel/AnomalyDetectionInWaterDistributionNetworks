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


def get_dataset(path_init, sensors, correlation_type, data_type, width):
    sensors_init = ['1', '2', '6', '9', '10']
    combos = list(combinations(sensors_init, 2))
    combos_str = []
    for combo in combos:
        if ((combo[0] in sensors) and (combo[1] in sensors)):
            combos_str.append(get_combo_name(combo))
    combos_str.append('y')
    path = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\dataset_'+ data_type + '_' + correlation_type +'_' + str(width) + '.csv'
    df = pd.read_csv(path, index_col=0)
    df = df.loc[:, combos_str]
    return df

def get_combo_name(combo):
    return str(combo[0]) + "-" + str(combo[1])

def get_instances_confusion_matrix(cnf_matrix):
    TN = cnf_matrix[0][0]
    FP = cnf_matrix[0][1]
    FN = cnf_matrix[1][0]
    TP = cnf_matrix[1][1]
    return TN, FP, FN, TP

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
    
def plot_scatter_plot_results(df):
    fig, ax = plt.subplots()
    x = 1-df.loc[:,'TNR'].to_numpy()
    y = df.loc[:,'TPR'].to_numpy()
    c = df.loc[:,'c'].to_numpy()
    index = df.index.to_numpy()
    scatter = ax.scatter(x, y, c=c)
    for i in index:
        ax.annotate(i, (x[i], y[i]), textcoords="offset points", xytext=(5,0), ha='left')
    title = "[" + correlation_type + " w/ " + classifier_type + "] TPR/FPR Variation w/ #sensors"
    ax.set(xlabel='False Positive Rate [0-1]', ylabel='True Positive Rate [0-1]', title=title)
    legend1 = ax.legend(*scatter.legend_elements(),loc="lower right", title="#sens")
    ax.add_artist(legend1)
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

def execute_train_test(path_init, sensors, correlation_type, classifier_type, data_type, width):

    df = get_dataset(path_init, sensors, correlation_type, data_type, width)
    
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
            li_svc = LinearSVC(random_state=1)
            li_svc_fit = li_svc.fit(X_train, y_train)
            y_pred = li_svc_fit.predict(X_test)
            y_scores = li_svc_fit.decision_function(X_test)
        elif classifier_type == "NuSVC":
            nu_svc = NuSVC(random_state=1, kernel='rbf')
            nu_svc_fit = nu_svc.fit(X_train, y_train)
            y_pred = nu_svc_fit.predict(X_test)
            y_scores = nu_svc_fit.decision_function(X_test)
        
        
        fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        y_pred = (y_scores >= optimal_threshold).astype(bool)
        
        cnf_matrix = confusion_matrix(y_test, y_pred, [0,1])
        TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix) 
        
        point1 = {}
        point1['fpr'] = 1-(TN/(TN+FP))
        point1['tpr'] = TP/(TP+FN)
        point1['label'] = "Optimal"
        point1['color'] = "red"
        
        plot_roc_curve(fpr, tpr, [point1])
        
        results = {}
        
        results['TPR'] = TP/(TP+FN)
        results['TNR'] = TN/(TN+FP)
        results['PPV'] = TP/(TP+FP)
        results['NPV'] = TN/(TN+FN)
        results['ACC'] = (TP+TN)/(TP+FN+TN+FP)
        
        df_results = df_results.append(results, ignore_index=True)
        
        #plot_confusion_matrix(cnf_matrix, [0,1])
        #print(results)
        n_fold += 1
        
    results = round(df_results.describe().loc['mean',:],2).to_dict()
    print(round(df_results,2))  
    print(results)
    return results

def test_different_number_sensors(sensors, correlation_type, classifier_type, data_type, width):
    combos = []
    df = pd.DataFrame()
    for i in range(3, 6):
        combos.append(list(combinations(sensors, i)))
    for combo in combos:
        for sensor_list in combo:
            results = execute_train_test(path_init, list(sensor_list), correlation_type, classifier_type, data_type, width)
            results['c'] = int(len(sensor_list))
            results['combo'] = str(list(sensor_list))
            df = df.append(results, ignore_index=True)
    print(df)
    plot_scatter_plot_results(df)
    return df

def test_different_widths(path_init, sensors, correlation_type, classifier_type, data_type):
    df = pd.DataFrame()
    for width in range(15, 41, 5):
          results = execute_train_test(path_init, sensors, correlation_type, classifier_type, data_type, width)
          results['width'] = width
          df = df.append(results, ignore_index=True)
    print(df)
    return df

config = Configuration()
path_init = config.path

correlation_type = "dcca"
classifier_type = "GaussianNB" #"GaussianNB" "NuSVC" "LinearSVC"
data_type = "q"
width = 40

sensors = []

if correlation_type == "pearson":
    #sensors = ['1', '2', '9']
    sensors = ['1', '6', '9']

elif correlation_type == "dcca":
    sensors = ['1', '2', '6', '9', '10'] 


#results = execute_train_test(path_init, sensors, correlation_type, classifier_type, data_type, width)
test_different_number_sensors(sensors, correlation_type, classifier_type, data_type, width)
#test_different_widths(path_init, sensors, correlation_type, classifier_type, data_type)












    
