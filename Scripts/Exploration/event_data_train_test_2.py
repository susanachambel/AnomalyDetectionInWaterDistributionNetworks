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
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
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
    path = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\dataset_'+ data_type + '_' + correlation_type.lower() +'_' + str(width) + '.csv'
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

def update_results(df_results, TN, FP, FN, TP):
    results = {}
    results['TPR'] = TP/(TP+FN)
    results['TNR'] = TN/(TN+FP)
    results['PPV'] = TP/(TP+FP)
    results['NPV'] = TN/(TN+FN)
    results['ACC'] = (TP+TN)/(TP+FN+TN+FP)
    df_results = df_results.append(results, ignore_index=True)
    return df_results

def optimize_y_pred(y_scores, x_curve, y_curve, thresholds):  
    optimal_idx = np.argmax(y_curve - x_curve)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_scores >= optimal_threshold).astype(bool)
    return y_pred

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
    
def plot_scatter_plot_results(df, label):
    fig, ax = plt.subplots()
    x = 1-df.loc[:,'TNR'].to_numpy()
    y = df.loc[:,'TPR'].to_numpy()
    z = df.loc[:,'PPV'].to_numpy()
    c = df.loc[:,'c'].to_numpy()
    index = df.index.to_numpy()
    
    i1 = np.argmax(y-x)
    i2 = np.argmax(2*((y*z)/(y+z)))
    
    arrow_title = 'F1'
    if(i1 != i2):
        ax.annotate('TPR-FPR', xy=(x[i1], y[i1]),  xycoords='data',
            bbox=dict(boxstyle="round", fc="none", ec="gray"),
            xytext=(x[i1], y[i1]), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.2", shrinkB=5),
            horizontalalignment='right', verticalalignment='top')
    else:    
        arrow_title = 'F1|TPR-FPR'
    ax.annotate(arrow_title, xy=(x[i2], y[i2]),  xycoords='data',
        bbox=dict(boxstyle="round", fc="none", ec="gray"),
        xytext=(x[i2], y[i2]), textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.2", shrinkB=5),
        horizontalalignment='right', verticalalignment='top')
    
    cm = plt.cm.get_cmap('Blues')
    
    scatter = ax.scatter(x, y, c=c, cmap=cm, edgecolors='black')
    for i in index:
        ax.annotate(int(c[i]), (x[i], y[i]), textcoords="offset points", xytext=(5,0), ha='left')
    title = "[" + correlation_type + " w/ " + classifier_type + "]\nTPR/FPR Variation w/ " + label
    ax.set(xlabel='False Positive Rate [0-1]', ylabel='True Positive Rate [0-1]', title=title)
    plt.colorbar(scatter, ticks=ticker.MaxNLocator(integer=True))
    plt.show()

def plot_roc_curve(fpr, tnr, points):
    lw = 2
    label = 'ROC curve (area = %0.2f)' % auc(fpr, tnr)
    xlabel = "False Positive Rate [0-1]"
    ylabel = "True Positive Rate [0-1]"
    title = "ROC Curve"
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot(fpr, tnr, color='darkorange', lw=lw, label=label)
    for point in points:
        plt.plot(point['fpr'],point['tnr'],'ro',label=point['label'], color=point['color'])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def train_predict(classifier_type, X_train, y_train, X_test):
    
    clf = None
    y_pred = None
    y_scores = None
    
    if classifier_type == "GaussianNB":
        clf = GaussianNB()
    elif classifier_type == "LinearSVC":
        clf = LinearSVC(random_state=1)
    elif classifier_type == "NuSVC-rbf":
        clf = NuSVC(random_state=1, kernel='rbf')
    elif classifier_type == "NuSVC-poly":
        clf = NuSVC(random_state=1, kernel='poly')
        
    clf_fit = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    if classifier_type == "GaussianNB":
        y_scores = clf_fit.predict_proba(X_test)[:,1]
    else:
        y_scores = clf_fit.decision_function(X_test)
    
    return y_pred, y_scores
        
def execute_train_test(path_init, sensors, correlation_type, classifier_type, data_type, width, optimize):
    
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
        
        y_pred, y_scores = train_predict(classifier_type, X_train, y_train, X_test)
        
        cnf_matrix = confusion_matrix(y_test, y_pred, [0,1])
        TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix)

        fpr, tnr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
        roc_points = []
        point1 = {}
        point1['label'] = "Non optimal"
        point1['color'] = "purple"
        point1['fpr'] = 1-(TN/(TN+FP))
        point1['tnr'] = TP/(TP+FN)
        roc_points.append(point1)
        
        if optimize == 1:
            y_pred = optimize_y_pred(y_scores, fpr, tnr, thresholds)
            cnf_matrix = confusion_matrix(y_test, y_pred, [0,1])
            TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix)
            point2 = {}
            point2['label'] = "Optimal"
            point2['color'] = "red"
            point2['fpr'] = 1-(TN/(TN+FP))
            point2['tnr'] = TP/(TP+FN)
            roc_points.append(point2)
        
        df_results = update_results(df_results, TN, FP, FN, TP)
        
        #plot_roc_curve(fpr, tnr, roc_points)
        #plot_confusion_matrix(cnf_matrix, [0,1])
        n_fold += 1
        
    results = round(df_results.describe().loc['mean',:],2).to_dict()
    print(round(df_results,2))  
    print(results)
    return results

def test_different_number_sensors(path_init, correlation_type, classifier_type, data_type, width, optimize):
    sensors_init = ['1', '2', '6', '9', '10']
    combos = []
    df = pd.DataFrame()
    for i in range(3, 6):
        combos.append(list(combinations(sensors_init, i)))
    for combo in combos:
        for sensor_list in combo:
            results = execute_train_test(path_init, list(sensor_list), correlation_type, classifier_type, data_type, width, optimize)
            results['c'] = int(len(sensor_list))
            results['combo'] = str(list(sensor_list))
            df = df.append(results, ignore_index=True)
    print(df)
    plot_scatter_plot_results(df, "#Sensors")
    return df

def test_different_widths(path_init, sensors, correlation_type, classifier_type, data_type, optimize):
    df = pd.DataFrame()
    for width in range(15, 41, 5):
          results = execute_train_test(path_init, sensors, correlation_type, classifier_type, data_type, width, optimize)
          results['c'] = width
          df = df.append(results, ignore_index=True)
    print(df)
    plot_scatter_plot_results(df, "Width")
    return df


config = Configuration()
path_init = config.path


correlation_type = "DCCA" # "dcca" "Pearson"
classifier_type = "GaussianNB" #"GaussianNB" "NuSVC-poly" "NuSVC-rbf" "LinearSVC"
data_type = "q"
optimize = 0
width = 40

sensors = [] # Combination chosen

if correlation_type == "Pearson":
    sensors = ['1', '6', '9']

elif correlation_type == "DCCA":
    sensors = ['1', '2', '6', '9', '10'] 


results = execute_train_test(path_init, sensors, correlation_type, classifier_type, data_type, width, optimize)

# ROC chart para comparar os diferentes metodos de classificação e correlação. 
#Não esquecer de fazer apenas uma simples divisão teste / treino


#test_different_widths(path_init, sensors, correlation_type, classifier_type, data_type, optimize)
#test_different_number_sensors(path_init, correlation_type, classifier_type, data_type, width, optimize)

# Different classifiers and correlation methods
# Optimization

# Comparação com a pressão










    
