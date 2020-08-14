# -*- coding: utf-8 -*-
"""
Created on Fri Aug 5 20:54:48 2020

@author: susan
"""

import sys
sys.path.append('../Functions')
from configuration import *
from event_archive import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from itertools import combinations, product
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

def calculate_correlation_difference(df1, df2, sensors):
    x11 = df1.loc[:,sensors[0]].to_numpy()
    x12 = df1.loc[:,sensors[1]].to_numpy()    
    x21 = df2.loc[:,sensors[0]].to_numpy() 
    x22 = df2.loc[:,sensors[1]].to_numpy()    
    corr1 = stats.pearsonr(x11, x12)[0]
    corr2 = stats.pearsonr(x21, x22)[0]    
    return abs(corr1-corr2)

def update_df_diff(df1, df2, df_diff, combos):
    for combo in combos:
        sensor1 = combo[0]
        sensor2 = combo[1]
        diff = calculate_correlation_difference(df1, df2, [sensor1, sensor2])
        df_diff[get_combo_name(combo)].append(diff)
    return df_diff

def plot_histogram(x, combo):
    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    n, bins, patches = ax.hist(x, bins=bin_edges, color='darkturquoise', edgecolor='k')
    
    mean = np.mean(x)
        
    ax.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = ax.get_ylim()
    ax.text(mean+0.05, max_ylim*0.9, 'Mean: {:.3f}'.format(mean), bbox=dict(facecolor="w",alpha=0.5,boxstyle="round"))
    ax.set(xticks=bin_edges)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    title = "[ " + str(combo[0]) + " & " + str(combo[1]) + " ] Correlation Difference Histogram"
    ax.set(xlabel='Correlation difference [0-2]', ylabel='Number of observations', title=title)
    
    plt.show()

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
        title = 'Confusion matrix, without normalization'

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

def get_combo_name(combo):
    return str(combo[0]) + "-" + str(combo[1])
      
def process_set_data(df, combos, ea1, ea2):
    
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
        df_diff1 = update_df_diff(df1, df2, df_diff1, combos)
        print(2*'\x1b[2K\r' + "Progress " + str(event_id), flush=True, end="\r")
        
    print("")
    
    for event_id in X2:
        df = ea2.get_event(event_id)                      
        df1, df2 = split_df(df)
        df_diff2 = update_df_diff(df1, df2, df_diff2, combos)
        print(2*'\x1b[2K\r' + "Progress " + str(event_id), flush=True, end="\r")
    
            
    df_diff1 = pd.DataFrame(df_diff1)
    df_diff1['y'] = y1
    
    df_diff2 = pd.DataFrame(df_diff2)
    df_diff2['y'] = y2
    
    df = df_diff1.append(df_diff2, ignore_index=True)
    
    return df

def save_set_data(path_init, df):
    path_export = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\dataset.csv'
    df.to_csv(index=True, path_or_buf=path_export)

def get_set_data(path_init):
    path = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\dataset.csv'
    df = pd.read_csv(path, index_col=0)
    return df
    

config = Configuration()
path_init = config.path

ea1 = EventArchive(path_init, 1)
ea2 = EventArchive(path_init, 2)

"""
df_archive = get_complete_archive(ea1, ea2) 
sensors = ['1', '2', '6', '9', '10']
combos = list(combinations(sensors, 2))
df = process_set_data(df_archive, combos, ea1, ea2)
save_set_data(path_init, df)
"""

df = get_set_data(path_init)


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=True)

X = df.iloc[:,:-1]
y = df.loc[:,'y']

recall = []
precision = []
accuracy = []

n_fold = 1
for train_index, test_index in skf.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
    y_train = y_train.replace([2, 3, 4, 5, 6], 1)
    y_test = y_test.replace([2, 3, 4, 5, 6], 1)

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    
    cnf_matrix = confusion_matrix(y_test, y_pred, [0,1]) # conjunto de testes, as previsões e as labels
    
    TN = cnf_matrix[0][0]
    FP = cnf_matrix[0][1]
    FN = cnf_matrix[1][0]
    TP = cnf_matrix[1][1]
    
    recall.append(TP/(TP+FN))
    precision.append(TP/(TP+FP))
    accuracy.append((TP+TN)/(TP+FN+TN+FP))
    
    #plot_confusion_matrix(cnf_matrix, [0,1])
      
    n_fold += 1

print("%d Folds" % n_splits)    
print("Recall: %.2f" % np.mean(recall))
print("Precision: %.2f" % np.mean(precision))
print("Accuracy: %.2f" % np.mean(accuracy))
    
