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
    path = ''  
    if(data_type == 'r'):
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_3\\dataset_r_' + correlation_type.lower() +'_' + str(width) + '.csv'
    else:
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_3\\dataset_' + correlation_type.lower() +'_' + str(width) + '.csv'
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

def update_results(df_results, TN, FP, FN, TP, auc_score):
    results = get_results(TN, FP, FN, TP, auc_score)
    df_results = df_results.append(results, ignore_index=True)
    return df_results

def get_results(TN, FP, FN, TP, auc_score):
    results = {}
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    
    if(auc_score is not None):
        results['AUC'] = auc_score
    
    results['recall'] = recall
    results['TNR'] = TN/(TN+FP)
    results['precision'] = precision
    results['NPV'] = TN/(TN+FN)
    results['F1'] = 2*((precision*recall)/(precision+recall))
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

def select_sensors(sensors_init_aux):
    combos = list(combinations(sensors_init_aux, 2))
    combos_str =[]
    for combo in combos:
        if ((combo[0] in sensors_init_aux) and (combo[1] in sensors_init_aux)):
            combos_str.append(get_combo_name(combo))
    combos_str.append('y')
    return combos_str

def select_features(X_train, y_train, X_test, sensors_init_aux, selection_type, n_features):
    if selection_type == 'mine':
        combos_str, _ = get_combos_2(X_train, y_train, sensors_init_aux, n_features)
    elif selection_type == 'chi2':
        selector = SelectKBest(chi2, k=n_features) .fit(abs(X_train), y_train)
        combos_str = selector.get_support()
    elif selection_type == 'f_classif':
        selector = SelectKBest(f_classif, k=n_features).fit(X_train, y_train) 
        combos_str = selector.get_support()        
    elif selection_type == 'mutual_info_classif':
        selector = SelectKBest(mutual_info_classif, k=n_features).fit(X_train, y_train) 
        combos_str = selector.get_support()
    else:
        return X_train, X_test
    #print(len(combos_str))
    #combos_str = select_sensors(sensors_init_aux)
    #print(len(combos_str))
    
    #print(combos_str)
    X_train = X_train.loc[:,combos_str]
    X_test = X_test.loc[:,combos_str]
    #print(list(X_train.columns))
    return X_train, X_test

def select_features_2(X_train, y_train, X_test, sensors_init_aux, selection_type, n_features, n_top):
    if selection_type == 'mine':
        combos_str, scores = get_combos(X_train, y_train, sensors_init_aux, n_features, n_top)
    else:
        if selection_type == 'chi2':
            selector = SelectKBest(chi2, k=n_features).fit(abs(X_train), y_train)
        elif selection_type == 'f_classif':
            selector = SelectKBest(f_classif, k=n_features).fit(X_train, y_train)
        elif selection_type == 'mutual_info_classif':
            selector = SelectKBest(mutual_info_classif, k=n_features).fit(X_train, y_train) 
        combos_str = selector.get_support()
        scores = selector.scores_[combos_str]
        combos_str = list(X_train.loc[:,combos_str])
    d = dict(combos_str = combos_str, scores = scores)
    df_scores = pd.DataFrame(data=d).sort_values('scores',ascending = False)
    #print(df_scores)
    return list(df_scores['combos_str']), list(df_scores['scores'])

def get_combos(X_train, y_train, sensors_init_aux, n_features, n_top):
    
    df_train = X_train.copy()
    df_train['y'] = y_train
    df_aux = df_train.copy()
    sensors_init = []
    sensors = []
    for sensor in sensors_init_aux:
        sensors_init.append(str(sensor))
        sensors.append(str(sensor))
    
    dic_results = {}
    diff = []
    sensors_aux = []
    
    for sensor in sensors:
        df_aux = df_train.copy()
        df_aux['y'] = df_aux['y'].replace([2, 3, 4, 5, 6], 1)
        combos_str = []
        combos = list(combinations(sensors_init, 2))
        for combo in combos:
            if ((combo[0] == sensor) or (combo[1] == sensor)):
                combos_str.append(get_combo_name(combo))
        combos_str.append('y')
        df_aux = df_aux.loc[:,combos_str]
        df_aux = df_aux.groupby(['y']).mean().T
        #print(df_aux)
        df_aux['diff'] = abs(df_aux[0])
        df_aux = df_aux.nlargest(n_top, 'diff')
        diff.extend(df_aux['diff'].to_numpy().tolist())
        sensors_aux.extend(df_aux.index.to_numpy().tolist())
        #print(df_aux)
        
    dic_results['combo'] = sensors_aux
    dic_results['diff'] = diff   
    df_aux = pd.DataFrame.from_dict(dic_results)
    df_aux = df_aux.drop_duplicates(subset=['combo'])
    #print(len(df_aux))
    #print(df_aux.sort_values(by='diff', ascending=False))
    df_aux = df_aux.nlargest(n_features, 'diff')
    #print(df_aux)
    combos_str = list(df_aux['combo'])
    #combos_str=['4-17', '17-19','4-9','5-9', '5-17','4-20','2-20','2-18','1-4', '1-12']
    #combos_str.append('y')
    #print(len(combos_str))
    return combos_str, list(df_aux['diff'])


def get_combos_2(X_train, y_train, sensors_init_aux, n_features):
    df_train = X_train.copy()
    df_train['y'] = y_train
    df_aux = df_train.copy()
    sensors_init = []
    sensors = []
    for sensor in sensors_init_aux:
        sensors_init.append(str(sensor))
        sensors.append(str(sensor))
    dfs = []
    for sensor in sensors:
        df_aux = df_train.copy()
        df_aux['y'] = df_aux['y'].replace([2, 3, 4, 5, 6], 1)
        combos_str = []
        combos = list(combinations(sensors_init, 2))
        for combo in combos:
            if ((combo[0] == sensor) or (combo[1] == sensor)):
                combos_str.append(get_combo_name(combo))
        combos_str.append('y')
        df_aux = df_aux.loc[:,combos_str]
        df_aux = df_aux.groupby(['y']).mean().T
        del df_aux[1]
        df_aux[0] = abs(df_aux[0])
        df_aux = df_aux.sort_values(by=0, ascending=False)
        #print(df_aux)
        dfs.append(df_aux)
    i = 0
    features = []
    while len(features) < n_features:
        df_aux = pd.DataFrame()
        for df in dfs:
            df_aux = df_aux.append({'combo':df.index[i], 'value':df.iloc[i,0]}, ignore_index=True)
        df_aux = df_aux.drop_duplicates(subset=['combo']).reset_index(drop=True)
        df_aux_combo = list(df_aux['combo'])
        j = 0
        for combo in df_aux_combo:
            if(combo in features):
                df_aux = df_aux.drop([j])
            j+=1
        #print(df_aux.sort_values(by='value', ascending=False))
        n = n_features - len(features)
        features.extend(df_aux.nlargest(n, 'value')['combo'].to_numpy())
        i += 1
    #print(features)
    #print(len(features))            
    return features, None

def get_df_train_test(df):
    df = df.sample(frac = 1, random_state=1).reset_index(drop=True)
    X = df.iloc[:,:-2]
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
    
    combos_str, _ = get_combos(df_train)
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
        
        combos_str, _ = get_combos(df_train_fold)
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

def get_sensors(data_type, sensor_type):
    if data_type == 'r':
        flow = [1,2,6,9,10,12,14]
        pressure = [3,7,8,11,13,15]
        if sensor_type == 'p':
            return pressure
        elif sensor_type == 'f':
            return flow
        else:
            return flow.extend(pressure)
    else:
        if sensor_type == 'p':
            return list(range(1,22,1))
        elif sensor_type == 'f':
            return list(range(22,27,1))
        else:
            return list(range(1,27,1))

def train_predict(classifier_type, X_train, y_train, X_test, option):
    
    clf = None
    y_pred = None
    y_scores = None
    
    if classifier_type == "GaussianNB":
        clf = GaussianNB()
    elif classifier_type == "LinearSVC":
        clf = LinearSVC(random_state=1, max_iter=3000) #max_iter=10000
    elif classifier_type == "NuSVC-rbf":
        clf = NuSVC(random_state=1, kernel='rbf') # class_weight='balanced', nu=0.0000001, 
    elif classifier_type == "NuSVC-poly":
        clf = NuSVC(random_state=1, kernel='poly',degree=3)
        
    clf_fit = clf.fit(X_train, y_train)
    
    if option == 'prediction':
        return clf_fit.predict(X_test)
    elif option == 'scores':
        if classifier_type == "GaussianNB":
            return clf_fit.predict_proba(X_test)[:,1]
        else:
            return clf_fit.decision_function(X_test)
    else:
        y_pred = clf_fit.predict(X_test)
        if classifier_type == "GaussianNB":
            y_scores = clf_fit.predict_proba(X_test)[:,1]
        else:
            y_scores = clf_fit.decision_function(X_test)
        return y_pred, y_scores
    
def cross_validation_2(classifier_type, df_train, sensors_init_aux, selection_type, n_features):
    
    df_results = pd.DataFrame()
    
    X = df_train.iloc[:,:-1]
    y = df_train.loc[:,'y']
    
    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    for train_index, test_index in skf.split(X, y):
    
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        y_train = y_train.replace([2, 3, 4, 5, 6], 1)
        y_test = y_test.replace([2, 3, 4, 5, 6], 1)
        
        df_train_fold = X_train.copy()
        df_train_fold['y'] = y_train

        X_train, X_test = select_features(X_train, y_train, X_test, sensors_init_aux, selection_type, n_features)        
        
        y_pred, y_scores = train_predict(classifier_type, X_train, y_train, X_test, 'both')
        
        fpr, tnr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
        
        auc_score = auc(fpr, tnr)
        
        cnf_matrix = confusion_matrix(y_test, y_pred, [0,1])
        TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix)
        df_results = update_results(df_results, TN, FP, FN, TP, auc_score)
        
        plot_confusion_matrix(cnf_matrix, [0,1])
        plot_roc_curve(fpr, tnr)
        
    results = round(df_results.describe().loc['mean',:],2).to_dict()
    print(round(df_results,2))  
    #print(results)
    return results

def execute_plot_feature_selection_distribution(path_init):
    
    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    width = 40
    
    # Feature Selection Parameters
    n_features = 37
    n_top = 2
    
    df = get_dataset(path_init, correlation_type, data_type, width)
    df_train, df_test = get_df_train_test(df)
    sensors_init_aux = get_sensors(data_type, sensor_type)
    combos_str = select_sensors(sensors_init_aux)
    df_train = df_train.loc[:,combos_str]
    df_test = df_test.loc[:,combos_str]

    X_train = df_train.iloc[:,:-1]
    y_train = df_train.loc[:,'y']
    X_test = df_test.iloc[:,:-1]
    
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(11.5,7), sharey=True)
    selection_types = ['mine','chi2', 'f_classif'] # 
    
    i = 0
    
    for ax in axs.flat:
        
        selection_type = selection_types[i]
        combos_str, _ = select_features_2(X_train, y_train, X_test, sensors_init_aux, selection_type, n_features, n_top)
        
        features_results = {}
        for sensor in sensors_init_aux:
            features_results[str(sensor)] = 0
            
        for combo_str in combos_str:
            combo = combo_str.split('-')
            features_results[combo[0]] += 1
            features_results[combo[1]] += 1
            
        if selection_type == 'mine':
            title = 'Correlation Value'
        elif selection_type == 'f_classif':
            title = 'ANOVA F-value'
            ax.set_xlabel('Sensors')
        else:
            title = 'Chi-Squared Value'
            ax.set_ylabel('# Features')
            
        ax.set_title(title)
        ax.bar(features_results.keys(), features_results.values()) 
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, axis='y', alpha=0.3) 
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        i+=1
    
    fig.tight_layout()
    plt.show()
    plt.close() 

def execute_plot_feature_selection_top(path_init):
    
    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "p" # p, f, all
    width = 40
    
    # Feature Selection Parameters
    n_features = 37
    n_top = 2
    
    df = get_dataset(path_init, correlation_type, data_type, width)
    df_train, df_test = get_df_train_test(df)
    sensors_init_aux = get_sensors(data_type, sensor_type)
    combos_str = select_sensors(sensors_init_aux)
    df_train = df_train.loc[:,combos_str]
    df_test = df_test.loc[:,combos_str]

    X_train = df_train.iloc[:,:-1]
    y_train = df_train.loc[:,'y']
    X_test = df_test.iloc[:,:-1]
    
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(11.5,7))
    selection_types = ['mine', 'chi2', 'f_classif']
    
    combos_str_array = []
    i = 0
    for ax in axs.flat:
        
        selection_type = selection_types[i]
    
        combos_str, scores = select_features_2(X_train, y_train, X_test, sensors_init_aux, selection_type, n_features, n_top)
        combos_str_array.append(combos_str)
        
        features_results = {}
        for sensor in sensors_init_aux:
            features_results[str(sensor)] = 0
            
        for combo_str in combos_str:
            combo = combo_str.split('-')
            features_results[combo[0]] += 1
            features_results[combo[1]] += 1
            
        if selection_type == 'mine':
            title = 'Correlation Value'
        elif selection_type == 'f_classif':
            title = 'ANOVA F-value'
            ax.set_xlabel('Sensor Pairs')
        else:
            title = 'Chi-Squared Value'
            ax.set_ylabel('Score')
            
        ax.set_title(title)
        ax.bar(combos_str, scores) 
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, axis='y', alpha=0.3) 
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        i+=1
    
    fig.tight_layout()
    plt.show()
    plt.close()
    
    print("\nPairs in common: " + str(len(np.intersect1d(combos_str_array[0], combos_str_array[1]))))
    print("Pairs in common: " + str(len(np.intersect1d(combos_str_array[0], combos_str_array[2]))))
    print("Pairs in common: " + str(len(np.intersect1d(combos_str_array[1], combos_str_array[2]))))
 
def execute_main_results(path_init):

    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    width = 40
    
    # Feature Selection Parameters
    selection_type = "None" # mine, chi2, f_classif, mutual_info_classif, None
    n_features = 99
    
    # Prediciton Parameters
    classifier_type = "NuSVC-rbf" # GaussianNB LinearSVC NuSVC-poly NuSVC-rbf
    
    df = get_dataset(path_init, correlation_type, data_type, width)
    df_train, df_test = get_df_train_test(df)
    
    sensors_init_aux = get_sensors(data_type, sensor_type)
    combos_str = select_sensors(sensors_init_aux)
    
    df_train = df_train.loc[:,combos_str]
    df_test = df_test.loc[:,combos_str]
    
    #print(df_train)
    #print(df_test)
    
    # Cross validation
    results = cross_validation_2(classifier_type, df_train, sensors_init_aux, selection_type, n_features)
    print(results)
    
def execute_feature_selection_evolution(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'

    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    width = 40
    
    # Feature Selection Parameters
    selection_type = "chi2" # mine, chi2, f_classif, mutual_info_classif, None
    
    # Prediciton Parameters
    classifier_type = "NuSVC-rbf" # GaussianNB LinearSVC NuSVC-poly NuSVC-rbf
   
    results_fs = pd.DataFrame()
    
    
    for n_features in range(15, 100, 4):
              
        df = get_dataset(path_init, correlation_type, data_type, width)
        df_train, df_test = get_df_train_test(df)
        
        sensors_init_aux = get_sensors(data_type, sensor_type)
        combos_str = select_sensors(sensors_init_aux)
        
        df_train = df_train.loc[:,combos_str]
        df_test = df_test.loc[:,combos_str]
        
        # Cross validation
        print(n_features)
        results = cross_validation_2(classifier_type, df_train, sensors_init_aux, selection_type, n_features)
        results['x'] = n_features
        results_fs = results_fs.append(results, ignore_index=True)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11.5,7))
    ax.plot(results_fs['x'], results_fs['precision'], color=color1, label='Precision')
    ax.plot(results_fs['x'], results_fs['recall'], color=color2, label='Recall') # marker='o', markersize=4,
    ax.plot(results_fs['x'], results_fs['AUC'], color=color3, label='AUC')
    
    ax.grid(True, axis='y', alpha=0.3) 
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend() #loc='lower left'
    
    fig.tight_layout()
    plt.show()
    plt.close()
    
    path_export = path_init + '\\Results\\feature_extraction_evolution_' + classifier_type + '_' + selection_type + '.csv'
    results_fs.to_csv(index=True, path_or_buf=path_export, sep=';', decimal=',')

def execute_plot_feature_selection_evolution(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    
    selection_types = ["mine","chi2", "f_classif"] # 
    
    for classifier_type in ["GaussianNB","LinearSVC"]: # 
        
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10.5,5), sharey=True, sharex=True)
        i = 0
        for ax in axs.flat:
            selection_type = selection_types[i]
        
            path = path_init + '\\Results\\feature_extraction_evolution_' + classifier_type + '_' + selection_type + '.csv'
            df = pd.read_csv(path, index_col=0, sep=';', decimal=',')
            
            df = df[df['x']<100]
            #print(df)
            
            ax.plot(df['x'], df['precision'], color=color1, label='Precision')
            ax.plot(df['x'], df['recall'], color=color2, label='Recall') # marker='o', markersize=4,
            ax.plot(df['x'], df['AUC'], color=color3, label='AUC')
            
            ax.grid(True, axis='y', alpha=0.3) 
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            
            title = "Score Function"
            if selection_type == 'mine':
                title += '\nCorrelation Value'
                ax.legend(loc='lower right')
            elif selection_type == 'f_classif':
                title += '\nANOVA F-value'
            else:
                title += '\nChi-Squared Test'
                ax.set_xlabel('Number of Features')
            
            ax.set_title(title)
            i+=1
        
        plt.ylim(0.53, 1.02)
        fig.tight_layout()
        plt.show()
        plt.close()

    
config = Configuration()
path_init = config.path

#execute_plot_feature_selection_distribution(path_init)
#execute_plot_feature_selection_top(path_init)

execute_feature_selection_evolution(path_init)
#execute_plot_feature_selection_evolution(path_init) 
    
#execute_main_results(path_init)
    
    
    
    
    




