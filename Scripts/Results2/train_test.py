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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, NuSVC

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from itertools import combinations, product
import statistics as stats
import pandas as pd
import numpy as np
import decimal

from functools import partial

from scipy.stats import shapiro, normaltest, mannwhitneyu
from scipy.stats.mstats import kruskalwallis

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
    
    if(TP+FP == 0):
        precision = 0
    else:
        precision = TP/(TP+FP)
        
    if(TP+FN == 0):
        recall = 0
    else:
        recall = TP/(TP+FN)
        
    if(TN+FN == 0):
        results['NPV'] = 0
    else:
        results['NPV'] = TN/(TN+FN)
        
    if(TN+FP == 0):
        results['TNR'] = 0
    else:
        results['TNR'] = TN/(TN+FP)
    
    if(auc_score is not None):
        results['AUC'] = auc_score
        
    if((precision == 0) & (recall == 0)):
        results['F1'] = 0
    else:
        results['F1'] = 2*((precision*recall)/(precision+recall)) 
    
    results['recall'] = recall
    results['precision'] = precision
    
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
    plt.close()

def plot_roc_curve_mine(fpr, tnr):
    lw = 2
    label = 'ROC curve (AUC = %0.2f)' % auc(fpr, tnr)
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.plot(fpr, tnr, color='tab:orange', lw=lw, label=label)
    plt.plot([0, 1], [0, 1], color='tab:grey', lw=lw, linestyle='--', label='Chance (AUC = 0.50)')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, axis='y', alpha=0.3)
    #plt.title("ROC Curve")
    plt.legend(loc="lower right")
    #plt.savefig(path_init + '\\Images\\roc_curve_example.png', format='png', dpi=300, bbox_inches='tight')
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
        my_score = partial(mutual_info_classif, discrete_features=False, random_state=1)
        selector = SelectKBest(score_func=my_score, k=n_features).fit(X_train, y_train) 
        combos_str = selector.get_support()
    elif selection_type == 'kruskalwallis':
        combos_str = rank_kruskalwallis(X_train, y_train, n_features)
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
            my_score = partial(mutual_info_classif, discrete_features=False, random_state=1)
            selector = SelectKBest(score_func=my_score, k=n_features).fit(X_train, y_train)
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

def final_test(classifier_type, df_train, df_test, sensors_init_aux, selection_type, n_features, optimal_threshold):

    X_train = df_train.iloc[:,:-1]
    y_train = df_train.loc[:,'y']
    X_test = df_test.iloc[:,:-1]
    y_test = df_test.loc[:,'y']
    
    df_test_aux = X_test.copy()
    df_test_aux['y'] = y_test
        
    y_train = y_train.replace([2, 3, 4, 5, 6], 1)
    y_test = y_test.replace([2, 3, 4, 5, 6], 1)
    
    X_train, X_test = select_features(X_train, y_train, X_test, sensors_init_aux, selection_type, n_features)
    
    if (optimal_threshold == None):
        y_pred, y_scores = train_predict(classifier_type, X_train, y_train, X_test, "both")
    else:
        y_scores = train_predict(classifier_type, X_train, y_train, X_test, "scores")
        y_pred = (y_scores >= optimal_threshold).astype(bool)
        
    cnf_matrix = confusion_matrix(y_test, y_pred, [0,1])
    TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix)
    
    print(TN, FP, FN, TP)
        
    results = get_results(TN, FP, FN, TP, 0)
    
    #plot_confusion_matrix(cnf_matrix, [0,1])
    
    df_test_aux['score'] = y_scores
    df_test_aux['pred'] = y_pred
    df_test_aux['y1'] = y_test
    
    return results, y_scores, df_test_aux

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
        plot_roc_curve_mine(fpr, tnr)
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
            scores = clf_fit.predict_proba(X_test)
            c = scores[:,1]-scores[:,0]
            y_scores = scores[:,1]
            y_scores = c
            return y_scores
        else:
            return clf_fit.decision_function(X_test)
    else:
        y_pred = clf_fit.predict(X_test)
        if classifier_type == "GaussianNB":
            scores = clf_fit.predict_proba(X_test)
            c = scores[:,1]-scores[:,0]
            y_scores = scores[:,1]
            y_scores = c
        else:
            y_scores = clf_fit.decision_function(X_test)
        return y_pred, y_scores
    
def cross_validation_2(classifier_type, df_train, sensors_init_aux, selection_type, n_features):
    optimal_thresholds = []
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
        
        _ , optimal_threshold = optimize_y_pred(y_scores, fpr, tnr, thresholds)
        
        optimal_thresholds.append(optimal_threshold)
        
        cnf_matrix = confusion_matrix(y_test, y_pred, [0,1])
        TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix)
        df_results = update_results(df_results, TN, FP, FN, TP, auc_score)
        
        #plot_confusion_matrix(cnf_matrix, [0,1])
        #plot_roc_curve_mine(fpr, tnr)
    
    optimal_thresholds.append(round(stats.mean(optimal_thresholds),4))
    results = round(df_results.describe().loc['mean',:],2).to_dict()
    #print(round(df_results,2))  
    #print(results)
    return results, optimal_thresholds

def cross_validation_3(df_train, sensors_init_aux, params):
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5), sharey=True)
    
    for j, ax in enumerate(axs.flat):    
        classifier_type = params[j][0]
        selection_type = params[j][1]
        n_features = params[j][2]
        
        if classifier_type == "GaussianNB":
            clf = GaussianNB()
            response_method = "predict_proba"
            title = "Naive Bayes"
        elif classifier_type == "LinearSVC":
            clf = LinearSVC(random_state=1, max_iter=3000) #max_iter=10000
            response_method = "decision_function"
            title = "SVM w/ Linear Kernel"
            ax.set_ylabel('')
        elif classifier_type == "NuSVC-rbf":
            clf = NuSVC(random_state=1, kernel='rbf') # class_weight='balanced', nu=0.0000001,
            response_method = "decision_function"
            title = "SVM (RBF)"
        elif classifier_type == "NuSVC-poly":
            clf = NuSVC(random_state=1, kernel='poly',degree=3)
            response_method = "decision_function"
            title = "SVM (Poly)"
            
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        X = df_train.iloc[:,:-1]
        y = df_train.loc[:,'y']
        skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            y_train = y_train.replace([2, 3, 4, 5, 6], 1)
            y_test = y_test.replace([2, 3, 4, 5, 6], 1)
            
            df_train_fold = X_train.copy()
            df_train_fold['y'] = y_train
    
            X_train, X_test = select_features(X_train, y_train, X_test, sensors_init_aux, selection_type, n_features)        
    
            clf_fit = clf.fit(X_train, y_train)
            viz = plot_roc_curve(clf_fit, X_test, y_test,
                        name='ROC fold {}'.format(i),
                        alpha=0.3, lw=1, ax=ax, response_method=response_method)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='tab:grey',
            label='Chance', alpha=.8)
    
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='tab:orange',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
        
        if j==1:
            ax.set_ylabel('')
            ax.plot(0.01, 0.98, 'ro', color="red", label="Optimal threshold = -0.74")
        else:
            ax.plot(0.055, 0.86, 'ro', color="red", label="Optimal threshold = -0.99")
        
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title=title)
        ax.legend(loc="lower right")
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.grid(True, axis='y', alpha=0.3, which='both')
        
        
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results2\\ROC Curves\\roc_curves.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
def cross_validation_3_ea(df_train, sensors_init_aux, params):
    
    for j in range(0,len(params)):
                           
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4.5))
                           
        classifier_type = params[j][0]
        selection_type = params[j][1]
        n_features = params[j][2]
        
        if classifier_type == "GaussianNB":
            clf = GaussianNB()
            response_method = "predict_proba"
            title = "Naive Bayes"
        elif classifier_type == "LinearSVC":
            clf = LinearSVC(random_state=1, max_iter=3000) #max_iter=10000
            response_method = "decision_function"
            title = "ROC Plot"
            ax.set_ylabel('')
        elif classifier_type == "NuSVC-rbf":
            clf = NuSVC(random_state=1, kernel='rbf') # class_weight='balanced', nu=0.0000001,
            response_method = "decision_function"
            title = "SVM (RBF)"
        elif classifier_type == "NuSVC-poly":
            clf = NuSVC(random_state=1, kernel='poly',degree=3)
            response_method = "decision_function"
            title = "SVM (Poly)"
            
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        X = df_train.iloc[:,:-1]
        y = df_train.loc[:,'y']
        skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            y_train = y_train.replace([2, 3, 4, 5, 6], 1)
            y_test = y_test.replace([2, 3, 4, 5, 6], 1)
            
            df_train_fold = X_train.copy()
            df_train_fold['y'] = y_train
    
            X_train, X_test = select_features(X_train, y_train, X_test, sensors_init_aux, selection_type, n_features)        
    
            clf_fit = clf.fit(X_train, y_train)
            viz = plot_roc_curve(clf_fit, X_test, y_test,
                        name='ROC fold {}'.format(i),
                        alpha=0.3, lw=1, ax=ax, response_method=response_method)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='tab:grey',
            label='Chance', alpha=.8)
    
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='tab:orange',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
        
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.set_title(title, fontsize=14)
        
        ax.plot(0.01, 0.98, 'ro', color="red", label="Optimal threshold = -0.74")
        
        ax.set_ylabel("True Positive Rate", fontsize=14)
        ax.set_xlabel("False Positive Rate", fontsize=14)
        ax.legend(loc="lower right")
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.grid(True, axis='y', alpha=0.3, which='both')
        
        
        
        fig.tight_layout()
        plt.savefig(path_init + '\\Images\\Results2\\ROC Curves\\roc_curve_'+ classifier_type +'_ea.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def cross_validation_4(classifier_type, df_train, sensors_init_aux, selection_type, n_features, optimal_threshold):
    
    df_results = pd.DataFrame()
    df_general_results = pd.DataFrame()
    
    X = df_train.iloc[:,:-1]
    y = df_train.loc[:,'y']
    
    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    for train_index, test_index in skf.split(X, y):
    
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        df_fold_results = pd.DataFrame()
        df_fold_results['y'] = y_test.copy()
        
        y_train = y_train.replace([2, 3, 4, 5, 6], 1)
        y_test = y_test.replace([2, 3, 4, 5, 6], 1)
        
        df_train_fold = X_train.copy()
        df_train_fold['y'] = y_train

        X_train, X_test = select_features(X_train, y_train, X_test, sensors_init_aux, selection_type, n_features)        
        
        if(optimal_threshold == None):
            y_pred, y_scores = train_predict(classifier_type, X_train, y_train, X_test, 'both')
        else:
            y_scores = train_predict(classifier_type, X_train, y_train, X_test, 'scores')
            y_pred = (y_scores >= optimal_threshold).astype(bool)
               
        cnf_matrix = confusion_matrix(y_test, y_pred, [0,1])
        TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix)
        df_results = update_results(df_results, TN, FP, FN, TP, 0)
        
        df_fold_results['y1'] = y_test
        df_fold_results['score'] = y_scores
        df_fold_results['pred'] = y_pred
        
        df_general_results = pd.concat([df_general_results, df_fold_results], sort=False)
        
        #plot_confusion_matrix(cnf_matrix, [0,1])
    df_general_results = df_general_results.sort_index()
    results = round(df_results.describe().loc['mean',:],2).to_dict()
    #print(round(df_results,2))  
    #print(results)
    return results, df_fold_results

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
    
    # Prediciton Parameters
    classifier_type = "LinearSVC" # GaussianNB LinearSVC NuSVC-poly NuSVC-rbf
    
    # Feature Selection Parameters
    
    if(classifier_type == "GaussianNB"):
        selection_type = "mine"
        n_features = 43
    elif (classifier_type == "LinearSVC"):
        selection_type = "kruskalwallis" #"f_classif"
        n_features = 83 #75
        
    
    df = get_dataset(path_init, correlation_type, data_type, width)
    df_train, df_test = get_df_train_test(df)
    
    sensors_init_aux = get_sensors(data_type, sensor_type)
    combos_str = select_sensors(sensors_init_aux)
    
    df_train = df_train.loc[:,combos_str]
    df_test = df_test.loc[:,combos_str]
    
    # Cross validation
    results, optimal_thresholds = cross_validation_2(classifier_type, df_train, sensors_init_aux, selection_type, n_features)
    print(results)
        
    for optimal_threshold in optimal_thresholds:
        print(optimal_threshold)
        results, _ = cross_validation_4(classifier_type, df_train, sensors_init_aux, selection_type, n_features, optimal_threshold)
        print(results)
        
def execute_final_results(path_init):
    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    width = 40
    
    classifier_type = "LinearSVC"
    
    if (classifier_type == "GaussianNB"):
        selection_type = "mine"
        n_features = 43
        optimal_threshold = -0.67 #-0.67 -0.99   
    elif (classifier_type == "LinearSVC"):
        selection_type = "kruskalwallis"
        n_features = 83
        optimal_threshold = -0.66 #-0.53 -0.70
    
    
    df = get_dataset(path_init, correlation_type, data_type, width)
    
    #mask = (df['event']<697)|(df['event']>702)
    #df = df[mask]
    
    df_train, df_test = get_df_train_test(df)
    
    sensors_init_aux = get_sensors(data_type, sensor_type)
    combos_str = select_sensors(sensors_init_aux)
    
    df_train = df_train.loc[:,combos_str]
    df_test = df_test.loc[:,combos_str]
    
    results, _, _ = final_test(classifier_type, df_train, df_test, sensors_init_aux, selection_type, n_features, optimal_threshold)       
    print(results)
    
def execute_feature_selection_evolution(path_init):
    
    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    width = 40
    
    # Feature Selection Parameters
    selection_type = "kruskalwallis" # mine, chi2, f_classif, mutual_info_classif, None
    
    # Prediciton Parameters
    classifier_type = "NuSVC-rbf" # GaussianNB LinearSVC NuSVC-poly NuSVC-rbf
   
    results_fs = pd.DataFrame()
    
    for n_features in range(15, 100, 2):
              
        df = get_dataset(path_init, correlation_type, data_type, width)
        df_train, df_test = get_df_train_test(df)
        
        sensors_init_aux = get_sensors(data_type, sensor_type)
        combos_str = select_sensors(sensors_init_aux)
        
        df_train = df_train.loc[:,combos_str]
        df_test = df_test.loc[:,combos_str]
        
        # Cross validation
        print(n_features)
        results, _ = cross_validation_2(classifier_type, df_train, sensors_init_aux, selection_type, n_features)
        results['x'] = n_features
        results_fs = results_fs.append(results, ignore_index=True)
    
    path_export = path_init + '\\Results\\feature_extraction_evolution_' + classifier_type + '_' + selection_type + '.csv'
    results_fs.to_csv(index=True, path_or_buf=path_export, sep=';', decimal=',')

def execute_plot_feature_selection_evolution(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    #color4 = 'tab:red'
    
    selection_types = ["mine", "kruskalwallis"] # mannwhitneyu f_classif
    
    for classifier_type in ["GaussianNB" ,"LinearSVC", "NuSVC-rbf"]: # "GaussianNB" ,"LinearSVC", "NuSVC-rbf"
        
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9,5), sharey=True, sharex=True)
        i = 0
        for ax in axs.flat:
            selection_type = selection_types[i]
        
            path = path_init + '\\Results\\feature_extraction_evolution_' + classifier_type + '_' + selection_type + '.csv'
            df = pd.read_csv(path, index_col=0, sep=';', decimal=',')
            df = df[df['x']<100]
            ax.plot(df['x'], df['precision'], color=color1, label=r'$Precision_{P}$')
            ax.plot(df['x'], df['recall'], color=color2, label=r'$Recall_{P}$')
            ax.plot(df['x'], df['F1'], color=color3, label = r'$F_{P}$')
            
            bottom_x = 0.37 #0.53
            ax.grid(True, axis='y', alpha=0.3, which='both') 
            if (classifier_type == 'GaussianNB'):
                if selection_type == 'mine':
                    title = 'Our Feature Ranking Method'
                    x = 43
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+1, 1.0-0.005,'1.00', ha="left", va="top", color=color1)
                    ax.text(x+1, 0.84+0.005,'0.84', ha="left", va="bottom", color=color3)
                    ax.text(x+1, 0.73,'0.73', ha="left", va="bottom", color=color2)
                    ax.text(x+1, bottom_x,str(x), ha="left", va="center", color='k', alpha=0.5)
                    
                    """
                    x = 36
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-2, 1.0-0.005,'1.00', ha="right", va="top", color=color1)
                    ax.text(x-1, 0.83+0.005,'0.83', ha="right", va="bottom", color=color3)
                    ax.text(x-1, 0.71,'0.71', ha="right", va="bottom", color=color2)
                    ax.text(x-1, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                    """
                    ax.legend(loc='lower right')
                else:
                    title = 'Kruskal–Wallis H Test'
                    x = 63
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-1, 1.0-0.005,'1.00', ha="right", va="top", color=color1)
                    ax.text(x-1, 0.81+0.005,'0.81', ha="right", va="bottom", color=color3)
                    ax.text(x-1, 0.68+0.005,'0.68', ha="right", va="bottom", color=color2)
                    ax.text(x-1, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                    
                    """
                    x = 36
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-1, 0.91+0.005,'0.91', ha="right", va="bottom", color=color1)
                    ax.text(x-1, 0.66+0.005,'0.66', ha="right", va="bottom", color=color3)
                    ax.text(x-1, 0.51+0.005,'0.51', ha="right", va="bottom", color=color2)
                    ax.text(x-1, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                    """
                    
            elif (classifier_type == 'LinearSVC'):
                if selection_type == 'mine':
                    title = 'Our Feature Ranking Method'
                    x = 95
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-1, 1.0+0.002,'1.00', ha="right", va="bottom", color=color1)
                    ax.text(x-1, 0.98-0.005,'0.98', ha="right", va="bottom", color=color3)
                    ax.text(x-1, 0.95-0.005,'0.95', ha="right", va="bottom", color=color2)
                    ax.text(x-1, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                    
                    """
                    x = 36
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+1, 0.99-0.005,'0.99', ha="left", va="top", color=color1)
                    ax.text(x+1, 0.91-0.005,'0.91', ha="left", va="top", color=color3)
                    ax.text(x+1, 0.84-0.005,'0.84', ha="left", va="top", color=color2)
                    ax.text(x+1, bottom_x,str(x), ha="left", va="center", color='k', alpha=0.5)
                    """
                    ax.legend(loc='lower left')
                    
                else:
                    title = 'Kruskal–Wallis H Test'
                    x = 83
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-1, 1.0+0.002,'1.00', ha="right", va="bottom", color=color1)
                    ax.text(x-1, 0.98-0.005,'0.98', ha="right", va="bottom", color=color3)
                    ax.text(x-1, 0.95-0.005,'0.95', ha="right", va="bottom", color=color2)
                    ax.text(x-1, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                    
                    """
                    x = 36
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+1, 0.95,'0.95', ha="left", va="center", color=color1)
                    ax.text(x+1, 0.88,'0.88', ha="left", va="center", color=color3)
                    ax.text(x+1, 0.82-0.005,'0.82', ha="left", va="top", color=color2)
                    ax.text(x+1, bottom_x,str(x), ha="left", va="center", color='k', alpha=0.5)
                    """
            else:
                if selection_type == 'mine':
                    title = 'Our Feature Ranking Method'
                    x = 87
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-1, 1.0-0.005,'1.00', ha="right", va="top", color=color1)
                    ax.text(x-1, 0.74+0.005,'0.74', ha="right", va="bottom", color=color2)
                    ax.text(x-1, 0.85+0.005,'0.85', ha="right", va="bottom", color=color3)
                    ax.text(x-1, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                    
                    """
                    x = 36
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+1, 1.0-0.005,'1.00', ha="left", va="top", color=color1)
                    ax.text(x+1, 0.84+0.005,'0.84', ha="left", va="bottom", color=color3)
                    ax.text(x+1, 0.72+0.005,'0.72', ha="left", va="bottom", color=color2)
                    ax.text(x+1, bottom_x,str(x), ha="left", va="center", color='k', alpha=0.5)
                    """
                    ax.legend(loc='lower left')
                else:
                    title = 'Kruskal–Wallis H Test'
                    x = 23
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+1, 1.0-0.005,'1.00', ha="left", va="top", color=color1)
                    ax.text(x+1, 0.94-0.005,'0.94', ha="left", va="bottom", color=color3)
                    ax.text(x+1, 0.88,'0.88', ha="left", va="bottom", color=color2)
                    ax.text(x+1, bottom_x,str(x), ha="left", va="center", color='k', alpha=0.5)
                    
                    """
                    x = 36
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+1, 1.0-0.005,'1.00', ha="left", va="top", color=color1)
                    ax.text(x+1, 0.91,'0.91', ha="left", va="bottom", color=color3)
                    ax.text(x+1, 0.84,'0.84', ha="left", va="bottom", color=color2)
                    ax.text(x+1, bottom_x,str(x), ha="left", va="center", color='k', alpha=0.5)
                    """
                    
            ax.set_xlabel('Top k Features')
            plt.ylim(0.35, 1.03)
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
            ax.set_title(title)
            i+=1
        
        fig.tight_layout()
        plt.savefig(path_init + '\\Images\\Results2\\Feature Selection\\' + classifier_type + '.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def execute_plot_feature_selection_evolution_ea(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    #color4 = 'tab:red'
    
    for classifier_type in ["GaussianNB" ,"LinearSVC"]: # "GaussianNB" ,"LinearSVC", "NuSVC-rbf"
        
        for selection_type in ["mine", "kruskalwallis"]:
        
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4.7))
        
            path = path_init + '\\Results\\feature_extraction_evolution_' + classifier_type + '_' + selection_type + '.csv'
            df = pd.read_csv(path, index_col=0, sep=';', decimal=',')
            df = df[df['x']<100]
            ax.plot(df['x'], df['precision'], color=color1, label=r'$Prec_{P}$')
            ax.plot(df['x'], df['recall'], color=color2, label=r'$Rec_{P}$')
            ax.plot(df['x'], df['F1'], color=color3, label = r'$F_{P}$')
            
            bottom_x = 0.575 #0.53
            ax.grid(True, axis='y', alpha=0.3, which='both') 
            if (classifier_type == 'GaussianNB'):
                if selection_type == 'mine':
                    title = 'Naive Bayes w/\nOur Feature Ranking Method'
                    x = 43
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+1, 1.0-0.005,'1.00', ha="left", va="top", color=color1)
                    ax.text(x+1, 0.84+0.005,'0.84', ha="left", va="bottom", color=color3)
                    ax.text(x+1, 0.73,'0.73', ha="left", va="bottom", color=color2)
                    ax.text(x+1, bottom_x,str(x), ha="left", va="center", color='k', alpha=0.5)
                    
                    ax.legend(loc='lower right', fontsize=13, ncol=2)
                else:
                    title = 'Naive Bayes w/\nKruskal–Wallis H Test'
                    x = 63
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-1, 1.0-0.005,'1.00', ha="right", va="top", color=color1)
                    ax.text(x-1, 0.81+0.005,'0.81', ha="right", va="bottom", color=color3)
                    ax.text(x-1, 0.68+0.005,'0.68', ha="right", va="bottom", color=color2)
                    ax.text(x-1, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                    
            elif (classifier_type == 'LinearSVC'):
                if selection_type == 'mine':
                    title = 'SVM (Linear) w/\nOur Feature Ranking Method'
                    x = 95
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-1, 1.0+0.002,'1.00', ha="right", va="bottom", color=color1)
                    ax.text(x-1, 0.98-0.005,'0.98', ha="right", va="bottom", color=color3)
                    ax.text(x-1, 0.95-0.005,'0.95', ha="right", va="bottom", color=color2)
                    ax.text(x-1, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                    
                else:
                    title = 'SVM (Linear) w/\nKruskal–Wallis H Test'
                    x = 83
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-1, 1.0+0.002,'1.00', ha="right", va="bottom", color=color1)
                    ax.text(x-1, 0.98-0.005,'0.98', ha="right", va="bottom", color=color3)
                    ax.text(x-1, 0.95-0.005,'0.95', ha="right", va="bottom", color=color2)
                    ax.text(x-1, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                    
                    ax.legend(loc='lower left', fontsize=13, ncol=2)
                    
                    
            ax.set_xlabel('Top k Features', fontsize="14")
            plt.ylim(0.56, 1.03)
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
            ax.set_title(title, fontsize="14")
            
            
        
            fig.tight_layout()
            plt.savefig(path_init + '\\Images\\Results2\\Feature Selection\\' + classifier_type + '_' + selection_type + '_ea.png', format='png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()



def execute_time_window_widths(path_init):
    
    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    
    # Prediciton Parameters
    classifier_type = "LinearSVC"
    
    # Feature Selection Parameters
    if (classifier_type == "GaussianNB"):
        selection_type = "mine"
        n_features = 43
    elif (classifier_type == "LinearSVC"):
        selection_type = "kruskalwallis" #"f_classif"
        n_features = 83 #75
    elif (classifier_type == "NuSVC-rbf"):
        selection_type = "mine"
        n_features = 87

    results_fs = pd.DataFrame()
    
    for width in range(16, 41, 2):
              
        df = get_dataset(path_init, correlation_type, data_type, width)
        df_train, df_test = get_df_train_test(df)
        
        sensors_init_aux = get_sensors(data_type, sensor_type)
        combos_str = select_sensors(sensors_init_aux)
        
        df_train = df_train.loc[:,combos_str]
        df_test = df_test.loc[:,combos_str]
        
        # Cross validation
        print(width)
        results, _ = cross_validation_2(classifier_type, df_train, sensors_init_aux, selection_type, n_features)
        results['x'] = width
        results_fs = results_fs.append(results, ignore_index=True)
    
   
    path_export = path_init + '\\Results\\time_window_widths_' + classifier_type + '.csv'
    results_fs.to_csv(index=True, path_or_buf=path_export, sep=';', decimal=',')

def execute_plot_time_window_widths(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    #color4 = 'tab:red'
    
    classifier_types = ["GaussianNB","LinearSVC"]
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9,5), sharey=True, sharex=True)
        
    i = 0
    for ax in axs.flat:
        
        classifier_type =  classifier_types[i]
        
        path = path_init + '\\Results\\time_window_widths_' + classifier_type + '.csv'
        df = pd.read_csv(path, index_col=0, sep=';', decimal=',')
        ax.plot(df['x'], df['precision'], color=color1, marker='o', markersize=3, label=r'$Precision_{P}$')
        ax.plot(df['x'], df['recall'], color=color2, marker='o', markersize=3, label=r'$Recall_{P}$')
        ax.plot(df['x'], df['F1'], color=color3, marker='o', markersize=3, label = r'$F_{P}$')
        ax.grid(True, axis='y', alpha=0.3, which='both') 
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
        
        
        if (classifier_type=="GaussianNB"):
            title = "Naive Bayes"
            x = 40
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x-0.2, 1.0+0.01,'1.00', ha="right", va="bottom", color=color1)
            ax.text(x-0.2, 0.84+0.01,'0.84', ha="right", va="bottom", color=color3)
            ax.text(x-0.2, 0.73+0.01,'0.73', ha="right", va="bottom", color=color2)
            ax.text(x-0.2, 0.485,str(x), ha="right", va="center", color='k', alpha=0.5)
        elif (classifier_type=="LinearSVC"):
            title = "SVM w/ Linear Kernel"
            x = 40
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x-0.2, 1.0+0.01,'1.00', ha="right", va="bottom", color=color1)
            ax.text(x-0.2, 0.97+0.003,'0.97', ha="right", va="bottom", color=color3)
            ax.text(x-0.2, 0.95-0.01,'0.95', ha="right", va="top", color=color2)
            ax.text(x-0.2, 0.485,str(x), ha="right", va="center", color='k', alpha=0.5)
            ax.legend(loc='lower left')
        else:
            title = "SVM w/ RBF Kernel"
            
        ax.set_xlabel('Time Window Size')    
        ax.set_title(title)
        i+=1
    plt.ylim(0.47, 1.04)
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results2\\Time Window Widths\\time_windows.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
def execute_plot_time_window_widths_ea(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    #color4 = 'tab:red'
    
    for classifier_type in ["GaussianNB","LinearSVC"]:
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4.5))
        
        path = path_init + '\\Results\\time_window_widths_' + classifier_type + '.csv'
        df = pd.read_csv(path, index_col=0, sep=';', decimal=',')
        ax.plot(df['x'], df['precision'], color=color1, marker='o', markersize=3, label=r'$Prec_{P}$')
        ax.plot(df['x'], df['recall'], color=color2, marker='o', markersize=3, label=r'$Rec_{P}$')
        ax.plot(df['x'], df['F1'], color=color3, marker='o', markersize=3, label = r'$F_{P}$')
        ax.grid(True, axis='y', alpha=0.3, which='both') 
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
        
        if (classifier_type=="GaussianNB"):
            title = "Naive Bayes"
            x = 40
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x-0.2, 1.0+0.01,'1.00', ha="right", va="bottom", color=color1)
            ax.text(x-0.2, 0.84+0.01,'0.84', ha="right", va="bottom", color=color3)
            ax.text(x-0.2, 0.73+0.01,'0.73', ha="right", va="bottom", color=color2)
            ax.text(x-0.2, 0.485,str(x), ha="right", va="center", color='k', alpha=0.5)
            ax.legend(loc='lower right', bbox_to_anchor=(0.88,0))
        elif (classifier_type=="LinearSVC"):
            title = "SVM (Linear)"
            x = 40
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x-0.2, 1.0+0.01,'1.00', ha="right", va="bottom", color=color1)
            ax.text(x-0.2, 0.97+0.003,'0.97', ha="right", va="bottom", color=color3)
            ax.text(x-0.2, 0.95-0.01,'0.95', ha="right", va="top", color=color2)
            ax.text(x-0.2, 0.485,str(x), ha="right", va="center", color='k', alpha=0.5)
            ax.legend(loc='lower left', fontsize=13)
            
        ax.set_xlabel('Time Window Size', fontsize=14)    
        ax.set_title(title, fontsize=14)
        
        plt.ylim(0.47, 1.04)
        fig.tight_layout()
        plt.savefig(path_init + '\\Images\\Results2\\Time Window Widths\\time_windows_' + classifier_type + '_ea.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def execute_plot_roc_curves_cv(path_init):
    
    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    width = 40
    
    # Feature Selection & Classification Parameters
    #params = [["GaussianNB","mine",43], ["LinearSVC","f_classif",75]]
    params = [["GaussianNB","mine",43], ["LinearSVC","kruskalwallis",83]]
    
    df = get_dataset(path_init, correlation_type, data_type, width)
    df_train, df_test = get_df_train_test(df)
    
    sensors_init_aux = get_sensors(data_type, sensor_type)
    combos_str = select_sensors(sensors_init_aux)
    
    df_train = df_train.loc[:,combos_str]
    df_test = df_test.loc[:,combos_str]
    
    #print(df_train)
    #print(df_test)
    
    # Cross validation
    cross_validation_3(df_train, sensors_init_aux, params)
    
def execute_plot_roc_curves_cv_ea(path_init):
    
    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    width = 40
    
    # Feature Selection & Classification Parameters
    #params = [["GaussianNB","mine",43], ["LinearSVC","f_classif",75]]
    params = [["LinearSVC","kruskalwallis",83]]
    
    df = get_dataset(path_init, correlation_type, data_type, width)
    df_train, df_test = get_df_train_test(df)
    
    sensors_init_aux = get_sensors(data_type, sensor_type)
    combos_str = select_sensors(sensors_init_aux)
    
    df_train = df_train.loc[:,combos_str]
    df_test = df_test.loc[:,combos_str]
    
    #print(df_train)
    #print(df_test)
    
    # Cross validation
    cross_validation_3_ea(df_train, sensors_init_aux, params)
    
def execute_thresholds(path_init):

    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    width = 40
    
    
    # Prediciton Parameters
    classifier_type = "LinearSVC" #"GaussianNB"
    
    # Feature Selection Parameters
    if(classifier_type == "GaussianNB"):
        selection_type = "mine"
        n_features = 43
    elif (classifier_type == "LinearSVC"):
        selection_type = "kruskalwallis" #"f_classif"
        n_features = 83 #75
    
    df = get_dataset(path_init, correlation_type, data_type, width)
    df_train, df_test = get_df_train_test(df)
    
    sensors_init_aux = get_sensors(data_type, sensor_type)
    combos_str = select_sensors(sensors_init_aux)
    
    df_train = df_train.loc[:,combos_str]
    df_test = df_test.loc[:,combos_str]
    
    optimal_thresholds = []
    start = round(-1,2)
    while start <= 1.05:
        start = round(start,2)
        optimal_thresholds.append(start)
        start += 0.01
    
    #print(optimal_thresholds)
    
    df_results = pd.DataFrame()
    for optimal_threshold in optimal_thresholds:
        print(optimal_threshold)
        results, _ = cross_validation_4(classifier_type, df_train, sensors_init_aux, selection_type, n_features, optimal_threshold)
        #print(results)
        results['optimal_threshold'] = optimal_threshold
        df_results = df_results.append(results, ignore_index=True)
    print(df_results)
    
    path_export = path_init + '\\Results\\thresholds_' + classifier_type + '.csv'
    df_results.to_csv(index=True, path_or_buf=path_export, sep=';', decimal=',')
    
def execute_plot_thresholds(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    
    classifier_types = ["GaussianNB","LinearSVC"] #"GaussianNB" #LinearSVC
    
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6), sharex=True, sharey=True)
    for i, ax in enumerate(axs):
    
        classifier_type = classifier_types[i]
        
        path = path_init + '\\Results\\thresholds_' + classifier_type + '.csv'
        df = pd.read_csv(path, index_col=0, sep=';', decimal=',')
        #print(df)
        
        df = df[(df.index<201)]
            
        ax.plot(df['optimal_threshold'], df['precision'], color=color1, label=r'$Precision_{P}$')
        ax.plot(df['optimal_threshold'], df['recall'], color=color2, label=r'$Recall_{P}$')
        ax.plot(df['optimal_threshold'], df['F1'], color=color3, label = r'$F_{P}$')
        
        if(classifier_type == "GaussianNB"):
            
            x = -0.99
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.02, 0.94+0.01,'0.94', ha="left", va="bottom", color=color1)
            ax.text(x+0.02, 0.88+0.01,'0.88', ha="left", va="bottom", color=color3)
            ax.text(x+0.02, 0.82,'0.82', ha="left", va="center", color=color2)
            ax.text(x+0.02, 0.37, str(x), ha="left", va="center", color='k', alpha=0.5)
            
            x = -0.67
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.02, 1.00-0.01,'1.00', ha="left", va="top", color=color1)
            ax.text(x+0.02, 0.86+0.01,'0.86', ha="left", va="bottom", color=color3)
            ax.text(x+0.02, 0.75+0.01,'0.75', ha="left", va="bottom", color=color2)
            ax.text(x+0.02, 0.37, str(x), ha="left", va="center", color='k', alpha=0.5)
            
            x = 0
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.02, 1.00-0.01,'1.00', ha="left", va="top", color=color1)
            ax.text(x+0.02, 0.84+0.01,'0.84', ha="left", va="bottom", color=color3)
            ax.text(x+0.02, 0.73,'0.73', ha="left", va="bottom", color=color2)
            ax.text(x+0.02, 0.37, "0.0", ha="left", va="center", color='k', alpha=0.5)
        
            ax.set_title('Naive Bayes')
            ax.legend(loc='lower right',bbox_to_anchor=(0.92, 0))
        else:
            
            x = -0.74
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x-0.04, 1.00-0.025,'1.00', ha="right", va="top", color=color1)
            ax.text(x-0.04, 0.99-0.042,'0.99', ha="right", va="top", color=color3)
            ax.text(x-0.04, 0.98-0.060,'0.98', ha="right", va="top", color=color2)
            ax.text(x-0.02, 0.37, str(x), ha="right", va="center", color='k', alpha=0.5)
            
            x = -0.66
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.02, 1.00-0.035,'1.00', ha="left", va="top", color=color1)
            ax.text(x+0.02, 0.99-0.052,'0.99', ha="left", va="top", color=color3)
            ax.text(x+0.02, 0.98-0.070,'0.98', ha="left", va="top", color=color2)
            ax.text(x+0.02, 0.37, str(x), ha="left", va="center", color='k', alpha=0.5)
            
            x = 0
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.04, 1.00-0.005,'1.00', ha="left", va="top", color=color1)
            ax.text(x+0.04, 0.97-0.005,'0.97', ha="left", va="top", color=color3)
            ax.text(x+0.04, 0.95-0.02,'0.95', ha="left", va="top", color=color2)
            ax.text(x+0.04, 0.37, "0.0", ha="left", va="center", color='k', alpha=0.5)
        
            ax.set_title('SVM w/ Linear Kernel')
        
        ax.set_xlabel('Threshold')
        ax.set_ylim(0.35, 1.03)
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.grid(True, axis='y', alpha=0.3, which='both')
        
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results2\\ROC Curves\\thresholds.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def execute_plot_thresholds_ea(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    
    
    for classifier_type in ["LinearSVC"]:
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4.5))
    
        path = path_init + '\\Results\\thresholds_' + classifier_type + '.csv'
        df = pd.read_csv(path, index_col=0, sep=';', decimal=',')
        #print(df)
        
        df = df[(df.index<201)]
            
        ax.plot(df['optimal_threshold'], df['precision'], color=color1, label=r'$Prec_{P}$')
        ax.plot(df['optimal_threshold'], df['recall'], color=color2, label=r'$Rec_{P}$')
        ax.plot(df['optimal_threshold'], df['F1'], color=color3, label = r'$F_{P}$')
        
        if(classifier_type == "GaussianNB"):
            
            x = -0.99
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.02, 0.94+0.01,'0.94', ha="left", va="bottom", color=color1)
            ax.text(x+0.02, 0.88+0.01,'0.88', ha="left", va="bottom", color=color3)
            ax.text(x+0.02, 0.82,'0.82', ha="left", va="center", color=color2)
            ax.text(x+0.02, 0.37, str(x), ha="left", va="center", color='k', alpha=0.5)
            
            x = -0.67
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.02, 1.00-0.01,'1.00', ha="left", va="top", color=color1)
            ax.text(x+0.02, 0.86+0.01,'0.86', ha="left", va="bottom", color=color3)
            ax.text(x+0.02, 0.75+0.01,'0.75', ha="left", va="bottom", color=color2)
            ax.text(x+0.02, 0.37, str(x), ha="left", va="center", color='k', alpha=0.5)
            
            x = 0
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.02, 1.00-0.01,'1.00', ha="left", va="top", color=color1)
            ax.text(x+0.02, 0.84+0.01,'0.84', ha="left", va="bottom", color=color3)
            ax.text(x+0.02, 0.73,'0.73', ha="left", va="bottom", color=color2)
            ax.text(x+0.02, 0.37, "0.0", ha="left", va="center", color='k', alpha=0.5)
        
            ax.set_title('Naive Bayes')
            ax.legend(loc='lower right',bbox_to_anchor=(0.92, 0))
        else:
            
            x = -0.74
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x-0.04, 1.00-0.025,'1.00', ha="right", va="top", color=color1)
            ax.text(x-0.04, 0.99-0.042,'0.99', ha="right", va="top", color=color3)
            ax.text(x-0.04, 0.98-0.060,'0.98', ha="right", va="top", color=color2)
            ax.text(x-0.02, 0.517, str(x), ha="right", va="center", color='k', alpha=0.5)
            
            x = -0.66
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.02, 1.00-0.035,'1.00', ha="left", va="top", color=color1)
            ax.text(x+0.02, 0.99-0.052,'0.99', ha="left", va="top", color=color3)
            ax.text(x+0.02, 0.98-0.070,'0.98', ha="left", va="top", color=color2)
            ax.text(x+0.02, 0.517, str(x), ha="left", va="center", color='k', alpha=0.5)
            
            x = 0
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.04, 1.00-0.005,'1.00', ha="left", va="top", color=color1)
            ax.text(x+0.04, 0.97-0.005,'0.97', ha="left", va="top", color=color3)
            ax.text(x+0.04, 0.95-0.02,'0.95', ha="left", va="top", color=color2)
            ax.text(x+0.04, 0.517, "0.0", ha="left", va="center", color='k', alpha=0.5)
        
            ax.set_title('Performance w/ Different Thresholds', fontsize=14)
            ax.legend(loc='lower right', fontsize=14)
        
        ax.set_xlabel('Threshold', fontsize=14)
        ax.set_ylim(0.5, 1.03)
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
        ax.grid(True, axis='y', alpha=0.3, which='both')
        
        fig.tight_layout()
        plt.savefig(path_init + '\\Images\\Results2\\ROC Curves\\threshold_' + classifier_type + '_ea.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


def before_after_threshold(path_init):
    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    width = 40
    
    classifier_types = ["GaussianNB", "LinearSVC"]
    
    labels = ['0','0.05', '0.1', '0.5', '1.0', '1.5', '2.0']
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9,5))
    for i, ax in enumerate(axs):
        classifier_type = classifier_types[i]
        
        if (classifier_type == "GaussianNB"):
            selection_type = "mine"
            n_features = 43
            optimal_threshold = None
        elif (classifier_type == "LinearSVC"):
            selection_type = "kruskalwallis"
            n_features = 83
            optimal_threshold = None
            
        df = get_dataset(path_init, correlation_type, data_type, width)
        df_train, df_test = get_df_train_test(df)
        
        sensors_init_aux = get_sensors(data_type, sensor_type)
        combos_str = select_sensors(sensors_init_aux)
        
        df_train = df_train.loc[:,combos_str]
        df_test = df_test.loc[:,combos_str]
        
        results, df_results = cross_validation_4(classifier_type, df_train, sensors_init_aux, selection_type, n_features, optimal_threshold)
        
        data_to_plot = []
        positions = []
        for y in range(0,7,1):
            data_to_plot.append(df_results[(df_results['y']==y)]['score'])
            positions.append(y)
        
        if (classifier_type == "GaussianNB"):
            ax.set_title("Naive Bayes")
            ax.set_ylabel('Difference between the Probability for each Class')
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.20))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.10))
        else:
            ax.set_title("SVM w/ Linear Kernel")
            ax.set_ylabel('Distance to the Separating Hyperplane')
            ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        
        ax.boxplot(data_to_plot, positions=positions, labels=labels, showfliers=False)
        ax.set_xlabel('Leakage Coefficient')
        ax.grid(True, axis='y', alpha=0.3, which='both') 
    
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results2\\ROC Curves\\box_plots.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def before_after_threshold_ea(path_init):
    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    width = 40
    
    labels = ['0','0.05', '0.1', '0.5', '1.0', '1.5', '2.0']
    
    for classifier_type in ["LinearSVC"]:
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4.5))
        
        if (classifier_type == "GaussianNB"):
            selection_type = "mine"
            n_features = 43
            optimal_threshold = None
        elif (classifier_type == "LinearSVC"):
            selection_type = "kruskalwallis"
            n_features = 83
            optimal_threshold = None
            
        df = get_dataset(path_init, correlation_type, data_type, width)
        df_train, df_test = get_df_train_test(df)
        
        sensors_init_aux = get_sensors(data_type, sensor_type)
        combos_str = select_sensors(sensors_init_aux)
        
        df_train = df_train.loc[:,combos_str]
        df_test = df_test.loc[:,combos_str]
        
        results, df_results = cross_validation_4(classifier_type, df_train, sensors_init_aux, selection_type, n_features, optimal_threshold)
        
        data_to_plot = []
        positions = []
        for y in range(0,7,1):
            data_to_plot.append(df_results[(df_results['y']==y)]['score'])
            positions.append(y)
        
        if (classifier_type == "GaussianNB"):
            ax.set_title("Naive Bayes")
            ax.set_ylabel('Difference between the Probability for each Class')
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.20))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.10))
        else:
            ax.set_title("Box Plots of Instances' Scores", fontsize=14)
            ax.set_ylabel('Distance to the Separating Hyperplane', fontsize=14)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        
        ax.boxplot(data_to_plot, positions=positions, labels=labels, showfliers=False)
        ax.set_xlabel('Leakage Coefficient', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3, which='both') 
    
        fig.tight_layout()
        plt.savefig(path_init + '\\Images\\Results2\\ROC Curves\\box_plot_' + classifier_type + '.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.0f}%'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 3, height),
                    xytext=(6, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    rotation=30,
                    ha='center', va='bottom', color='k')
        

def before_after_threshold_2(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'

    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    width = 40
    
    # Prediciton Parameters
    classifiers_type = ["GaussianNB","LinearSVC"]
    
    df = get_dataset(path_init, correlation_type, data_type, width)
    df_train, df_test = get_df_train_test(df)
    
    sensors_init_aux = get_sensors(data_type, sensor_type)
    combos_str = select_sensors(sensors_init_aux)
    
    df_train = df_train.loc[:,combos_str]
    df_test = df_test.loc[:,combos_str]
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13,5), sharey=True)
    for i, ax in enumerate(axs):
        
        classifier_type = classifiers_type[i] 
        
        if (classifier_type == "GaussianNB"):
            selection_type = "mine"
            n_features = 43
            optimal_threshold = None
            optimal_threshold1 = -0.67
            optimal_threshold2 = -0.99
        elif (classifier_type == "LinearSVC"):
            selection_type = "kruskalwallis"
            n_features = 83
            optimal_threshold = None
            optimal_threshold1 = -0.66
            optimal_threshold2 = -0.74
    
        results, df_results = cross_validation_4(classifier_type, df_train, sensors_init_aux, selection_type, n_features, optimal_threshold)
        results1, df_results1 = cross_validation_4(classifier_type, df_train, sensors_init_aux, selection_type, n_features, optimal_threshold1)
        results2, df_results2 = cross_validation_4(classifier_type, df_train, sensors_init_aux, selection_type, n_features, optimal_threshold2)
        
        print(results)
        print(results1)
        print(results2)
        
        g1 = []
        g2 = []
        g3 = []
        positions = []
        labels = ['0', '0.05', '0.1', '0.5', '1.0', '1.5', '2.0']
        df_results = df_results.groupby(['y','pred']).count()['y1']
        df_results1 = df_results1.groupby(['y','pred']).count()['y1']
        df_results2 = df_results2.groupby(['y','pred']).count()['y1']
        
        print(df_results)
        
        for y in range(0,7,1):
            positions.append(y)
            g = df_results[y]
            if(y==0):
                if(len(g)==1):
                    g1.append(0)
                else:
                    g1.append((g[1]*100)/(g[0]+g[1]))
            else:
                if(len(g)==1):
                    g1.append(0)
                else:
                    g1.append((g[0]*100)/(g[0]+g[1]))
                
        for y in range(0,7,1):
            g = df_results1[y]
            if(y==0):
                if(len(g)==1):
                    g2.append(0)
                else:
                    g2.append((g[1]*100)/(g[0]+g[1]))
            else:
                if(len(g)==1):
                    g2.append(0)
                else:
                    g2.append((g[0]*100)/(g[0]+g[1]))
                
        for y in range(0,7,1):
            g = df_results2[y]
            if(y==0):
                if(len(g)==1):
                    g3.append(0)
                else:
                    g3.append((g[1]*100)/(g[0]+g[1]))
            else:
                if(len(g)==1):
                    g3.append(0)
                else:
                    g3.append((g[0]*100)/(g[0]+g[1]))
            
        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars
        
        label1 = 'Threshold = 0'
        label2 = 'Threshold = ' + str(optimal_threshold1)
        label3 = 'Threshold = ' + str(optimal_threshold2)
        
        
        rects1 = ax.bar(x - width, g1, width, label=label1, color=color1)
        rects2 = ax.bar(x, g2, width, label=label2, color=color2)
        rects3 = ax.bar(x + width, g3, width, label=label3, color=color3)
        
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        
        if (i==0):
            ax.set_ylabel('Percentage of Instances Incorrecly Classified')
            ax.set_title('Naive Bayes')
            ax.set_xlabel('Leakage Coefficient')
            ax.legend()
        else:
            ax.set_title('SVM w/ Linear Kernel')
            ax.set_xlabel('Leakage Coefficient')
            ax.legend()
            
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis='y', alpha=0.3, which='both')
        autolabel(rects1, ax)
        autolabel(rects2, ax)
        autolabel(rects3, ax)
    
    plt.ylim(0, 75)
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results2\\ROC Curves\\bar_chart.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def before_after_threshold_2_ea(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'

    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    width = 40
    
    # Prediciton Parameters
    
    df = get_dataset(path_init, correlation_type, data_type, width)
    df_train, df_test = get_df_train_test(df)
    
    sensors_init_aux = get_sensors(data_type, sensor_type)
    combos_str = select_sensors(sensors_init_aux)
    
    df_train = df_train.loc[:,combos_str]
    df_test = df_test.loc[:,combos_str]
    
    
    for classifier_type in ["LinearSVC"]:
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4.5))
        
        if (classifier_type == "GaussianNB"):
            selection_type = "mine"
            n_features = 43
            optimal_threshold = None
            optimal_threshold1 = -0.67
            optimal_threshold2 = -0.99
        elif (classifier_type == "LinearSVC"):
            selection_type = "kruskalwallis"
            n_features = 83
            optimal_threshold = None
            optimal_threshold1 = -0.66
            optimal_threshold2 = -0.74
    
        results, df_results = cross_validation_4(classifier_type, df_train, sensors_init_aux, selection_type, n_features, optimal_threshold)
        results1, df_results1 = cross_validation_4(classifier_type, df_train, sensors_init_aux, selection_type, n_features, optimal_threshold1)
        results2, df_results2 = cross_validation_4(classifier_type, df_train, sensors_init_aux, selection_type, n_features, optimal_threshold2)
        
        print(results)
        print(results1)
        print(results2)
        
        g1 = []
        g2 = []
        g3 = []
        positions = []
        labels = ['0', '0.05', '0.1', '0.5', '1.0', '1.5', '2.0']
        df_results = df_results.groupby(['y','pred']).count()['y1']
        df_results1 = df_results1.groupby(['y','pred']).count()['y1']
        df_results2 = df_results2.groupby(['y','pred']).count()['y1']
        
        print(df_results)
        
        for y in range(0,7,1):
            positions.append(y)
            g = df_results[y]
            if(y==0):
                if(len(g)==1):
                    g1.append(0)
                else:
                    g1.append((g[1]*100)/(g[0]+g[1]))
            else:
                if(len(g)==1):
                    g1.append(0)
                else:
                    g1.append((g[0]*100)/(g[0]+g[1]))
                
        for y in range(0,7,1):
            g = df_results1[y]
            if(y==0):
                if(len(g)==1):
                    g2.append(0)
                else:
                    g2.append((g[1]*100)/(g[0]+g[1]))
            else:
                if(len(g)==1):
                    g2.append(0)
                else:
                    g2.append((g[0]*100)/(g[0]+g[1]))
                
        for y in range(0,7,1):
            g = df_results2[y]
            if(y==0):
                if(len(g)==1):
                    g3.append(0)
                else:
                    g3.append((g[1]*100)/(g[0]+g[1]))
            else:
                if(len(g)==1):
                    g3.append(0)
                else:
                    g3.append((g[0]*100)/(g[0]+g[1]))
            
        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars
        
        label1 = 'Threshold = 0'
        label2 = 'Threshold = ' + str(optimal_threshold1)
        label3 = 'Threshold = ' + str(optimal_threshold2)
        
        
        rects1 = ax.bar(x - width, g1, width, label=label1, color=color1)
        rects2 = ax.bar(x, g2, width, label=label2, color=color2)
        rects3 = ax.bar(x + width, g3, width, label=label3, color=color3)
        
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
        

        ax.set_title('Instances Incorrecly Classified', fontsize=12)
        ax.set_xlabel('Leakage Coefficient', fontsize=14)
        ax.set_ylabel('Percentage of Instances Incorrecly Classified', fontsize=12)
        ax.legend(fontsize=14)
            
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis='y', alpha=0.3, which='both')
        autolabel(rects1, ax)
        autolabel(rects2, ax)
        autolabel(rects3, ax)
    
    plt.ylim(0, 25)
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results2\\ROC Curves\\bar_chart_' + classifier_type + '_ea.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def before_after_threshold_3(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'

    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    width = 40
    
    # Prediciton Parameters
    classifiers_type = ["GaussianNB","LinearSVC"]
    
    df = get_dataset(path_init, correlation_type, data_type, width)
    df_train, df_test = get_df_train_test(df)
    
    sensors_init_aux = get_sensors(data_type, sensor_type)
    combos_str = select_sensors(sensors_init_aux)
    
    df_train = df_train.loc[:,combos_str]
    df_test = df_test.loc[:,combos_str]
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11,5), sharey=True)
    for i, ax in enumerate(axs):
        
        classifier_type = classifiers_type[i] 
        
        if (classifier_type == "GaussianNB"):
            selection_type = "mine"
            n_features = 43
            optimal_threshold1 = -0.67
        elif (classifier_type == "LinearSVC"):
            selection_type = "kruskalwallis"
            n_features = 83
            optimal_threshold1 = -0.66
    
        results, df_results = cross_validation_4(classifier_type, df_train, sensors_init_aux, selection_type, n_features, optimal_threshold1)
        results1, _, df_results1 = final_test(classifier_type, df_train, df_test, sensors_init_aux, selection_type, n_features, optimal_threshold1)
        
        print(results)
        print(results1)
        
        g1 = []
        g2 = []
        positions = []
        labels = ['0', '0.05', '0.1', '0.5', '1.0', '1.5', '2.0']
        df_results = df_results.groupby(['y','pred']).count()['y1']
        df_results1 = df_results1.groupby(['y','pred']).count()['y1']
        
        print(df_results)
        
        for y in range(0,7,1):
            positions.append(y)
            g = df_results[y]
            if(y==0):
                if(len(g)==1):
                    g1.append(0)
                else:
                    g1.append((g[1]*100)/(g[0]+g[1]))
            else:
                if(len(g)==1):
                    g1.append(0)
                else:
                    g1.append((g[0]*100)/(g[0]+g[1]))
                
        for y in range(0,7,1):
            g = df_results1[y]
            if(y==0):
                if(len(g)==1):
                    g2.append(0)
                else:
                    g2.append((g[1]*100)/(g[0]+g[1]))
            else:
                if(len(g)==1):
                    g2.append(0)
                else:
                    g2.append((g[0]*100)/(g[0]+g[1]))
                
            
        x = np.arange(len(labels))  # the label locations
        width = 0.4  # the width of the bars
        
        label1 = 'Training Set (5-Fold Cross-Validation)'
        label2 = 'Test Set'
        
        rects1 = ax.bar(x - width, g1, width, label=label1, color=color1)
        rects2 = ax.bar(x, g2, width, label=label2, color=color2)
        
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        
        if (i==0):
            ax.set_ylabel('Percentage of Instances Incorrecly Classified')
            ax.set_title('Naive Bayes')
            ax.set_xlabel('Leakage Coefficient')
            ax.legend()
        else:
            ax.set_title('SVM w/ Linear Kernel')
            ax.set_xlabel('Leakage Coefficient')
            ax.legend()
            
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis='y', alpha=0.3, which='both')
        autolabel(rects1, ax)
        autolabel(rects2, ax)
    
    plt.ylim(0, 70)
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results2\\Final Results\\bar_chart.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()   
   
def execute_leakage_assessment(path_init):
    # Dataset Parameters
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    sensor_type = "all" # p, f, all
    width = 40
    
    classifier_types = ["GaussianNB", "LinearSVC"]
    
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(11.4,6.4), sharex=True)
    for i, ax in enumerate(axs):
    
        classifier_type = classifier_types[i];
        
        print(classifier_type)
        
        if (classifier_type == "GaussianNB"):
            selection_type = "mine"
            n_features = 43
            optimal_threshold = -0.67
        elif (classifier_type == "LinearSVC"):
            selection_type = "f_classif"
            n_features = 75
            optimal_threshold = -0.53
    
        df = get_dataset(path_init, correlation_type, data_type, width)
    
        df_train, df_test = get_df_train_test(df)
    
        sensors_init_aux = get_sensors(data_type, sensor_type)
        combos_str = select_sensors(sensors_init_aux)
        
        df_train = df_train.loc[:,combos_str]
        df_test = df_test.loc[:,combos_str]
    
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_4\\dataset_702_' + correlation_type.lower() +'_' + str(width) + '.csv'
        df_test = pd.read_csv(path, index_col=0)
        df_test = df_test.loc[:,combos_str]
    
        results, y_scores, _ = final_test(classifier_type, df_train, df_test, sensors_init_aux, selection_type, n_features, optimal_threshold)       
        
        df_test['score'] = y_scores
        x = range(0, len(df_test)*600, 600)
        
        ax.plot(x,df_test['score'], color='tab:blue')
        
        ax.axvspan(48600, 63000, color='tab:red', alpha=0.1, label="Positive Instances (coef=2.0)")
        ax.axhline(y=optimal_threshold, color='tab:red', linestyle='--', linewidth=1.25, alpha=0.7, label="Threshold = " + str(optimal_threshold))
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(8*600))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(4*600))
        
        if i == 0:
            ax.set_title("Naive Bayes")
            ax.set_ylabel('Difference between the\nProbability for each Class')
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            ax.legend(bbox_to_anchor=(0.515, 1));
        else:
            ax.set_ylabel('Distance to the\nSeparating Hyperplane')
            ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))
            ax.set_title("SVM w/ Linear Kernel")
            ax.set_xlabel('Time Point')
            ax.legend();
        
        ax.grid(True, axis='y', alpha=0.3, which='both')
        
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results2\\Final Results\\leakage_assessment.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def rank_kruskalwallis(X_train, y_train, n_features):
    
    columns_x = X_train.columns
    y = y_train
    columns = []
    values = []
    pvalues = []
    
    for column in columns_x:
        x = X_train[column]
        statistic, pvalue = kruskalwallis(x.values,y.values)
        columns.append(column)
        values.append(statistic)
        pvalues.append(pvalue)
            
    columns_df = pd.DataFrame();
    columns_df = columns_df.from_dict({'column':columns,'value':values,'pvalue':pvalues}).sort_values(by=['value','pvalue'], ascending = [True,False])
    
    columns_final = columns_df.nlargest(n_features, 'value')['column']
    
    return columns_final
    
    
def execute_histograms(path_init):
    
    correlation_type = "DCCA" # DCCA, Pearson
    data_type = "s" # s, r
    width = 40
    n_features = 100
    
    columns = ['1-4','22-25','1-25']
    
    df = get_dataset(path_init, correlation_type, data_type, width)
    
    df_train, df_test = get_df_train_test(df)
        
    print(len(df_train))
    print(len(df_test))
    
    print(len(df_train[df_train['y']==0]))
    print(len(df_train[df_train['y']>0]))
    print(len(df_test[df_test['y']==0]))
    print(len(df_test[df_test['y']>0]))
    
    df_pos = df[df['y']>0]
    df_neg = df[df['y']==0]
    
    #print(df_pos)
    #print(df_neg)
    
    #print(df.iloc[:,:-2])
    
    
    
    rank_kruskalwallis(df.iloc[:,:-2], df['y'], n_features)
    
    #for column in columns:
        
        #data = df_pos[column]
        
    """
        stat, p = normaltest(data)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
        	print('Sample looks Gaussian (fail to reject H0)')
        else:
        	print('Sample does not look Gaussian (reject H0)')
    
        plt.hist(data, bins='auto')
        plt.show()
        plt.close()
    """
        
        #statistic, pvalue = kruskalwallis(data,df['y'])
        #print(statistic, pvalue)
    
config = Configuration()
path_init = config.path

#execute_plot_feature_selection_distribution(path_init)
#execute_plot_feature_selection_top(path_init)

#execute_feature_selection_evolution(path_init)
#execute_plot_feature_selection_evolution(path_init) 
#execute_plot_feature_selection_evolution_ea(path_init) 
  
#execute_time_window_widths(path_init)
#execute_plot_time_window_widths(path_init)
#execute_plot_time_window_widths_ea(path_init)

#execute_plot_roc_curves_cv(path_init)  
#execute_plot_roc_curves_cv_ea(path_init)  
  
#execute_thresholds(path_init)
#execute_plot_thresholds(path_init)
#execute_plot_thresholds_ea(path_init)


#before_after_threshold(path_init) # Boxplots
#before_after_threshold_ea(path_init)
#before_after_threshold_2(path_init) # Barchart Thresholds
#before_after_threshold_2_ea(path_init)
#before_after_threshold_3(path_init) # Barchart Test/Training

#execute_histograms(path_init)

#execute_main_results(path_init)
#execute_final_results(path_init)

#execute_leakage_assessment(path_init)
    

    
    




