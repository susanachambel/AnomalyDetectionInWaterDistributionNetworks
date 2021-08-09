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

from scipy.stats.mstats import kruskalwallis

from functools import partial

def get_dataset(path_init, correlation_type, width):
    dcca_k = int(width/4)
    path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_3\\dataset_r_' + correlation_type.lower() +'_' + str(width) + '_' + str(dcca_k) + '.csv'
    df = pd.read_csv(path, index_col=0)
    return df

def split_pos_neg(df):
    
    df_neg = df[df['y']==0]
    df_pos = df[df['y']>0]

    df_neg = df_neg.fillna(df_neg.mean())
    df_pos = df_pos.fillna(df_pos.mean())
    
    events_id = df_pos['y'].unique()
    
    dfs_pos = []
    for event_id in events_id:
        dfs_pos.append(df_pos[df_pos['y']==event_id])
    
    df_neg = df_neg.sample(frac = 1, random_state=1).reset_index(drop=True)
    dfs_neg = np.array_split(df_neg, len(events_id))
        
    return dfs_neg, dfs_pos

def get_train_test(df, frac):
    
    dfs_neg, dfs_pos = split_pos_neg(df)
    
    test_id = 2
    df_neg_test = dfs_neg[test_id]
    df_pos_test = dfs_pos[test_id]
    
    #if(frac is not None):
        #df_neg_test = df_neg_test.sample(frac=frac, random_state=1)
    
    dfs_neg_train = []
    dfs_pos_train = []
    
    for i in range(0,len(dfs_neg)):
        if (i!=test_id):
            
            df_neg_train = dfs_neg[i]
            df_pos_train = dfs_pos[i]
            
            if(frac is not None):
                df_neg_train = df_neg_train.sample(frac=frac, random_state=1)
                
            dfs_neg_train.append(df_neg_train)
            dfs_pos_train.append(df_pos_train)
    
    return dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test
        
def get_fold_elements(dfs_neg, dfs_pos, n_fold):
    
    X_test_neg = dfs_neg[n_fold].iloc[:,:-3]
    y_test_neg = dfs_neg[n_fold]['y']
    
    X_test_pos = dfs_pos[n_fold].iloc[:,:-3]
    y_test_pos = dfs_pos[n_fold]['y'].copy()
    
    y_test_pos[y_test_pos > 0] = 1
    
    X_test = X_test_neg.append(X_test_pos).reset_index(drop=True)
    y_test = y_test_neg.append(y_test_pos).reset_index(drop=True)
    
    dfs_train_pos = dfs_pos[:n_fold] + dfs_pos[n_fold+1:]
    dfs_train_neg = dfs_neg[:n_fold] + dfs_neg[n_fold+1:]
    
    df_train = dfs_train_pos[0]
    df_train = df_train.append(dfs_train_neg[0])
    for i in range(0,len(dfs_train_neg)):
        if i != 0:
            df_train = df_train.append(dfs_train_neg[i])
            df_train = df_train.append(dfs_train_pos[i])
    
    X_train = df_train.iloc[:,:-3]
    y_train = df_train['y'].copy()
    y_train[y_train > 0] = 1
    
    return X_train, y_train, X_test, y_test

def get_elements(dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test):
    
    df_train = dfs_pos_train[0]
    df_train = df_train.append(dfs_neg_train[0])
    for i in range(0,len(dfs_neg_train)):
        if i != 0:
            df_train = df_train.append(dfs_neg_train[i])
            df_train = df_train.append(dfs_pos_train,[i])
    
    X_train = df_train.iloc[:,:-3].reset_index(drop=True)
    y_train = df_train['y'].reset_index(drop=True).copy()
    y_train[y_train > 0] = 1
    
    df_test = df_neg_test.append(df_pos_test).reset_index(drop=True)
    X_test = df_test.iloc[:,:-3]
    y_test = df_test['y'].copy()
    y_test[y_test > 0] = 1

    return X_train, y_train, X_test, y_test
 
def train_predict(classifier_type, X_train, y_train, X_test, option):
    
    clf = None
    y_pred = None
    y_scores = None
    
    if classifier_type == "GaussianNB":
        clf = GaussianNB()
    elif classifier_type == "LinearSVC":
        clf = LinearSVC(random_state=1, max_iter=3000) #max_iter=10000
    elif classifier_type == "NuSVC-rbf":
        clf = NuSVC(random_state=1, kernel='rbf', nu=0.000001) # class_weight='balanced', , nu=0.000001) 
    elif classifier_type == "NuSVC-poly":
        clf = NuSVC(random_state=1, kernel='poly',degree=3,nu=0.000001) #nu=0.000001
    elif classifier_type =="NuSVC-sigmoid":
        clf = NuSVC(random_state=1, kernel='sigmoid', nu=0.001)
        
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
    plt.close()

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
        
    if(TP > 0):
        results['leakage'] = 1
    else:
        results['leakage'] = 0
    
    results['recall'] = recall
    results['precision'] = precision
    
    results['ACC'] = (TP+TN)/(TP+FN+TN+FP)
    return results

def update_results(df_results, TN, FP, FN, TP, auc_score):
    results = get_results(TN, FP, FN, TP, auc_score)
    df_results = df_results.append(results, ignore_index=True)
    return df_results

def get_mean_results(df_results):
    leakages_identified = round(df_results['leakage'].sum()/11,2)
    dic_results = round(df_results.describe().loc['mean',:],2).to_dict()
    dic_results_std = round(df_results.describe().loc['std',:],2).to_dict()
    #print(dic_results_std)
    dic_results['leakage'] = leakages_identified
    return dic_results

def cross_validation(dfs_neg, dfs_pos, sensors, classifier_type, selection_type, n_features, optimal_threshold):
    
    #print(dfs_neg)
    #print(dfs_pos)
    
    df_results = pd.DataFrame()
    n_folds = len(dfs_pos)
    
    for n_fold in range(0,n_folds,1):
        
        X_train, y_train, X_test, y_test = get_fold_elements(dfs_neg, dfs_pos, n_fold)
        X_train, X_test = select_features(X_train, y_train, X_test, sensors, selection_type, n_features)
        
        if(optimal_threshold == None):
            y_pred, _ = train_predict(classifier_type, X_train, y_train, X_test, 'both')
        else:
            y_scores = train_predict(classifier_type, X_train, y_train, X_test, 'scores')
            y_pred = (y_scores >= optimal_threshold).astype(bool)
        
        #df_y = pd.DataFrame()
        #df_y['pred'] = y_pred
        #df_y['test'] = y_test
        #print(df_y[df_y['test']==1])
        
        cnf_matrix = confusion_matrix(y_test, y_pred, [0,1])
        TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix)
        df_results = update_results(df_results, TN, FP, FN, TP, None)
        
        #plot_confusion_matrix(cnf_matrix, [0,1])
        
    return df_results

def cross_validation_roc(dfs_neg, dfs_pos, sensors, classifier_types, selection_type, n_features):
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5), sharey=True)
    for i, ax in enumerate(axs.flat):    
            
        classifier_type = classifier_types[i]
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        if classifier_type == "GaussianNB":
            clf = GaussianNB()
            response_method = "predict_proba"
            title = "Naive Bayes"
        elif classifier_type == "LinearSVC":
            clf = LinearSVC(random_state=1, max_iter=3000)
            response_method = "decision_function"
            title = "SVM w/ Linear Kernel"
            ax.set_ylabel('')
        elif classifier_type == "NuSVC-rbf":
            clf = NuSVC(random_state=1, kernel='rbf', nu=0.05) #nu=0.000001
            response_method = "decision_function"
            title = "SVM w/ RBF Kernel"
        
        n_folds = len(dfs_pos)
        for n_fold in range(0,n_folds,1):
            
            X_train, y_train, X_test, y_test = get_fold_elements(dfs_neg, dfs_pos, n_fold)
            X_train, X_test = select_features(X_train, y_train, X_test, sensors, selection_type, n_features)
            
            clf_fit = clf.fit(X_train, y_train)
            viz = plot_roc_curve(clf_fit, X_test, y_test,
                        name='ROC fold {}'.format(n_fold),
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
                label= 'Mean ROC\n' + r'(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
        
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
        
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.grid(True, axis='y', alpha=0.3, which='both')
        
        if i!=0:
            ax.set_ylabel('')
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-3:], labels[-3:],loc="lower right")
        
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results3\\ROC Curves\\roc_curves_redux.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def cross_validation_roc_ea(dfs_neg, dfs_pos, sensors, classifier_types, selection_type, n_features):
    
    
    for classifier_type in classifier_types:
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4.5))
        
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        if classifier_type == "GaussianNB":
            clf = GaussianNB()
            response_method = "predict_proba"
            title = "ROC Plot"
        elif classifier_type == "LinearSVC":
            clf = LinearSVC(random_state=1, max_iter=3000)
            response_method = "decision_function"
            title = "SVM (Linear)"
            ax.set_ylabel('')
        elif classifier_type == "NuSVC-rbf":
            clf = NuSVC(random_state=1, kernel='rbf', nu=0.05) #nu=0.000001
            response_method = "decision_function"
            title = "SVM (RBF)"
        
        n_folds = len(dfs_pos)
        for n_fold in range(0,n_folds,1):
            
            X_train, y_train, X_test, y_test = get_fold_elements(dfs_neg, dfs_pos, n_fold)
            X_train, X_test = select_features(X_train, y_train, X_test, sensors, selection_type, n_features)
            
            clf_fit = clf.fit(X_train, y_train)
            viz = plot_roc_curve(clf_fit, X_test, y_test,
                        name='ROC fold {}'.format(n_fold),
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
                label= 'Mean ROC\n' + r'(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
        
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.set_title(title, fontsize=14)
        
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.grid(True, axis='y', alpha=0.3, which='both')
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-3:], labels[-3:],loc="lower right")
        
        ax.set_ylabel("True Positive Rate", fontsize=14)
        ax.set_xlabel("False Positive Rate", fontsize=14)
        
        fig.tight_layout()
        plt.savefig(path_init + '\\Images\\Results3\\ROC Curves\\roc_curves_redux_'+ classifier_type +'_ea.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def cross_validation_roc_1(dfs_neg, dfs_pos, sensors, classifier_type, selection_type, n_features):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.3,4.9)) #(5.3,4.9)
      
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    if classifier_type == "GaussianNB":
        clf = GaussianNB()
        response_method = "predict_proba"
    elif classifier_type == "LinearSVC":
        clf = LinearSVC(random_state=1, max_iter=3000)
        response_method = "decision_function"
        ax.set_ylabel('')
    elif classifier_type == "NuSVC-rbf":
        clf = NuSVC(random_state=1, kernel='rbf', nu=0.000001)
        response_method = "decision_function"
    
    n_folds = len(dfs_pos)
    for n_fold in range(0,n_folds,1):
        
        X_train, y_train, X_test, y_test = get_fold_elements(dfs_neg, dfs_pos, n_fold)
        X_train, X_test = select_features(X_train, y_train, X_test, sensors, selection_type, n_features)
        
        clf_fit = clf.fit(X_train, y_train)
        viz = plot_roc_curve(clf_fit, X_test, y_test,
                    name='ROC fold {}'.format(n_fold),
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
            label= 'Mean ROC\n' + r'(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.grid(True, axis='y', alpha=0.3, which='both')
    
    #ax.set_ylabel("True Positive Rate", fontsize=14)
    #ax.set_xlabel("False Positive Rate", fontsize=14)
    #ax.set_title("SVM (RBF)", fontsize=14)
    
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-3:], labels[-3:],loc="lower right")
        
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results3\\ROC Curves\\roc_curves_RBF.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def final_evaluation(dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test, sensors, classifier_type, selection_type, n_features, optimal_threshold):
    
    df_results = pd.DataFrame()
    
    X_train, y_train, X_test, y_test = get_elements(dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test)
    X_train, X_test = select_features(X_train, y_train, X_test, sensors, selection_type, n_features)
    
    if(optimal_threshold == None):
        y_pred, _ = train_predict(classifier_type, X_train, y_train, X_test, 'both')
    else:
        y_scores = train_predict(classifier_type, X_train, y_train, X_test, 'scores')
        y_pred = (y_scores >= optimal_threshold).astype(bool)
    
    #df_y = pd.DataFrame()
    #df_y['pred'] = y_pred
    #df_y['test'] = y_test
    #print(df_y[df_y['test']==1])
        
    cnf_matrix = confusion_matrix(y_test, y_pred, [0,1])
    TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix)
    df_results = update_results(df_results, TN, FP, FN, TP, None)        
    #plot_confusion_matrix(cnf_matrix, [0,1])
    return df_results


def select_features(X_train, y_train, X_test, sensors_init_aux, selection_type, n_features):
    if selection_type == 'mine':
        combos_str = our_feature_ranking_method(X_train, y_train, sensors_init_aux, n_features)
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

def get_combo_name(combo):
    return str(combo[0]) + "-" + str(combo[1])

def our_feature_ranking_method(X_train, y_train, sensors_init_aux, n_features):
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
    return features

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
    
def get_initial_results(path_init):
    sensors = [1, 2, 6, 9, 10, 12, 14, 3, 7]
    width = 120
    correlation_types = ['DCCA','Pearson']
    classifier_types = ["GaussianNB","LinearSVC","NuSVC-rbf"]
    selection_type = None
    n_features = None
    frac = None
    optimal_threshold = None
    
    for classifier_type in classifier_types:
        print("-> " + classifier_type)
        for correlation_type in correlation_types:
            print("---> " + correlation_type)
            df = get_dataset(path_init, correlation_type, width)
            dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test = get_train_test(df, frac)
            df_results = cross_validation(dfs_neg_train, dfs_pos_train, sensors, classifier_type, selection_type, n_features, optimal_threshold)
            mean_results = get_mean_results(df_results)
            #print(df_results)
            print(mean_results) 

def get_main_results(path_init):
    sensors = [1, 2, 6, 9, 10, 12, 14, 3, 7]
    width = 120
    correlation_type = 'Pearson'
    classifier_type = 'GaussianNB'
    selection_type = None #'mine'
    n_features = None #19
    frac = 0.01
    optimal_threshold = None #-0.8
    
    print("---> " + correlation_type)
    df = get_dataset(path_init, correlation_type, width)
    
    #columns = ['1-14', '12-7', '6-7', '12-3', '10-14', '9-12', '2-6', '6-12', '3-7', '1-12', '2-3', '1-9', '10-12', '6-3', '12-14', '1-3', '9-3', '2-12', '9-10','y']
    #df = df.loc[:,columns]
    
    dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test = get_train_test(df, frac)
    df_results = cross_validation(dfs_neg_train, dfs_pos_train, sensors, classifier_type, selection_type, n_features, optimal_threshold)
    mean_results = get_mean_results(df_results)
    print(df_results)
    print(mean_results)

def get_final_results(path_init):
    sensors = [1, 2, 6, 9, 10, 12, 14, 3, 7]
    width = 120
    correlation_type = 'Pearson'
    classifier_type = "GaussianNB" #"GaussianNB","LinearSVC","NuSVC-rbf"
    selection_type = None #'mine'
    n_features = 19
    frac = 0.01
    optimal_threshold = None #-0.52
    
    print("---> " + correlation_type)
    df = get_dataset(path_init, correlation_type, width)
    dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test = get_train_test(df, frac)
    df_results = final_evaluation(dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test, sensors, classifier_type, selection_type, n_features, optimal_threshold)
    results = round(df_results.describe().loc['mean',:],2).to_dict()
    print(results)

def execute_feature_selection(path_init):
    
    sensors = [1, 2, 6, 9, 10, 12, 14, 3, 7]
    width = 120
    frac = 0.01
    correlation_type = 'Pearson'
    optimal_threshold = None
    
    classifier_types = ['NuSVC-rbf'] #'GaussianNB','LinearSVC','NuSVC-rbf'
    selection_types = ['mine','kruskalwallis']
    
    for classifier_type in classifier_types:
        print("-> " + classifier_type)
        
        for selection_type in selection_types:
            print("-----> " + selection_type)
            results_fs = pd.DataFrame()
            for n_features in range(15, 37, 1):
                print(n_features)
                df = get_dataset(path_init, correlation_type, width)
                dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test = get_train_test(df, frac)
                df_results = cross_validation(dfs_neg_train, dfs_pos_train, sensors, classifier_type, selection_type, n_features, optimal_threshold)
                mean_results = get_mean_results(df_results)
                #print(df_results)
                #print(mean_results)
                mean_results['x'] = n_features
                results_fs = results_fs.append(mean_results, ignore_index=True)
            path_export = path_init + '\\Results\\Real Dataset\\r_feature_extraction_evolution_' + classifier_type + '_' + selection_type + '.csv'
            results_fs.to_csv(index=True, path_or_buf=path_export, sep=';', decimal=',')
    
def plot_feature_selection(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    color4 = 'tab:brown'
    color5 = 'tab:red'
    
    classifier_types = ['GaussianNB','LinearSVC','NuSVC-rbf'] #'GaussianNB','LinearSVC','NuSVC-rbf'
    selection_types = ['mine','kruskalwallis']
    
    for classifier_type in classifier_types:
        print("-> " + classifier_type)
        
        i = 0
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11,5.5), sharey=True, sharex=True)
        axs_aux = axs.flat
        for selection_type in selection_types:
            print("-----> " + selection_type)
            
            path = path_init + '\\Results\\Real Dataset\\r_feature_extraction_evolution_' + classifier_type + '_' + selection_type + '.csv'
            df = pd.read_csv(path, index_col=0, sep=';', decimal=',')

            ax = axs_aux[i]
            
            ax.plot(df['x'], df['TNR'], color=color4, label = r'$Recall_{N}$')
            ax.plot(df['x'], df['precision'], color=color1, label=r'$Precision_{P}$')
            ax.plot(df['x'], df['recall'], color=color2, label=r'$Recall_{P}$')
            ax.plot(df['x'], df['F1'], color=color3, label = r'$F_{P}$')
            ax.plot(df['x'], df['leakage'], color=color5, label = 'L')
            
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            
            ax.set_xlabel('Top k Features')
            
            bottom_x = 0
            x = 36
            if classifier_type == "GaussianNB":
                if i==0:
                    ax.set_title('Our Feature Ranking Method')
                    ax.legend(loc='lower left', ncol=3)
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.2, 0.83+0.01,'0.83', ha="right", va="bottom", color=color4)
                    ax.text(x-0.2, 0.39+0.02,'0.39', ha="right", va="bottom", color=color1)
                    ax.text(x-0.2, 0.36-0.02,'0.36', ha="right", va="top", color=color2)
                    ax.text(x-0.2, 0.35-0.05,'0.35', ha="right", va="top", color=color3)
                    ax.text(x-0.2, 0.91+0.01,'0.91', ha="right", va="bottom", color=color5)
                    ax.text(x-0.2, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
                elif i==1:
                    ax.set_title('Kruskal Wallis H test')
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.2, 0.83-0.02,'0.83', ha="right", va="top", color=color4)
                    ax.text(x-0.2, 0.39+0.01,'0.39', ha="right", va="bottom", color=color1)
                    ax.text(x-0.2, 0.36-0.02,'0.36', ha="right", va="top", color=color2)
                    ax.text(x-0.2, 0.35-0.05,'0.35', ha="right", va="top", color=color3)
                    ax.text(x-0.2, 0.91+0.01,'0.91', ha="right", va="bottom", color=color5)
                    ax.text(x-0.2, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
            elif classifier_type == "LinearSVC":
                if i==0:
                    ax.set_title('Our Feature Ranking Method')
                    ax.legend(loc='lower left', ncol=3)
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.2, 0.89+0.01,'0.89', ha="right", va="bottom", color=color4)
                    ax.text(x-0.2, 0.29+0.01,'0.29', ha="right", va="bottom", color=color1)
                    ax.text(x-0.2, 0.27-0.015,'0.27', ha="right", va="top", color=color2)
                    ax.text(x-0.2, 0.27-0.06,'0.27', ha="right", va="top", color=color3)
                    ax.text(x-0.2, 0.64+0.01,'0.64', ha="right", va="bottom", color=color5)
                    ax.text(x-0.2, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                elif i==1:
                    ax.set_title('Kruskal Wallis H test')
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.2, 0.89+0.01,'0.89', ha="right", va="bottom", color=color4)
                    ax.text(x-0.2, 0.29+0.01,'0.29', ha="right", va="bottom", color=color1)
                    ax.text(x-0.2, 0.27-0.035,'0.27', ha="right", va="top", color=color2)
                    ax.text(x-0.2, 0.27-0.080,'0.27', ha="right", va="top", color=color3)
                    ax.text(x-0.2, 0.64+0.01,'0.64', ha="right", va="bottom", color=color5)
                    ax.text(x-0.2, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
            elif classifier_type == 'NuSVC-rbf':
                if i==0:
                    ax.set_title('Our Feature Ranking Method')
                    ax.legend(loc='lower left', ncol=3)
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.2, 0.60-0.03,'0.60', ha="right", va="top", color=color4)
                    ax.text(x-0.2, 0.22-0.02,'0.22', ha="right", va="top", color=color1)
                    ax.text(x-0.2, 0.37+0.01,'0.37', ha="right", va="bottom", color=color2)
                    ax.text(x-0.2, 0.27+0.01,'0.27', ha="right", va="bottom", color=color3)
                    ax.text(x-0.2, 0.64+0.01,'0.64', ha="right", va="bottom", color=color5)
                    ax.text(x-0.2, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                elif i==1:
                    ax.set_title('Kruskal Wallis H test')
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.2, 0.60-0.03,'0.60', ha="right", va="top", color=color4)
                    ax.text(x-0.2, 0.22-0.02,'0.22', ha="right", va="top", color=color1)
                    ax.text(x-0.2, 0.37+0.01,'0.37', ha="right", va="bottom", color=color2)
                    ax.text(x-0.2, 0.27+0.01,'0.27', ha="right", va="bottom", color=color3)
                    ax.text(x-0.2, 0.64+0.01,'0.64', ha="right", va="bottom", color=color5)
                    ax.text(x-0.2, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
            
            ax.grid(True, axis='y', alpha=0.3, which='both') 
            i+=1
           
        plt.ylim(-0.03, 1.03)
        fig.tight_layout()
        #plt.savefig(path_init + '\\Images\\Results3\\Feature Selection\\' + classifier_type + '.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        
def plot_feature_selection_ea(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    color4 = 'tab:brown'
    color5 = 'tab:red'
    
    classifier_types = ['GaussianNB'] #'GaussianNB','LinearSVC','NuSVC-rbf'
    selection_types = ['mine']
    
    for classifier_type in classifier_types:
        print("-> " + classifier_type)
        
        for selection_type in selection_types:
            
            
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4.7))
            
            print("-----> " + selection_type)
            
            path = path_init + '\\Results\\Real Dataset\\r_feature_extraction_evolution_' + classifier_type + '_' + selection_type + '.csv'
            df = pd.read_csv(path, index_col=0, sep=';', decimal=',')

            ax.plot(df['x'], df['TNR'], color=color4, label = r'$Rec_{N}$')
            ax.plot(df['x'], df['precision'], color=color1, label=r'$Prec_{P}$')
            ax.plot(df['x'], df['recall'], color=color2, label=r'$Rec_{P}$')
            ax.plot(df['x'], df['F1'], color=color3, label = r'$F_{P}$')
            ax.plot(df['x'], df['leakage'], color=color5, label = 'L')
            
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            
            ax.set_xlabel('Top k Features', fontsize=14)
            
            bottom_x = 0
            x = 36
            if classifier_type == "GaussianNB":
                if selection_type == "mine":
                    ax.set_title('Naive Bayes w/\nOur Feature Ranking Method', fontsize=14)
                    ax.legend(loc='lower left', ncol=3, fontsize=12)
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.2, 0.83+0.01,'0.83', ha="right", va="bottom", color=color4)
                    ax.text(x-0.2, 0.39+0.02,'0.39', ha="right", va="bottom", color=color1)
                    ax.text(x-0.2, 0.36-0.02,'0.36', ha="right", va="top", color=color2)
                    ax.text(x-0.2, 0.35-0.05,'0.35', ha="right", va="top", color=color3)
                    ax.text(x-0.2, 0.91+0.01,'0.91', ha="right", va="bottom", color=color5)
                    ax.text(x-0.2, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
                else:
                    ax.set_title('Kruskal Wallis H test')
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.2, 0.83-0.02,'0.83', ha="right", va="top", color=color4)
                    ax.text(x-0.2, 0.39+0.01,'0.39', ha="right", va="bottom", color=color1)
                    ax.text(x-0.2, 0.36-0.02,'0.36', ha="right", va="top", color=color2)
                    ax.text(x-0.2, 0.35-0.05,'0.35', ha="right", va="top", color=color3)
                    ax.text(x-0.2, 0.91+0.01,'0.91', ha="right", va="bottom", color=color5)
                    ax.text(x-0.2, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
            elif classifier_type == "LinearSVC":
                if selection_type == "mine":
                    ax.set_title('Our Feature Ranking Method')
                    ax.legend(loc='lower left', ncol=3)
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.2, 0.89+0.01,'0.89', ha="right", va="bottom", color=color4)
                    ax.text(x-0.2, 0.29+0.01,'0.29', ha="right", va="bottom", color=color1)
                    ax.text(x-0.2, 0.27-0.015,'0.27', ha="right", va="top", color=color2)
                    ax.text(x-0.2, 0.27-0.06,'0.27', ha="right", va="top", color=color3)
                    ax.text(x-0.2, 0.64+0.01,'0.64', ha="right", va="bottom", color=color5)
                    ax.text(x-0.2, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                else:
                    ax.set_title('Kruskal Wallis H test')
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.2, 0.89+0.01,'0.89', ha="right", va="bottom", color=color4)
                    ax.text(x-0.2, 0.29+0.01,'0.29', ha="right", va="bottom", color=color1)
                    ax.text(x-0.2, 0.27-0.035,'0.27', ha="right", va="top", color=color2)
                    ax.text(x-0.2, 0.27-0.080,'0.27', ha="right", va="top", color=color3)
                    ax.text(x-0.2, 0.64+0.01,'0.64', ha="right", va="bottom", color=color5)
                    ax.text(x-0.2, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
            elif classifier_type == 'NuSVC-rbf':
                if selection_type == "mine":
                    ax.set_title('Our Feature Ranking Method')
                    ax.legend(loc='lower left', ncol=3)
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.2, 0.60-0.03,'0.60', ha="right", va="top", color=color4)
                    ax.text(x-0.2, 0.22-0.02,'0.22', ha="right", va="top", color=color1)
                    ax.text(x-0.2, 0.37+0.01,'0.37', ha="right", va="bottom", color=color2)
                    ax.text(x-0.2, 0.27+0.01,'0.27', ha="right", va="bottom", color=color3)
                    ax.text(x-0.2, 0.64+0.01,'0.64', ha="right", va="bottom", color=color5)
                    ax.text(x-0.2, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                else:
                    ax.set_title('Kruskal Wallis H test')
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.2, 0.60-0.03,'0.60', ha="right", va="top", color=color4)
                    ax.text(x-0.2, 0.22-0.02,'0.22', ha="right", va="top", color=color1)
                    ax.text(x-0.2, 0.37+0.01,'0.37', ha="right", va="bottom", color=color2)
                    ax.text(x-0.2, 0.27+0.01,'0.27', ha="right", va="bottom", color=color3)
                    ax.text(x-0.2, 0.64+0.01,'0.64', ha="right", va="bottom", color=color5)
                    ax.text(x-0.2, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
            
            ax.grid(True, axis='y', alpha=0.3, which='both') 
           
            plt.ylim(-0.07, 1.03)
            fig.tight_layout()
            plt.savefig(path_init + '\\Images\\Results3\\Feature Selection\\' + classifier_type + '_ea.png', format='png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()        
        
        
def plot_feature_selection_1(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    color4 = 'tab:brown'
    color5 = 'tab:red'
    
    classifier_types = ['NuSVC-rbf']
    selection_types = ['mine','kruskalwallis']
    
    for classifier_type in classifier_types:
        print("-> " + classifier_type)
        
        i = 0
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11,5.5), sharey=True, sharex=True)
        axs_aux = axs.flat
        for selection_type in selection_types:
            print("-----> " + selection_type)
            
            path = path_init + '\\Results\\Real Dataset\\r_feature_extraction_evolution_' + classifier_type + '_' + selection_type + '_1.csv'
            df = pd.read_csv(path, index_col=0, sep=';', decimal=',')

            ax = axs_aux[i]
            
            ax.plot(df['x'], df['TNR'], color=color4, label = r'$Recall_{N}$')
            ax.plot(df['x'], df['precision'], color=color1, label=r'$Precision_{P}$')
            ax.plot(df['x'], df['recall'], color=color2, label=r'$Recall_{P}$')
            ax.plot(df['x'], df['F1'], color=color3, label = r'$F_{P}$')
            ax.plot(df['x'], df['leakage'], color=color5, label = 'L')
            
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            
            ax.set_xlabel('Top k Features')
            
            bottom_x = -0.035
            if i==0:
                ax.set_title('Our Feature Ranking Method')
                ax.legend(loc='upper right', ncol=3, bbox_to_anchor=(0.925, 1))
                x = 19
                ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                ax.text(x-0.2, 0.72+0.03,'0.72', ha="right", va="top", color=color4)
                ax.text(x-0.2, 0.01+0.02,'0.01', ha="right", va="bottom", color=color1)
                ax.text(x-0.2, 0.50+0.005,'0.50', ha="right", va="bottom", color=color2)
                ax.text(x-0.2, 0.02+0.05,'0.02', ha="right", va="bottom", color=color3)
                ax.text(x-0.2, 0.82+0.005,'0.82', ha="right", va="bottom", color=color5)
                ax.text(x-0.2, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                
                x = 36
                ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                ax.text(x-0.2, 0.68-0.01,'0.68', ha="right", va="top", color=color4)
                ax.text(x-0.2, 0.00+0.02,'0.00', ha="right", va="bottom", color=color1)
                ax.text(x-0.2, 0.33+0.005,'0.33', ha="right", va="bottom", color=color2)
                ax.text(x-0.2, 0.01+0.05,'0.01', ha="right", va="bottom", color=color3)
                ax.text(x-0.2, 0.55-0.005,'0.55', ha="right", va="top", color=color5)
                ax.text(x-0.2, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
            elif i==1:
                ax.set_title('Kruskal Wallis H test')
                x = 36
                ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                ax.text(x-0.2, 0.68+0.01,'0.68', ha="right", va="bottom", color=color4)
                ax.text(x-0.2, 0.00+0.02,'0.00', ha="right", va="bottom", color=color1)
                ax.text(x-0.2, 0.33+0.04,'0.33', ha="right", va="bottom", color=color2)
                ax.text(x-0.2, 0.01+0.05,'0.01', ha="right", va="bottom", color=color3)
                ax.text(x-0.2, 0.55+0.01,'0.55', ha="right", va="bottom", color=color5)
                ax.text(x-0.2, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
            
            ax.grid(True, axis='y', alpha=0.3, which='both') 
            i+=1
           
        plt.ylim(-0.07, 1.03)
        fig.tight_layout()
        plt.savefig(path_init + '\\Images\\Results3\\Feature Selection\\' + classifier_type + '_2.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def plot_feature_selection_1_ea(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    color4 = 'tab:brown'
    color5 = 'tab:red'
    
    classifier_types = ['NuSVC-rbf']
    selection_types = ['mine']
    
    for classifier_type in classifier_types:
        print("-> " + classifier_type)
        
        for selection_type in selection_types:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4.7))
            
            print("-----> " + selection_type)
            
            path = path_init + '\\Results\\Real Dataset\\r_feature_extraction_evolution_' + classifier_type + '_' + selection_type + '_1.csv'
            df = pd.read_csv(path, index_col=0, sep=';', decimal=',')

            ax.plot(df['x'], df['TNR'], color=color4, label = r'$Rec_{N}$')
            ax.plot(df['x'], df['precision'], color=color1, label=r'$Prec_{P}$')
            ax.plot(df['x'], df['recall'], color=color2, label=r'$Rec_{P}$')
            ax.plot(df['x'], df['F1'], color=color3, label = r'$F_{P}$')
            ax.plot(df['x'], df['leakage'], color=color5, label = 'L')
            
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            
            ax.set_xlabel('Top k Features', fontsize=14)
            
            bottom_x = -0.035
            if selection_type == "mine":
                ax.set_title('SVM (RBF) w/\nOur Feature Ranking Method', fontsize=14)
                ax.legend(loc='upper left', ncol=3, fontsize=9, bbox_to_anchor=(0.22, 1))
                x = 19
                ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                ax.text(x-0.2, 0.72+0.03,'0.72', ha="right", va="top", color=color4)
                ax.text(x-0.2, 0.01+0.02,'0.01', ha="right", va="bottom", color=color1)
                ax.text(x-0.2, 0.50+0.005,'0.50', ha="right", va="bottom", color=color2)
                ax.text(x-0.2, 0.02+0.05,'0.02', ha="right", va="bottom", color=color3)
                ax.text(x-0.2, 0.82+0.005,'0.82', ha="right", va="bottom", color=color5)
                ax.text(x-0.2, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
                
                x = 36
                ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                ax.text(x-0.2, 0.68-0.01,'0.68', ha="right", va="top", color=color4)
                ax.text(x-0.2, 0.00+0.02,'0.00', ha="right", va="bottom", color=color1)
                ax.text(x-0.2, 0.33+0.005,'0.33', ha="right", va="bottom", color=color2)
                ax.text(x-0.2, 0.01+0.05,'0.01', ha="right", va="bottom", color=color3)
                ax.text(x-0.2, 0.55-0.005,'0.55', ha="right", va="top", color=color5)
                ax.text(x-0.2, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
            else:
                ax.set_title('Kruskal Wallis H test')
                x = 36
                ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                ax.text(x-0.2, 0.68+0.01,'0.68', ha="right", va="bottom", color=color4)
                ax.text(x-0.2, 0.00+0.02,'0.00', ha="right", va="bottom", color=color1)
                ax.text(x-0.2, 0.33+0.04,'0.33', ha="right", va="bottom", color=color2)
                ax.text(x-0.2, 0.01+0.05,'0.01', ha="right", va="bottom", color=color3)
                ax.text(x-0.2, 0.55+0.01,'0.55', ha="right", va="bottom", color=color5)
                ax.text(x-0.2, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
            
            ax.grid(True, axis='y', alpha=0.3, which='both') 
           
        plt.ylim(-0.07, 1.03)
        fig.tight_layout()
        plt.savefig(path_init + '\\Images\\Results3\\Feature Selection\\' + classifier_type + '_2_ea.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


def execute_time_window_sizes(path_init):
    
    sensors = [1, 2, 6, 9, 10, 12, 14, 3, 7]
 
    classifier_types = ["NuSVC-rbf"] #"GaussianNB" "LinearSVC" 'NuSVC-rbf'
    widths = [60, 120, 180, 240]
    frac = None #0.01
    optimal_threshold = None
    selection_type = 'mine'
    n_features = 19
    correlation_type ='Pearson'
    
    for classifier_type in classifier_types:
        print("-> " + classifier_type)
        
        for width in widths:
            print("-----> " + str(width))

            df = get_dataset(path_init, correlation_type, width)
            dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test = get_train_test(df, frac)
            df_results = cross_validation(dfs_neg_train, dfs_pos_train, sensors, classifier_type, selection_type, n_features, optimal_threshold)
            mean_results = get_mean_results(df_results)
            
            print(mean_results)
        
        print("\n")

def count_intances(dfs):
    count = 0
    for df in dfs:
        count += len(df)
    return count

def negative_set_reduction(path_init):
    sensors = [1, 2, 6, 9, 10, 12, 14, 3, 7]
    width = 120
    correlation_type = 'DCCA'
    classifier_type = 'NuSVC-rbf' #NuSVC-rbf GaussianNB LinearSVC
    selection_type = None
    n_features = None
    optimal_threshold = None
    
    fracs = []
    start = round(0.005,4) #0.005
    while start <= 0.1:
        start = round(start,4)
        fracs.append(start)
        start += 0.001
    
    df_results_aux = pd.DataFrame()
    for frac in fracs:
        
        print(frac)
        df = get_dataset(path_init, correlation_type, width)
        dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test = get_train_test(df, frac)
        
        try :
            df_results = cross_validation(dfs_neg_train, dfs_pos_train, sensors, classifier_type, selection_type, n_features, optimal_threshold)
            mean_results = get_mean_results(df_results)
            #print(df_results)
            #print(mean_results)
            mean_results['frac'] = frac
            mean_results['neg'] = count_intances(dfs_neg_train)
            mean_results['pos'] = count_intances(dfs_pos_train)
            #print(mean_results)
            df_results_aux = df_results_aux.append(mean_results, ignore_index=True)
        except ValueError:
            path_export = path_init + '\\Results\\Real Dataset\\set_reduction_' + classifier_type + '_' + correlation_type + '.csv'
            df_results_aux.to_csv(index=True, path_or_buf=path_export, sep=';', decimal=',')
            raise
    
    path_export = path_init + '\\Results\\Real Dataset\\set_reduction_' + classifier_type + '_' + correlation_type + '.csv'
    df_results_aux.to_csv(index=True, path_or_buf=path_export, sep=';', decimal=',')
    
def plot_dataset_reduction(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    color4 = 'tab:brown'
    color5 = 'tab:red'
    
    correlation_types = ['DCCA','Pearson']
    classifier_types = ['GaussianNB','LinearSVC','NuSVC-rbf'] #'GaussianNB','LinearSVC','NuSVC-rbf'
    
    for classifier_type in classifier_types:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6), sharey=True, sharex=True)
        for i, ax in enumerate(axs):
            correlation_type = correlation_types[i]
            
            path_import = path_init + '\\Results\\Real Dataset\\set_reduction_' + classifier_type + '_' + correlation_type + '.csv'
            df = pd.read_csv(path_import, index_col=0, sep=';', decimal=',')
            
            ax.plot(df['frac']*100, df['TNR'], color=color4, label = r'$Recall_{N}$')
            ax.plot(df['frac']*100, df['precision'], color=color1, label=r'$Precision_{P}$')
            ax.plot(df['frac']*100, df['recall'], color=color2, label=r'$Recall_{P}$')
            ax.plot(df['frac']*100, df['F1'], color=color3, label = r'$F_{P}$')
            ax.plot(df['frac']*100, df['leakage'], color=color5, label = "L")
              
            bottom_x = 0
            if classifier_type == 'GaussianNB':
                if i == 0:
                    ax.set_title('DCCA')
                    x = 0.5
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.1, 0.66+0.005,'0.66', ha="right", va="bottom", color=color4)
                    ax.text(x-0.1, 0.36+0.045,'0.36', ha="right", va="bottom", color=color1)
                    ax.text(x-0.1, 0.32+0.005,'0.32', ha="right", va="bottom", color=color2)
                    ax.text(x-0.1, 0.33+0.035,'0.33', ha="right", va="bottom", color=color3)
                    ax.text(x-0.1, 0.64-0.01,'0.64', ha="right", va="bottom", color=color5)
                    ax.text(x-0.1, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
                    
                    x = 1
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+0.1, 0.77-0.03,'0.77', ha="left", va="bottom", color=color4)
                    ax.text(x+0.1, 0.30+0.07,'0.30', ha="left", va="bottom", color=color1)
                    ax.text(x+0.1, 0.26+0.034,'0.26', ha="left", va="bottom", color=color2)
                    ax.text(x+0.1, 0.27+0.06,'0.27', ha="left", va="bottom", color=color3)
                    ax.text(x+0.1, 0.73-0.04,'0.73', ha="left", va="bottom", color=color5)
                    ax.text(x+0.1, bottom_x,str(x)+'%', ha="left", va="center", color='k', alpha=0.5)
                    ax.legend(loc='lower right',bbox_to_anchor=(0.98, 0.135))
                else:
                    ax.set_title('PCC')
                    x = 0.5
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.1, 0.73,'0.73', ha="right", va="center", color=color4)
                    ax.text(x-0.1, 0.53,'0.53', ha="right", va="center", color=color1)
                    ax.text(x-0.1, 0.46+0.005,'0.46', ha="right", va="bottom", color=color2)
                    ax.text(x-0.1, 0.46-0.035,'0.46', ha="right", va="bottom", color=color3)
                    ax.text(x-0.1, 0.91,'0.91', ha="right", va="center", color=color5)
                    ax.text(x-0.1, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
                    
                    x = 1
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+0.1, 0.83+0.03,'0.83', ha="left", va="bottom", color=color4)
                    ax.text(x+0.1, 0.39+0.06,'0.39', ha="left", va="bottom", color=color1)
                    ax.text(x+0.1, 0.36+0.05,'0.36', ha="left", va="bottom", color=color2)
                    ax.text(x+0.1, 0.35+0.02,'0.35', ha="left", va="bottom", color=color3)
                    ax.text(x+0.1, 0.91,'0.91', ha="left", va="center", color=color5)
                    ax.text(x+0.1, bottom_x,str(x)+'%', ha="left", va="center", color='k', alpha=0.5)
            elif classifier_type == 'LinearSVC':
                if i == 0:
                    ax.set_title('DCCA')
                    x = 0.5
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.1, 0.76+0.005,'0.76', ha="right", va="bottom", color=color4)
                    ax.text(x-0.1, 0.40,'0.40', ha="right", va="center", color=color1)
                    ax.text(x-0.1, 0.26-0.005,'0.26', ha="right", va="bottom", color=color2)
                    ax.text(x-0.1, 0.29+0.005,'0.29', ha="right", va="bottom", color=color3)
                    ax.text(x-0.1, 0.73-0.015,'0.73', ha="right", va="bottom", color=color5)
                    ax.text(x-0.1, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
                    
                    x = 1
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+0.1, 0.86-0.03,'0.86', ha="left", va="bottom", color=color4)
                    ax.text(x+0.2, 0.19+0.07,'0.19', ha="left", va="bottom", color=color1)
                    ax.text(x+0.2, 0.15+0.034,'0.15', ha="left", va="bottom", color=color2)
                    ax.text(x+0.2, 0.15+0.07,'0.15', ha="left", va="bottom", color=color3)
                    ax.text(x+0.1, 0.36+0.005,'0.36', ha="left", va="bottom", color=color5)
                    ax.text(x+0.1, bottom_x,str(x)+'%', ha="left", va="center", color='k', alpha=0.5)
                    ax.legend(loc='lower right',bbox_to_anchor=(0.98, 0.05))
                else:
                    ax.set_title('PCC')
                    x = 0.5
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.1, 0.76,'0.76', ha="right", va="center", color=color4)
                    ax.text(x-0.1, 0.37,'0.37', ha="right", va="center", color=color1)
                    ax.text(x-0.1, 0.37+0.025,'0.37', ha="right", va="bottom", color=color2)
                    ax.text(x-0.1, 0.35-0.035,'0.35', ha="right", va="bottom", color=color3)
                    ax.text(x-0.1, 0.64,'0.64', ha="right", va="center", color=color5)
                    ax.text(x-0.1, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
                    
                    x = 1
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+0.1, 0.89,'0.89', ha="left", va="center", color=color4)
                    ax.text(x+0.3, 0.29+0.03,'0.29', ha="left", va="bottom", color=color1)
                    ax.text(x+0.3, 0.27-0.01,'0.27', ha="left", va="bottom", color=color2)
                    ax.text(x+0.3, 0.27+0.02,'0.27', ha="left", va="bottom", color=color3)
                    ax.text(x+0.3, 0.64,'0.64', ha="left", va="center", color=color5)
                    ax.text(x+0.1, bottom_x,str(x)+'%', ha="left", va="center", color='k', alpha=0.5)
            else:
                if i == 0:
                    ax.set_title('DCCA')
                    x = 0.5
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.1, 0.64+0.005,'0.64', ha="right", va="bottom", color=color4)
                    ax.text(x-0.1, 0.33-0.02,'0.33', ha="right", va="bottom", color=color1)
                    ax.text(x-0.1, 0.34+0.01,'0.34', ha="right", va="bottom", color=color2)
                    ax.text(x-0.1, 0.31-0.04,'0.31', ha="right", va="bottom", color=color3)
                    ax.text(x-0.1, 0.73+0.01,'0.73', ha="right", va="bottom", color=color5)
                    ax.text(x-0.1, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
                    
                    x = 1
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+0.1, 0.60+0.015,'0.60', ha="left", va="bottom", color=color4)
                    ax.text(x+0.1, 0.21-0.008,'0.21', ha="left", va="top", color=color1)
                    ax.text(x+0.1, 0.34+0.01,'0.34', ha="left", va="bottom", color=color2)
                    ax.text(x+0.1, 0.25,'0.25', ha="left", va="bottom", color=color3)
                    ax.text(x+0.1, 0.73+0.01,'0.73', ha="left", va="bottom", color=color5)
                    ax.text(x+0.1, bottom_x,str(x)+'%', ha="left", va="center", color='k', alpha=0.5)
                    ax.legend(loc='lower right',ncol=1,bbox_to_anchor=(0.98, 0.39) )
                else:
                    ax.set_title('PCC')
                    x = 0.5
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.1, 0.59,'0.59', ha="right", va="center", color=color4)
                    ax.text(x-0.1, 0.32-0.015,'0.32', ha="right", va="center", color=color1)
                    ax.text(x-0.1, 0.37+0.01,'0.37', ha="right", va="bottom", color=color2)
                    ax.text(x-0.1, 0.33,'0.33', ha="right", va="bottom", color=color3)
                    ax.text(x-0.1, 0.64+0.01,'0.64', ha="right", va="bottom", color=color5)
                    ax.text(x-0.1, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
                    
                    x = 1
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+0.1, 0.60-0.005,'0.60', ha="left", va="top", color=color4)
                    ax.text(x+0.1, 0.22-0.002,'0.22', ha="left", va="top", color=color1)
                    ax.text(x+0.1, 0.37+0.01,'0.37', ha="left", va="bottom", color=color2)
                    ax.text(x+0.1, 0.27,'0.27', ha="left", va="bottom", color=color3)
                    ax.text(x+0.1, 0.64+0.01,'0.64', ha="left", va="bottom", color=color5)
                    ax.text(x+0.1, bottom_x,str(x)+'%', ha="left", va="center", color='k', alpha=0.5)
            
            ax.set_xlabel('Percentage of Negative Instances')
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.grid(True, axis='y', alpha=0.3, which='both')
            
        plt.ylim(-0.03, 1.03) 
        plt.xlim(-0.4, 10.3) 
        fig.tight_layout()
        plt.savefig(path_init + '\\Images\\Results3\\Dataset Reduction\\' + classifier_type + '.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def plot_dataset_reduction_ea(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    color4 = 'tab:brown'
    color5 = 'tab:red'
    
    for classifier_type in ['GaussianNB','LinearSVC']:
        
        for correlation_type in ['Pearson']:
            
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4.5))
            
            path_import = path_init + '\\Results\\Real Dataset\\set_reduction_' + classifier_type + '_' + correlation_type + '.csv'
            df = pd.read_csv(path_import, index_col=0, sep=';', decimal=',')
            
            ax.plot(df['frac']*100, df['TNR'], color=color4, label = r'$Rec_{N}$')
            ax.plot(df['frac']*100, df['precision'], color=color1, label=r'$Prec_{P}$')
            ax.plot(df['frac']*100, df['recall'], color=color2, label=r'$Rec_{P}$')
            ax.plot(df['frac']*100, df['F1'], color=color3, label = r'$F_{P}$')
            ax.plot(df['frac']*100, df['leakage'], color=color5, label = "L")
              
            bottom_x = 0
            if classifier_type == 'GaussianNB':
                if correlation_type == 'DCCA':
                    ax.set_title('DCCA')
                    x = 0.5
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.1, 0.66+0.005,'0.66', ha="right", va="bottom", color=color4)
                    ax.text(x-0.1, 0.36+0.045,'0.36', ha="right", va="bottom", color=color1)
                    ax.text(x-0.1, 0.32+0.005,'0.32', ha="right", va="bottom", color=color2)
                    ax.text(x-0.1, 0.33+0.035,'0.33', ha="right", va="bottom", color=color3)
                    ax.text(x-0.1, 0.64-0.01,'0.64', ha="right", va="bottom", color=color5)
                    ax.text(x-0.1, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
                    
                    x = 1
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+0.1, 0.77-0.03,'0.77', ha="left", va="bottom", color=color4)
                    ax.text(x+0.1, 0.30+0.07,'0.30', ha="left", va="bottom", color=color1)
                    ax.text(x+0.1, 0.26+0.034,'0.26', ha="left", va="bottom", color=color2)
                    ax.text(x+0.1, 0.27+0.06,'0.27', ha="left", va="bottom", color=color3)
                    ax.text(x+0.1, 0.73-0.04,'0.73', ha="left", va="bottom", color=color5)
                    ax.text(x+0.1, bottom_x,str(x)+'%', ha="left", va="center", color='k', alpha=0.5)
                    ax.legend(loc='lower right',bbox_to_anchor=(0.98, 0.135))
                else:
                    ax.set_title('Naive Bayes w/ PCC', fontsize=14)
                    x = 0.5
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.1, 0.73,'0.73', ha="right", va="center", color=color4)
                    ax.text(x-0.1, 0.53,'0.53', ha="right", va="center", color=color1)
                    ax.text(x-0.1, 0.46+0.005,'0.46', ha="right", va="bottom", color=color2)
                    ax.text(x-0.1, 0.46-0.035,'0.46', ha="right", va="bottom", color=color3)
                    ax.text(x-0.1, 0.91,'0.91', ha="right", va="center", color=color5)
                    ax.text(x-0.04, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
                    
                    x = 1
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+0.1, 0.83+0.035,'0.83', ha="left", va="bottom", color=color4)
                    ax.text(x+0.1, 0.39+0.06,'0.39', ha="left", va="bottom", color=color1)
                    ax.text(x+0.1, 0.36+0.05,'0.36', ha="left", va="bottom", color=color2)
                    ax.text(x+0.1, 0.35+0.02,'0.35', ha="left", va="bottom", color=color3)
                    ax.text(x+0.1, 0.91+0.02,'0.91', ha="left", va="center", color=color5)
                    ax.text(x+0.1, bottom_x,str(x)+'%', ha="left", va="center", color='k', alpha=0.5)
                    ax.legend(loc='lower right',bbox_to_anchor=(0.98, 0.385), fontsize=12)
            elif classifier_type == 'LinearSVC':
                if correlation_type == 'DCCA':
                    ax.set_title('DCCA')
                    x = 0.5
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.1, 0.76+0.005,'0.76', ha="right", va="bottom", color=color4)
                    ax.text(x-0.1, 0.40,'0.40', ha="right", va="center", color=color1)
                    ax.text(x-0.1, 0.26-0.005,'0.26', ha="right", va="bottom", color=color2)
                    ax.text(x-0.1, 0.29+0.005,'0.29', ha="right", va="bottom", color=color3)
                    ax.text(x-0.1, 0.73-0.015,'0.73', ha="right", va="bottom", color=color5)
                    ax.text(x-0.1, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
                    
                    x = 1
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+0.1, 0.86-0.03,'0.86', ha="left", va="bottom", color=color4)
                    ax.text(x+0.2, 0.19+0.07,'0.19', ha="left", va="bottom", color=color1)
                    ax.text(x+0.2, 0.15+0.034,'0.15', ha="left", va="bottom", color=color2)
                    ax.text(x+0.2, 0.15+0.07,'0.15', ha="left", va="bottom", color=color3)
                    ax.text(x+0.1, 0.36+0.005,'0.36', ha="left", va="bottom", color=color5)
                    ax.text(x+0.1, bottom_x,str(x)+'%', ha="left", va="center", color='k', alpha=0.5)
                    ax.legend(loc='lower right',bbox_to_anchor=(0.98, 0.05))
                else:
                    ax.set_title('SVM (Linear) w/ PCC', fontsize=14)
                    x = 0.5
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.1, 0.76,'0.76', ha="right", va="center", color=color4)
                    ax.text(x-0.1, 0.37,'0.37', ha="right", va="center", color=color1)
                    ax.text(x-0.1, 0.37+0.025,'0.37', ha="right", va="bottom", color=color2)
                    ax.text(x-0.1, 0.35-0.045,'0.35', ha="right", va="bottom", color=color3)
                    ax.text(x-0.1, 0.64,'0.64', ha="right", va="center", color=color5)
                    ax.text(x-0.04, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
                    
                    x = 1
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+0.1, 0.89-0.01,'0.89', ha="left", va="center", color=color4)
                    ax.text(x+0.3, 0.29+0.02,'0.29', ha="left", va="bottom", color=color1)
                    ax.text(x+0.3, 0.27-0.04,'0.27', ha="left", va="bottom", color=color2)
                    ax.text(x+0.3, 0.27,'0.27', ha="left", va="bottom", color=color3)
                    ax.text(x+0.3, 0.64,'0.64', ha="left", va="center", color=color5)
                    ax.text(x+0.1, bottom_x,str(x)+'%', ha="left", va="center", color='k', alpha=0.5)
                    ax.legend(loc='lower right',bbox_to_anchor=(0.98, 0.05), fontsize=12)
            else:
                if correlation_type == 'DCCA':
                    ax.set_title('DCCA')
                    x = 0.5
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.1, 0.64+0.005,'0.64', ha="right", va="bottom", color=color4)
                    ax.text(x-0.1, 0.33-0.02,'0.33', ha="right", va="bottom", color=color1)
                    ax.text(x-0.1, 0.34+0.01,'0.34', ha="right", va="bottom", color=color2)
                    ax.text(x-0.1, 0.31-0.04,'0.31', ha="right", va="bottom", color=color3)
                    ax.text(x-0.1, 0.73+0.01,'0.73', ha="right", va="bottom", color=color5)
                    ax.text(x-0.1, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
                    
                    x = 1
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+0.1, 0.60+0.015,'0.60', ha="left", va="bottom", color=color4)
                    ax.text(x+0.1, 0.21-0.008,'0.21', ha="left", va="top", color=color1)
                    ax.text(x+0.1, 0.34+0.01,'0.34', ha="left", va="bottom", color=color2)
                    ax.text(x+0.1, 0.25,'0.25', ha="left", va="bottom", color=color3)
                    ax.text(x+0.1, 0.73+0.01,'0.73', ha="left", va="bottom", color=color5)
                    ax.text(x+0.1, bottom_x,str(x)+'%', ha="left", va="center", color='k', alpha=0.5)
                    ax.legend(loc='lower right',ncol=1,bbox_to_anchor=(0.98, 0.39) )
                else:
                    ax.set_title('PCC')
                    x = 0.5
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x-0.1, 0.59,'0.59', ha="right", va="center", color=color4)
                    ax.text(x-0.1, 0.32-0.015,'0.32', ha="right", va="center", color=color1)
                    ax.text(x-0.1, 0.37+0.01,'0.37', ha="right", va="bottom", color=color2)
                    ax.text(x-0.1, 0.33,'0.33', ha="right", va="bottom", color=color3)
                    ax.text(x-0.1, 0.64+0.01,'0.64', ha="right", va="bottom", color=color5)
                    ax.text(x-0.1, bottom_x,str(x)+'%', ha="right", va="center", color='k', alpha=0.5)
                    
                    x = 1
                    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
                    ax.text(x+0.1, 0.60-0.005,'0.60', ha="left", va="top", color=color4)
                    ax.text(x+0.1, 0.22-0.002,'0.22', ha="left", va="top", color=color1)
                    ax.text(x+0.1, 0.37+0.01,'0.37', ha="left", va="bottom", color=color2)
                    ax.text(x+0.1, 0.27,'0.27', ha="left", va="bottom", color=color3)
                    ax.text(x+0.1, 0.64+0.01,'0.64', ha="left", va="bottom", color=color5)
                    ax.text(x+0.1, bottom_x,str(x)+'%', ha="left", va="center", color='k', alpha=0.5)
            
            ax.set_xlabel('Percentage of Negative Instances', fontsize=14)
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.grid(True, axis='y', alpha=0.3, which='both')
            
            plt.ylim(-0.03, 1.03) 
            plt.xlim(-0.4, 10.3) 
            fig.tight_layout()
            plt.savefig(path_init + '\\Images\\Results3\\Dataset Reduction\\' + classifier_type + '_ea.png', format='png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

        
def execute_thresholds(path_init):
    
    sensors = [1, 2, 6, 9, 10, 12, 14, 3, 7]
 
    classifier_types = ['NuSVC-rbf'] #"GaussianNB" "LinearSVC" 'NuSVC-rbf'
    width = 120
    frac = None #0.01
    selection_type = 'mine'
    n_features = 19
    correlation_type = 'Pearson'
    
    optimal_thresholds = []
    start = round(-3,2)
    while start <= 2.00:
        start = round(start,2)
        optimal_thresholds.append(start)
        start += 0.01
    
    for classifier_type in classifier_types:
        print("-> " + classifier_type)
        
        results_fs = pd.DataFrame()
        for optimal_threshold in optimal_thresholds:
            print(optimal_threshold)
            df = get_dataset(path_init, correlation_type, width)
            dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test = get_train_test(df, frac)
            df_results = cross_validation(dfs_neg_train, dfs_pos_train, sensors, classifier_type, selection_type, n_features, optimal_threshold)
            mean_results = get_mean_results(df_results)
            #print(df_results)
            #print(mean_results)
            mean_results['threshold'] = optimal_threshold
            results_fs = results_fs.append(mean_results, ignore_index=True)
        
        path_export = path_init + '\\Results\\Real Dataset\\threshold_evolution_' + classifier_type + '.csv'
        results_fs.to_csv(index=True, path_or_buf=path_export, sep=';', decimal=',')
    
def plot_thresholds(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    color4 = 'tab:brown'
    color5 = 'tab:red'
    
    classifier_types = ['GaussianNB','LinearSVC']
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6), sharey=True)
    for i, ax in enumerate(axs):
        classifier_type = classifier_types[i]
        
        path_import = path_init + '\\Results\\Real Dataset\\threshold_evolution_' + classifier_type + '.csv'
        df = pd.read_csv(path_import, index_col=0, sep=';', decimal=',')
        
        df = df[df['threshold']<=1.2]
        
        ax.plot(df['threshold'], df['TNR'], color=color4, label = r'$Recall_{N}$')
        ax.plot(df['threshold'], df['precision'], color=color1, label=r'$Precision_{P}$')
        ax.plot(df['threshold'], df['recall'], color=color2, label=r'$Recall_{P}$')
        ax.plot(df['threshold'], df['F1'], color=color3, label = r'$F_{P}$')
        ax.plot(df['threshold'], df['leakage'], color=color5, label = "L")
        
        bottom_x = -0.027
        if i==1:
            ax.set_title("SVM w/ Linear Kernel")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.4))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            x=0
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.05, 0.89,'0.89', ha="left", va="top", color=color4)
            ax.text(x+0.05, 0.29+0.02,'0.29', ha="left", va="bottom", color=color1)
            ax.text(x+0.05, 0.27-0.1,'0.27', ha="left", va="top", color=color2)
            ax.text(x+0.05, 0.27-0.14,'0.27', ha="left", va="top", color=color3)
            ax.text(x+0.05, 0.64+0.01,'0.64', ha="left", va="bottom", color=color5)
            ax.text(x+0.05, bottom_x,str(x), ha="left", va="center", color='k', alpha=0.5)
            
            x=-0.44
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x-0.05, 0.72,'0.72', ha="right", va="center", color=color4)
            ax.text(x-0.05, 0.40-0.015,'0.40', ha="right", va="bottom", color=color1)
            ax.text(x-0.05, 0.55-0.01,'0.55', ha="right", va="top", color=color2)
            ax.text(x-0.05, 0.43+0.02,'0.43', ha="right", va="bottom", color=color3)
            ax.text(x-0.05, 0.91+0.01,'0.91', ha="right", va="bottom", color=color5)
            ax.text(x-0.05, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
            ax.legend(loc='upper right', ncol=1, bbox_to_anchor=(0.965, 0.9))
        else:
            ax.set_title("Naive Bayes")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            x = 0
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.03, 0.83-0.025,'0.83', ha="left", va="top", color=color4)
            ax.text(x+0.03, 0.39+0.01,'0.39', ha="left", va="bottom", color=color1)
            ax.text(x+0.03, 0.36-0.05,'0.36', ha="left", va="top", color=color2)
            ax.text(x+0.03, 0.35-0.08,'0.35', ha="left", va="top", color=color3)
            ax.text(x+0.03, 0.91+0.01,'0.91', ha="left", va="bottom", color=color5)
            ax.text(x+0.03, bottom_x,str(x), ha="left", va="center", color='k', alpha=0.5)
            
            
            x = -0.52
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.03, 0.72,'0.72', ha="left", va="top", color=color4)
            ax.text(x+0.03, 0.37-0.01,'0.37', ha="left", va="top", color=color1)
            ax.text(x+0.03, 0.55,'0.55', ha="left", va="bottom", color=color2)
            ax.text(x+0.03, 0.42+0.01,'0.42', ha="left", va="bottom", color=color3)
            ax.text(x+0.03, 0.91+0.01,'0.91', ha="left", va="bottom", color=color5)
            ax.text(x+0.03, bottom_x,str(x), ha="left", va="center", color='k', alpha=0.5)
        
        ax.set_xlabel('Threshold')
        
        
        #ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.grid(True, axis='y', alpha=0.3, which='both')
    
    #plt.ylim(-0.03, 1.03)  
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results3\\ROC Curves\\thresholds_redux.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_thresholds_ea(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    color4 = 'tab:brown'
    color5 = 'tab:red'
    
    for classifier_type in ['GaussianNB']:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4.5))
        
        path_import = path_init + '\\Results\\Real Dataset\\threshold_evolution_' + classifier_type + '.csv'
        df = pd.read_csv(path_import, index_col=0, sep=';', decimal=',')
        
        df = df[df['threshold']<=1.2]
        
        ax.plot(df['threshold'], df['TNR'], color=color4, label = r'$Rec_{N}$')
        ax.plot(df['threshold'], df['precision'], color=color1, label=r'$Prec_{P}$')
        ax.plot(df['threshold'], df['recall'], color=color2, label=r'$Rec_{P}$')
        ax.plot(df['threshold'], df['F1'], color=color3, label = r'$F_{P}$')
        ax.plot(df['threshold'], df['leakage'], color=color5, label = "L")
        
        bottom_x = -0.02
        if classifier_type == "LinearSVC":
            ax.set_title("SVM w/ Linear Kernel")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.4))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            x=0
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.05, 0.89,'0.89', ha="left", va="top", color=color4)
            ax.text(x+0.05, 0.29+0.02,'0.29', ha="left", va="bottom", color=color1)
            ax.text(x+0.05, 0.27-0.1,'0.27', ha="left", va="top", color=color2)
            ax.text(x+0.05, 0.27-0.14,'0.27', ha="left", va="top", color=color3)
            ax.text(x+0.05, 0.64+0.01,'0.64', ha="left", va="bottom", color=color5)
            ax.text(x+0.05, bottom_x,str(x), ha="left", va="center", color='k', alpha=0.5)
            
            x=-0.44
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x-0.05, 0.72,'0.72', ha="right", va="center", color=color4)
            ax.text(x-0.05, 0.40-0.015,'0.40', ha="right", va="bottom", color=color1)
            ax.text(x-0.05, 0.55-0.01,'0.55', ha="right", va="top", color=color2)
            ax.text(x-0.05, 0.43+0.02,'0.43', ha="right", va="bottom", color=color3)
            ax.text(x-0.05, 0.91+0.01,'0.91', ha="right", va="bottom", color=color5)
            ax.text(x-0.05, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
            ax.legend(loc='upper right', ncol=1, bbox_to_anchor=(0.965, 0.9))
        else:
            ax.set_title("Performance w/ Different Thresholds", fontsize=14)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            x = 0
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.03, 0.83-0.025,'0.83', ha="left", va="top", color=color4)
            ax.text(x+0.03, 0.39+0.01,'0.39', ha="left", va="bottom", color=color1)
            ax.text(x+0.03, 0.36-0.05,'0.36', ha="left", va="top", color=color2)
            ax.text(x+0.03, 0.35-0.08,'0.35', ha="left", va="top", color=color3)
            ax.text(x+0.03, 0.91+0.01,'0.91', ha="left", va="bottom", color=color5)
            ax.text(x+0.03, bottom_x,str(x), ha="left", va="center", color='k', alpha=0.5)
            
            
            x = -0.52
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
            ax.text(x+0.03, 0.72,'0.72', ha="left", va="top", color=color4)
            ax.text(x+0.03, 0.37-0.01,'0.37', ha="left", va="top", color=color1)
            ax.text(x+0.03, 0.55,'0.55', ha="left", va="bottom", color=color2)
            ax.text(x+0.03, 0.42+0.01,'0.42', ha="left", va="bottom", color=color3)
            ax.text(x+0.03, 0.91+0.01,'0.91', ha="left", va="bottom", color=color5)
            ax.text(x+0.03, bottom_x,str(x), ha="left", va="center", color='k', alpha=0.5)
            ax.legend(loc='upper right', ncol=1, bbox_to_anchor=(0.965, 0.85), fontsize=9)
        
        ax.set_xlabel('Threshold', fontsize=14)
        
        
        #ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.grid(True, axis='y', alpha=0.3, which='both')
    
        #plt.ylim(-0.1, 1.03)  
        fig.tight_layout()
        plt.savefig(path_init + '\\Images\\Results3\\ROC Curves\\thresholds_redux_'+ classifier_type +'_ea.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
  
def plot_thresholds_1(path_init):
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    color4 = 'tab:brown'
    color5 = 'tab:red'
    
    classifier_type = 'NuSVC-rbf'
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.2,6), sharey=True)
        
    path_import = path_init + '\\Results\\Real Dataset\\threshold_evolution_' + classifier_type + '.csv'
    df = pd.read_csv(path_import, index_col=0, sep=';', decimal=',')
    
    #df = df[df['threshold']<=1.2]
    
    ax.plot(df['threshold'], df['TNR'], color=color4, label = r'$Recall_{N}$')
    ax.plot(df['threshold'], df['precision'], color=color1, label=r'$Precision_{P}$')
    ax.plot(df['threshold'], df['recall'], color=color2, label=r'$Recall_{P}$')
    ax.plot(df['threshold'], df['F1'], color=color3, label = r'$F_{P}$')
    ax.plot(df['threshold'], df['leakage'], color=color5, label = "L")
    
    bottom_x = -0.027
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.4))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    x=0
    ax.axvline(x=x, color='k', linestyle='--', linewidth=1.25, alpha=0.3)
    ax.text(x-0.08, 0.72,'0.72', ha="right", va="bottom", color=color4)
    ax.text(x-0.08, 0.01+0.02,'0.01', ha="right", va="bottom", color=color1)
    ax.text(x-0.08, 0.50,'0.50', ha="right", va="center", color=color2)
    ax.text(x-0.08, 0.02+0.05,'0.02', ha="right", va="bottom", color=color3)
    ax.text(x-0.08, 0.82+0.01,'0.82', ha="right", va="bottom", color=color5)
    ax.text(x-0.08, bottom_x,str(x), ha="right", va="center", color='k', alpha=0.5)
    
    ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(0.03, 0.85))
    
    ax.set_xlabel('Threshold')
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.grid(True, axis='y', alpha=0.3, which='both')
    
    #plt.ylim(-0.03, 1.03)  
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results3\\ROC Curves\\thresholds_RBF.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
def plot_roc_curves(path_init):

    sensors = [1, 2, 6, 9, 10, 12, 14, 3, 7]
 
    classifier_types = ["GaussianNB","LinearSVC"] #"GaussianNB"  'NuSVC-rbf'
    width = 120
    frac = 0.01
    selection_type = None
    n_features = None
    correlation_type ='Pearson'
    
    df = get_dataset(path_init, correlation_type, width)
    dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test = get_train_test(df, frac)
    cross_validation_roc(dfs_neg_train, dfs_pos_train, sensors, classifier_types, selection_type, n_features)     
    
    
    classifier_type = 'NuSVC-rbf'
    frac = None
    selection_type = 'mine'
    n_features = 19
    dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test = get_train_test(df, frac)
    cross_validation_roc_1(dfs_neg_train, dfs_pos_train, sensors, classifier_type, selection_type, n_features)
    #dfs_neg_train, dfs_pos_train, df_neg_test, df_pos_test = get_train_test(df, frac)
    #cross_validation_roc_1(dfs_neg_train, dfs_pos_train, sensors, classifier_types, selection_type, n_features)
    

config = Configuration()
path_init = config.path

correlation_type = 'DCCA' # DCCA Pearson
width = 120 #60 120 180 240
classifier_type = "NuSVC-rbf" #"GaussianNB" , "LinearSVC", "NuSVC-rbf", "NuSVC-poly"


#get_main_results(path_init)
#get_initial_results(path_init)
#get_final_results(path_init)

#negative_set_reduction(path_init)  
#plot_dataset_reduction(path_init)  
#plot_dataset_reduction_ea(path_init)  

#execute_feature_selection(path_init)
#plot_feature_selection(path_init)
#plot_feature_selection_ea(path_init)
#plot_feature_selection_1(path_init)
#plot_feature_selection_1_ea(path_init)

#execute_time_window_sizes(path_init)

#execute_thresholds(path_init)
#plot_thresholds(path_init)
#plot_thresholds_ea(path_init)
#plot_thresholds_1(path_init)

plot_roc_curves(path_init)
