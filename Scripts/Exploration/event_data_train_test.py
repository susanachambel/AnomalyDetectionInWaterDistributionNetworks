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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, NuSVC
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from itertools import combinations, product
import pandas as pd
import numpy as np

def get_dataset(path_init, sensors, correlation_type, data_type, width):
    sensors_init = []
    combos_str = []
    path = ''
    
    if(data_type == 'r'):
        sensors_init_f = ['1', '9', '10', '12', '14', '2', '6']
        sensors_init_p = ['3', '7', '8', '11', '15']
        combos_f = list(combinations(sensors_init_f, 2))
        combos_p = list(combinations(sensors_init_p, 2))
        for combos in [combos_f, combos_p]:
            for combo in combos:
                if ((combo[0] in sensors) and (combo[1] in sensors)):
                    combos_str.append(get_combo_name(combo))
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\dataset_'+ data_type + '_' + correlation_type.lower() + '.csv'
    else:
        if(data_type == 'q'):
            sensors_init = ['1', '2', '6', '9', '10']
        elif(data_type == 'p'):
            sensors_init_aux = list(range(1, 21, 4))
            for sensor in sensors_init_aux:
                sensors_init.append(str(sensor))
        
        combos = list(combinations(sensors_init, 2))
        for combo in combos:
            if ((combo[0] in sensors) and (combo[1] in sensors)):
                combos_str.append(get_combo_name(combo))
        
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\dataset_'+ data_type + '_' + correlation_type.lower() +'_' + str(width) + '.csv'
    
    combos_str.append('y')
    
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
    c = df.loc[:,label.lower()].to_numpy()
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
    
    if isinstance(c[0], str):
        scatter = ax.scatter(x, y, edgecolors='black')
    else:
        cm = plt.cm.get_cmap('Blues')
        scatter = ax.scatter(x, y, c=c, cmap=cm, edgecolors='black')
        plt.colorbar(scatter, ticks=ticker.MaxNLocator(integer=True), label=label)

    for i in index:
        c_aux = c[i]
        if not isinstance(c_aux, str):
            c_aux = int(c_aux)
        ax.annotate(c_aux, (x[i], y[i]), textcoords="offset points", xytext=(5,0), ha='left')
    
    title = "TPR/FPR Variation\n[" + correlation_type + " w/ " + classifier_type + "]"
    ax.set(xlabel='False Positive Rate [0-1]', ylabel='True Positive Rate [0-1]', title=title)
    plt.show()

def plot_roc_curve(fpr, tnr, points):
    lw = 2
    label = 'ROC curve (auc=%0.2f)' % auc(fpr, tnr)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot(fpr, tnr, color='darkorange', lw=lw, label=label)
    for point in points:
        plt.plot(point['fpr'],point['tnr'],'ro',label=point['label'], color=point['color'])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    #plt.savefig('C:\\Users\\susan\\Desktop\\roc_curve.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curves(roc_curves, title, legend_title):   
    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve " + title)
    
    for index, curve in roc_curves.iterrows():
        plt.plot(curve['fpr'], curve['tnr'], lw=lw, label=curve['label'])
    
    plt.legend(loc="lower right", title=legend_title)
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
        clf = NuSVC(random_state=1, kernel='rbf') # class_weight='balanced', nu=0.0000001, 
    elif classifier_type == "NuSVC-poly":
        clf = NuSVC(random_state=1, kernel='poly')
        
    clf_fit = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    if classifier_type == "GaussianNB":
        y_scores = clf_fit.predict_proba(X_test)[:,1]
    else:
        y_scores = clf_fit.decision_function(X_test)
    
    return y_pred, y_scores

def execute_train_test_real(path_init, correlation_type, classifier_type, data_type, width, optimize):

    data_type = "r"
    sensors = ['1', '9', '10', '12', '14', '2', '6', '3', '7', '8', '11', '15'] 
    sensors = ['1', '9', '12', '14', '2']     
       
    df = get_dataset(path_init, sensors, correlation_type, data_type, width)
    
    n_loc = 7
    if correlation_type == "Pearson":
        n_loc = 9
    
    sensors_in = df.isnull().sum().sort_values(ascending = False).iloc[n_loc:].index.to_numpy()
    df = df.loc[:, sensors_in].dropna()
        
    df = df.sample(frac = 1, random_state=1).reset_index(drop=True)
     
    n_splits = len(df[df['y']==1])
    skf = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=True)
        
    X = df.iloc[:,:-1]
    y = df.loc[:,'y']
     
    df_results = pd.DataFrame()
    
    n_fold = 1
    for train_index, test_index in skf.split(X, y):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]     
           
        y_pred, y_scores = train_predict(classifier_type, X_train, y_train, X_test)
            
        cnf_matrix = confusion_matrix(y_test, y_pred, [0,1])
        TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix)      
            
        df_results = update_results(df_results, TN, FP, FN, TP)  
        fpr, tnr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
        
        roc_points = []
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
        
        plot_roc_curve(fpr, tnr, roc_points)
        #plot_confusion_matrix(cnf_matrix, [0,1])
        n_fold += 1
            
    results = round(df_results.describe().loc['mean',:],2).to_dict()
    print(round(df_results,2))  
    print(results)
       
def execute_train_test(path_init, sensors, correlation_type, classifier_type, data_type, width, optimize):
    
    df = get_dataset(path_init, sensors, correlation_type, data_type, width)
    df = df.sample(frac = 1, random_state=1).reset_index(drop=True)
    
    
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

def execute_train_test_simple(path_init, sensors, correlation_type, classifier_type, data_type, width, optimize, label):
    
    df = get_dataset(path_init, sensors, correlation_type, data_type, width)
    df = df.sample(frac = 1, random_state=1).reset_index(drop=True)
            
    X = df.iloc[:,:-1]
    y = df.loc[:,'y']
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)
            
    y_train = y_train.replace([2, 3, 4, 5, 6], 1)
    y_test = y_test.replace([2, 3, 4, 5, 6], 1)
                
    y_pred, y_scores = train_predict(classifier_type, X_train, y_train, X_test)
    fpr, tnr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
                
    if optimize == 1:
        y_pred = optimize_y_pred(y_scores, fpr, tnr, thresholds)
                
    cnf_matrix = confusion_matrix(y_test, y_pred, [0,1])
    TN, FP, FN, TP = get_instances_confusion_matrix(cnf_matrix)
    
    auc_value = auc(fpr, tnr)
    label = label + ' (a=%0.2f)' % auc_value
    roc_curve_results = {'fpr':fpr,'tnr':tnr,'label':label, 'auc':auc_value}
                
    results = get_results(TN, FP, FN, TP)

    for k, v in results.items():
        results[k] = round(v, 2)
        
    results['corr'] = correlation_type
    results['clf'] = classifier_type
    results['width'] = width
    results['opt'] = optimize
    
    return roc_curve_results, results

def test_different_number_sensors(path_init, data_type, width, optimize):
    correlation_types = ["DCCA"] #, "Pearson"
    classifier_types = ["NuSVC-rbf"] #"GaussianNB","LinearSVC","NuSVC-poly",
    
    sensors_init = []
    
    if(data_type == "q"):
        sensors_init = ['1', '2', '6', '9', '10']
    else:
        sensors_init = ['1', '5', '9', '13', '17']
    
    combos = []
    
    for i in range(3, 6):
        combos.append(list(combinations(sensors_init, i)))
        
    for correlation_type in correlation_types:
        for classifier_type in classifier_types:
            if(optimize == 0):
                print("\n" + correlation_type  + " & " + classifier_type + " w/o optimization\n")
            else:
                print("\n" + correlation_type  + " & " + classifier_type + " w/ optimization\n")
            roc_curves = pd.DataFrame()
            df_results = pd.DataFrame()
            for combo in combos:
                for sensors in combo:
                    roc_curve_results, results = execute_train_test_simple(path_init, sensors, correlation_type, classifier_type, data_type, width, optimize, str(list(sensors)))
                    roc_curve_results['c'] = len(list(sensors))
                    roc_curves = roc_curves.append(roc_curve_results, ignore_index=True)
                    results['combo'] = str(list(sensors))
                    df_results = df_results.append(results, ignore_index=True)
            
            idx = roc_curves.groupby(['c'], sort=False)['auc'].max()
            idx = roc_curves.groupby(['c'])['auc'].transform(max) == roc_curves['auc']
            roc_curves = roc_curves[idx]
            
            print(df_results)
            #plot_scatter_plot_results(df_results, "Combo")
            plot_roc_curves(roc_curves, "\n[" + correlation_type  + " w/ " + classifier_type + "]", "Combo")

def test_different_widths(path_init, sensors, data_type, optimize):
    correlation_types = ["DCCA", "Pearson"]
    classifier_types = ["NuSVC-rbf"] #"GaussianNB","LinearSVC","NuSVC-poly",
    
    for correlation_type in correlation_types:
        
        for classifier_type in classifier_types:
            
            if(optimize == 0):
                print("\n" + correlation_type  + " & " + classifier_type + " w/o optimization\n")
            else:
                print("\n" + correlation_type  + " & " + classifier_type + " w/ optimization\n")
            
            roc_curves = pd.DataFrame()
            df_results = pd.DataFrame()
            for width in range(30, 41, 5): #15
                roc_curve_results, results = execute_train_test_simple(path_init, sensors, correlation_type, classifier_type, data_type, width, optimize, str(width))
                roc_curves = roc_curves.append(roc_curve_results, ignore_index=True)
                df_results = df_results.append(results, ignore_index=True)
            
            print(df_results)
            #plot_scatter_plot_results(df_results, "Width")
            plot_roc_curves(roc_curves, "\n[" + correlation_type  + " w/ " + classifier_type + "]", "Width")

def test_different_classifiers_correlations(path_init, sensors, data_type, width, optimize):

    correlation_types = ["DCCA"] #"Pearson",
    classifier_types = ["GaussianNB","LinearSVC", "NuSVC-poly","NuSVC-rbf"] #"NuSVC-poly","NuSVC-rbf"
        
    for correlation_type in correlation_types:
        roc_curves = pd.DataFrame()
        df_results = pd.DataFrame()
        
        if(optimize == 0):
            print("\n" + correlation_type  + " w/o optimization\n")
        else:
            print("\n" + correlation_type  + " w/ optimization\n")
            
        for classifier_type in classifier_types:
            roc_curve_results, results = execute_train_test_simple(path_init, sensors, correlation_type, classifier_type, data_type, width, optimize, classifier_type)
            roc_curves = roc_curves.append(roc_curve_results, ignore_index=True)
            df_results = df_results.append(results, ignore_index=True)
        
        print(df_results)
        #plot_scatter_plot_results(df_results, "Clf")
        plot_roc_curves(roc_curves, "w/ " + correlation_type, "Classifier")


config = Configuration()
path_init = config.path

sensors = [] # Combination chosen
correlation_type = "DCCA" # "DCCA" "Pearson"
classifier_type = "LinearSVC" #"GaussianNB" "NuSVC-poly" "NuSVC-rbf" "LinearSVC"
data_type = "q"
optimize = 1
width = 40


if(data_type == "q"):
    sensors = ['1', '2', '6', '9', '10']
elif(data_type == "p"):
    #sensors_aux = list(range(1, 21, 2))
    sensors = ['1', '5', '9', '13', '17']
    #sensors = ['3', '7', '11', '15', '19']
 
execute_train_test(path_init, sensors, correlation_type, classifier_type, data_type, width, optimize)
#execute_train_test_real(path_init, correlation_type, classifier_type, data_type, width, optimize)

#test_different_widths(path_init, sensors, data_type, optimize)
#test_different_number_sensors(path_init, data_type, width, optimize)
#test_different_classifiers_correlations(path_init, sensors, data_type, width, optimize)




        
              
    
