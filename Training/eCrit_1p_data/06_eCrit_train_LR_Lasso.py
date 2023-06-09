#!/usr/bin/env python
# coding: utf-8
# author: Maruthi Mutnuri (maruthi2008@gmail.com)

# loading libraries
import pandas as pd 
import numpy as np
import random
from sklearn.model_selection import  PredefinedSplit, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, Lasso 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn import metrics

# loading master files
'''
master files have data for all the outcomes. 
outcome dependent excluding criteria was not applied on master files. 
these exclusing criteria will be applied as and when needed for each outcome.
So for each outcome, train, validation, and test datasets will be extracted from master files.

Outcomes:
1. 30_DAYS_EXPIRE (30 days mortality)
2. AKI (Acute kidney injury)
3. H_LOS (Hospital length of stay)
4. ICU_LOS (ICU length of stay)
'''

X_train_df_original = pd.read_pickle('Data/eCrit_master_X_train.pickle', compression = 'zip')
X_val_df = pd.read_pickle('Data/eCrit_master_X_val.pickle', compression = 'zip')
X_test_df = pd.read_pickle('Data/eCrit_master_X_test.pickle', compression = 'zip')

y_train_df_original = pd.read_pickle('Data/eCrit_master_y_train.pickle', compression = 'zip')
y_val_df = pd.read_pickle('Data/eCrit_master_y_val.pickle', compression = 'zip')
y_test_df = pd.read_pickle('Data/eCrit_master_y_test.pickle', compression = 'zip')

# printing check point
print('files loaded')


# creating 10 random states for avoiding selection bias while selecting data subsets 
# using random seed for reproducibility
random.seed(32)
random_states = [random.randint(1,1000) for x in range(10)]

# looping through each random state for training models and saving results at each random state
for random_state in random_states:
    
    # creating data subset for traininig
    _, X_train_df, _, y_train_df = train_test_split(X_train_df_original, y_train_df_original, test_size=0.01, random_state = random_state)
    
# # 1. 30_DAYS_EXPIRE processing started
    
    # Extracting train dataset from master files
    # dropping PATIENT_ID column
    X_train = X_train_df.drop(columns = ['PATIENT_ID'])
    # selecting 30_DAYS_EXPIRE column
    y_train = y_train_df['30_DAYS_EXPIRE']
    
    # Applying smote, first 50% oversampling and then 100% undersampling
    over = SMOTE(sampling_strategy=0.5)
    under = RandomUnderSampler(sampling_strategy=1.0)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X_r, y_r = pipeline.fit_resample(X_train, y_train)

    # Extracting validation dataset from master files
    X_val = X_val_df.drop(columns = ['PATIENT_ID'])
    y_val = y_val_df['30_DAYS_EXPIRE']
    
    # creating predefined split for using validation data in gridsearchcv instead of cv split
    # this is similar to how validation data used in tf/keras deep learning models, -1 for training set and 0 for validation set 
    split_index = [-1]*len(X_r) + [0]*len(X_val)
    X = np.concatenate((X_r, X_val), axis=0)
    y = np.concatenate((y_r, y_val), axis=0)
    pds = PredefinedSplit(test_fold = split_index)
    
    # parameter grid definition
    param_grid = [{'solver':['newton-cg','lbfgs','sag'],'penalty':['l2'],'C':np.arange(0.01,10.0,0.01)},
                  {'solver': ['liblinear','saga'],'penalty': ['l1','l2'],'C':np.arange(0.01,10.0,0.01)},
                  {'solver':['newton-cg', 'lbfgs', 'sag', 'saga'], 'penalty': ['none']}]
    
    # grisdearch definition and fitting              
    gs = GridSearchCV(LogisticRegression(), param_grid, cv=pds,  
                      scoring=['roc_auc','accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy'], refit = 'balanced_accuracy')
    gs.fit(X, y)
    
    # printing check point
    print('30_DAYS_EXPIRE GridSearch completed')
    
    # Extracting test dataset from master files    
    X_test = X_test_df.drop(columns = ['PATIENT_ID'])
    y_test = y_test_df['30_DAYS_EXPIRE']
    
    # BOOTSTRAP
    # concatenating X and y test dataset for bootstrapping
    data = pd.concat([X_test,y_test], axis=1).reset_index(drop=True)
    
    # creating and initializing DataFrame for storing bootstrap results
    Boot_stats = pd.DataFrame(columns=['AUC CI','Accuracy CI','F1 Score CI','Precision CI','Recall CI','Balanced Accuracy CI'])
    N = 100 # number of bootstrap samples for each random state
    
    # creating random states and using random seed for reproducibility
    random.seed(random_state) 
    r_states = [random.randint(1,10000) for x in range(N)]
    
    # loop for bootstrap sampling
    for i in range(N):
    
        # extracting bootstrap samples with replacement from test dataset
        X_test_boot = data.sample(frac=1, replace=True, random_state=r_states[i])
        
        # splitting bootstrap samples into X and y
        y_test_boot = X_test_boot['30_DAYS_EXPIRE']
        X_test_boot = X_test_boot.drop(columns = ['30_DAYS_EXPIRE'])
        
        # calculating and updating dataframe with metrics using test dataset bootstrap samples
        y_pred_proba = gs.best_estimator_.predict_proba(X_test_boot)
        fpr, tpr, thresholds = metrics.roc_curve(y_test_boot, y_pred_proba[:,1], pos_label=1)
        y_pred = gs.best_estimator_.predict(X_test_boot)
        
        # using temp dictionary to store bootstrap results
        temp_dict = {'AUC CI': metrics.auc(fpr, tpr), 
                              'Accuracy CI': metrics.accuracy_score(y_test_boot, y_pred), 'F1 Score CI': metrics.f1_score(y_test_boot, y_pred),
                              'Precision CI': metrics.precision_score(y_test_boot, y_pred), 'Recall CI': metrics.recall_score(y_test_boot, y_pred),
                              'Balanced Accuracy CI': metrics.balanced_accuracy_score(y_test_boot, y_pred)}
        
        # converting temp dictionary into DataFrame
        temp = pd.DataFrame([temp_dict])
        
        # updating Boot_stats DataFrame with the boostrap result 
        Boot_stats = pd.concat([Boot_stats,temp], axis=0).reset_index(drop=True)
    
    # saving bootstrap results for each random state and for each outcome
    Boot_stats.to_csv('Results/eCrit_30_DAYS_EXPIRE_Bootstrap_1p_'+str(random_state)+'.csv')
    
    
# # 2. AKI processing started
    # Extracting train dataset from master files
    # AKI_data_miss indicates missing data, which prevents identifying presence or absence of AKI in patients
    y_train = y_train_df[y_train_df['AKI_data_miss'] != 1]
    # AKI_adm indicates presence of AKI during admission, which is an exclusion criteria  
    y_train = y_train[y_train['AKI_adm'] != 1]
    # selecting X_train rows for the corresponsing y_train rows using PATIENT_ID
    X_train = X_train_df.loc[X_train_df['PATIENT_ID'].isin(y_train['PATIENT_ID'])]
    # dropping PATIENT_ID column
    X_train = X_train.drop(columns = ['PATIENT_ID'])
    # selecting AKI column
    y_train = y_train['AKI']
    
    # Applying smote, first 50% oversampling and then 100% undersampling
    over = SMOTE(sampling_strategy=0.5)
    under = RandomUnderSampler(sampling_strategy=1.0)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X_r, y_r = pipeline.fit_resample(X_train, y_train)

    # Extracting validation dataset from master files
    y_val = y_val_df[y_val_df['AKI_data_miss'] != 1]
    y_val = y_val[y_val['AKI_adm'] != 1]
    X_val = X_val_df.loc[X_val_df['PATIENT_ID'].isin(y_val['PATIENT_ID'])]
    X_val = X_val.drop(columns = ['PATIENT_ID'])
    y_val = y_val['AKI']
    
    # creating predefined split for using validation data in gridsearchcv instead of cv split
    split_index = [-1]*len(X_r) + [0]*len(X_val)
    X = np.concatenate((X_r, X_val), axis=0)
    y = np.concatenate((y_r, y_val), axis=0)
    pds = PredefinedSplit(test_fold = split_index)    
    
    # parameter grid definition
    param_grid = [{'solver':['newton-cg','lbfgs','sag'],'penalty':['l2'],'C':np.arange(0.01,10.0,0.01)},
                  {'solver': ['liblinear','saga'],'penalty': ['l1','l2'],'C':np.arange(0.01,10.0,0.01)},
                  {'solver':['newton-cg', 'lbfgs', 'sag', 'saga'], 'penalty': ['none']}]
    
    # grisdearch definition and fitting               
    gs = GridSearchCV(LogisticRegression(), param_grid, cv=pds, 
                      scoring=['roc_auc','accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy'], refit = 'balanced_accuracy')
    gs.fit(X, y)
    
    # printing check point
    print('AKI GridSearch completed')

    # Extracting test dataset from master files 
    y_test = y_test_df[y_test_df['AKI_data_miss'] != 1]
    y_test = y_test[y_test['AKI_adm'] != 1]
    X_test = X_test_df.loc[X_test_df['PATIENT_ID'].isin(y_test['PATIENT_ID'])]
    X_test = X_test.drop(columns = ['PATIENT_ID'])
    y_test = y_test['AKI']
    
    # BOOTSTRAP
    # concatenating X and y test dataset for bootstrapping
    data = pd.concat([X_test,y_test], axis=1).reset_index(drop=True)
    
    # creating and initializing DataFrame for storing bootstrap results
    Boot_stats = pd.DataFrame(columns=['AUC CI','Accuracy CI','F1 Score CI','Precision CI','Recall CI','Balanced Accuracy CI'])
    
    # loop for bootstrap sampling
    for i in range(N):
        # extracting bootstrap samples with replacement from test dataset
        X_test_boot = data.sample(frac=1, replace=True, random_state=r_states[i])
        
        # splitting bootstrap samples into X and y
        y_test_boot = X_test_boot['AKI']
        X_test_boot = X_test_boot.drop(columns = ['AKI'])
        
        # calculating and updating dataframe with metrics using test dataset bootstrap samples
        y_pred_proba = gs.best_estimator_.predict_proba(X_test_boot)
        fpr, tpr, thresholds = metrics.roc_curve(y_test_boot, y_pred_proba[:,1], pos_label=1)
        y_pred = gs.best_estimator_.predict(X_test_boot)
        
        # using temp dictionary to store bootstrap results
        temp_dict = {'AUC CI': metrics.auc(fpr, tpr), 
                              'Accuracy CI': metrics.accuracy_score(y_test_boot, y_pred), 'F1 Score CI': metrics.f1_score(y_test_boot, y_pred),
                              'Precision CI': metrics.precision_score(y_test_boot, y_pred), 'Recall CI': metrics.recall_score(y_test_boot, y_pred),
                              'Balanced Accuracy CI': metrics.balanced_accuracy_score(y_test_boot, y_pred)}
        
        # converting temp dictionary into DataFrame
        temp = pd.DataFrame([temp_dict])
        
        # updating Boot_stats DataFrame with the boostrap result
        Boot_stats = pd.concat([Boot_stats,temp], axis=0).reset_index(drop=True)

    # saving bootstrap results for each random state and for each outcome
    Boot_stats.to_csv('Results/eCrit_AKI_Bootstrap_1p_'+str(random_state)+'.csv')
    
    
# # 3. H_LOS processing started

    # Extracting train, validation and test datasets from master files
    # H_LOS_miss indicates missing data, which prevents identifying presence or absence of H_LOS in patients
    y_train = y_train_df[y_train_df['H_LOS_miss'] != 1]
    # To avoid outliers, only bottom 98 percentile data is included
    y_train = y_train[y_train.H_LOS < np.percentile(y_train.H_LOS,98)]
    # selecting X_train rows for the corresponsing y_train rows using PATIENT_ID
    X_train = X_train_df.loc[X_train_df['PATIENT_ID'].isin(y_train['PATIENT_ID'])]
    # dropping PATIENT_ID column
    X_train = X_train.drop(columns = ['PATIENT_ID'])
    # selecting H_LOS column
    y_train = y_train['H_LOS']
    
    y_val = y_val_df[y_val_df['H_LOS_miss'] != 1]
    y_val = y_val[y_val.H_LOS < np.percentile(y_val.H_LOS,98)]
    X_val = X_val_df.loc[X_val_df['PATIENT_ID'].isin(y_val['PATIENT_ID'])]
    X_val = X_val.drop(columns = ['PATIENT_ID'])
    y_val = y_val['H_LOS']    
    
    y_test = y_test_df[y_test_df['H_LOS_miss'] != 1]
    y_test = y_test[y_test.H_LOS < np.percentile(y_test.H_LOS,98)]
    X_test = X_test_df.loc[X_test_df['PATIENT_ID'].isin(y_test['PATIENT_ID'])]
    X_test = X_test.drop(columns = ['PATIENT_ID'])
    y_test = y_test['H_LOS']

    # creating predefined split for using validation data in gridsearchcv instead of cv split    
    split_index = [-1]*len(X_train) + [0]*len(X_val)
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    pds = PredefinedSplit(test_fold = split_index)    
    
    # parameter grid definition    
    param_grid = [{'alpha': np.arange(0.01,1,0.01)}]
    
    # grisdearch definition and fitting      
    gs = GridSearchCV(Lasso() , param_grid, cv=pds, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'], refit='neg_mean_squared_error')
    gs.fit(X, y)
    
    # printing check point    
    print('H_LOS Lasso GridSearch completed')
    
    # BOOTSTRAP
    # concatenating X and y test dataset for bootstrapping    
    data = pd.concat([X_test,y_test], axis=1).reset_index(drop=True)
    
    # creating and initializing DataFrame for storing bootstrap results
    Boot_stats = pd.DataFrame(columns=['MSE CI','MAE CI'])
    
    # loop for bootstrap sampling    
    for i in range(N):
    
        # extracting bootstrap samples with replacement from test dataset
        X_test_boot = data.sample(frac=1, replace=True, random_state=r_states[i])
        
        # splitting bootstrap samples into X and y        
        y_test_boot = X_test_boot['H_LOS']
        X_test_boot = X_test_boot.drop(columns = ['H_LOS'])

        # calculating and updating dataframe with metrics using test dataset bootstrap samples
        y_pred = gs.best_estimator_.predict(X_test_boot)
        temp_dict = {'MSE CI': metrics.mean_squared_error(y_test_boot, y_pred), 'MAE CI': metrics.mean_absolute_error(y_test_boot, y_pred)}
        temp = pd.DataFrame([temp_dict])
        Boot_stats = pd.concat([Boot_stats,temp], axis=0).reset_index(drop=True)

    # saving bootstrap results for each random state and for each outcome
    Boot_stats.to_csv('Results/eCrit_H_LOS_Lasso_Bootstrap_1p_'+str(random_state)+'.csv')
    
    
# # 4. ICU_LOS

    # Extracting train, validation and test datasets from master files
    # To avoid outliers, only bottom 98 percentile data is included
    y_train = y_train_df[y_train_df.ICU_LOS < np.percentile(y_train_df.ICU_LOS,98)]
    # selecting X_train rows for the corresponsing y_train rows using PATIENT_ID
    X_train = X_train_df.loc[X_train_df['PATIENT_ID'].isin(y_train['PATIENT_ID'])]
    # dropping PATIENT_ID column
    X_train = X_train.drop(columns = ['PATIENT_ID'])
    # selecting ICU_LOS column
    y_train = y_train['ICU_LOS']
    
    y_val = y_val_df[y_val_df.ICU_LOS < np.percentile(y_val_df.ICU_LOS,98)]
    X_val = X_val_df.loc[X_val_df['PATIENT_ID'].isin(y_val['PATIENT_ID'])]
    X_val = X_val.drop(columns = ['PATIENT_ID'])
    y_val = y_val['ICU_LOS']   
    
    y_test = y_test_df[y_test_df.ICU_LOS < np.percentile(y_test_df.ICU_LOS,98)]
    X_test = X_test_df.loc[X_test_df['PATIENT_ID'].isin(y_test['PATIENT_ID'])]
    X_test = X_test.drop(columns = ['PATIENT_ID'])
    y_test = y_test['ICU_LOS']

    # creating predefined split for using validation data in gridsearchcv instead of cv split      
    split_index = [-1]*len(X_train) + [0]*len(X_val)
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    pds = PredefinedSplit(test_fold = split_index)     

    # parameter grid definition    
    param_grid = [{'alpha': np.arange(0.01, 1, 0.01)}]
    
    # grisdearch definition and fitting      
    gs = GridSearchCV(Lasso() , param_grid, cv=pds, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'], refit='neg_mean_squared_error')
    gs.fit(X, y)
    
    # printing check point    
    print('ICU_LOS Lasso GridSearch completed')
    
    # BOOTSTRAP
    # concatenating X and y test dataset for bootstrapping    
    data = pd.concat([X_test,y_test], axis=1).reset_index(drop=True)
    
    # creating and initializing DataFrame for storing bootstrap results    
    Boot_stats = pd.DataFrame(columns=['MSE CI','MAE CI'])

    # loop for bootstrap sampling
    for i in range(N):
        # extracting bootstrap samples with replacement from test dataset
        X_test_boot = data.sample(frac=1, replace=True, random_state=r_states[i])
        
        # splitting bootstrap samples into X and y
        y_test_boot = X_test_boot['ICU_LOS']
        X_test_boot = X_test_boot.drop(columns = ['ICU_LOS'])

        # calculating and updating dataframe with metrics using test dataset bootstrap samples        
        y_pred = gs.best_estimator_.predict(X_test_boot)
        temp_dict = {'MSE CI': metrics.mean_squared_error(y_test_boot, y_pred), 'MAE CI': metrics.mean_absolute_error(y_test_boot, y_pred)}
        temp = pd.DataFrame([temp_dict])      
        Boot_stats = pd.concat([Boot_stats,temp], axis=0).reset_index(drop=True)

    # saving bootstrap results for each random state and for each outcome    
    Boot_stats.to_csv('Results/eCrit_ICU_LOS_Lasso_Bootstrap_1p_'+str(random_state)+'.csv')
