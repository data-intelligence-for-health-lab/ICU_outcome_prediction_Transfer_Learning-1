#!/usr/bin/env python
# coding: utf-8
# author: Maruthi Mutnuri (maruthi2008@gmail.com)

# loading libraries
import pandas as pd 
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn import metrics
from tensorflow.keras.metrics import AUC, Precision, Recall, binary_accuracy, MeanAbsoluteError,MeanSquaredError

# main routine starts

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

X_train_df = pd.read_pickle('Data/MIMIC_master_X_train.pickle', compression = 'zip')
X_val_df = pd.read_pickle('Data/MIMIC_master_X_val.pickle', compression = 'zip')
X_test_df = pd.read_pickle('Data/MIMIC_master_X_test.pickle', compression = 'zip')

y_train_df = pd.read_pickle('Data/MIMIC_master_y_train.pickle', compression = 'zip')
y_val_df = pd.read_pickle('Data/MIMIC_master_y_val.pickle', compression = 'zip')
y_test_df = pd.read_pickle('Data/MIMIC_master_y_test.pickle', compression = 'zip')

 # printing check point
print('files loaded')

 
# # 1. 30_DAYS_EXPIRE processing started  
# Extracting train dataset from master files
# dropping SUBJECT_ID column
X_train = X_train_df.drop(columns = ['SUBJECT_ID'])
# selecting 30_DAYS_EXPIRE column
y_train = y_train_df['30_DAYS_EXPIRE']

# Applying smote, first 50% oversampling and then 100% undersampling   
over = SMOTE(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=1.0)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
X_r, y_r = pipeline.fit_resample(X_train, y_train)

# Extracting validation and test dataset from master files     
X_val = X_val_df.drop(columns = ['SUBJECT_ID'])
y_val = y_val_df['30_DAYS_EXPIRE']

X_test = X_test_df.drop(columns = ['SUBJECT_ID'])
y_test = y_test_df['30_DAYS_EXPIRE']    

# defining early stopping condition
early_stop_fcnn = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10)

# loading pre-trained model
model_path = "/home/maruthikumar.mutnuri/Final_Models/"
model_name = model_path+"eCrit_30_DAYS_EXPIRE_R20_SM_929.h5"
new_model = tf.keras.models.load_model(model_name)

# removing top layers and replacing them with custom new hidden layer along with new output layer	
# All the pre-trained layers are changed to not trainable, only new custom layers are trainable	
retrained_model = tf.keras.models.Sequential()
for layer in new_model.layers[:-3]:
    layer.trainable = False
    retrained_model.add(layer)
retrained_model.add((tf.keras.layers.Dense(32,activation='relu',kernel_initializer='HeUniform',kernel_regularizer=tf.keras.regularizers.l1(l1=0.001),name='new_dense')))
retrained_model.add(tf.keras.layers.Dropout(0.3,name='new_dropout'))
retrained_model.add(tf.keras.layers.Dense(1, activation='sigmoid',name='main_output'))
input_shape = (None, 104)
retrained_model.build(input_shape)

# compiling and fitting the model
opt = tf.keras.optimizers.Adam(lr=0.001)    
retrained_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[AUC(),Precision(),Recall()])
history = retrained_model.fit(X_r, y_r,epochs = 50, \
          verbose = 0,  callbacks= [early_stop_fcnn],batch_size = 128,validation_data=(X_val, y_val))
          
# changing all the layers to be trainable, including the pre-trained layers.
retrained_model.trainable = True

# compiling and fitting the model
opt = tf.keras.optimizers.Adam(lr=0.001)
retrained_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[AUC(),Precision(),Recall()])
history = retrained_model.fit(X_r, y_r,epochs = 50, \
          verbose = 0,  callbacks= [early_stop_fcnn],batch_size = 128,validation_data=(X_val, y_val))


# BOOTSTRAP
# concatenating X and y test dataset for bootstrapping
data = pd.concat([X_test,y_test], axis=1).reset_index(drop=True)

# creating and initializing DataFrame for storing bootstrap results     
Boot_stats = pd.DataFrame(columns=['AUC CI','Accuracy CI','F1 Score CI','Precision CI','Recall CI','Balanced Accuracy CI'])
N = 1000 # number of bootstrap samples for each random state

# creating random states and using random seed for reproducibility       
random.seed(32)
r_states = [random.randint(1,10000) for x in range(N)]

# loop for bootstrap sampling    
for i in range(N):
    # extracting bootstrap samples with replacement from test dataset
    X_test_boot = data.sample(frac=1, replace=True, random_state=r_states[i])
    
    # splitting bootstrap samples into X and y
    y_test_boot = X_test_boot['30_DAYS_EXPIRE']
    X_test_boot = X_test_boot.drop(columns = ['30_DAYS_EXPIRE'])

    # calculating and updating dataframe with metrics using test dataset bootstrap samples        
    y_pred_proba = retrained_model.predict(X_test_boot)
    fpr, tpr, thresholds = metrics.roc_curve(y_test_boot, y_pred_proba[:,0], pos_label=1)    
    y_pred = (retrained_model.predict(X_test_boot) > 0.5).astype("int32").flatten()

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
Boot_stats.to_csv('Results/eCrit_to_MIMIC_Domain_TL_30_DAYS_EXPIRE_Bootstrap_100p.csv')


# # 2. AKI processing started

# Extracting train dataset from master files
# AKI_data_miss indicates missing data, which prevents identifying presence or absence of AKI in patients
y_train = y_train_df[y_train_df['AKI_data_miss'] != 1]
# AKI_adm indicates presence of AKI during admission, which is an exclusion criteria  
y_train = y_train[y_train['AKI_adm'] != 1]
# selecting X_train rows for the corresponsing y_train rows using SUBJECT_ID
X_train = X_train_df.loc[X_train_df['SUBJECT_ID'].isin(y_train['SUBJECT_ID'])]
# dropping SUBJECT_ID column
X_train = X_train.drop(columns = ['SUBJECT_ID'])
# selecting AKI column
y_train = y_train['AKI']

# Applying smote, first 50% oversampling and then 100% undersampling    
over = SMOTE(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=1.0)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps) 
X_r, y_r = pipeline.fit_resample(X_train, y_train)

# Extracting validation and test dataset from master files  
y_val = y_val_df[y_val_df['AKI_data_miss'] != 1]
y_val = y_val[y_val['AKI_adm'] != 1]
X_val = X_val_df.loc[X_val_df['SUBJECT_ID'].isin(y_val['SUBJECT_ID'])]    
X_val = X_val.drop(columns = ['SUBJECT_ID'])
y_val = y_val['AKI']

y_test = y_test_df[y_test_df['AKI_data_miss'] != 1]
y_test = y_test[y_test['AKI_adm'] != 1]
X_test = X_test_df.loc[X_test_df['SUBJECT_ID'].isin(y_test['SUBJECT_ID'])]    
X_test = X_test.drop(columns = ['SUBJECT_ID'])
y_test = y_test['AKI']

# loading pre-trained model   
model_name = model_path+"eCrit_AKI_R20_SM_1708.h5"
new_model = tf.keras.models.load_model(model_name)

# removing top layers and replacing them with custom new hidden layer along with new output layer	
# All the pre-trained layers are changed to not trainable, only new custom layers are trainable		
retrained_model = tf.keras.models.Sequential()
for layer in new_model.layers[:-3]:
    layer.trainable = False
    retrained_model.add(layer)
retrained_model.add((tf.keras.layers.Dense(64,activation='PReLU',kernel_initializer='HeNormal',kernel_regularizer=tf.keras.regularizers.l1(l1=0.001),name='new_dense')))
retrained_model.add(tf.keras.layers.Dropout(0.3,name='new_dropout'))    
retrained_model.add(tf.keras.layers.Dense(1, activation='sigmoid',name='main_output'))
input_shape = (None, 104)
retrained_model.build(input_shape)

# compiling and fitting the model    
opt = tf.keras.optimizers.Adam(lr=0.0001)
retrained_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[AUC(),Precision(),Recall()])
history = retrained_model.fit(X_r, y_r,epochs = 50, \
          verbose = 0,  callbacks= [early_stop_fcnn],batch_size = 32,validation_data=(X_val, y_val))
          
# changing all the layers to be trainable, including the pre-trained layers
retrained_model.trainable = True

# compiling and fitting the model
opt = tf.keras.optimizers.Adam(lr=0.0001)
retrained_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[AUC(),Precision(),Recall()])
history = retrained_model.fit(X_r, y_r,epochs = 50, \
          verbose = 0,  callbacks= [early_stop_fcnn],batch_size = 32,validation_data=(X_val, y_val))


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
    y_pred_proba = retrained_model.predict(X_test_boot)
    fpr, tpr, thresholds = metrics.roc_curve(y_test_boot, y_pred_proba[:,0], pos_label=1)    
    y_pred = (retrained_model.predict(X_test_boot) > 0.5).astype("int32").flatten()

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
Boot_stats.to_csv('Results/eCrit_to_MIMIC_Domain_TL_AKI_Bootstrap_100p.csv')    


# # 3. H_LOS processing started

# Extracting train, validation and test datasets from master files
# H_LOS_miss indicates missing data, which prevents identifying presence or absence of H_LOS in patients
y_train = y_train_df[y_train_df['H_LOS_miss'] != 1]
# To avoid outliers, only bottom 98 percentile data is included
y_train = y_train[y_train.H_LOS < np.percentile(y_train.H_LOS,98)]
# selecting X_train rows for the corresponsing y_train rows using SUBJECT_ID
X_train = X_train_df.loc[X_train_df['SUBJECT_ID'].isin(y_train['SUBJECT_ID'])]
# dropping SUBJECT_ID column
X_train = X_train.drop(columns = ['SUBJECT_ID'])
# selecting H_LOS column
y_train = y_train['H_LOS']

y_val = y_val_df[y_val_df['H_LOS_miss'] != 1]
y_val = y_val[y_val.H_LOS < np.percentile(y_val.H_LOS,98)]
X_val = X_val_df.loc[X_val_df['SUBJECT_ID'].isin(y_val['SUBJECT_ID'])]
X_val = X_val.drop(columns = ['SUBJECT_ID'])
y_val = y_val['H_LOS']

y_test = y_test_df[y_test_df['H_LOS_miss'] != 1]
y_test = y_test[y_test.H_LOS < np.percentile(y_test.H_LOS,98)]
X_test = X_test_df.loc[X_test_df['SUBJECT_ID'].isin(y_test['SUBJECT_ID'])]
X_test = X_test.drop(columns = ['SUBJECT_ID'])
y_test = y_test['H_LOS']

# loading pre-trained model
model_name = model_path+"eCrit_H_LOS_R20_LM_827.h5"
new_model = tf.keras.models.load_model(model_name)

# removing top layers and replacing them with custom new hidden layer along with new output layer	
# All the pre-trained layers are changed to not trainable, only new custom layers are trainable	
retrained_model = tf.keras.models.Sequential()
for layer in new_model.layers[:-3]:
    layer.trainable = False
    retrained_model.add(layer)
retrained_model.add((tf.keras.layers.Dense(64,activation='PReLU',kernel_initializer='HeNormal',kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3),name='new_dense')))
retrained_model.add(tf.keras.layers.Dropout(0.4,name='new_dropout'))
retrained_model.add(tf.keras.layers.Dense(1, activation='linear',name='main_output'))    
input_shape = (None, 104)
retrained_model.build(input_shape)

# compiling and fitting the model
opt = tf.keras.optimizers.Adam(lr=0.001)
retrained_model.compile(optimizer=opt, loss='mae', metrics=['mae','mse'])
history = retrained_model.fit(X_train, y_train,epochs = 50, \
          verbose = 0,  callbacks= [early_stop_fcnn],batch_size = 64,validation_data=(X_val, y_val))
          
# changing all the layers to be trainable, including the pre-trained layers
retrained_model.trainable = True

# compiling and fitting the model
opt = tf.keras.optimizers.Adam(lr=0.001)
retrained_model.compile(optimizer=opt, loss='mae', metrics=['mae','mse'])
history = retrained_model.fit(X_train, y_train,epochs = 50, \
          verbose = 0,  callbacks= [early_stop_fcnn],batch_size = 64,validation_data=(X_val, y_val))

# BOOTSTRAP
# concatenating X and y test dataset for bootstrapping
data = pd.concat([X_test,y_test], axis=1).reset_index(drop=True)

# creating and initializing DataFrame for storing bootstrap results    
Boot_stats = pd.DataFrame(columns=['MSE CI', 'MAE CI'])

# loop for bootstrap sampling
for i in range(N):
    # extracting bootstrap samples with replacement from test dataset
    X_test_boot = data.sample(frac=1, replace=True, random_state=r_states[i])
    
    # splitting bootstrap samples into X and y        
    y_test_boot = X_test_boot['H_LOS']
    X_test_boot = X_test_boot.drop(columns = ['H_LOS'])

    # calculating and updating dataframe with metrics using test dataset bootstrap samples
    y_pred = retrained_model.predict(X_test_boot)        
    temp_dict = {'MSE CI': metrics.mean_squared_error(y_test_boot, y_pred), 'MAE CI': metrics.mean_absolute_error(y_test_boot, y_pred)}      
    temp = pd.DataFrame([temp_dict])      
    Boot_stats = pd.concat([Boot_stats,temp], axis=0).reset_index(drop=True)

# saving bootstrap results for each random state and for each outcome
Boot_stats.to_csv('Results/eCrit_to_MIMIC_Domain_TL_H_LOS_Bootstrap_100p.csv')

# # 4. ICU_LOS processing started

# Extracting train, validation and test datasets from master files
# To avoid outliers, only bottom 98 percentile data is included
y_train = y_train_df[y_train_df.ICU_LOS < np.percentile(y_train_df.ICU_LOS,98)]
# selecting X_train rows for the corresponsing y_train rows using SUBJECT_ID
X_train = X_train_df.loc[X_train_df['SUBJECT_ID'].isin(y_train['SUBJECT_ID'])]
# dropping SUBJECT_ID column
X_train = X_train.drop(columns = ['SUBJECT_ID'])
# selecting ICU_LOS column
y_train = y_train['ICU_LOS'] 

y_val = y_val_df[y_val_df.ICU_LOS < np.percentile(y_val_df.ICU_LOS,98)]
X_val = X_val_df.loc[X_val_df['SUBJECT_ID'].isin(y_val['SUBJECT_ID'])]
X_val = X_val.drop(columns = ['SUBJECT_ID'])
y_val = y_val['ICU_LOS']

y_test = y_test_df[y_test_df.ICU_LOS < np.percentile(y_test_df.ICU_LOS,98)]
X_test = X_test_df.loc[X_test_df['SUBJECT_ID'].isin(y_test['SUBJECT_ID'])]
X_test = X_test.drop(columns = ['SUBJECT_ID'])
y_test = y_test['ICU_LOS']

# loading pre-trained model
model_name = model_path+"eCrit_ICU_LOS_R20_LM_772.h5"
new_model = tf.keras.models.load_model(model_name)

# removing top layers and replacing them with custom new hidden layer along with new output layer	
# All the pre-trained layers are changed to not trainable, only new custom layers are trainable		
retrained_model = tf.keras.models.Sequential()
for layer in new_model.layers[:-3]:
    layer.trainable = False
    retrained_model.add(layer)
retrained_model.add((tf.keras.layers.Dense(64,activation='LeakyReLU',kernel_initializer='HeNormal',kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3),name='new_dense')))
retrained_model.add(tf.keras.layers.Dropout(0.3,name='new_dropout'))
retrained_model.add(tf.keras.layers.Dense(1, activation='linear',name='main_output'))
input_shape = (None, 104)
retrained_model.build(input_shape)

# compiling and fitting the model    
opt = tf.keras.optimizers.Adam(lr=0.0001)
retrained_model.compile(optimizer=opt, loss='mae', metrics=['mae','mse'])
history = retrained_model.fit(X_train, y_train,epochs = 50, \
          verbose = 0,  callbacks= [early_stop_fcnn],batch_size = 32,validation_data=(X_val, y_val))

# changing all the layers to be trainable, including the pre-trained layers
retrained_model.trainable = True

# compiling and fitting the model
opt = tf.keras.optimizers.Adam(lr=0.0001)
retrained_model.compile(optimizer=opt, loss='mae', metrics=['mae','mse'])
history = retrained_model.fit(X_train, y_train,epochs = 50, \
          verbose = 0,  callbacks= [early_stop_fcnn],batch_size = 32,validation_data=(X_val, y_val))

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
    y_pred = retrained_model.predict(X_test_boot)
    temp_dict = {'MSE CI': metrics.mean_squared_error(y_test_boot, y_pred), 'MAE CI': metrics.mean_absolute_error(y_test_boot, y_pred)}      
    temp = pd.DataFrame([temp_dict])      
    Boot_stats = pd.concat([Boot_stats,temp], axis=0).reset_index(drop=True)

# saving bootstrap results for each random state and for each outcome
Boot_stats.to_csv('Results/eCrit_to_MIMIC_Domain_TL_ICU_LOS_Bootstrap_100p.csv')

