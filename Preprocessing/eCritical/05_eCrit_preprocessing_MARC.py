#!/usr/bin/env python
# coding: utf-8
# author: Maruthi Mutnuri (maruthi2008@gmail.com)

# loading libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# loading the feature output file from previous script, 04_eCrit_mount_MEASUREMENTS.py.
X_feature_df = pd.read_pickle('Temp/eCrit_Features.pickle', compression = 'zip')

# one-hot encoding the Gender column
X_feature_df['GENDER'] = list(map(lambda a: 1 if a == 'M' else 0, X_feature_df['GENDER']))

# calculating the missingness (columns) and storing it in a dataframe
miss_df = pd.DataFrame(X_feature_df.isnull().sum(), columns = ['missing'])

# getting a list of features with more than 30% missingness
missing_list = miss_df[miss_df['missing']>len(X_feature_df)*0.3].index

# dropping features with more than 30% missing values
X_feature_df = X_feature_df.drop(columns = missing_list)

# calculating the missingness (rows) and storing it in a dataframe
miss_row_df = pd.DataFrame(X_feature_df.isnull().sum(axis=1), columns = ['missing'])

# getting a list of rows with more than 20% missingness
missing_row_list = miss_row_df[miss_row_df['missing']>X_feature_df.shape[1]*0.2].index

# dropping rows with more than 20% missing values
X_feature_df = X_feature_df.drop(index = missing_row_list)

# creating a list of binary features
binary_features = ['GENDER', 'Dialysis_Flag', 'Mechanical_Ventilation_Flag', 'Norepinephrine', 'PhenyLEPHrine', 
                   'Vasopressin', 'DOBUTamine', 'DOPamine', 'EPINEPHrine'] 

# creating a list of continuous features
cont_features = list(set(X_feature_df.columns.tolist()) - set(binary_features+['PATIENT_ID', 'ADMISSION_ID']))

# loading the target output file from previous script, 04_eCrit_mount_MEASUREMENTS.py.
Y_target_df = pd.read_pickle('Temp/eCrit_Target.pickle', compression = 'zip')

# splitting data into train, validation and test datasets.
X_train_df, X, y_train_df, y = train_test_split(X_feature_df, Y_target_df, test_size=0.2, random_state = 5)
X_val_df, X_test_df, y_val_df, y_test_df = train_test_split(X, y, test_size=0.5, random_state = 5)

# resetting index and dropping columns
y_train_df.reset_index(inplace = True, drop=True)
X_train = X_train_df.drop(columns=['ADMISSION_ID','PATIENT_ID']).reset_index(drop=True)
y_val_df.reset_index(inplace = True, drop=True)
X_val = X_val_df.drop(columns=['ADMISSION_ID','PATIENT_ID']).reset_index(drop=True)
y_test_df.reset_index(inplace = True, drop=True)
X_test = X_test_df.drop(columns=['ADMISSION_ID','PATIENT_ID']).reset_index(drop=True)


# imputing missing values using IterativeImputer with linear regression as estimator.
# imputing train, validation and test datasets separately
lr = LinearRegression()
imp = IterativeImputer(estimator=lr,missing_values=np.nan, max_iter=30, verbose=0, imputation_order='roman',random_state=5)
X_train_imp=pd.DataFrame(imp.fit_transform(X_train), columns = X_train.columns)
X_val_imp=pd.DataFrame(imp.fit_transform(X_val), columns = X_val.columns)
X_test_imp=pd.DataFrame(imp.fit_transform(X_test), columns = X_test.columns)


# scaling continuous features using standard scaler
# scaling train, validation and test datasets separately
scaler = StandardScaler()
scaler.fit(X_train_imp[cont_features])
X_train_imp_scale = pd.DataFrame(scaler.transform(X_train_imp[cont_features]), columns = cont_features)
X_val_imp_scale = pd.DataFrame(scaler.transform(X_val_imp[cont_features]), columns = cont_features)
X_test_imp_scale = pd.DataFrame(scaler.transform(X_test_imp[cont_features]), columns = cont_features)

# concatinating continuous and binary features
X_train_imp_scale = pd.concat([X_train_imp_scale,X_train_imp[binary_features]], axis=1)
X_val_imp_scale = pd.concat([X_val_imp_scale,X_val_imp[binary_features]], axis=1)
X_test_imp_scale = pd.concat([X_test_imp_scale,X_test_imp[binary_features]], axis=1)

# adding PATIENT_ID column back
X_train_imp_scale = X_train_imp_scale.assign(PATIENT_ID = X_train_df['PATIENT_ID'].values)
X_val_imp_scale = X_val_imp_scale.assign(PATIENT_ID = X_val_df['PATIENT_ID'].values)
X_test_imp_scale = X_test_imp_scale.assign(PATIENT_ID = X_test_df['PATIENT_ID'].values)

# saving master train, validation and test files
X_train_imp_scale.to_pickle('Data/eCrit_master_X_train.pickle', compression = 'zip')
X_train_imp_scale.to_csv('Data/eCrit_master_X_train.csv')

X_val_imp_scale.to_pickle('Data/eCrit_master_X_val.pickle', compression = 'zip')
X_val_imp_scale.to_csv('Data/eCrit_master_X_val.csv')

X_test_imp_scale.to_pickle('Data/eCrit_master_X_test.pickle', compression = 'zip')
X_test_imp_scale.to_csv('Data/eCrit_master_X_test.csv')

y_train_df.to_pickle('Data/eCrit_master_y_train.pickle', compression = 'zip')
y_train_df.to_csv('Data/eCrit_master_y_train.csv')

y_val_df.to_pickle('Data/eCrit_master_y_val.pickle', compression = 'zip')
y_val_df.to_csv('Data/eCrit_master_y_val.csv')

y_test_df.to_pickle('Data/eCrit_master_y_test.pickle', compression = 'zip')
y_test_df.to_csv('Data/eCrit_master_y_test.csv')

# printing check point
print('All eCrit X and Y files saved')



