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

# loading the feature output file from previous script, 04_MIMIC_mount_CHARTEVENTS.py.
X_feature_df = pd.read_pickle('Temp/MIMIC_Features.pickle', compression = 'zip')

# selecting columns
X_feature_df = X_feature_df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'GENDER', 'AGE_r', 'Urine_Volumes', 'Norepinephrine', 
                             'PhenyLEPHrine', 'Vasopressin', 'DOBUTamine', 'DOPamine', 'EPINEPHrine', 'Eye_Opening_5percentile', 
                             'Eye_Opening_median', 'Eye_Opening_IQR', 'Eye_Opening_95percentile', 'Verbal_Response_5percentile', 
                             'Verbal_Response_median', 'Verbal_Response_IQR', 'Verbal_Response_95percentile', 'Motor_Response_5percentile', 
                             'Motor_Response_median', 'Motor_Response_IQR', 'Motor_Response_95percentile', 'Heart_Rate_5percentile', 'Heart_Rate_median',
                             'Heart_Rate_IQR', 'Heart_Rate_95percentile', 'SpO2_5percentile', 'SpO2_median', 'SpO2_IQR', 'SpO2_95percentile', 
                             'PH_Arterial_5percentile', 'PH_Arterial_median', 'PH_Arterial_IQR', 'PH_Arterial_95percentile', 'BP_Systolic_5percentile',
                             'BP_Systolic_median', 'BP_Systolic_IQR', 'BP_Systolic_95percentile', 'FiO2_5percentile', 'FiO2_median', 'FiO2_IQR', 
                             'FiO2_95percentile', 'BP_Diastolic_5percentile', 'BP_Diastolic_median', 'BP_Diastolic_IQR', 'BP_Diastolic_95percentile', 
                             'Urea_Blood_5percentile', 'Urea_Blood_median', 'Urea_Blood_IQR', 'Urea_Blood_95percentile', 'Creatinine_Blood_5percentile',
                             'Creatinine_Blood_median', 'Creatinine_Blood_IQR', 'Creatinine_Blood_95percentile', 'CO2_Content_Blood_5percentile', 
                             'CO2_Content_Blood_median', 'CO2_Content_Blood_IQR', 'CO2_Content_Blood_95percentile', 'Glucose_Blood_5percentile', 
                             'Glucose_Blood_median', 'Glucose_Blood_IQR', 'Glucose_Blood_95percentile', 'Potassium_Blood_5percentile', 
                             'Potassium_Blood_median', 'Potassium_Blood_IQR', 'Potassium_Blood_95percentile', 'Sodium_Blood_5percentile', 
                             'Sodium_Blood_median', 'Sodium_Blood_IQR', 'Sodium_Blood_95percentile', 'Respiratory_Rate_5percentile', 
                             'Respiratory_Rate_median', 'Respiratory_Rate_IQR', 'Respiratory_Rate_95percentile', 'PCO2_Arterial_5percentile', 
                             'PCO2_Arterial_median', 'PCO2_Arterial_IQR', 'PCO2_Arterial_95percentile', 'PO2_Arterial_5percentile', 
                             'PO2_Arterial_median', 'PO2_Arterial_IQR', 'PO2_Arterial_95percentile', 'Hemoglobin_5percentile', 'Hemoglobin_median', 
                             'Hemoglobin_IQR', 'Hemoglobin_95percentile', 'Hematocrit_5percentile', 'Hematocrit_median', 'Hematocrit_IQR', 
                             'Hematocrit_95percentile', 'RBC_5percentile', 'RBC_median', 'RBC_IQR', 'RBC_95percentile', 'WBC_5percentile', 
                             'WBC_median', 'WBC_IQR', 'WBC_95percentile', 'Dialysis_Flag', 'Mechanical_Ventilation_Flag', 'ADMISSION_WEIGHT', 
                             'GCS_5percentile', 'GCS_95percentile', 'GCS_IQR', 'GCS_median']]


# one-hot encoding the Gender column
X_feature_df['GENDER'] = list(map(lambda a: 1 if a == 'M' else 0, X_feature_df['GENDER']))

# to be compatible with eCritical database, columns (features) should be same. 
# So, missingness for columns is calculated, saved to file and printed but columns are not dropped.
miss_df = pd.DataFrame(X_feature_df.isnull().sum(), columns = ['missing'])
miss_df.to_csv('Temp/MIMIC_missing_data.csv')
missing_list = miss_df[miss_df['missing']>len(X_feature_df)*0.3].index
print(missing_list)
print(miss_df.loc[missing_list])

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
cont_features = list(set(X_feature_df.columns.tolist()) - set(binary_features+['SUBJECT_ID','HADM_ID','ICUSTAY_ID']))

# loading the target output file from previous script, 04_MIMIC_mount_CHARTEVENTS.py.
Y_target_df = pd.read_pickle('Temp/MIMIC_Target.pickle', compression = 'zip')

# selecting columns
Y_target_df = Y_target_df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'DIAGNOSIS', 'H_LOS', 'ICU_LOS', '30_DAYS_EXPIRE', 
                           'AKI_data_miss', 'AKI', 'HRF', 'HRF_data_miss', 'Delirium', 'Delirium_data_miss', 'AKI_adm', 
                           'HRF_adm', 'Delirium_adm']]

# splitting data into train, validation and test datasets.
X_train_df, X, y_train_df, y = train_test_split(X_feature_df, Y_target_df, test_size=0.3, random_state = 5)
X_val_df, X_test_df, y_val_df, y_test_df = train_test_split(X, y, test_size=0.5, random_state = 5)

# resetting index and dropping columns
y_train_df.reset_index(inplace = True, drop=True)
X_train = X_train_df.drop(columns=['SUBJECT_ID','HADM_ID','ICUSTAY_ID']).reset_index(drop=True)
y_val_df.reset_index(inplace = True, drop=True)
X_val = X_val_df.drop(columns=['SUBJECT_ID','HADM_ID','ICUSTAY_ID']).reset_index(drop=True)
y_test_df.reset_index(inplace = True, drop=True)
X_test = X_test_df.drop(columns=['SUBJECT_ID','HADM_ID','ICUSTAY_ID']).reset_index(drop=True)


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


# adding SUBJECT_ID column back
X_train_imp_scale = X_train_imp_scale.assign(SUBJECT_ID = X_train_df['SUBJECT_ID'].values)
X_val_imp_scale = X_val_imp_scale.assign(SUBJECT_ID = X_val_df['SUBJECT_ID'].values)
X_test_imp_scale = X_test_imp_scale.assign(SUBJECT_ID = X_test_df['SUBJECT_ID'].values)

# saving master train, validation and test files
X_train_imp_scale.to_pickle('Data/MIMIC_master_X_train.pickle', compression = 'zip')
X_train_imp_scale.to_csv('Data/MIMIC_master_X_train.csv')

X_val_imp_scale.to_pickle('Data/MIMIC_master_X_val.pickle', compression = 'zip')
X_val_imp_scale.to_csv('Data/MIMIC_master_X_val.csv')

X_test_imp_scale.to_pickle('Data/MIMIC_master_X_test.pickle', compression = 'zip')
X_test_imp_scale.to_csv('Data/MIMIC_master_X_test.csv')

y_train_df.to_pickle('Data/MIMIC_master_y_train.pickle', compression = 'zip')
y_train_df.to_csv('Data/MIMIC_master_y_train.csv')

y_val_df.to_pickle('Data/MIMIC_master_y_val.pickle', compression = 'zip')
y_val_df.to_csv('Data/MIMIC_master_y_val.csv')

y_test_df.to_pickle('Data/MIMIC_master_y_test.pickle', compression = 'zip')
y_test_df.to_csv('Data/MIMIC_master_y_test.csv')

# printing check point
print('files saved')


