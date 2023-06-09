#!/usr/bin/env python
# coding: utf-8
# author: Maruthi Mutnuri (maruthi2008@gmail.com)

# loading libraries
import pandas as pd 
import numpy as np

# loading database file
PRESCRIPTIONS_s = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/PRESCRIPTIONS.pickle', compression = 'zip')

# selecting columns
PRESCRIPTIONS_s = PRESCRIPTIONS_s[['PATIENT_ID','ADMISSION_ID','ITEM_ID', 'START_DATETIME','VALUE_NUM']]

# renaming column
PRESCRIPTIONS_s.rename(columns={'START_DATETIME': 'DATETIME'}, inplace = True)

# sorting by patient ID
PRESCRIPTIONS_s.sort_values(by = 'PATIENT_ID',  inplace = True)

# converting to datetime datatype 
PRESCRIPTIONS_s['DATETIME'] = pd.to_datetime(PRESCRIPTIONS_s['DATETIME'])

# loading the output file from previous script, 02_eCrit_mount_INTERVENTIONS.py.
df_eCrit = pd.read_csv('Temp/eCrit_Features_INT.csv', sep = ',')

# converting to datetime datatype 
df_eCrit['ICU_ADMIT_DATETIME'] = pd.to_datetime(df_eCrit['ICU_ADMIT_DATETIME'])

# creating a temporary dataframe with just admission ID.
temp = df_eCrit[['ADMISSION_ID']]

# merging dataframes to filter rows using ADMISSION_ID from eCrit_Features_INT file.
PRESCRIPTIONS_s = pd.merge(PRESCRIPTIONS_s, temp, on='ADMISSION_ID', how='inner')


# creating a reference dictionary for features.
feature_names = {'Norepinephrine': ['I698'], 'PhenyLEPHrine': ['I702'], 'Vasopressin': ['I706'], 'DOBUTamine': ['I707'], 
                 'DOPamine': ['I708'], 'EPINEPHrine': ['I709']}

# Initializing
data_list = []

# loop to create prescription feature columns.
for row in df_eCrit.iterrows():
    # copying the patient record from df_eCrit
    record = row[1].copy()
    
    # filtering rows to keep records of the selected patient in each iteration
    df_prescription = PRESCRIPTIONS_s.loc[PRESCRIPTIONS_s['ADMISSION_ID'] == row[1]['ADMISSION_ID']]
    
    # filtering rows to keep records from first 24 hours of ICU admission   
    df_prescription = df_prescription[(df_prescription['DATETIME'] - record['ICU_ADMIT_DATETIME']).between(pd.Timedelta('0s'),pd.Timedelta('1d'))]

    # creating prescription feature columns
    record['Norepinephrine'] = 1 if df_prescription['ITEM_ID'].isin(feature_names['Norepinephrine']).sum() >= 1 else 0
    record['PhenyLEPHrine'] = 1 if df_prescription['ITEM_ID'].isin(feature_names['PhenyLEPHrine']).sum() >= 1 else 0
    record['Vasopressin'] = 1 if df_prescription['ITEM_ID'].isin(feature_names['Vasopressin']).sum() >= 1 else 0
    record['DOBUTamine'] = 1 if df_prescription['ITEM_ID'].isin(feature_names['DOBUTamine']).sum() >= 1 else 0
    record['DOPamine'] = 1 if df_prescription['ITEM_ID'].isin(feature_names['DOPamine']).sum() >= 1 else 0
    record['EPINEPHrine'] = 1 if df_prescription['ITEM_ID'].isin(feature_names['EPINEPHrine']).sum() >= 1 else 0

    # appending updated patient record to the list
    data_list.append(record)

# converting list to dataframe         
X_feature_df = pd.DataFrame(data_list)

# saving file
X_feature_df.to_csv('Temp/eCrit_Features_Rx.csv')

# printing check point
print('eCrit_Features_Rx files saved')


