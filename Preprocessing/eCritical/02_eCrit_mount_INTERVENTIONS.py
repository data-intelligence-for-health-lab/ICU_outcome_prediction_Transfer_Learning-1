#!/usr/bin/env python
# coding: utf-8
# author: Maruthi Mutnuri (maruthi2008@gmail.com)

# loading libraries
import pandas as pd 
import numpy as np

# loading database file
INTERVENTIONS_s = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/INTERVENTIONS.pickle', compression = 'zip')
# selecting columns
INTERVENTIONS_s = INTERVENTIONS_s[['PATIENT_ID','ADMISSION_ID','ITEM_ID', 'START_DATETIME', 'VALUE_NUM']]

# renaming column
INTERVENTIONS_s.rename(columns={'START_DATETIME': 'DATETIME'}, inplace = True)

# sorting by patient ID
INTERVENTIONS_s.sort_values(by = 'PATIENT_ID',  inplace = True)

# converting to datetime datatype   
INTERVENTIONS_s['DATETIME'] = pd.to_datetime(INTERVENTIONS_s['DATETIME'])

# loading the output file from previous script, 01_eCrit_mount_ADMISSIONS.py.
df_eCrit = pd.read_csv('Temp/eCrit_ADMISSION_PATIENTS.csv', sep = ',')

# converting to datetime datatype 
df_eCrit['ICU_ADMIT_DATETIME'] = pd.to_datetime(df_eCrit['ICU_ADMIT_DATETIME'])

# creating a temporary dataframe with just admission ID.
temp = df_eCrit[['ADMISSION_ID']]

# merging dataframes to filter rows using ADMISSION_ID from eCrit_ADMISSION_PATIENTS file.
INTERVENTIONS_s = pd.merge(INTERVENTIONS_s, temp, on='ADMISSION_ID', how='inner')

# creating a dictionary of required ITEM_ID's, to extract features from the database files.
lab_items = {'INTERVENTIONS': ['I299','I339','I399','I436','I455','I456','I334','I356']}

# creating a reference dictionary for features.
feature_names = {'Dialysis_Flag': ['I118','I262'], 'Mechanical_Ventilation_Flag': ['I263', 'I584']}


# function to extract the ITEM_ID's from the range 'I118 - I262' and 'I263 - I584'
def get_list(feature):
    first = int(feature[0][1:])
    last = int(feature[1][1:])+1
    return(['I'+x for x in list(map(str,range(first,last)))])

# getting the list of ITEM_ID's
Dialysis_feature_list = get_list(feature_names['Dialysis_Flag'])
MV_feature_list = get_list(feature_names['Mechanical_Ventilation_Flag'])

# Initializing
X_data_list = []

# loop to create Dialysis_Flag and Mechanical_Ventilation_Flag columns. 
for row in df_eCrit.iterrows():
    # filtering rows to keep records of the selected patient in each iteration
    df_intervention = INTERVENTIONS_s.loc[INTERVENTIONS_s['ADMISSION_ID'] == row[1]['ADMISSION_ID']]
    
    # copying the patient record from df_eCrit
    record = row[1].copy()
    
    # filtering rows to keep records from first 24 hours of ICU admission
    df_intervention = df_intervention[(df_intervention['DATETIME'] - record['ICU_ADMIT_DATETIME']).between(pd.Timedelta('0s'),pd.Timedelta('1d'))]
    
    # creating Dialysis_Flag and Mechanical_Ventilation_Flag columns
    record['Dialysis_Flag'] = 1 if df_intervention['ITEM_ID'].isin(Dialysis_feature_list).sum() >= 1 else 0
    record['Mechanical_Ventilation_Flag'] = 1 if df_intervention['ITEM_ID'].isin(MV_feature_list).sum() >= 1 else 0

    # appending updated patient record to the list
    X_data_list.append(record)

# converting list to dataframe     
X_feature_df = pd.DataFrame(X_data_list)

# filtering rows using lab_items. keeping records of only required ITEM_ID's
INTERVENTIONS_s = INTERVENTIONS_s[INTERVENTIONS_s['ITEM_ID'].isin(lab_items['INTERVENTIONS'])]

# saving files
INTERVENTIONS_s.to_csv('Temp/INTERVENTIONS_processed.csv')
X_feature_df.to_csv('Temp/eCrit_Features_INT.csv')

# printing check point
print('eCrit_Features_INT file saved')






