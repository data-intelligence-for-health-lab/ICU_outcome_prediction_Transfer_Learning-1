#!/usr/bin/env python
# coding: utf-8
# author: Maruthi Mutnuri (maruthi2008@gmail.com)

# loading libraries
import pandas as pd 
import numpy as np


# loading database file
INPUTEVENTS_CV = pd.read_csv("/home/maruthikumar.mutnuri/physionet/files/mimiciii/data/INPUTEVENTS_CV.csv.gz", sep = ',', compression = 'gzip',
                            usecols = ['SUBJECT_ID','HADM_ID','ICUSTAY_ID','ITEMID', 'CHARTTIME', 'AMOUNT'])

# renaming column
INPUTEVENTS_CV.rename(columns={'CHARTTIME': 'DATETIME'}, inplace = True)

# converting to datetime datatype
INPUTEVENTS_CV['DATETIME'] = pd.to_datetime(INPUTEVENTS_CV['DATETIME'])

# converting to int datatype
INPUTEVENTS_CV['HADM_ID'] = (INPUTEVENTS_CV['HADM_ID']).astype(pd.Int64Dtype())
INPUTEVENTS_CV['ICUSTAY_ID'] = (INPUTEVENTS_CV['ICUSTAY_ID']).astype(pd.Int64Dtype())

# dropping rows with missing values in 'HADM_ID','ICUSTAY_ID', and 'AMOUNT' columns
INPUTEVENTS_CV.dropna(subset=['HADM_ID','ICUSTAY_ID','AMOUNT'],inplace=True)

# creating a dictionary of required ITEMID's, to extract features from the database files.
lab_items = {'CHARTEVENTS': [6, 51, 184, 211, 442, 454, 455, 490, 618, 646, 723, 777, 778, 779, 780, 781, 791, 807, 811, 
                             813, 814, 833, 837, 861, 1126, 1127, 1162, 1525, 1529, 1536, 1542, 1673, 3420, 3422, 3651, 
                             3737, 3750, 3784, 3785, 3799, 3803, 3808, 3809, 3810, 3834, 3835, 3836, 3837, 4197, 4200, 
                             4201, 4753, 6701, 8364, 8368, 8440, 8441, 8555, 220045, 220050, 220051, 220179, 220180, 
                             220210, 220227, 220228, 220235, 220277, 220545, 220546, 220615, 220621, 220640, 220645, 
                             220739, 223830, 223835, 223900, 223901, 224689, 224690, 225309, 225310, 225624, 225664, 
                             225698, 226534, 226535, 226537, 226540, 226756, 226757, 226758, 227011, 227012, 227014, 
                             227442, 227464, 228112,148, 152,582, 225126,227124,505,506,3555,720,223849,220339,224686,684,
                            763,3580,3693,226512,224639,228300,228301,228302,228303,228332],
            'OUTPUTEVENTS': [40055, 40056, 40057, 40061, 40065, 40069, 40085, 40086, 40094, 40096, 40288,
                             40405, 40428, 40473, 40651, 40715, 42676, 43175, 43373, 43431, 44325, 
                             226557, 226558, 226559, 226560, 226561, 226563, 226564, 226565, 226567, 
                             226584, 226627, 226631, 227488, 227489],
            'INPUTEVENTS_CV': [30047,30120,30044,30119,30309,30127,30128,30051,30043,30307,30042,30306],
            'INPUTEVENTS_MV': [221906,221289,221749,222315,221662,221653],
            'LABEVENTS': [50804, 50809, 50810, 50811, 50816, 50818, 50820, 50821, 50822, 50824, 50912, 50931, 50971, 
                          50983, 51006, 51221, 51222, 51279, 51300, 51301, 51529],
            'Norepinephrine': [30047,30120,221906],
            'PhenyLEPHrine': [30127,30128,221749],
            'Vasopressin': [30051,222315],
            'DOBUTamine': [30042,30306,221653],
            'DOPamine': [30043,30307,221662],
            'EPINEPHrine': [30044,30119,30309,221289]}

# filtering rows using lab_items. keeping records of only required ITEM_ID's
INPUTEVENTS_CV = INPUTEVENTS_CV[INPUTEVENTS_CV['ITEMID'].isin(lab_items['INPUTEVENTS_CV'])]

# loading database file
INPUTEVENTS_MV = pd.read_csv("/home/maruthikumar.mutnuri/physionet/files/mimiciii/data/INPUTEVENTS_MV.csv.gz", sep = ',', compression = 'gzip',
                            usecols = ['SUBJECT_ID','HADM_ID','ICUSTAY_ID','ITEMID', 'STARTTIME', 'AMOUNT'])
# renaming column
INPUTEVENTS_MV.rename(columns={'STARTTIME': 'DATETIME'}, inplace = True)

# converting to datetime datatype
INPUTEVENTS_MV['DATETIME'] = pd.to_datetime(INPUTEVENTS_MV['DATETIME'])

# converting to int datatype
INPUTEVENTS_MV['HADM_ID'] = (INPUTEVENTS_MV['HADM_ID']).astype(pd.Int64Dtype())
INPUTEVENTS_MV['ICUSTAY_ID'] = (INPUTEVENTS_MV['ICUSTAY_ID']).astype(pd.Int64Dtype())

# dropping rows with missing values in 'HADM_ID','ICUSTAY_ID', and 'AMOUNT' columns
INPUTEVENTS_MV.dropna(subset=['HADM_ID','ICUSTAY_ID','AMOUNT'],inplace=True)

# filtering rows using lab_items. keeping records of only required ITEM_ID's
INPUTEVENTS_MV = INPUTEVENTS_MV[INPUTEVENTS_MV['ITEMID'].isin(lab_items['INPUTEVENTS_MV'])]

# concatenating dataframes
INPUTEVENTS = pd.concat([INPUTEVENTS_CV,INPUTEVENTS_MV])

# loading the output file from previous script, 02_MIMIC_mount_LABEVENTS_OUTPUTEVENTS.py.
df_MIMIC = pd.read_pickle('Temp/MIMIC_Features_OUT.pickle', compression = 'zip')

# selecting columns
df_MIMIC = df_MIMIC[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'ICU_LOS', 'GENDER',
       'H_LOS', 'AGE_r', '30_DAYS_EXPIRE', 'Urine_Volumes']]
       
# converting to datetime datatype       
df_MIMIC['INTIME'] = pd.to_datetime(df_MIMIC['INTIME'])

# creating a temporary dataframe with just ICUSTAY_ID column.
temp = df_MIMIC[['ICUSTAY_ID']]

# merging dataframes to filter rows using HADM_ID from MIMIC_Features_OUT file. Then sorting
INPUTEVENTS = pd.merge(INPUTEVENTS, temp, on='ICUSTAY_ID', how='inner').sort_values(by = ['SUBJECT_ID','DATETIME'])

# Initializing
data_list = []

# loop to create prescription feature columns.
for row in df_MIMIC.iterrows():
    # copying the patient record from df_MIMIC
    record = row[1].copy()
    
    # filtering rows to keep records of the selected patient in each iteration
    df_INPUTEVENTS = INPUTEVENTS.loc[INPUTEVENTS['ICUSTAY_ID'] == record['ICUSTAY_ID']] 
    
    # filtering rows to keep records from first 24 hours of ICU admission  
    df_INPUTEVENTS = df_INPUTEVENTS[(df_INPUTEVENTS['DATETIME'] - record['INTIME']).between(pd.Timedelta('0s'),pd.Timedelta('1d'))]

    # creating prescription feature columns
    record['Norepinephrine'] = 1 if df_INPUTEVENTS['ITEMID'].isin(lab_items['Norepinephrine']).sum() >= 1 else 0
    record['PhenyLEPHrine'] = 1 if df_INPUTEVENTS['ITEMID'].isin(lab_items['PhenyLEPHrine']).sum() >= 1 else 0
    record['Vasopressin'] = 1 if df_INPUTEVENTS['ITEMID'].isin(lab_items['Vasopressin']).sum() >= 1 else 0
    record['DOBUTamine'] = 1 if df_INPUTEVENTS['ITEMID'].isin(lab_items['DOBUTamine']).sum() >= 1 else 0
    record['DOPamine'] = 1 if df_INPUTEVENTS['ITEMID'].isin(lab_items['DOPamine']).sum() >= 1 else 0
    record['EPINEPHrine'] = 1 if df_INPUTEVENTS['ITEMID'].isin(lab_items['EPINEPHrine']).sum() >= 1 else 0

    # appending updated patient record to the list
    data_list.append(record)

# converting list to dataframe 
X_feature_df = pd.DataFrame(data_list)
# display(X_feature_df)


# saving file
X_feature_df.to_pickle('Temp/MIMIC_Features_INPUT.pickle', compression = 'zip')

# printing check point
print('MIMIC_Features_sample_INPUT files saved')

