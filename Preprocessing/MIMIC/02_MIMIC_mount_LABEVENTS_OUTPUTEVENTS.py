#!/usr/bin/env python
# coding: utf-8
# author: Maruthi Mutnuri (maruthi2008@gmail.com)

# loading libraries
import pandas as pd 
import numpy as np

# loading database file
LABEVENTS = pd.read_csv("/home/maruthikumar.mutnuri/physionet/files/mimiciii/data/LABEVENTS.csv.gz", sep = ',', compression = 'gzip', 
                        usecols = ['SUBJECT_ID','HADM_ID','ITEMID', 'CHARTTIME', 'VALUENUM'])

# renaming column
LABEVENTS.rename(columns={'CHARTTIME': 'DATETIME'}, inplace = True)

# sorting by SUBJECT_ID,'ITEMID', and 'DATETIME'
LABEVENTS.sort_values(by = ['SUBJECT_ID','ITEMID','DATETIME'],  inplace = True)

# converting to datetime datatype
LABEVENTS['DATETIME'] = pd.to_datetime(LABEVENTS['DATETIME'])

# converting to int datatype
LABEVENTS['HADM_ID'] = (LABEVENTS['HADM_ID']).astype(pd.Int64Dtype())

# dropping rows with missing values in VALUENUM column
LABEVENTS.dropna(subset=['VALUENUM'], inplace=True)

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
LABEVENTS = LABEVENTS[LABEVENTS['ITEMID'].isin(lab_items['LABEVENTS'])]

# creating a temporary dataframe with rows which have missing values in HADM_ID.
temp_df = LABEVENTS[LABEVENTS['HADM_ID'].isna()]

# filtering rows. keeping records only if HADM_ID is not NULL.
LABEVENTS = LABEVENTS[~LABEVENTS['HADM_ID'].isna()]

# loading the output file from previous script, 01_MIMIC_mount_ADMISSIONS.py.
df_MIMIC = pd.read_pickle('Temp/MIMIC_ADMISSION_PATIENTS.pickle', compression = 'zip')

# selecting columns
df_MIMIC = df_MIMIC[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'ICU_LOS', 'GENDER',
       'H_LOS', 'AGE_r', '30_DAYS_EXPIRE']]
       
# sorting by 'SUBJECT_ID' and 'INTIME'       
df_MIMIC.sort_values(by = ['SUBJECT_ID','INTIME'],  inplace = True)

# creating a temporary dataframe with just HADM_ID.
temp = df_MIMIC[['HADM_ID']]

# merging dataframes to filter rows using HADM_ID from MIMIC_ADMISSION_PATIENTS file.
LABEVENTS = pd.merge(LABEVENTS, temp, on='HADM_ID', how='inner')

# creating a temporary dataframe with just 'SUBJECT_ID' and'INTIME'.
temp = df_MIMIC[['SUBJECT_ID','INTIME']]
# merging dataframes to filter rows using SUBJECT_ID from MIMIC_ADMISSION_PATIENTS file.
temp = pd.merge(temp_df, temp, on='SUBJECT_ID', how='inner')
# filtering rows. keeping rows with INTIME greater than datetime
temp = temp[temp['INTIME'] > temp['DATETIME']]
# filtering rows. keeping rows with ITEMID = 50912 (creatinine)
temp = temp[temp['ITEMID'] == 50912]
# dropping column
temp.drop(columns = 'INTIME',inplace=True)

# concatinating dataframes to add back creatinine rows with missing HADM_ID values
LABEVENTS = pd.concat([LABEVENTS, temp])

# loading database file
OUTPUTEVENTS = pd.read_csv("/home/maruthikumar.mutnuri/physionet/files/mimiciii/data/OUTPUTEVENTS.csv.gz", sep = ',', compression = 'gzip', 
                        usecols = ['SUBJECT_ID','HADM_ID','ICUSTAY_ID','ITEMID', 'CHARTTIME', 'VALUE'])

# renaming column
OUTPUTEVENTS.rename(columns={'CHARTTIME': 'DATETIME'}, inplace = True)

# sorting by SUBJECT_ID
OUTPUTEVENTS.sort_values(by = 'SUBJECT_ID',  inplace = True)

# converting to datetime datatype 
OUTPUTEVENTS['DATETIME'] = pd.to_datetime(OUTPUTEVENTS['DATETIME'])

# converting to int datatype 
OUTPUTEVENTS['HADM_ID'] = (OUTPUTEVENTS['HADM_ID']).astype(pd.Int64Dtype())
OUTPUTEVENTS['ICUSTAY_ID'] = (OUTPUTEVENTS['ICUSTAY_ID']).astype(pd.Int64Dtype())

# dropping rows with missing values
OUTPUTEVENTS.dropna(inplace=True)

# filtering rows using lab_items. keeping records of only required ITEM_ID's. Then sorting.
OUTPUTEVENTS = OUTPUTEVENTS[OUTPUTEVENTS['ITEMID'].isin(lab_items['OUTPUTEVENTS'])].sort_values(by = ['SUBJECT_ID','DATETIME'])

# Initializing
X_data_list = []

# loop to create Urine_Volumes column.
for row in df_MIMIC.iterrows():

    # filtering rows to keep records of the selected patient in each iteration
    df_OUTPUTEVENTS = OUTPUTEVENTS.loc[OUTPUTEVENTS['HADM_ID'] == row[1]['HADM_ID']]
    
    # copying the patient record from df_MIMIC
    record = row[1].copy()

    # filtering rows to keep records from first 24 hours of ICU admission
    df_OUTPUTEVENTS = df_OUTPUTEVENTS[(df_OUTPUTEVENTS['DATETIME'] - record['INTIME']).between(pd.Timedelta('0s'),pd.Timedelta('1d'))]

    # creating Urine_Volumes column
    record['Urine_Volumes'] = df_OUTPUTEVENTS['VALUE'].sum()

    # appending updated patient record to the list
    X_data_list.append(record)

# converting list to dataframe      
X_feature_df = pd.DataFrame(X_data_list)

# saving files
LABEVENTS.to_pickle('Temp/MIMIC_LABEVENTS_processed.pickle', compression = 'zip')
X_feature_df.to_pickle('Temp/MIMIC_Features_OUT.pickle', compression = 'zip')

# printing check point
print('files saved')

