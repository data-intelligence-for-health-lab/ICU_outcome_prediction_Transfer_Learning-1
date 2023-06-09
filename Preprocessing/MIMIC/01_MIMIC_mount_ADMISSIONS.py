#!/usr/bin/env python
# coding: utf-8
# author: Maruthi Mutnuri (maruthi2008@gmail.com)

# loading libraries
import pandas as pd 
import numpy as np

# loading database files
adm_mimic = pd.read_csv("/home/maruthikumar.mutnuri/physionet/files/mimiciii/data/ADMISSIONS.csv.gz", sep = ',', compression = 'gzip', usecols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME','DISCHTIME'])
pat_mimic = pd.read_csv("/home/maruthikumar.mutnuri/physionet/files/mimiciii/data/PATIENTS.csv.gz", sep = ',', compression = 'gzip', usecols = ['SUBJECT_ID', 'GENDER', 'DOB','DOD'])
icu_stays_mimic = pd.read_csv("/home/maruthikumar.mutnuri/physionet/files/mimiciii/data/ICUSTAYS.csv.gz", sep = ',', compression = 'gzip',
                              usecols = ['SUBJECT_ID','HADM_ID','ICUSTAY_ID','INTIME', 'OUTTIME','LOS'])
                              
# creating H_LOS column
adm_mimic['H_LOS'] = ((pd.to_datetime(adm_mimic['DISCHTIME'].values) - pd.to_datetime(adm_mimic['ADMITTIME'].values)).astype(int)/1000000000/3600/24)

# renaming column
icu_stays_mimic.rename(columns={'LOS': 'ICU_LOS'}, inplace = True)

# merging dataframes
df_mimic = pd.merge(icu_stays_mimic, pat_mimic, on='SUBJECT_ID', how='inner').sort_values(by='INTIME')

# dropping duplicates, to keep only the first admissions of the patients.
df_mimic.drop_duplicates(subset='SUBJECT_ID',keep='first', inplace=True)

# merging dataframes
df_mimic_1 = pd.merge(df_mimic, adm_mimic, on=['SUBJECT_ID','HADM_ID'], how='inner').sort_values(by='INTIME')

# Keeping patient records with more than 1 day ICU LOS.
df_mimic_1 = df_mimic_1[df_mimic_1['ICU_LOS'] > 1.0]

# creating AGE column. Age as of ICU admission.
df_mimic_1['AGE_1'] = ((pd.to_datetime(df_mimic_1['INTIME'].values).astype(int) - pd.to_datetime(df_mimic_1['DOB'].values).astype(int))/1000000000/3600/24/365.242)

# patients aged 89 or more have their DOB masked. 
# Those parients median age is 91.4. refer: https://doi.org/10.1038/s41597-021-00864-4
df_mimic_1['AGE'] = list(map(lambda a: 91.4 if a < 0.0 else a, df_mimic_1['AGE_1']))

# keeping patient records with age more than 18 years.
df_mimic_2 = df_mimic_1[df_mimic_1['AGE'] >= 18.0]

# rounding Age to the nearest year.
df_mimic_2['AGE_r'] = df_mimic_2['AGE'].round()

# creating 30_DAYS_EXPIRE column. 30_DAYS_EXPIRE (30_DAYS_MORTALITY) is defined as mortality within 30 days of ICU admission
df_mimic_2['30_DAYS_EXPIRE'] = list(map(lambda a,b: 1 if (a - b).days <= 30 and (a - b).days > 0 else 0, pd.to_datetime(df_mimic_2['DOD'].values), pd.to_datetime(df_mimic_2['INTIME'].values)))

# selecting the columns
df_mimic_2 = df_mimic_2[['SUBJECT_ID', 'HADM_ID','ICUSTAY_ID', 'INTIME', 'ICU_LOS', 'GENDER',
        'H_LOS', 'AGE_r', '30_DAYS_EXPIRE']]
# converting to datetime datatype         
df_mimic_2['INTIME'] = pd.to_datetime(df_mimic_2['INTIME'])

# sorting by patient ID
df_mimic_2.sort_values(by = 'SUBJECT_ID',  inplace = True)

# saving file
df_mimic_2.to_pickle('Temp/MIMIC_ADMISSION_PATIENTS.pickle', compression = 'zip')

# printing check point
print('MIMIC_ADMISSION_PATIENTS file saved')
