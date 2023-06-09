#!/usr/bin/env python
# coding: utf-8
# author: Maruthi Mutnuri (maruthi2008@gmail.com)

# loading libraries
import pandas as pd 
import numpy as np

# loading database files
adm = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/ADMISSIONS.pickle', compression = 'zip')
pat = pd.read_pickle('/project/M-ABeICU176709/ABeICU/data/PATIENTS.pickle', compression = 'zip')

# selecting columns
adm = adm[['ADMISSION_ID', 'PATIENT_ID', 'HOSP_ADMIT_DATETIME', 'HOSP_DISCH_DATETIME',
           'ICU_ADMIT_DATETIME', 'ICU_DISCH_DATETIME', 'ADMISSION_TYPE', 'ADMISSION_CLASS', 
           'ADMISSION_WEIGHT']]

# creating H_LOS_miss column, which indicates missing hospital admit or discharge datetime information.
adm['H_LOS_miss'] = list(map(lambda a,b: 1 if a or b else 0, adm['HOSP_DISCH_DATETIME'].isnull(), adm['HOSP_ADMIT_DATETIME'].isnull()))

# creating H_LOS and ICU_LOS columns
adm['H_LOS'] = ((pd.to_datetime(adm['HOSP_DISCH_DATETIME'].values) - pd.to_datetime(adm['HOSP_ADMIT_DATETIME'].values)).astype(int)/1000000000/3600/24)
adm['ICU_LOS'] = ((pd.to_datetime(adm['ICU_DISCH_DATETIME'].values) - pd.to_datetime(adm['ICU_ADMIT_DATETIME'].values)).astype(int)/1000000000/3600/24)

# selecting columns
pat = pat[['PATIENT_ID', 'GENDER', 'HEIGHT', 'DOB', 'DOD']]

# merging admissons and patients dataframes
df_eCrit = pd.merge(adm, pat, on='PATIENT_ID', how='inner').sort_values(by='ICU_ADMIT_DATETIME')

# sorting and dropping duplicates. Keeping only the first admissions of the patients.
df_eCrit.sort_values(by = 'ICU_ADMIT_DATETIME', inplace=True)
df_eCrit.drop_duplicates(subset='PATIENT_ID',keep='first', inplace=True)

# Keeping patient records with more than 1 day ICU LOS.
df_eCrit = df_eCrit[df_eCrit['ICU_LOS'] > 1.0]

# creating AGE column. Age as of ICU admission.
df_eCrit['AGE'] = ((pd.to_datetime(df_eCrit['ICU_ADMIT_DATETIME'].values) - pd.to_datetime(df_eCrit['DOB'].values)).astype(int)/1000000000/3600/24/365.242)                     

# rounding Age to the nearest year.
df_eCrit['AGE_r'] = df_eCrit['AGE'].round()

# keeping patient records with age more than 18 years.
df_eCrit = df_eCrit[df_eCrit['AGE'] >= 18.0 ]

# creating 30_DAYS_EXPIRE column. 30_DAYS_EXPIRE (30_DAYS_MORTALITY) is defined as mortality within 30 days of ICU admission
df_eCrit['30_DAYS_EXPIRE'] = list(map(lambda a,b: 1 if (a - b).days <= 30 and (a - b).days > 0 else 0, pd.to_datetime(df_eCrit['DOD'].values), pd.to_datetime(df_eCrit['ICU_ADMIT_DATETIME'].values)))

# selecting the columns
df_eCrit = df_eCrit[['ADMISSION_ID', 'PATIENT_ID', 'ICU_ADMIT_DATETIME', 'ADMISSION_WEIGHT',
                    'H_LOS', 'ICU_LOS', 'GENDER', 'HEIGHT', 'AGE_r','30_DAYS_EXPIRE', 'H_LOS_miss']]

# sorting by patient ID
df_eCrit.sort_values(by = 'PATIENT_ID',  inplace = True)

# saving file
df_eCrit.to_csv('Temp/eCrit_ADMISSION_PATIENTS.csv')

# printing check point
print('eCrit_ADMISSION_PATIENTS file saved')
