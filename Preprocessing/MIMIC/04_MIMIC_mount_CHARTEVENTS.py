#!/usr/bin/env python
# coding: utf-8
# author: Maruthi Mutnuri (maruthi2008@gmail.com)

# loading libraries
import pandas as pd 
import numpy as np

# creating a dictionary of required ITEM_ID's, to extract features from the database files.
lab_items = {'CHARTEVENTS': [6, 51, 184, 211, 442, 454, 455, 490, 618, 646, 723, 777, 778, 779, 780, 781, 791, 807, 811, 
                             813, 814, 833, 837, 861, 1126, 1127, 1162, 1525, 1529, 1536, 1542, 1673, 3420, 3422, 3651, 
                             3737, 3750, 3784, 3785, 3799, 3803, 3808, 3809, 3810, 3834, 3835, 3836, 3837, 4197, 4200, 
                             4201, 4753, 6701, 8364, 8368, 8440, 8441, 8555, 220045, 220050, 220051, 220179, 220180, 
                             220210, 220227, 220228, 220235, 220277, 220545, 220546, 220615, 220621, 220640, 220645, 
                             220739, 223830, 223835, 223900, 223901, 224689, 224690, 225309, 225310, 225624, 225664, 
                             225698, 226534, 226535, 226537, 226540, 226756, 226757, 226758, 227011, 227012, 227014, 
                             227442, 227464, 228112,148, 152,582, 225126,227124,505,506,3555,720,223849,220339,224686,684,
                            763,3580,3693,226512,224639,228300,228301,228302,228303,228332,190,2981],
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

# creating a reference dictionary for features.
feature_names = {'Eye_Opening': [184,220739,227011,226756], 'Verbal_Response': [723,227014,228112,223900,226758],
                 'Motor_Response': [454,227012,223901,226757], 'Heart_Rate': [211,220045],
                 'SpO2': [220277, 646, 220227], 'PH_Arterial': [50820,780,1126,1673,4753,223830],
                 'BP_Systolic': [6,51,442,455,6701,220179,220050,225309], 'FiO2': [3420,223835,3422,190,2981,50816],
                 'BP_Diastolic': [8364,8368,8440,8441,8555,220051,220180,225310], 'Urea_Blood': [51006,781,1162,3737,225624], 
                 'Creatinine_Blood': [50912,791,1525,3750,220615],'CO2_Content_Blood': [50804,225698,3810,3808,3809,777], 
                 'Glucose_Blood': [50809,50931,51529,807,811,1529,220621,225664,226537], 
                 'Potassium_Blood': [50971,227442,220640,50822,227464,226535], 
                 'Sodium_Blood': [50824,50983,837,1536,3803,220645,226534], 'Respiratory_Rate': [618,220210,224689,224690],
                 'PCO2_Arterial': [50818,220235,3651,778,3784,3835,3836,4201], 'PO2_Arterial': [50821,490,779,3785,3837],
                 'Hemoglobin': [50811,51222,814,220228], 'Hematocrit': [50810,51221,813,220545,226540], 
                 'RBC': [51279,833,3799,4197], 'WBC': [51300,51301,861,1127,1542,3834,4200,220546],
                 'Dialysis_Flag': [148, 152,582, 225126,227124], 'ADMISSION_WEIGHT': [763,226512],
                 'Mechanical_Ventilation_Flag': [505,506,3555,720,223849,220339,224686,684]}


# loading database file
CHARTEVENTS = pd.read_csv("/home/maruthikumar.mutnuri/physionet/files/mimiciii/data/CHARTEVENTS.csv.gz", sep = ',', compression = 'gzip',
                         usecols = ['SUBJECT_ID','HADM_ID','ICUSTAY_ID','ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUE'])

# renaming columns
CHARTEVENTS.rename(columns={'CHARTTIME': 'DATETIME', 'VALUENUM': 'VALUE_NUM'}, inplace = True)

# converting to datetime datatype
CHARTEVENTS['DATETIME'] = pd.to_datetime(CHARTEVENTS['DATETIME'])

# converting to int datatype
CHARTEVENTS['ICUSTAY_ID'] = (CHARTEVENTS['ICUSTAY_ID']).astype(pd.Int64Dtype())

# dropping rows with missing values in VALUE_NUM column
CHARTEVENTS.dropna(subset=['VALUE_NUM'], inplace=True)

# filtering rows using lab_items. keeping records of only required ITEM_ID's
CHARTEVENTS = CHARTEVENTS[CHARTEVENTS['ITEMID'].isin(lab_items['CHARTEVENTS'])]

# creating a temporary dataframe with rows which have missing values in ICUSTAY_ID.
temp_df = CHARTEVENTS[CHARTEVENTS['ICUSTAY_ID'].isna()]

# filtering rows. keeping records only if ICUSTAY_ID is not NULL.
CHARTEVENTS = CHARTEVENTS[~CHARTEVENTS['ICUSTAY_ID'].isna()]

# loading the output file from previous script, 03_MIMIC_mount_INPUTEVENTS.py.
df_MIMIC = pd.read_pickle('Temp/MIMIC_Features_INPUT.pickle', compression = 'zip')

# selecting columns
df_MIMIC = df_MIMIC[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'ICU_LOS', 'GENDER',
       'H_LOS', 'AGE_r', '30_DAYS_EXPIRE', 'Urine_Volumes',
       'Norepinephrine', 'PhenyLEPHrine', 'Vasopressin', 'DOBUTamine',
       'DOPamine', 'EPINEPHrine']]

# converting to datetime datatype  
df_MIMIC['INTIME'] = pd.to_datetime(df_MIMIC['INTIME'])

print("MIMIC_Features_sample_INPUT file loaded")

# creating a temporary dataframe with just ICUSTAY_ID column.
temp = df_MIMIC[['ICUSTAY_ID']]

# merging dataframes to filter rows using ICUSTAY_ID from MIMIC_Features_OUT file. Then sorting
CHARTEVENTS = pd.merge(CHARTEVENTS, temp, on='ICUSTAY_ID', how='inner').sort_values(by = ['SUBJECT_ID','ITEMID','DATETIME'])

# creating a temporary dataframe with just 'SUBJECT_ID' and'INTIME'.
temp = df_MIMIC[['SUBJECT_ID','INTIME']]
# merging dataframes to filter rows using SUBJECT_ID from MIMIC_ADMISSION_PATIENTS file.
temp = pd.merge(temp_df, temp, on='SUBJECT_ID', how='inner')
# filtering rows. keeping rows with INTIME greater than datetime
temp = temp[temp['INTIME'] > temp['DATETIME']]
# filtering rows. keeping rows with ITEMID's of creatinine.
temp = temp[temp['ITEMID'].isin(feature_names['Creatinine_Blood'])]
# dropping column
temp.drop(columns = 'INTIME',inplace=True)

# concatinating dataframes to add back creatinine rows with missing ICUSTAY_ID values
CHARTEVENTS = pd.concat([CHARTEVENTS, temp])

# coverting values for decimals to percentage to be compatible with values from other FiO2 ITEMID's
CHARTEVENTS.loc[CHARTEVENTS['ITEMID'] == 190, 'VALUE_NUM'] = CHARTEVENTS[CHARTEVENTS['ITEMID'] == 190]['VALUE_NUM'] * 100

# loading the output file from previous script, 02_MIMIC_mount_LABEVENTS_OUTPUTEVENTS.py.
LABEVENTS_df = pd.read_pickle('Temp/MIMIC_LABEVENTS_processed.pickle', compression = 'zip')

# converting to datetime datatype 
LABEVENTS_df['DATETIME'] = pd.to_datetime(LABEVENTS_df['DATETIME'])

# renaming column
LABEVENTS_df.rename(columns={'VALUENUM': 'VALUE_NUM'}, inplace = True)

# creating a dataframe for target variable (y, Dependent variables)
target_df = df_MIMIC[['SUBJECT_ID','HADM_ID','ICUSTAY_ID', 'INTIME', 
                       'H_LOS', 'ICU_LOS', '30_DAYS_EXPIRE']]

# creating a dataframe for features (X, Independent variables)
df_MIMIC = df_MIMIC.drop(columns=['H_LOS', 'ICU_LOS', '30_DAYS_EXPIRE'])


# printing check point
print(" X_feature_df Processing Started...")

# Initializing
data_list = []

# loop to create the feature (X) columns.
for row in df_MIMIC.iterrows():
    # copying the patient record from df_MIMIC
    record = row[1].copy()
    # filtering rows to keep records of the selected patient in each iteration
    df_chartevents = CHARTEVENTS.loc[CHARTEVENTS['SUBJECT_ID'] == record['SUBJECT_ID']]
    df_labevents = LABEVENTS_df.loc[LABEVENTS_df['SUBJECT_ID'] == record['SUBJECT_ID']]
    
    # concatenating dataframes and sorting by datetime and then by ITEM_ID
    df_chartevents = pd.concat([df_chartevents, df_labevents]).sort_values(by = ['DATETIME','ITEMID'])
    
    # filtering rows to keep records from first 24 hours of ICU admission 
    df_chartevents = df_chartevents[(df_chartevents['DATETIME'] - record['INTIME']).between(pd.Timedelta('0s'),pd.Timedelta('1d'))]

    # looping through all the features and creating feature columns
    for feature in feature_names.keys():            

        # filtering rows to keep records of the selected feature in each iteration
        df_item = df_chartevents.loc[df_chartevents['ITEMID'].isin(feature_names[feature])]

        # converting values based on the conversion factors of the units of measure of MIMIC and eCritical.
        # MIMIC feature values are converted to be compatible with corresponding feature eCritical values
        if feature == 'Urea_Blood':
            df_item['VALUE_NUM'] = df_item['VALUE_NUM'] * 0.3571
        elif feature == 'Creatinine_Blood':
            df_item['VALUE_NUM'] = df_item['VALUE_NUM'] * 88.4
        elif feature == 'Glucose_Blood':
            df_item['VALUE_NUM'] = df_item['VALUE_NUM'] * 0.0555
        elif feature == 'Hemoglobin':
            df_item['VALUE_NUM'] = df_item['VALUE_NUM'] * 10
        elif feature == 'Dialysis_Flag':
            # creating Dialysis_Flag column
            record[feature] = 0 if df_item.empty else 1 
            continue
        elif feature == 'Mechanical_Ventilation_Flag':
            # creating Mechanical_Ventilation_Flag column
            record[feature] = 0 if df_item.empty else 1 
            continue
        elif feature == 'ADMISSION_WEIGHT':
            if not df_item.empty:
                # creating ADMISSION_WEIGHT column
                record[feature] = df_item.iloc[0]['VALUE_NUM']
                continue
            else:
                continue
        # for other features, median, IQR, 5th and 95th percentile are calculated and new feature columns created
        record[feature+'_5percentile'] = df_item['VALUE_NUM'].quantile(0.05)
        record[feature+'_median'] = df_item['VALUE_NUM'].median()
        record[feature+'_IQR'] = df_item['VALUE_NUM'].quantile(0.75) - df_item['VALUE_NUM'].quantile(0.25)
        record[feature+'_95percentile'] = df_item['VALUE_NUM'].quantile(0.95)
        
    # appending updated patient record to the list    
    data_list.append(record)

# converting list to dataframe  
X_feature_df = pd.DataFrame(data_list)

# creating GCS columns 
X_feature_df['GCS_5percentile'] = X_feature_df['Eye_Opening_5percentile'] + X_feature_df['Motor_Response_5percentile'] + X_feature_df['Verbal_Response_5percentile']
X_feature_df['GCS_95percentile'] = X_feature_df['Eye_Opening_95percentile'] + X_feature_df['Motor_Response_95percentile'] + X_feature_df['Verbal_Response_95percentile']
X_feature_df['GCS_IQR'] = X_feature_df['Eye_Opening_IQR'] + X_feature_df['Motor_Response_IQR'] + X_feature_df['Verbal_Response_IQR']
X_feature_df['GCS_median'] = X_feature_df['Eye_Opening_median'] + X_feature_df['Motor_Response_median'] + X_feature_df['Verbal_Response_median']

# printing check point
print("target_df Processing Started...")

# Initializing
data_list = []

# loop to create the target (y) columns.
for row in target_df.iterrows():
    # copying the patient record from target_df
    record = row[1].copy()

    # filtering rows to keep records of the selected patient in each iteration
    df_chartevents = CHARTEVENTS.loc[CHARTEVENTS['SUBJECT_ID'] == record['SUBJECT_ID']]
    df_labevents = LABEVENTS_df.loc[LABEVENTS_df['SUBJECT_ID'] == record['SUBJECT_ID']]
    
    # concatenating dataframes and sorting by datetime and then by ITEM_ID    
    df_chartevents = pd.concat([df_chartevents, df_labevents]).sort_values(by = ['DATETIME','ITEMID'])

    # filtering rows to keep all available Blood Creatinine records, to calculate baseline creatinine        
    X_df_item = df_chartevents.loc[df_chartevents['ITEMID'].isin(feature_names['Creatinine_Blood'])]

    # filtering rows to keep records from first 24 hours of ICU admission
    df_chartevents_adm = df_chartevents[(df_chartevents['DATETIME'] - record['INTIME']).between(pd.Timedelta('0s'),pd.Timedelta('1d'))]
    
    # filtering rows to keep records from after 24 hours of ICU admission    
    df_chartevents = df_chartevents[(df_chartevents['DATETIME'] - record['INTIME']) > pd.Timedelta('1d')]


# AKI - after 24 hours of ICU admission

    # filtering rows to keep Blood Creatinine records, after 24 hours of ICU admission
    df_item = df_chartevents.loc[df_chartevents['ITEMID'].isin(feature_names['Creatinine_Blood'])]

    # Initializing
    AKI_base = 0
    AKI_SCr = 0  

    # check if there is atleast one creatinine value after 24 hours
    if df_item.empty:
        Creatinine_miss = 1 
    else:
        Creatinine_miss = 0  
        
        # get historical creatinine values (any lab record upto the first 24 hours of ICU admission)
        # if there are more than one lab record then using mean of the historical values as baseline               
        baseline = pd.concat([X_df_item,df_item]).drop_duplicates(keep=False)['VALUE_NUM'].mean() 

        # check if there is atleast one baseline creatinine value 
        if np.isnan(baseline):
            Creatinine_miss = 1     
            
            # check if any Creatinine value is greater than 1.5 times baseline. 
            # 1 for presence of AKI and 0 for absence of AKI  
            AKI_base = 1 if (df_item['VALUE_NUM']/baseline >= 1.5).any() else 0
        
        # loop to check if there is a Creatinine increase > 0.3 mg/dl within 2 days
        for item in df_item.iterrows():
            # filtering rows to keep records from within past two days of the selected patient in each iteration        
            df_item_1 = df_item[(item[1]['DATETIME'] - df_item['DATETIME']).between(pd.Timedelta('0d 1s'),pd.Timedelta('2d'))]
            
            # calculating difference between the selected lab value and minimum of lab value within the past two days            
            SCr_diff = item[1]['VALUE_NUM']-df_item_1['VALUE_NUM'].min()
            
            # checking if difference is greater than or equal to 0.3 mg/dl            
            if SCr_diff >= 0.3:
                AKI_SCr = 1 # indicates presence of AKI
                break
    
    # creating AKI_data_miss and AKI columns                   
    record['AKI_data_miss'] = 1 if Creatinine_miss else 0
    record['AKI'] = 1 if AKI_base or AKI_SCr else 0

    
 
    

# AKI Admission Diagnosis within 24 hours of ICU admission
    # Initializing
    AKI_base_adm = 0
    AKI_SCr_adm = 0 

    # check if there is atleast one creatinine value         
    if not X_df_item.empty:
        baseline = X_df_item['VALUE_NUM'].iloc[0]

        # filtering rows to keep Blood Creatinine records
        df_item_adm = df_chartevents_adm.loc[df_chartevents_adm['ITEMID'].isin(feature_names['Creatinine_Blood'])]

        # check if there is atleast one creatinine value after 24 hours of ICU admission  
        if not df_item_adm.empty:
            # check if any Creatinine value is greater than 1.5 times baseline        
            AKI_base_adm = 1 if (df_item_adm['VALUE_NUM']/baseline >= 1.5).any() else 0
            
            # loop to check if there is a Creatinine increase > 26.5 within 2 days
            for item in df_item_adm.iterrows():
                # filtering rows to keep records from within past two days of the selected patient in each iteration            
                df_item_adm_1 = df_item_adm[(item[1]['DATETIME'] - df_item_adm['DATETIME']).between(pd.Timedelta('0d 1s'),pd.Timedelta('1d'))]
                # calculating difference between the selected lab value and minimum of lab value within the past two days                
                SCr_diff_adm = item[1]['VALUE_NUM']-df_item_adm_1['VALUE_NUM'].min()
                # checking if difference is greater than or equal to 0.3 mg/dl                
                if SCr_diff_adm >= 0.3:
                    AKI_SCr_adm = 1 # indicates presence of AKI at admission
                    break
    # creating AKI_adm column
    record['AKI_adm'] = 1 if AKI_base_adm or AKI_SCr_adm else 0
 
    # appending updated patient record to the list       
    data_list.append(record)

# converting list to dataframe        
Y_feature_df = pd.DataFrame(data_list) 

# dropping columns
Y_feature_df = Y_feature_df.drop(columns=['INTIME'])
X_feature_df = X_feature_df.drop(columns=['INTIME'])

# saving files
X_feature_df.to_pickle('Temp/MIMIC_Features.pickle', compression = 'zip')
Y_feature_df.to_pickle('Temp/MIMIC_Target.pickle', compression = 'zip')

# printing check point
print('Features and Target files saved')

