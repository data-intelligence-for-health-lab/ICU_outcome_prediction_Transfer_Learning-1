#!/usr/bin/env python
# coding: utf-8
# author: Maruthi Mutnuri (maruthi2008@gmail.com)

# loading libraries
import pandas as pd 
import numpy as np

# loading database file
MEASUREMENTS_df = pd.read_pickle('/project/M-ABeICU176709/ABeICU/Temp/MEASUREMENTS.pickle', compression = 'zip')

# selecting columns
MEASUREMENTS_df = MEASUREMENTS_df[['PATIENT_ID', 'ADMISSION_ID','ITEM_ID', 'DATETIME', 'VALUE_NUM']]

# converting to datetime datatype 
MEASUREMENTS_df['DATETIME'] = pd.to_datetime(MEASUREMENTS_df['DATETIME'])

# creating a dictionary of required ITEM_ID's, to extract features from the database files.
lab_items = {'MEASUREMENTS': ['I000','I001','I002','I003','I023','I024','I025','I026','I027','I028','I029','I030',
                              'I035','I047','I050','I031','I036','I048','I051','I033','I034','I052','I038','I039',
                              'I040','I041','I042','I043','I044','I045','I046','I053','I054','I055','I056','I057',
                              'I062','I080','I091','I058','I059','I060','I061','I063','I064','I065','I066','I067',
                              'I077','I068','I076','I069','I070','I071','I072','I073','I074','I082','I075','I078',
                              'I079','I081','I083','I084','I085','I086','I087','I088','I089','I090','I092']}

# creating a reference dictionary for features.
feature_names = {'Eye_Opening': ['I000'], 'Verbal_Response': ['I001'], 'GCS': ['I002'], 'Motor_Response': ['I003'],
                 'Urine_Volumes': ['I023', 'I024', 'I025', 'I026', 'I027', 'I028'], 'Heart_Rate': ['I029'],
                 'BP_Systolic': ['I030', 'I035', 'I047', 'I050'], 'BP_Diastolic': ['I031', 'I036', 'I048', 'I051'],
                 'SpO2': ['I033', 'I034', 'I052'], 'PCO2_Venous': ['I038', 'I081'], 
                 'Respiratory_Rate': ['I039','I040','I299','I339','I399','I436','I455','I456'],
                 'Temperature': ['I041', 'I042', 'I043', 'I044', 'I045', 'I046', 'I053'], 'Urea_Blood': ['I054'],
                 'CO2_Content_Blood': ['I055'], 'Creatinine_Blood': ['I056'], 
                 'Glucose_Blood': ['I057', 'I062', 'I080'], 'Potassium_Blood': ['I058','I069'], 
                 'Sodium_Blood': ['I059', 'I060'], 'PCO2_Arterial': ['I061'], 'FiO2': ['I063', 'I334', 'I356'],
                 'O2_Gradient_Arterial': ['I064'], 'PH_Arterial': ['I065'], 'PO2_Arterial': ['I066'],
                 'Hemoglobin': ['I067', 'I077'], 'Hematocrit': ['I068', 'I076'],
                 'Albumin_Blood': ['I070'], 'Alkaline_Phosphatase': ['I071'], 'ALT': ['I072'], 'AST': ['I073'],
                 'Bilirubin': ['I074', 'I082'], 'GGT': ['I075'], 'RBC': ['I078'], 'WBC': ['I079'],
                 'HCO3_Calculated_Venous': ['I083'], 'Lactate_Blood': ['I084'], 'Total_Protein_Blood': ['I085'],
                 'HCO3_Calculated_Manual_Entry': ['I086'], 'Ammonia_Blood': ['I087'], 'BNP': ['I088'], 
                 'CRP': ['I089'], 'Alveolar_Arterial_O2': ['I090'], 'Glucose_Fasting': ['I091'],
                 'HCO3_Calculated_Capillary': ['I092']}

# filtering rows using lab_items. keeping records of only required ITEM_ID's
MEASUREMENTS_df = MEASUREMENTS_df[MEASUREMENTS_df['ITEM_ID'].isin(lab_items['MEASUREMENTS'])]


# loading the output file from previous script, 03_eCrit_mount_PRESCRIPTIONS.py.
df_eCrit = pd.read_csv('Temp/eCrit_Features_Rx.csv', sep = ',')

# converting to datetime datatype 
df_eCrit['ICU_ADMIT_DATETIME'] = pd.to_datetime(df_eCrit['ICU_ADMIT_DATETIME'])
print("eCrit_Features_sample_Rx file loaded")

# creating a temporary dataframe with just admission ID.
temp = df_eCrit[['ADMISSION_ID']]

# merging dataframes to filter rows using ADMISSION_ID from eCrit_Features_Rx file.
MEASUREMENTS_df = pd.merge(MEASUREMENTS_df, temp, on='ADMISSION_ID', how='inner')


# loading the output file from the script, 02_eCrit_mount_INTERVENTIONS.py.
INTERVENTIONS_df = pd.read_csv('Temp/INTERVENTIONS_processed.csv', sep = ',')

# converting to datetime datatype
INTERVENTIONS_df['DATETIME'] = pd.to_datetime(INTERVENTIONS_df['DATETIME'])

# creating a dataframe for target variable (y, Dependent variables)
target_df = df_eCrit[['ADMISSION_ID', 'PATIENT_ID', 'ICU_ADMIT_DATETIME',  
       'ADMISSION_WEIGHT', 'H_LOS', 'ICU_LOS', '30_DAYS_EXPIRE', 'H_LOS_miss']]

# creating a dataframe for features (X, Independent variables)
df_eCrit = df_eCrit.drop(columns=['H_LOS', 'ICU_LOS', 'H_LOS_miss', '30_DAYS_EXPIRE'])

# printing check point
print(" X_feature_df Processing Started...")

# Initializing
data_list = []

# loop to create the feature (X) columns.
for row in df_eCrit.iterrows():
    # copying the patient record from df_eCrit
    record = row[1].copy()
    
    # filtering rows to keep records of the selected patient in each iteration
    df_measure = MEASUREMENTS_df.loc[MEASUREMENTS_df['ADMISSION_ID'] == record['ADMISSION_ID']]
    df_intervention = INTERVENTIONS_df.loc[INTERVENTIONS_df['ADMISSION_ID'] == record['ADMISSION_ID']]
    
    # concatenating dataframes and sorting by datetime and then by ITEM_ID
    df_measure = pd.concat([df_measure, df_intervention]).sort_values(by = ['DATETIME','ITEM_ID'])
    
    # filtering rows to keep records from first 24 hours of ICU admission  
    df_measure = df_measure[(df_measure['DATETIME'] - record['ICU_ADMIT_DATETIME']).between(pd.Timedelta('0s'),pd.Timedelta('1d'))]

    # looping through all the features and creating feature columns
    for feature in feature_names.keys():            

        # filtering rows to keep records of the selected feature in each iteration
        df_item = df_measure.loc[df_measure['ITEM_ID'].isin(feature_names[feature])]
        
        # for Urine volumes, values are aggregated
        if feature == 'Urine_Volumes':
            record[feature] = df_item['VALUE_NUM'].sum()        
        else:
        # for other features, median, IQR, 5th and 95th percentile are calculated and new feature columns created
            record[feature+'_5percentile'] = df_item['VALUE_NUM'].quantile(0.05)
            record[feature+'_median'] = df_item['VALUE_NUM'].median()
            record[feature+'_IQR'] = df_item['VALUE_NUM'].quantile(0.75) - df_item['VALUE_NUM'].quantile(0.25)
            record[feature+'_95percentile'] = df_item['VALUE_NUM'].quantile(0.95)

    # appending updated patient record to the list
    data_list.append(record)

# converting list to dataframe            
X_feature_df = pd.DataFrame(data_list)    


# printing check point
print("target_df Processing Started...")

# Initializing
data_list = []

# loop to create the target (y) columns.
for row in target_df.iterrows():
    # copying the patient record from target_df
    record = row[1].copy()
    
    # filtering rows to keep records of the selected patient in each iteration
    df_measure = MEASUREMENTS_df.loc[MEASUREMENTS_df['ADMISSION_ID'] == record['ADMISSION_ID']]
    df_intervention = INTERVENTIONS_df.loc[INTERVENTIONS_df['ADMISSION_ID'] == record['ADMISSION_ID']]
    
    # concatenating dataframes and sorting by datetime and then by ITEM_ID    
    df_measure = pd.concat([df_measure, df_intervention]).sort_values(by = ['DATETIME','ITEM_ID'])

    # filtering rows to keep all available Blood Creatinine records, to calculate baseline creatinine        
    X_df_item = df_measure.loc[df_measure['ITEM_ID'].isin(feature_names['Creatinine_Blood'])]

    # filtering rows to keep records from first 24 hours of ICU admission
    df_measure_adm = df_measure[(df_measure['DATETIME'] - record['ICU_ADMIT_DATETIME']).between(pd.Timedelta('0s'),pd.Timedelta('1d'))]
    
    # filtering rows to keep records from after 24 hours of ICU admission    
    df_measure = df_measure[(df_measure['DATETIME'] - record['ICU_ADMIT_DATETIME']) > pd.Timedelta('1d')]

    
# AKI - after 24 hours of ICU admission

    # filtering rows to keep Blood Creatinine records, after 24 hours of ICU admission
    df_item = df_measure.loc[df_measure['ITEM_ID'].isin(feature_names['Creatinine_Blood'])]

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
        
        # loop to check if there is a Creatinine increase > 26.5 (Mu)mol/l within 2 days
        for item in df_item.iterrows():
            # filtering rows to keep records from within past two days of the selected patient in each iteration
            df_item_1 = df_item[(item[1]['DATETIME'] - df_item['DATETIME']).between(pd.Timedelta('0d 1s'),pd.Timedelta('2d'))]
            
            # calculating difference between the selected lab value and minimum of lab value within the past two days
            SCr_diff = item[1]['VALUE_NUM']-df_item_1['VALUE_NUM'].min()
            
            # checking if difference is greater than or equal to 26.5 (Mu)mol/l
            if SCr_diff >= 26.5:
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
        # first value is used as baseline
        baseline = X_df_item['VALUE_NUM'].iloc[0]

        # filtering rows to keep Blood Creatinine records
        df_item_adm = df_measure_adm.loc[df_measure_adm['ITEM_ID'].isin(feature_names['Creatinine_Blood'])]  
      
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
                # checking if difference is greater than or equal to 26.5
                if SCr_diff_adm >= 26.5:
                    AKI_SCr_adm = 1 # indicates presence of AKI at admission
                    break
    # creating AKI_adm column
    record['AKI_adm'] = 1 if AKI_base_adm or AKI_SCr_adm else 0

    # appending updated patient record to the list
    data_list.append(record)
    
# converting list to dataframe        
Y_feature_df = pd.DataFrame(data_list) 


# dropping columns
Y_feature_df = Y_feature_df.drop(columns=['ICU_ADMIT_DATETIME', 'ADMISSION_WEIGHT'])
X_feature_df = X_feature_df.drop(columns=['ICU_ADMIT_DATETIME'])


# saving files
X_feature_df.to_pickle('Temp/eCrit_Features.pickle', compression = 'zip')
Y_feature_df.to_pickle('Temp/eCrit_Target.pickle', compression = 'zip')

# printing check point
print('eCrit_Features and eCrit_Target files saved')


