# ICU_outcome_prediction_Transfer_Learning
Using domain adaptation (DA) and inductive transfer learning (ITL) to improve patient outcome prediction in the intensive care unit

Code used to predict patient outcomes in critically ill patients. Two databases are used. First, a database from Alberta, Canada was used for modelling, this data set is not publicly available. Second, a publicly avaialble MIMIC-III database was used, which can obtained from https://physionet.org/content/mimiciii/1.4/. To access this dataset, create a PhysioNet profile at https://physionet.org/register/ and sign the DUA.

OBS: This code was used on a specific computing clusters at the University of Calgary called MARC and ARC. It needs to be modified to run in a different computing environment.

Pre-trained models folder has the trained best performing transfer learning models for each outcome.

The following 4 patient outcomes were predicted: 
1. 30-day mortality (30_DAYS_EXPIRE)
2. Acute kidney injury (AKI) 
3. Hospital length of stay (H_LOS)
4. ICU length of stay (ICU_LOS)

Entire coding is in python and requirements file has package versions used. 
Python scripts are available in three folders: Preprocessing, Training, Postprocessing.

Sample script name: 01_eCrit_mount_ADMISSIONS
01 indicates the sequence to run the script. eCrit indicates the database used for training. 

Preprocessing has two folders, eCritical and MIMIC, one each for the database. Scripts 01 to 05 are available in each folder.

Seven different data subsets were used for trianing models: 1%, 5%, 10%, 25%, 50%, 75%, and 100%.
Training folder has 14 folders one each for combination of database and data subset (eCrit_1p_data...MIMIC_100p_data). 
Each of these 14 folders has three scripts: MIMIC has 06,07,08 and eCrit has 06,07,09.
For ITL run all eCrit scripts (01 to 09) and for DA run all MIMIC scripts (01 to 08).

Finally run the script "10_eCrit_MIMIC_Statistical_Testing.py" from Postprocessing to get the results.
