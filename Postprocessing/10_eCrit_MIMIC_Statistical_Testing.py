#!/usr/bin/env python
# coding: utf-8
# author: Maruthi Mutnuri (maruthi2008@gmail.com)

# loading libraries
import scipy.stats as stats
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# function for loading files from all the random states and concatenate them
def file_load(path):    
    df = pd.DataFrame()
    # loop to concatenate bootstrap files from all the random states  
    for files in glob.glob(path):
        # load bootstrap file of the single random state
        temp = pd.read_csv(files, index_col=0)
        # concatenate with bootstrap files of other random states
        df = pd.concat([df,temp], axis=0).reset_index(drop=True)
    return(df) # returning concatenated file

# function for calculating the statistics of metrics and perform paired wilcoxon rank sum statistical test for classification models
def stat_CL(df1,df2):
    # Initializing
    stats1 = {}
    stats2 = {}
    viz_stat1 = {}
    viz_stat2 = {}

    # extracting metrics from the dataframes. 
    # df1 is for transfer learning (TL) models. 
    # df2 is for baseline models.    
    BA1 = df1['Balanced Accuracy CI']
    BA2 = df2['Balanced Accuracy CI']
    AUC1 = df1['AUC CI']
    AUC2 = df2['AUC CI']
    ACC1 = df1['Accuracy CI']
    ACC2 = df2['Accuracy CI']
    F1_1 = df1['F1 Score CI']
    F1_2 = df2['F1 Score CI']
    PR1 = df1['Precision CI']
    PR2 = df2['Precision CI']
    RE1 = df1['Recall CI']
    RE2 = df2['Recall CI']
    
    # calculating paired wilcoxon rank sum statistical test results
    stat_BA = stats.wilcoxon(x=BA1, y=BA2, alternative = 'two-sided')
    stat_AUC = stats.wilcoxon(x=AUC1, y=AUC2, alternative = 'two-sided')
    stat_ACC = stats.wilcoxon(x=ACC1, y=ACC2, alternative = 'two-sided')
    stat_F1 = stats.wilcoxon(x=F1_1, y=F1_2, alternative = 'two-sided')
    stat_PR = stats.wilcoxon(x=PR1, y=PR2, alternative = 'two-sided')
    stat_RE = stats.wilcoxon(x=RE1, y=RE2, alternative = 'two-sided')    

    # calculating statistics of metrics: median and 95% confidence interval    
    stats1['BA'] = str(round(BA1.median(),4)) + '(' + str(round(np.quantile(BA1, 0.025),4)) + ',' + str(round(np.quantile(BA1, 0.975),4)) + ')'
    stats2['BA'] = str(round(BA2.median(),4)) + '(' + str(round(np.quantile(BA2, 0.025),4)) + ',' + str(round(np.quantile(BA2, 0.975),4)) + ')'
    stats1['AUC'] = str(round(AUC1.median(),4)) + '(' + str(round(np.quantile(AUC1, 0.025),4)) + ',' + str(round(np.quantile(AUC1, 0.975),4)) + ')'
    stats2['AUC'] = str(round(AUC2.median(),4)) + '(' + str(round(np.quantile(AUC2, 0.025),4)) + ',' + str(round(np.quantile(AUC2, 0.975),4)) + ')'
    stats1['ACC'] = str(round(ACC1.median(),4)) + '(' + str(round(np.quantile(ACC1, 0.025),4)) + ',' + str(round(np.quantile(ACC1, 0.975),4)) + ')'
    stats2['ACC'] = str(round(ACC2.median(),4)) + '(' + str(round(np.quantile(ACC2, 0.025),4)) + ',' + str(round(np.quantile(ACC2, 0.975),4)) + ')'
    stats1['F1'] = str(round(F1_1.median(),4)) + '(' + str(round(np.quantile(F1_1, 0.025),4)) + ',' + str(round(np.quantile(F1_1, 0.975),4)) + ')'
    stats2['F1'] = str(round(F1_2.median(),4)) + '(' + str(round(np.quantile(F1_2, 0.025),4)) + ',' + str(round(np.quantile(F1_2, 0.975),4)) + ')'
    stats1['PR'] = str(round(PR1.median(),4)) + '(' + str(round(np.quantile(PR1, 0.025),4)) + ',' + str(round(np.quantile(PR1, 0.975),4)) + ')'
    stats2['PR'] = str(round(PR2.median(),4)) + '(' + str(round(np.quantile(PR2, 0.025),4)) + ',' + str(round(np.quantile(PR2, 0.975),4)) + ')'
    stats1['RE'] = str(round(RE1.median(),4)) + '(' + str(round(np.quantile(RE1, 0.025),4)) + ',' + str(round(np.quantile(RE1, 0.975),4)) + ')'
    stats2['RE'] = str(round(RE2.median(),4)) + '(' + str(round(np.quantile(RE2, 0.025),4)) + ',' + str(round(np.quantile(RE2, 0.975),4)) + ')'

    # calculating statistics of metrics useful for visualization    
    viz_stat1['BA'], viz_stat1['BA_LL'], viz_stat1['BA_UL'],viz_stat1['AUC'], viz_stat1['AUC_LL'], viz_stat1['AUC_UL'] = BA1.median(),np.quantile(BA1, 0.025),np.quantile(BA1, 0.975),AUC1.median(),np.quantile(AUC1, 0.025),np.quantile(AUC1, 0.975)
    viz_stat2['BA'], viz_stat2['BA_LL'], viz_stat2['BA_UL'],viz_stat2['AUC'], viz_stat2['AUC_LL'], viz_stat2['AUC_UL'] = BA2.median(),np.quantile(BA2, 0.025),np.quantile(BA2, 0.975),AUC2.median(),np.quantile(AUC2, 0.025),np.quantile(AUC2, 0.975)

    # returning all the stats    
    return(stat_BA,stat_AUC,stat_ACC,stat_F1,stat_PR,stat_RE,stats1,stats2,viz_stat1,viz_stat2)

# function for calculating the statistics of metrics and perform paired wilcoxon rank sum statistical test for regression models
def stat_REG(df1,df2):
    # Initializing
    stats1 = {}
    stats2 = {}
    viz_stat1 = {}
    viz_stat2 = {}

    # extracting metrics from the dataframes. 
    # df1 is for transfer learning models. 
    # df2 is for baseline models.    
    MSE1 = df1['MSE CI']
    MSE2 = df2['MSE CI']
    MAE1 = df1['MAE CI']
    MAE2 = df2['MAE CI']

    # calculating paired wilcoxon rank sum statistical test results    
    stat_MSE = stats.wilcoxon(x=df1['MSE CI'], y=df2['MSE CI'], alternative = 'two-sided')
    stat_MAE = stats.wilcoxon(x=df1['MAE CI'], y=df2['MAE CI'], alternative = 'two-sided')

    # calculating statistics of metrics: median and 95% confidence interval (CI)     
    stats1['MSE'] = str(round(MSE1.median(),4)) + '(' + str(round(np.quantile(MSE1, 0.025),4)) + ',' + str(round(np.quantile(MSE1, 0.975),4)) + ')'
    stats2['MSE'] = str(round(MSE2.median(),4)) + '(' + str(round(np.quantile(MSE2, 0.025),4)) + ',' + str(round(np.quantile(MSE2, 0.975),4)) + ')'
    stats1['MAE'] = str(round(MAE1.median(),4)) + '(' + str(round(np.quantile(MAE1, 0.025),4)) + ',' + str(round(np.quantile(MAE1, 0.975),4)) + ')'
    stats2['MAE'] = str(round(MAE2.median(),4)) + '(' + str(round(np.quantile(MAE2, 0.025),4)) + ',' + str(round(np.quantile(MAE2, 0.975),4)) + ')'

    # calculating statistics of metrics useful for visualization
    viz_stat1['MAE'], viz_stat1['MAE_LL'], viz_stat1['MAE_UL'],viz_stat1['MSE'], viz_stat1['MSE_LL'], viz_stat1['MSE_UL'] = MAE1.median(),np.quantile(MAE1, 0.025),np.quantile(MAE1, 0.975),MSE1.median(),np.quantile(MSE1, 0.025),np.quantile(MSE1, 0.975)
    viz_stat2['MAE'], viz_stat2['MAE_LL'], viz_stat2['MAE_UL'],viz_stat2['MSE'], viz_stat2['MSE_LL'], viz_stat2['MSE_UL'] = MAE2.median(),np.quantile(MAE2, 0.025),np.quantile(MAE2, 0.975),MSE2.median(),np.quantile(MSE2, 0.025),np.quantile(MSE2, 0.975)

    # returning all the stats  
    return(stat_MSE,stat_MAE, stats1, stats2,viz_stat1,viz_stat2)    

# function for visualizations
def viz(df,TL,base,outcome):
    # plot with two sub plots
    fig, axs = plt.subplots(1, 2)

    # visualizations for classification models 
    # if one of the baseline model is logistic regression (LR)   
    if base == 'LR':
        # renaming outcome
        if outcome == '30_DAYS_EXPIRE':
            outcome = '30-Day_mortality'
            
        # setting label, suptitle, and file path & name
        if TL == 'ITL':
            label = 'Inductive Transfer Learning'
            suptitle ='(A) ' + outcome + ', ITL vs LR vs FCNN'
            filename = "Results/" + outcome + '_ITL_LR.svg'
        else:
            label = 'Domain Adaptation'
            suptitle = '(B) ' + outcome + ', DA vs LR vs FCNN'
            filename = "Results/" + outcome + '_DA_LR.svg'
            
# Balanced Accuracy visualization
        # TL visualization
        # extracting x,y,upper limit and lower limit values from dataframe
        x = df[df['Model'] == TL]['Data set %']
        y = df[df['Model'] == TL]['BA']
        ul = df[df['Model'] == TL]['BA_UL']
        ll = df[df['Model'] == TL]['BA_LL']
        # plotting the median line
        axs[0].plot(x,y, label = label)
        # plotting the area for 95% CI
        axs[0].fill_between(x, ll, ul, color='b', alpha=.1)

        # baseline LR visualization
        x = df[df['Model'] == 'LR']['Data set %']
        y = df[df['Model'] == 'LR']['BA']
        ul = df[df['Model'] == 'LR']['BA_UL']
        ll = df[df['Model'] == 'LR']['BA_LL']
        axs[0].plot(x,y,color='r', label = 'Logistic Regression')
        axs[0].fill_between(x, ll, ul, color='r', alpha=.1)

        # baseline FCNN visualization        
        x = df[df['Model'] == 'FCNN']['Data set %']
        y = df[df['Model'] == 'FCNN']['BA']
        ul = df[df['Model'] == 'FCNN']['BA_UL']
        ll = df[df['Model'] == 'FCNN']['BA_LL']
        axs[0].plot(x,y,color='g', label = 'Fully Connected Neural Network')
        axs[0].fill_between(x, ll, ul, color='g', alpha=.1)        

        # setting x and y labels, subplot title, legend location
        axs[0].set_xlabel("Data set %")
        axs[0].set_ylabel("Balanced Accuracy")
        axs[0].set_title(label = "Balanced Accuracy")
        axs[0].legend(loc='lower right')

# AUC visualizations
        # TL visualization
        x = df[df['Model'] == TL]['Data set %']
        y = df[df['Model'] == TL]['AUC']
        ul = df[df['Model'] == TL]['AUC_UL']
        ll = df[df['Model'] == TL]['AUC_LL']
        axs[1].plot(x,y, label = label)
        axs[1].fill_between(x, ll, ul, color='b', alpha=.1)

        # baseline LR visualization
        x = df[df['Model'] == 'LR']['Data set %']
        y = df[df['Model'] == 'LR']['AUC']
        ul = df[df['Model'] == 'LR']['AUC_UL']
        ll = df[df['Model'] == 'LR']['AUC_LL']
        axs[1].plot(x,y,color='r', label = 'Logistic Regression')
        axs[1].fill_between(x, ll, ul, color='r', alpha=.1)

        # baseline FCNN visualization        
        x = df[df['Model'] == 'FCNN']['Data set %']
        y = df[df['Model'] == 'FCNN']['AUC']
        ul = df[df['Model'] == 'FCNN']['AUC_UL']
        ll = df[df['Model'] == 'FCNN']['AUC_LL']
        axs[1].plot(x,y,color='g', label = 'Fully Connected Neural Network')
        axs[1].fill_between(x, ll, ul, color='g', alpha=.1)         

        # setting x and y labels, subplot title, legend location
        axs[1].set_xlabel("Data set %")
        axs[1].set_ylabel("AUC")
        axs[1].set_title(label = "AUC")
        axs[1].legend(loc='lower right')

        # setting plot title, its size, saving the file.
        plt.rc('figure', titlesize=22)  # fontsize of the figure title
        fig.suptitle(suptitle)
        fig.set_size_inches(12, 5)
        plt.subplots_adjust(top=0.85)
        plt.savefig(filename, format="svg")
        plt.show()

    # visualizations for regression models 
    # if none of baseline model are LR
    else:
        if outcome == '30_DAYS_EXPIRE':
            outcome = '30-Day_mortality'
        if TL == 'ITL':
            label = 'Inductive Transfer Learning'
            suptitle ='(A) ' + outcome + ', ITL vs Lasso vs FCNN'
            filename = "Results/" + outcome + '_ITL_Lasso.svg'
        else:
            label = 'Domain Adaptation'
            suptitle = '(B) ' + outcome + ', DA vs Lasso vs FCNN'
            filename = "Results/" + outcome + '_DA_Lasso.svg'
            
# MAE visualizations
        x = df[df['Model'] == TL]['Data set %']
        y = df[df['Model'] == TL]['MAE']
        ul = df[df['Model'] == TL]['MAE_UL']
        ll = df[df['Model'] == TL]['MAE_LL']
        axs[0].plot(x,y, label = label)
        axs[0].fill_between(x, ll, ul, color='b', alpha=.1)


        x = df[df['Model'] == 'Lasso']['Data set %']
        y = df[df['Model'] == 'Lasso']['MAE']
        ul = df[df['Model'] == 'Lasso']['MAE_UL']
        ll = df[df['Model'] == 'Lasso']['MAE_LL']
        axs[0].plot(x,y,color='r', label = 'Lasso')
        axs[0].fill_between(x, ll, ul, color='r', alpha=.1)
        
        x = df[df['Model'] == 'FCNN']['Data set %']
        y = df[df['Model'] == 'FCNN']['MAE']
        ul = df[df['Model'] == 'FCNN']['MAE_UL']
        ll = df[df['Model'] == 'FCNN']['MAE_LL']
        axs[0].plot(x,y,color='g', label = 'Fully Connected Neural Network')
        axs[0].fill_between(x, ll, ul, color='g', alpha=.1)        

        axs[0].set_xlabel("Data set %")
        axs[0].set_ylabel("MAE")
        axs[0].set_title(label = "MAE")
        axs[0].legend(loc='upper right')

# MSE visualizations
        x = df[df['Model'] == TL]['Data set %']
        y = df[df['Model'] == TL]['MSE']
        ul = df[df['Model'] == TL]['MSE_UL']
        ll = df[df['Model'] == TL]['MSE_LL']
        axs[1].plot(x,y, label = label)
        axs[1].fill_between(x, ll, ul, color='b', alpha=.1)


        x = df[df['Model'] == 'Lasso']['Data set %']
        y = df[df['Model'] == 'Lasso']['MSE']
        ul = df[df['Model'] == 'Lasso']['MSE_UL']
        ll = df[df['Model'] == 'Lasso']['MSE_LL']
        axs[1].plot(x,y,color='r', label = 'Lasso')
        axs[1].fill_between(x, ll, ul, color='r', alpha=.1)
        
        x = df[df['Model'] == 'FCNN']['Data set %']
        y = df[df['Model'] == 'FCNN']['MSE']
        ul = df[df['Model'] == 'FCNN']['MSE_UL']
        ll = df[df['Model'] == 'FCNN']['MSE_LL']
        axs[1].plot(x,y,color='g', label = 'Fully Connected Neural Network')
        axs[1].fill_between(x, ll, ul, color='g', alpha=.1)        

        axs[1].set_xlabel("Data set %")
        axs[1].set_ylabel("MSE")
        axs[1].set_title(label = "MSE")
        axs[1].legend(loc='upper right')


        plt.rc('figure', titlesize=22)  # fontsize of the figure title
        fig.suptitle(suptitle)
        fig.set_size_inches(12, 5)
        plt.subplots_adjust(top=0.85)
        plt.savefig(filename, format="svg")
        plt.show()


# main routine starts

# Initializing variables
results = pd.DataFrame() #Dataframe for results
cohorts = ['eCrit','MIMIC']
subsets = ['1p','5p','10p', '25p', '50p', '75p', '100p'] # subset values for file names
outcomes = ['30_DAYS_EXPIRE', 'AKI', 'H_LOS', 'ICU_LOS']
subsets_p = [1,5,10, 25, 50, 75, 100] # subset values for viz
extension = '*.csv'

# Looping through all files obtained from model training (bootstrap), to compile results and viz data.
# looping through all cohorts
for cohort in cohorts:
    # looping through all outcomes
    for outcome in outcomes:
        viz_stats = pd.DataFrame() #Dataframe for visualizations
        name_base = "_"+outcome +"_Bootstrap_" # file name of baseline LR models
        name_base_FCNN = "_"+outcome +"_FCNN_Bootstrap_" # file name of baseline FCNN models
        
        if cohort == 'eCrit':
            name_TL = "eCrit_to_eCrit_Inductive_TL_"+ outcome +"_Bootstrap_" # file name of TL models
            model = 'ITL' 
        else:
            name_TL = "eCrit_to_MIMIC_Domain_TL_"+ outcome +"_Bootstrap_" # file name of TL models
            model = 'DA'
        # looping through all subsets    
        for i,subset in enumerate(subsets):
            # file path to results folder
            # make sure path points to the folder which has results from bootstrap samples
            # example path and file: Training/eCrit_1p_data/Results/eCrit_30_DAYS_EXPIRE_Bootstrap_1p_25.csv
            path = "Training/"+cohort+"_" + subset + "_data/Results/" 

            # regression models            
            if outcome == 'H_LOS' or outcome == 'ICU_LOS':
                name_base = "_"+outcome + "_Lasso_Bootstrap_" # file name of baseline Lasso models
                df_TL = file_load(path+name_TL+subset+extension) # loading TL files
                df_base = file_load(path+cohort+name_base+subset+extension) # loading baseline Lasso files
                df_base_FCNN = file_load(path+cohort+name_base_FCNN+subset+extension) # loading baseline FCNN files

                
                #TL and baseline (Lasso) stats extraction from bootstrap data
                stat_MSE,stat_MAE,stats_TL,stats_base,viz_stat_TL,viz_stat_base = stat_REG(df_TL, df_base)                
                
                # creating temp TL dataframe with results
                temp_TL = pd.DataFrame({'DB': cohort,'Model': model,'Outcome': outcome,'Data set %': subsets_p[i],
                     'MAE': stats_TL['MAE'],'p_MAE':'','MSE': stats_TL['MSE'],'p_MSE':''}, index=[0])

                # creating temp baseline Lasso dataframe with results                
                temp_base = pd.DataFrame({'DB': cohort,'Model': 'Lasso','Outcome': outcome,'Data set %': subsets_p[i],
                     'MAE': stats_base['MAE'],'p_MAE':stat_MAE[1],'MSE': stats_base['MSE'],'p_MSE':stat_MSE[1]}, index=[0])                                

                # updating visualization dataframe of TL and baseline Lasso with model, outcome, subsets values 
                viz_stat_TL['Model'], viz_stat_TL['Outcome'],viz_stat_TL['Data set %'] = model,outcome,subsets_p[i]
                viz_stat_base['Model'], viz_stat_base['Outcome'],viz_stat_base['Data set %'] = 'Lasso',outcome,subsets_p[i]
                
                # baseline (FCNN) stats extraction from bootstrap data
                stat_MSE,stat_MAE,_,stats_base,_,viz_stat_base_FCNN = stat_REG(df_TL, df_base_FCNN)

                # creating temp baseline FCNN dataframe with results                 
                temp_base_FCNN = pd.DataFrame({'DB': cohort,'Model': 'FCNN','Outcome': outcome,'Data set %': subsets_p[i],
                     'MAE': stats_base['MAE'],'p_MAE':stat_MAE[1],'MSE': stats_base['MSE'],'p_MSE':stat_MSE[1]}, index=[0])

                # updating visualization dataframe of baseline FCNN with model, outcome, subsets values                 
                viz_stat_base_FCNN['Model'], viz_stat_base_FCNN['Outcome'],viz_stat_base_FCNN['Data set %'] = 'FCNN',outcome,subsets_p[i]

            # classification models                                  
            else:            
                df_TL = file_load(path+name_TL+subset+extension) # loading TL files
                df_base = file_load(path+cohort+name_base+subset+extension) # loading baseline Lasso files
                df_base_FCNN = file_load(path+cohort+name_base_FCNN+subset+extension) # loading baseline FCNN files
                
                #TL and baseline (LR) stats extraction from bootstrap data                
                stat_BA,stat_AUC,stat_ACC,stat_F1,stat_PR,stat_RE,stats_TL,stats_base,viz_stat_TL,viz_stat_base = stat_CL(df_TL, df_base)
                
                # creating temp TL dataframe with results
                temp_TL = pd.DataFrame({'DB': cohort,'Model': model,'Outcome': outcome,'Data set %': subsets_p[i],
                         'BA': stats_TL['BA'],'p_BA':'','AUC': stats_TL['AUC'],'p_AUC':'',
                         'ACC': stats_TL['ACC'],'p_ACC':'','F1': stats_TL['F1'],'p_F1':'',
                         'PR': stats_TL['PR'],'p_PR':'','RE': stats_TL['RE'],'p_RE':''}, index=[0])

                # creating temp baseline LR dataframe with results                 
                temp_base = pd.DataFrame({'DB': cohort,'Model': 'LR','Outcome': outcome,'Data set %': subsets_p[i],
                         'BA': stats_base['BA'],'p_BA':stat_BA[1],'AUC': stats_base['AUC'],'p_AUC':stat_AUC[1],
                         'ACC': stats_base['ACC'],'p_ACC':stat_ACC[1],'F1': stats_base['F1'],'p_F1':stat_F1[1],
                         'PR': stats_base['PR'],'p_PR':stat_PR[1],'RE': stats_base['RE'],'p_RE':stat_RE[1]}, index=[0])
                
                # updating visualization dataframe of TL and baseline LR with model, outcome, subsets values                                 
                viz_stat_TL['Model'], viz_stat_TL['Outcome'],viz_stat_TL['Data set %'] = model,outcome,subsets_p[i]
                viz_stat_base['Model'], viz_stat_base['Outcome'],viz_stat_base['Data set %'] = 'LR',outcome,subsets_p[i]

                # baseline (FCNN) stats extraction from bootstrap data                
                stat_BA,stat_AUC,stat_ACC,stat_F1,stat_PR,stat_RE,_,stats_base,_,viz_stat_base_FCNN = stat_CL(df_TL, df_base_FCNN)

                # creating temp baseline FCNN dataframe with results                  
                temp_base_FCNN = pd.DataFrame({'DB': cohort,'Model': 'FCNN','Outcome': outcome,'Data set %': subsets_p[i],
                         'BA': stats_base['BA'],'p_BA':stat_BA[1],'AUC': stats_base['AUC'],'p_AUC':stat_AUC[1],
                         'ACC': stats_base['ACC'],'p_ACC':stat_ACC[1],'F1': stats_base['F1'],'p_F1':stat_F1[1],
                         'PR': stats_base['PR'],'p_PR':stat_PR[1],'RE': stats_base['RE'],'p_RE':stat_RE[1]}, index=[0])
                
                # updating visualization dataframe of baseline FCNN with model, outcome, subsets values                 
                viz_stat_base_FCNN['Model'], viz_stat_base_FCNN['Outcome'],viz_stat_base_FCNN['Data set %'] = 'FCNN',outcome,subsets_p[i]

            # updating the results dataframe with recently extracted stats    
            results = pd.concat([results,temp_TL], axis=0).reset_index(drop=True)
            results = pd.concat([results,temp_base], axis=0).reset_index(drop=True)
            results = pd.concat([results,temp_base_FCNN], axis=0).reset_index(drop=True)

            # updating the visualization dataframe with recently extracted stats 
            viz_stats = pd.concat([viz_stats,pd.DataFrame([viz_stat_TL])], axis=0).reset_index(drop=True)
            viz_stats = pd.concat([viz_stats,pd.DataFrame([viz_stat_base])], axis=0).reset_index(drop=True)
            viz_stats = pd.concat([viz_stats,pd.DataFrame([viz_stat_base_FCNN])], axis=0).reset_index(drop=True)

        
        if outcome == 'H_LOS' or outcome == 'ICU_LOS':
            viz(viz_stats,model,'Lasso',outcome) # calling the visualization function
        else:
            viz(viz_stats,model,'LR',outcome) # calling the visualization function

# saving the results file
results.to_csv('Results.csv')            
           

