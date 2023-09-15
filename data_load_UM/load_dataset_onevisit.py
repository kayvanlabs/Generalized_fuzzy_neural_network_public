from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import platform,os


def convert_gender_to_int(x):
    return 0 if x == 'M' else 1

def date_convert(date_str):
    d = datetime.strptime(date_str, '%m/%d/%y %H:%M').date()
    return d

def carry_forward_imputation(table,id_ls):
    df = pd.DataFrame()
    for idx in id_ls:
        sub = table.loc[table.HFID == idx,:] # table.loc[table.HFID == idx,:]
        sub = sub.fillna(method="ffill")
        df = pd.concat([df,sub])
    return df

def lstrip_word(str, word):
    if str.startswith(word):
        return str[len(word):]
    return str

    
def inspect_statistics(data, n_classes):
    print('The number of total patients is ', np.unique(data.HFID).shape[0])
    print('The number of total samples is ', data.shape[0])
    for label in range(1,n_classes+1):
        print(f'The number of patients with label {label} is ', len(np.unique(data[data.loc[:, 'Cohort'] == label].loc[:, 'HFID'])),
              f'\nThe number of observations with label {label} is ', data[data.loc[:, 'Cohort'] == label].shape[0])

def reject_outliers(data, m=2):
    return data[abs(data - np.nanmean(data)) < m * np.nanstd(data)]


def array_summary(ls):
    return np.stack([np.nanmax(ls),np.nanmin(ls),np.nanmean(ls),np.nanmedian(ls),np.nanstd(ls)],axis = 0)

def compute_summary_removing_outlier(sub):
    sub = np.array(sub).astype('float64')
    sub_stats = np.zeros((sub.shape[1],5))
    for i in range(sub.shape[1]):
        wo_outlier = reject_outliers(sub[:,i],m =1)
        sub_stats[i,:] = array_summary(wo_outlier)
    return sub_stats

def compute_summary(sub):
    sub = np.array(sub).astype('float64')
    sub_stats = np.zeros((sub.shape[1],5))
    for i in range(sub.shape[1]):
        sub_max = np.nanmax(np.array(sub[:,i]).astype('float64'), axis=0)
        sub_min = np.nanmin(np.array(sub[:,i]).astype('float64'), axis=0)
        sub_mean = np.nanmean(np.array(sub[:,i]).astype('float64'), axis=0)
        sub_median = np.nanmedian(np.array(sub[:,i]).astype('float64'), axis=0)
        sub_sd = np.nanstd(np.array(sub[:,i]).astype('float64'), axis=0)
        sub_q_25 = np.nanquantile(np.array(sub[:,i]).astype('float64'), q = 0.25,axis=0)
        sub_q_75 = np.nanquantile(np.array(sub[:,i]).astype('float64'), q = 0.75,axis=0)
        ls = np.stack([sub_max,sub_min,sub_mean,sub_median,sub_sd],axis = 0)
        sub_stats[i,:] = ls
    
    
    return sub_stats


def calculate_mean_based_on_class(table):
    
    pos_sub = table[table['Cohort'] == 1].drop(['Cohort'],axis = 1)
    neg_sub = table[table['Cohort'] == 2].drop(['Cohort'],axis = 1)
    columns = pos_sub.columns

    # pos_ls = compute_summary_removing_outlier(pos_sub)
    # neg_ls = compute_summary_removing_outlier(neg_sub)
    all_ls = compute_summary(table.drop(['Cohort'],axis = 1))
    pos_ls = compute_summary(pos_sub)
    neg_ls = compute_summary(neg_sub)

    sum_tbl = pd.DataFrame(np.concatenate([all_ls,pos_ls,neg_ls],axis = 1),
           columns = [
               
               'all max','all min','all mean','all median','all std',
               'pos max','pos min','pos mean','pos median','pos std',
                    'neg max','neg min','neg mean','neg median','neg std'],index = columns)
    return sum_tbl


# def calculate_mean_based_on_class(table):
    
#     table = table.iloc[:,:]
#     columns = table.columns

#     pos_sub = table[table['Cohort'] == 1]
#     neg_sub = table[table['Cohort'] == 2]


#     pos_ls = compute_summary_removing_outlier(pos_sub)
#     neg_ls = compute_summary_removing_outlier(neg_sub)


#     # pos_ls = compute_summary(pos_sub)
#     # neg_ls = compute_summary(neg_sub)
    
#     sum_tbl = pd.DataFrame(np.concatenate([pos_ls,neg_ls],axis = 1),
#            columns = ['pos max','pos min','pos mean','pos median','pos std',
#                     'neg max','neg min','neg mean','neg median','neg std'],index = columns)
#     return sum_tbl

def drop_variables(tbl):
    table = tbl.iloc[:,:-3]
    columns = table.columns
    missing_values = np.sum(np.isnan(np.array(table)), axis=0)/table.shape[0]
    missing_index_list = missing_values>0.6
    
    for index, name in enumerate(columns):
        if missing_index_list[index] == 1:
            print('Remove {} because of {:.2f}% missing values.'.format(columns[index],
                                                                    missing_values[index]*100))
            table = table.drop(columns[index], axis=1)
    
    table = pd.concat([table,tbl.iloc[:,-3:]],axis = 1)
    
    return table

def build_categorical_index(data_table):
    column_names = data_table.columns
    feature_names = [i for i in column_names if i not in ['HFID','EncID','Cohort']] 
    category = pd.Series(np.zeros([len(feature_names)]), feature_names)

    # If one variable is categorical, then set the number of categories
    category.GenderCode = 2
    category.MitralRegurg = 5

    comorbid_variables = ['AlcoholAbuse', 'BloodLossAnemia',
       'CardiacArrhythmias', 'ChronicPulmonaryDisease', 'Coagulopathy', 'DeficiencyAnemia', 'Depression',
       'DiabetesComplicated', 'DiabetesUncomplicated', 'DrugAbuse',
       'FluidElectrolyteDisorders', 'HypertensionComplicated',
       'HypertensionUncomplicated', 'Hypothyroidism', 'LiverDisease',
       'Lymphoma', 'MetastaticCancer', 'Obesity', 'OtherNeurologicalDisorders',
       'Paralysis', 'PepticUlcerDiseaseExcludingBleeding',
       'PeripheralVascularDisorders', 'Psychoses',
       'PulmonaryCirculationDisorders', 'RenalFailure',
       'RheumatoidArthritisCollagenVascularDiseases',
       'SolidTumorWithoutMetastasis', 'ValvularDisease', 'WeightLoss']

    comorbid_variables_ls = [i for i in comorbid_variables]
    # comorbid_variables_ls = ['previous_visit_' + i for i in comorbid_variables]
    for v in comorbid_variables_ls:
        if v in column_names:
            category[v] = 2
        else:
            continue

    column_names = category.index
    feature_names = column_names
    
    return np.array(category).astype(np.int32), feature_names.tolist()


def load_NSF_dataset(missing_value_tolerance,fill_missing = True, drop_last_features = True,show_statistics = True):
    if platform.system() == 'Windows':
        ROOT = 'G:/'
    else:
        ROOT = '/nfs/turbo/med-kayvan-lab/' # /nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Processed/Yufeng/Extended_data

    # read pre-processed data
    csv_file = os.path.join(ROOT, 'Projects/HeartFailure/Data/Processed/Yufeng/Extended_data/cohort123_lab_diff.csv')  
    if csv_file is None:
        raise ValueError
    table = pd.read_csv(csv_file)
    print('Original data has {} records'.format(table.shape[0]))

    
    table['GenderCode'] = table['GenderCode'].apply(lambda x: convert_gender_to_int(x))
    
    if drop_last_features:
        table = table.loc[:,~table.columns.str.startswith('last')]
    table = table.drop(['CREAT','BNP','RenalFailure'],axis = 1)
    table.rename(columns={'BP':'SBP'}, inplace=True)


    table.EncounterStart = table.EncounterStart.apply(lambda x: date_convert(x))
    table.EncounterEnd = table.EncounterEnd.apply(lambda x: date_convert(x))
    table.HFID = table.HFID.astype(np.uint64)
    table.EncID = table.EncID.astype(np.uint64)
    table.Cohort = table.Cohort.astype(np.uint64)
    
    table.iloc[:,2:-3] = table.iloc[:,2:-3].apply(pd.to_numeric,errors = 'coerce')
    
    missing_before_imputation = pd.DataFrame(data = np.sum(np.array(pd.isnull(table)),axis = 0)/table.shape[0] * 100,
                                            index = list(table.columns),
                                            columns = ['missing rate'])
                                            
    # missing_before_imputation.to_csv('files_stats/missing_before_imputation.csv')
    
    table = drop_variables(table)
    
    table.replace([np.inf, -np.inf], np.nan, inplace=True)
    print('Before removing records:',table.shape[0])
    table = table[np.sum(np.isnan(table.iloc[:, 2:-3]),1) < missing_value_tolerance] 
    print('After removing records with more than {} missing values the sample reduced to {}'.format(missing_value_tolerance,table.shape[0]))
    
    table = table.loc[table.Cohort != 3,:]
    
    uniq_ID = set(np.unique(table['HFID']))
    ID_ENC = {}
    ID_CNT = {}
    for idx in uniq_ID:
        ID_ENC[idx] = list(np.unique(table[table['HFID'] == idx]['EncID']))
        ID_CNT[idx] = len(list(np.unique(table[table['HFID'] == idx]['EncID'])))

    id_ls = []
    for idx in ID_ENC:
        if len(ID_ENC[idx]) >= 2:
            id_ls.append(idx)
    # table = carry_forward_imputation(table,id_ls)
    
    
    # missing_after_carry_forward = pd.DataFrame(data = np.sum(np.array(pd.isnull(table)),axis = 0)/table.shape[0] * 100,
    #                                         index = list(table.columns),
    #                                         columns = ['missing rate'])
                                            
    # missing_after_carry_forward.to_csv('files_stats/missing_after_carry_forward.csv')                                        
    
    
    
    colnames = ['previous_visit_' + i for i in table.columns]
    # colnames = table.columns
    
    new_tbl = pd.DataFrame()
    total_cnt = 0
    for idx in id_ls:
        sub = table.loc[table.HFID == idx,:]
        l = list(sub.EncID)
        if all(l[i] <= l[i+1] for i in range(len(l) - 1)) is False:
            sub = sub.sort_values(by = ['EncounterStart'], ascending = True)
        cnt = sub.shape[0]
        total_cnt += (cnt - 1)
        for i in range(cnt - 1):
            if (sub.iloc[i,1] == sub.iloc[i+1,1]):
                continue
            else:
                sub_sub = pd.DataFrame(sub.iloc[i,:]).transpose()
                sub_sub.columns = colnames
                sub_sub['Cohort'] = sub.iloc[i+1,-1]
                sub_sub['recent_visit_EncounterStart'] = sub.iloc[i+1,:]['EncounterStart']
                new_tbl = pd.concat([new_tbl,sub_sub],ignore_index=True)
                
    new_tbl['time_elapse'] = new_tbl['recent_visit_EncounterStart'] - new_tbl['previous_visit_EncounterStart']
    new_tbl['time_elapse'] = new_tbl['recent_visit_EncounterStart'] - new_tbl['previous_visit_EncounterStart']
    new_tbl['time_elapse'] = new_tbl['time_elapse'].apply(lambda x: x.days)
    
    new_tbl = new_tbl.drop(['previous_visit_EncounterStart','recent_visit_EncounterStart','time_elapse'],axis = 1)
    new_tbl = new_tbl.drop(['previous_visit_Cohort','previous_visit_EncounterEnd'],axis = 1)
    new_tbl.rename(columns={'previous_visit_HFID':'HFID',
                        'previous_visit_EncID':'EncID',
                        'previous_visit_GenderCode': 'GenderCode'}, inplace=True)
    
    
    
    new_tbl = new_tbl.drop('EncID',axis = 1)
    corrected_colnames = []
    for col in new_tbl.columns:
        if 'previous_visit_' in col:
            cc = lstrip_word(col,'previous_visit_')
            corrected_colnames.append(cc)
        else:
            corrected_colnames.append(col)
    new_tbl.columns = corrected_colnames
    new_tbl.rename(columns={'diff_percent_BNP':'BNP change',
                        'diff_percent_CREAT':'CREAT change'}, inplace=True)
    category_info, feature_names = build_categorical_index(new_tbl)

    missing_after_without_imputation = pd.DataFrame(data = np.sum(np.array(pd.isnull(new_tbl)),axis = 0)/new_tbl.shape[0] * 100,
                                            index = list(new_tbl.columns),
                                            columns = ['missing rate'])
    # missing_after_without_imputation.to_csv('files_stats/missing_after_without_imputation.csv') 

    
    
    print('Prediction Purpopose:',new_tbl.shape[0])
    
    # new_tbl.to_csv("files_stats/new_tbl.csv",index = False)
    new_stat = calculate_mean_based_on_class(new_tbl)
    # new_stat.to_csv("files_stats/new_stat.csv")

    data = np.array(new_tbl)
    # np.savetxt('post_HFID.txt', data[:,0].astype(np.float64), delimiter=',')
    labels = data[:, -1]
    labels = labels - 1 # 0 1 
    where_0 = np.where(labels == 0)
    where_1 = np.where(labels == 1)

    labels[where_0] = 1
    labels[where_1] = 0
    # data = np.concatenate([data[:,:-2],data[:,-1:]],axis = 1)
    data = data[:,:-1]

    neg_index = labels == 0
    pos_index = labels == 1

    neg_label = labels[neg_index]
    pos_label = labels[pos_index]

    neg_data = data[neg_index,:]
    pos_data = data[pos_index,:]

    # if fill_missing:
    #     neg_data[:, 1:] = fill_in_missing_value(neg_data[:, 1:])
    #     pos_data[:, 1:] = fill_in_missing_value(pos_data[:, 1:])
    data = np.concatenate([neg_data,pos_data],axis = 0)
    
    # data[:,1:] = fill_in_missing_value(data[:,1:])
    
    data[:,1:] = data[:,1:].astype(np.float64)
    data[:,0] = data[:,0].astype(np.int64)
    labels = np.concatenate([neg_label,pos_label],axis = 0)

    feature_names = list(feature_names)
    print('There are {} features in total'.format(len(feature_names)))
    
    # Show final statistics
    print('For the binary classification')
    print('The number of patients with negative samples ', np.unique(data[labels == 0, 0]).shape[0])
    print('The number of negative is ', data[labels == 0, 0].shape)
    print('The number of patients with positive samples ', np.unique(data[labels == 1, 0]).shape[0])
    print('The number of positive is ', data[labels == 1, 0].shape)
    #==================================================================================

    # initilaize the training network 
    rule_data =  [
                      {'Relation': [
                                    [feature_names.index('LVEF'), 0],
                                    [feature_names.index('SBP'), 0]], # Rule 9
                                'Out_weight': 1},

                      
                    {'Relation': [
                                    [feature_names.index('MAP'), 0],
                                    [feature_names.index('BMI'), 0],
                                    [feature_names.index('LVEF'), 0],
                                    [feature_names.index('LVEF'), 1],
                                    [feature_names.index('LiverDisease'), 0]
                                    
                                 ],
                                'Out_weight': 1},

                    {'Relation': [
                                    [feature_names.index('LVEF'), 0],
                                    [feature_names.index('CREAT change'), 1],
                                    [feature_names.index('CREAT change'), 2]], # Rule 9
                                'Out_weight': 1}, 

                    {'Relation': [
                                    [feature_names.index('MAP'), 0],
                                    [feature_names.index('CREAT change'), 2],
                                    [feature_names.index('BloodLossAnemia'), 0]], # Rule 9
                                'Out_weight': 1},

                    {'Relation': [
                                    [feature_names.index('SBP'), 0],
                                    [feature_names.index('HGB'), 0],
                                    [feature_names.index('DiabetesUncomplicated'), 0]], # Rule 9
                                'Out_weight': 1},
                      
                      {'Relation': [
                                    [feature_names.index('SBP'), 0],
                                    [feature_names.index('BNP change'), 2], 
                                ],
                                'Out_weight': 1},

                     {'Relation': [
                                    [feature_names.index('LVEF'), 0],
                                    [feature_names.index('SOD'), 0], 
                                ],
                                'Out_weight': 1},
                      

                        ]

  
                        
                        
                        
                        
                        
                        
                        
    print(len(rule_data))



    dataset = {
        'table': new_tbl,
        'variables': data,
        'response': labels,
        'feature_names': feature_names,
        'num_classes': 2,
        'category_info': category_info,
        'split_method': 'patient_wise',
        'rule_data': rule_data,
        }

    return dataset

if __name__ == '__main__':
    dataset = load_NSF_dataset(missing_value_tolerance = 10,fill_missing=True)
