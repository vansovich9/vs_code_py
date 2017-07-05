import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, Normalizer

def LoadFile(file_name ):
    data = pd.read_csv(file_name, header=0,
                    sep=';', na_values=['None'])

    #data['age'] = round(data['age']/365, 2)

    data['ap_hi'] = abs(data['ap_hi'])
    data['ap_lo'] = abs(data['ap_lo'])

    data.loc[(data['ap_lo'] == 0),'ap_lo'] = np.NaN
    data.loc[(data['ap_hi'] == 0),'ap_hi'] = np.NaN

    data['gender'] = data['gender'] - 1
    
    if(data[(data['ap_lo'] == 0)]['ap_lo'].count()>0):
        print("ap_lo execute 0")
    if(data[(data['ap_hi'] == 0)]['ap_hi'].count()>0):
        print("ap_hi execute 0")
    while (data[data.ap_lo > 200]['id'].count()>0):
        data.loc[(data['ap_lo'] > 200),'ap_lo'] = data[data.ap_lo > 200]['ap_lo'] // 10

    while (data[data.ap_lo <= 20]['id'].count()>0):
        data.loc[(data['ap_lo'] <= 20),'ap_lo'] = data[data.ap_lo <= 20]['ap_lo'] * 10

    while (data[data.ap_hi < 50]['id'].count()>0):
        data.loc[(data['ap_hi'] < 50),'ap_hi'] = data[data.ap_hi < 50]['ap_hi'] * 10

    while (data[data.ap_hi > 250]['id'].count()>0):
        data.loc[(data['ap_hi'] > 250),'ap_hi'] = data[data.ap_hi > 250]['ap_hi'] // 10
    
    data['ap_lo'] = data['ap_lo'].fillna(round(data[(data['age'] >= data.age)]['ap_lo'].mean()))
    data['ap_hi'] = data['ap_hi'].fillna(round(data[(data['age'] >= data.age)]['ap_hi'].mean()))
    
    if(data[(data['ap_lo'] == 0)]['ap_lo'].count()>0):
        print("ap_lo execute 0 s2")
    if(data[(data['ap_hi'] == 0)]['ap_hi'].count()>0):
        print("ap_hi execute 0 s2")


    null_idx = data['ap_hi'] <= data['ap_lo']
    
    data.loc[null_idx,['ap_hi','ap_lo']] = data.loc[null_idx,['ap_lo','ap_hi']].values

    data_bmi = pd.DataFrame(
        {'bmi': round(((data['weight']) / ((data['height'] / 100)**2)), 2),
        'weight_o': 50 +  round(0.75 *(data['height'] - 150), 2) + round((round(data['age']/365, 2) - 20) / 4, 2),#отимальный вес
        'weight_nfg_o': 45 +  round((data['height'] - 152.4) / 2.45 * 0.9, 2) + round(((round(data['age']/365, 2) - 20) / 4), 2)#отимальный вес формула Наглера начало расчета
        }
        )
    data_bmi['weight_nfg_o'] = data_bmi['weight_nfg_o'] * 1.1#отимальный вес формула Наглера завершение расчета
    data_bmi['weight_nfg_o_с'] = data['weight'] - data_bmi['weight_nfg_o']#отимальный вес формула Наглера отклонение
    data_bmi['weight_o_c'] = data['weight'] - data_bmi['weight_o']#отимальный вес отклонение

    data_bmi['weight_o'] = data_bmi['weight_o'].fillna(0)
    data_bmi['weight_nfg_o'] = data_bmi['weight_nfg_o'].fillna(0)
    data_bmi['weight_nfg_o_с'] = data_bmi['weight_nfg_o_с'].fillna(0)
    data_bmi['weight_o_c'] = data_bmi['weight_o_c'].fillna(0)

    #weight_o,weight_nfg_o,weight_nfg_o_с,weight_o_c
    #data_ap_hi_n = pd.DataFrame({'ap_hi_n': round(
    #    109 + (0.5 * round(data['age'] / 365, 2)) + (0.1 * data['weight']), 0)})
    #data_ap_lo_n = pd.DataFrame({'ap_lo_n': round(
        #64 + (0.1 * round(data['age'] / 365, 2)) + (0.15 * data['weight']), 0)})
    data_ap_hi_n = pd.DataFrame({'ap_hi_n': round(
        109 + (0.5 * round(data['age']/365, 2)) + (0.1 * data['weight']), 0)})
    data_ap_lo_n = pd.DataFrame({'ap_lo_n': round(
        64 + (0.1 * round(data['age']/365, 2)) + (0.15 * data['weight']), 0)})

    data = pd.concat((data, data_bmi, data_ap_hi_n, data_ap_lo_n), axis=1)

    data['ap_hi_n'] = data['ap_hi_n'].fillna(0)
    data['ap_lo_n'] = data['ap_lo_n'].fillna(0)

    #data['ap_hi_n'] = np.sqrt((data['ap_hi'] - data['ap_hi_n'])**2)
    #data['ap_lo_n'] = np.sqrt((data['ap_lo'] - data['ap_lo_n'])**2)
    data['ap_hi_n'] = data['ap_hi'] - data['ap_hi_n']
    data['ap_lo_n'] = data['ap_lo'] - data['ap_lo_n']

    data.at[(abs(data['ap_hi_n']) >= 20 ), 'ap_hi_c'] = 1.0
    data.at[(abs(data['ap_lo_n']) >= 10 ), 'ap_lo_c'] = 1.0

    data['ap_hi_c'] = data['ap_hi_c'].fillna(0)
    data['ap_lo_c'] = data['ap_lo_c'].fillna(0)

    data.at[(data['bmi'] <= 16 ), 'bmi_n_1'] = 1 	#Выраженный дефицит массы тела
    data.at[(data['bmi'] <= 18.5 ) & (data['bmi'] > 16 ), 'bmi_n_2'] = 1 	#Недостаточная (дефицит) масса тела
    data.at[(data['bmi'] >= 25 ) & (data['bmi'] < 30 ), 'bmi_n_4'] = 1 	#Избыточная масса тела (предожирение)
    data.at[(data['bmi'] >= 30 ) & (data['bmi'] < 35 ), 'bmi_n_5'] = 1 	#Ожирение первой степени
    data.at[(data['bmi'] >= 35 ) & (data['bmi'] < 40 ), 'bmi_n_6'] = 1 	#Ожирение второй степени
    data.at[(data['bmi'] >= 40 ), 'bmi_n_7'] = 1 	#Ожирение третьей степени (морбидное)

    data['bmi_n_1'] = data['bmi_n_1'].fillna(0)
    data['bmi_n_2'] = data['bmi_n_2'].fillna(0)
    data['bmi_n_4'] = data['bmi_n_4'].fillna(0)
    data['bmi_n_5'] = data['bmi_n_5'].fillna(0)
    data['bmi_n_6'] = data['bmi_n_6'].fillna(0)
    data['bmi_n_7'] = data['bmi_n_7'].fillna(0)
    #bmi
    data.at[(data['bmi'] < 18.5 ), 'bmi_r_1'] = 1 	#Risk of developing problems such as nutritional deficiency and osteoporosis	under 18.5
    data.at[(data['bmi'] >= 23) & (data['bmi'] < 27.5 ), 'bmi_r_3'] = 1 	    #Moderate risk of developing heart disease, high blood pressure, stroke, diabetes	23 to 27.5
    data.at[(data['bmi'] >= 27.5 ), 'bmi_r_4'] = 1 	#High risk of developing heart disease, high blood pressure, stroke, diabetes	over 27.5

    data['bmi_r_1'] = data['bmi_r_1'].fillna(0)
    data['bmi_r_3'] = data['bmi_r_3'].fillna(0)
    data['bmi_r_4'] = data['bmi_r_4'].fillna(0)
    data['bmi'] = data['bmi'].fillna(0)

    #cholesterol
    data.at[(data['cholesterol'] == 1 ), 'cholesterol_1'] = 1 	#High risk of developing heart disease, high blood pressure, stroke, diabetes	over 27.5
    data.at[(data['cholesterol'] == 2 ), 'cholesterol_2'] = 1 	#High risk of developing heart disease, high blood pressure, stroke, diabetes	over 27.5
    data.at[(data['cholesterol'] == 3 ), 'cholesterol_3'] = 1 	#High risk of developing heart disease, high blood pressure, stroke, diabetes	over 27.5
    data['cholesterol_1'] = data['cholesterol_1'].fillna(0)
    data['cholesterol_2'] = data['cholesterol_2'].fillna(0)
    data['cholesterol_3'] = data['cholesterol_3'].fillna(0)
    #gluc
    data.at[(data['gluc'] == 1 ), 'gluc_1'] = 1
    data.at[(data['gluc'] == 2 ), 'gluc_2'] = 1
    data.at[(data['gluc'] == 3 ), 'gluc_3'] = 1
    data['gluc_1'] = data['gluc_1'].fillna(0)
    data['gluc_2'] = data['gluc_2'].fillna(0)
    data['gluc_3'] = data['gluc_3'].fillna(0)

    data['risk']=0.0

    data.loc[(data['smoke']==1),'risk'] = data[(data.smoke==1)]['risk'].astype(float)+0.2
    idx_smoke = (data['smoke']==1) & ((data['age']//365-25)>0)
    data.loc[idx_smoke,'risk'] = data[idx_smoke]['risk'].astype(float)+((data[idx_smoke]['age']//365-25)//5).astype(float)*0.06

    data.loc[(data['alco']==1),'risk']=data[(data.alco==1)]['risk'].astype(float)+0.1
    idx_alko = (data['alco']==1) & ((data['age']//365-20)>0)
    data.loc[idx_alko,'risk'] = data[idx_alko]['risk'].astype(float)+((data[idx_alko]['age']//365-20)//5).astype(float)*0.03

    data.loc[(data['bmi_r_1']==1),'risk']=data[(data.bmi_r_1==1)]['risk'].astype(float)+0.2
    data.loc[(data['bmi_r_3']==1),'risk']=data[(data.bmi_r_3==1)]['risk'].astype(float)+0.4
    data.loc[(data['bmi_r_4']==1),'risk']=data[(data.bmi_r_4==1)]['risk'].astype(float)+0.7

        
    idx_ad = (data['cholesterol']==1)
    data.loc[idx_ad,'risk'] = data.loc[idx_ad,'risk'].astype(float)+0.1
    idx_ad = (data['cholesterol']==2)
    data.loc[idx_ad,'risk'] = data.loc[idx_ad,'risk'].astype(float)+0.15
    idx_ad = (data['cholesterol']==3)
    data.loc[idx_ad,'risk'] = data.loc[idx_ad,'risk'].astype(float)+0.2
    idx_ad = (data['gluc']==1)
    data.loc[idx_ad,'risk'] = data.loc[idx_ad,'risk'].astype(float)+0.1
    idx_ad = (data['gluc']==2)
    data.loc[idx_ad,'risk'] = data.loc[idx_ad,'risk'].astype(float)+0.15
    idx_ad = (data['gluc']==3)
    data.loc[idx_ad,'risk'] = data.loc[idx_ad,'risk'].astype(float)+0.2

    #Начиная с АД 115/75 мм рт. ст. с возрастанием АД на каждые 20/10 мм рт. ст. риск сердечно-сосудистых заболеваний увеличивается.
    data['ad_risk']=0.0
    idx_ad = (data['ap_hi']>140) & ((data['age']//365)>50)
    data.loc[idx_ad,'ad_risk']=data[idx_ad]['ad_risk'].astype(float)+0.5
    idx_ad = (data['ap_hi']>115)
    data.loc[idx_ad,'ad_risk']=((data[idx_ad]['ap_hi']-115)//20).astype(float)*0.1
    idx_ad = (data['ap_lo']>75)
    data.loc[idx_ad,'ad_risk']=((data[idx_ad]['ap_lo']-75)//10).astype(float)*0.1

    '''
    1 степень – давления свыше 140–159/90–99 мм рт. ст.;
    2 степень – 160-179/100–109 мм рт. ст.;
    3 степень – 180/100 мм рт. ст.
    '''
    data['gyperton']=0
    idx_ad = (data['ap_hi']>140) & (data['ap_lo']>90)
    data.loc[idx_ad,'gyperton']=1 #
    idx_ad = (data['ap_hi']>160) & (data['ap_lo']>100)
    data.loc[idx_ad,'gyperton']=2 #
    idx_ad = (data['ap_hi']>180) & (data['ap_lo']>100)
    data.loc[idx_ad,'gyperton']=3 #

    data_n = pd.DataFrame(StandardScaler().fit_transform(data[['age', 'ap_lo', 'ap_hi','height', 'weight', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c']]))
    data[['age', 'ap_lo', 'ap_hi', 'height', 'weight', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n', 'weight_o', 'weight_nfg_o', 'weight_nfg_o_с', 'weight_o_c']] = data_n
    
    ''' data_n = data[['age', 'ap_lo', 'ap_hi','height', 'weight', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c']]
    data_n = (data_n - data_n.mean()) / data_n.std()
    data[['age', 'ap_lo', 'ap_hi', 'height', 'weight', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n', 'weight_o', 'weight_nfg_o', 'weight_nfg_o_с', 'weight_o_c']] = data_n
    '''
    return data