from processing_data import LoadFile
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

data_p = LoadFile("ml5/test.csv")

'''data_n = pd.DataFrame(StandardScaler().fit_transform(data_p[['age', 'height', 'weight', 'ap_hi', 'ap_lo',
               'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c']]))
data_p[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n', 'weight_o', 'weight_nfg_o', 'weight_nfg_o_с', 'weight_o_c']
     ] = data_n'''

clf_smoke = joblib.load('training_models/smoke.pkl')
clf_alco = joblib.load('training_models/alko.pkl')
clf_cardio = joblib.load('training_models/cardio.pkl')
'''
    null_idx = data['ap_hi'] <= data['ap_lo']
    
    data.loc[null_idx,['ap_hi','ap_lo']] = data.loc[null_idx,['ap_lo','ap_hi']].values

'''
#Заполняем предсказанными данными 'alco'
data_na = data_p.loc[:]
#data_na_t = data_p.drop(['id','smoke','alco','active'], axis=1)
data_na_t = data_p.drop(["id", "smoke", "alco", "active","gluc_3" ,"bmi_n_4" ,"gluc_1" ,
    "gluc_2" ,"bmi_n_2" ,"bmi_r_4" ,"bmi_n_1" ,"bmi_r_1" ,"bmi_n_7" ,"bmi_n_6" ,
    "bmi_n_5" ,"ap_lo_c" ,"bmi_r_3"], axis=1)
#data_na_t = data_na_t.fillna(0, axis=0)
#data_na_t = data_na_t.fillna(0, axis=0)
data_na['p_smoke'] = clf_smoke.predict(data_na_t)
#Заполняем предсказанными данными 'smoke'
data_na['p_alko'] = clf_alco.predict(data_na_t)

data_na.loc[(pd.isnull(data_na['smoke'])),'smoke'] = data_na.loc[(pd.isnull(data_na['smoke'])),'p_smoke']
data_na.loc[(pd.isnull(data_na['alco'])),'alco'] = data_na.loc[(pd.isnull(data_na['alco'])),'p_alko']

data_p['alco'] = data_na.loc[:,'alco']
data_p['smoke'] = data_na.loc[:,'smoke'] 

data_p['risk']=0
data_p.loc[(data_p['smoke']==1),'risk']=data_p[(data_p.smoke==1)]['risk']+0.2
data_p.loc[(data_p['alco']==1),'risk']=data_p[(data_p.alco==1)]['risk']+0.1
data_p.loc[(data_p['bmi_r_1']==1),'risk']=data_p[(data_p.bmi_r_1==1)]['risk']+0.2
data_p.loc[(data_p['bmi_r_3']==1),'risk']=data_p[(data_p.bmi_r_3==1)]['risk']+0.4
data_p.loc[(data_p['bmi_r_4']==1),'risk']=data_p[(data_p.bmi_r_4==1)]['risk']+0.7

data_p = data_p.drop(['id','p_smoke','p_alko','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c','alco','bmi_r_4','bmi_n_7','bmi_r_1','bmi_n_2','bmi_n_1'], axis=1)  # Выбрасываем столбец 'class'.

#X = data_p[:]  # Выбрасываем столбец 'class'.

# Предсказание курения
print("start gbt Y_smoke")
#abc = ensemble.AdaBoostClassifier(n_estimators=300, random_state=264)
#gbt = ensemble.GradientBoostingClassifier(n_estimators=300, random_state=264)

data_p_s1 = pd.DataFrame(clf_cardio.predict_proba(data_p))
data_p_s2 = pd.DataFrame(clf_cardio.predict_log_proba(data_p))
data_p_s1 = data_p_s1.drop([0], axis=1)
data_p_s2 = data_p_s2.drop([0], axis=1)

data_p = pd.concat((data_p, data_p_s1, data_p_s2), axis=1)

data_p_s1.to_csv("result/test_predict1.csv", sep=';', index=False)
data_p.to_csv("result/test_predict.csv", sep=';', index=False)
print("File is save")