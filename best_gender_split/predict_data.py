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

clf_cardio0 = joblib.load('training_models/cardio_0.pkl')
clf_cardio1 = joblib.load('training_models/cardio_1.pkl')
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

data_p = data_p.drop(['id','p_smoke','p_alko','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c','alco','bmi_r_4','bmi_n_7','bmi_r_1','bmi_n_2','bmi_n_1'], axis=1)  # Выбрасываем столбец 'class'.

#X = data_p[:]  # Выбрасываем столбец 'class'.

# Предсказание курения
print("start gbt Y_smoke")
#abc = ensemble.AdaBoostClassifier(n_estimators=300, random_state=264)
#gbt = ensemble.GradientBoostingClassifier(n_estimators=300, random_state=264)
idx0 = data_p['gender'] == 0
idx1 = data_p['gender'] == 1

X0 = data_p.drop('gender', axis=1)
X1 = data_p.drop('gender', axis=1)

X0 = pd.DataFrame(clf_cardio0.predict_proba(X0))
X1 = pd.DataFrame(clf_cardio1.predict_proba(X1))

X0.columns = ['X0_0', 'X0_1']
X1.columns = ['X1_0', 'X1_1']

data_p = pd.concat((data_p, X0, X1), axis=1)
data_p.loc[idx0,'predict'] = X0.loc[idx0,'X0_1']
data_p.loc[idx1,'predict'] = X1.loc[idx1,'X1_1']

'''print (X0.describe())
print (X1.describe())
print (data_p.describe())'''

'''data_p_s1 = pd.DataFrame(clf_cardio.predict_proba(data_p))
data_p_s2 = pd.DataFrame(clf_cardio.predict_log_proba(data_p))
data_p_s1 = data_p_s1.drop([0], axis=1)
data_p_s2 = data_p_s2.drop([0], axis=1)

data_p = pd.concat((data_p, data_p_s1, data_p_s2), axis=1)'''

data_p['predict'].to_csv("result/test_predict1.csv", sep=';', index=False)
data_p.to_csv("result/test_predict.csv", sep=';', index=False)
print("File is save")