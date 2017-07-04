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

X = data_p.drop(['id', "smoke", "alco",'p_smoke','p_alko','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c','alco','bmi_r_4','bmi_n_7','bmi_r_1','bmi_n_2','bmi_n_1'], axis=1)  # Выбрасываем столбец 'class'.
#X = data_p[:]  # Выбрасываем столбец 'class'.

# Предсказание курения
print("start gbt Y_smoke")
#abc = ensemble.AdaBoostClassifier(n_estimators=300, random_state=264)
#gbt = ensemble.GradientBoostingClassifier(n_estimators=300, random_state=264)

data_p_s = pd.DataFrame(clf_cardio.predict_log_proba(X))
X = pd.DataFrame(clf_cardio.predict_proba(X))

X.columns = ['X_0', 'X_1']

data_p = pd.concat((data_p, X), axis=1)
data_p.loc[:,'predict'] = X.loc[:,'X_1']

'''idx = (data_p['smoke']==1)
data_p.loc[idx,'predict'] = data_p.loc[idx,'predict'].astype(float) + 0.2

idx = (data_p['alco']==1)
data_p.loc[idx,'predict'] = data_p.loc[idx,'predict'].astype(float) + 0.1'''

'''print (X0.describe())
print (X1.describe())
print (data_p.describe())'''

data_p_s.columns = ['plp_0', 'plp_1']

data_p = pd.concat((data_p, data_p_s), axis=1)

data_p['predict'].to_csv("result/test_predict1.csv", sep=';', index=False)
data_p.to_csv("result/test_predict.csv", sep=';', index=False)
print("File is save")