from processing_data import LoadFile
from sklearn.externals import joblib

import numpy as np
import pandas as pd

data_p = LoadFile("ml5/test.csv")

clf_smoke = joblib.load('training_models/smoke.pkl')
clf_alco = joblib.load('training_models/alko.pkl')
clf_cardio = joblib.load('training_models/cardio.pkl')
'''
    null_idx = data['ap_hi'] <= data['ap_lo']
    
    data.loc[null_idx,['ap_hi','ap_lo']] = data.loc[null_idx,['ap_lo','ap_hi']].values

'''
#Заполняем предсказанными данными 'alco'
data_na = data_p.loc[:]
data_na_t = data_p.drop(['id','smoke','alco','active'], axis=1)
data_na_t = data_na_t.fillna(0, axis=0)
#data_na_t = data_na_t.fillna(0, axis=0)
data_na['p_smoke'] = clf_smoke.predict(data_na_t)
#Заполняем предсказанными данными 'smoke'
data_na['p_alko'] = clf_alco.predict(data_na_t)

data_na.loc[(pd.isnull(data_na['smoke'])),'smoke'] = data_na.loc[(pd.isnull(data_na['smoke'])),'p_smoke']
data_na.loc[(pd.isnull(data_na['alco'])),'alco'] = data_na.loc[(pd.isnull(data_na['alco'])),'p_alko']

data_p['alco'] = data_na.loc[:,'alco']
data_p['smoke'] = data_na.loc[:,'smoke'] 

data_p = data_p.drop(['id','p_smoke','p_alko'], axis=1)  # Выбрасываем столбец 'class'.

data_p_s1 = pd.DataFrame(clf_cardio.predict_proba(data_p))
data_p_s2 = pd.DataFrame(clf_cardio.predict_log_proba(data_p))
data_p_s1 = data_p_s1.drop([0], axis=1)
data_p_s2 = data_p_s2.drop([0], axis=1)

data_p = pd.concat((data_p, data_p_s1, data_p_s2), axis=1)

data_p_s1.to_csv("f:/PY_Project/ML5/ml5/result/test_predict1.csv", sep=';', index=False)
data_p.to_csv("f:/PY_Project/ML5/ml5/result/test_predict.csv", sep=';', index=False)
print("File is save")