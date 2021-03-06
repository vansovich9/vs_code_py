import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from processing_data import LoadFile

data = pd.DataFrame(LoadFile("ml5/train.csv"))
# Нормализация данных
'''
data_n = data[['age', 'height', 'weight', 'ap_hi', 'ap_lo',
               'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n']]

data_n = (data_n - data_n.mean()) / data_n.std()

data[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n']
     ] = data_n[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n']]
'''
# age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio
# Предсказание курение алкоголь активность
#,'weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c'

'''
data_n = pd.DataFrame(StandardScaler().fit_transform(data[['age', 'height', 'weight', 'ap_hi', 'ap_lo',
               'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c']]))
data[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n', 'weight_o', 'weight_nfg_o', 'weight_nfg_o_с', 'weight_o_c']
     ] = data_n

X_train.loc[(X_train['predict']>0.1) & (X_train['smoke']==1),'predict']=X_train[(X_train.predict>0.1) & (X_train.smoke==1)]['predict']+0.2
X_test.loc[(X_test['predict']>0.1) & (X_test['smoke']==1),'predict']=X_test[(X_test.predict>0.1) & (X_test.smoke==1)]['predict']+0.2

X_train.loc[(X_train['predict']>0.1) & (X_train['alco']==1),'predict']=X_train[(X_train.predict>0.1) & (X_train.alco==1)]['predict']+0.2
X_test.loc[(X_test['predict']>0.1) & (X_test['alco']==1),'predict']=X_test[(X_test.predict>0.1) & (X_test.alco==1)]['predict']+0.2

X_train.loc[(X_train['predict']>0.1) & (X_train['bmi_r_1']==1),'predict']=X_train[(X_train.predict>0.1) & (X_train.bmi_r_1==1)]['predict']+0.2
X_test.loc[(X_test['predict']>0.1) & (X_test['bmi_r_1']==1),'predict']=X_test[(X_test.predict>0.1) & (X_test.bmi_r_1==1)]['predict']+0.2

X_train.loc[(X_train['predict']>0.1) & (X_train['bmi_r_3']==1),'predict']=X_train[(X_train.predict>0.1) & (X_train.bmi_r_3==1)]['predict']+0.3
X_test.loc[(X_test['predict']>0.1) & (X_test['bmi_r_3']==1),'predict']=X_test[(X_test.predict>0.1) & (X_test.bmi_r_3==1)]['predict']+0.3

X_train.loc[(X_train['predict']>0.1) & (X_train['bmi_r_4']==1),'predict']=X_train[(X_train.predict>0.1) & (X_train.bmi_r_4==1)]['predict']+0.4
X_test.loc[(X_test['predict']>0.1) & (X_test['bmi_r_4']==1),'predict']=X_test[(X_test.predict>0.1) & (X_test.bmi_r_4==1)]['predict']+0.4

0.253346938776 0.265761904762  X = data.drop(['cardio','id','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c','alco'], axis=1)  # Выбрасываем столбец 'class'.
'''


#print(X.columns)

# Предсказание курения
print("start gbt")
#abc = ensemble.AdaBoostClassifier(n_estimators=300, random_state=264)
#gbt = ensemble.GradientBoostingClassifier(n_estimators=300, random_state=264)
'''
0.240673469388 0.267523809524 err_sum 0.248728571429(n_estimators=300, random_state=264,learning_rate = 0.3)
0.254408163265 0.265238095238 err_sum 0.257657142857(n_estimators=300, random_state=264,learning_rate = 0.1)
0.0591836734694 0.282428571429 err_sum 0.126157142857(n_estimators=300, random_state=264,max_depth = 10)
0.253979591837 0.26480952381 err_sum 0.257228571429(n_estimators=300, random_state=264,min_samples_leaf = 3)
0.25387755102 0.264761904762 err_sum 0.257142857143(n_estimators=300, random_state=264,min_samples_leaf = 5)
0.254265306122 0.264428571429 err_sum 0.257314285714(n_estimators=300, random_state=264,min_samples_leaf = 5, subsample = 0.5)
start to nite
,criterion = 'mae'
'''
X = data[(data.gender==0)].drop(['cardio','id','gender','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c','alco','bmi_r_4','bmi_n_7','bmi_r_1','bmi_n_2','bmi_n_1'], axis=1)  # Выбрасываем столбец 'class'.
Y = data[(data.gender==0)]['cardio']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=11)
'''    
#gbt = ensemble.GradientBoostingClassifier(n_estimators=55, random_state=264,min_samples_leaf = 5, subsample = 0.5, verbose=0)
gender 0 0.257255812494 0.263050003661 err_sum 0.258994069844 gbt = MLPClassifier(alpha=0.0, random_state = 0)
gender 0 0.261052367356 0.262025038436 err_sum 0.26134416868 gbt = MLPClassifier(alpha=0.0, random_state = 0, hidden_layer_sizes = 50, verbose = 1)
'''
gbt = MLPClassifier(alpha=0.0, random_state = 0, activation = 'relu', hidden_layer_sizes=(50,), verbose = 0)
clf4 = gbt.fit(X_train, Y_train)

err_train = np.mean(Y_train != gbt.predict(X_train))
err_test = np.mean(Y_test != gbt.predict(X_test))
err_sum = np.mean(Y != gbt.predict(X))
joblib.dump(gbt, "training_models/cardio_0.pkl", compress=1)

print("gender 0",err_train, err_test, 'err_sum', err_sum)
print("gbt score %s" % clf4.score(X_train, Y_train))

X = data[(data.gender==1)].drop(['cardio','id','gender','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c','alco','bmi_r_4','bmi_n_7','bmi_r_1','bmi_n_2','bmi_n_1'], axis=1)  # Выбрасываем столбец 'class'.
Y = data[(data.gender==1)]['cardio']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=11)
#gbt = GradientBoostingClassifier(n_estimators=40, random_state=264,min_samples_leaf = 6, subsample = 0.1, verbose=0)
#gender 1 0.265164341176 0.263860509467 err_sum 0.264773191663
gbt = MLPClassifier(alpha=0.0, random_state = 0, activation = 'relu', hidden_layer_sizes=(120,), verbose = 0)
clf4 = gbt.fit(X_train, Y_train)

err_train = np.mean(Y_train != gbt.predict(X_train))
err_test = np.mean(Y_test != gbt.predict(X_test))
err_sum = np.mean(Y != gbt.predict(X))
joblib.dump(gbt, "training_models/cardio_1.pkl", compress=1)

print("gender 1",err_train, err_test, 'err_sum', err_sum)
print("gbt score %s" % clf4.score(X_train, Y_train))


'''print()
feature_names = X.columns
importances = gbt.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importances:")
for f, idx in enumerate(indices):
    print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))'''
#0.265285714286 0.265238095238 0.265238095238 
#0.254959183673 0.265714285714 0.254408163265 0.265238095238