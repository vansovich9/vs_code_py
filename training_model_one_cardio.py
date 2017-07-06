import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
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
X = data.drop(['cardio', 'smoke', 'alco', 'active','id','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c','alco','bmi_r_4','bmi_n_7','bmi_r_1','bmi_n_2','bmi_n_1'], axis=1)  # Выбрасываем столбец 'class'.
Y = data['cardio']
#print(X.describe())

'''    
#gbt = ensemble.GradientBoostingClassifier(n_estimators=55, random_state=264,min_samples_leaf = 5, subsample = 0.5, verbose=0)
gender 0 0.257255812494 0.263050003661 err_sum 0.258994069844 gbt = MLPClassifier(alpha=0.0, random_state = 0)
gender 0 0.261052367356 0.262025038436 err_sum 0.26134416868 gbt = MLPClassifier(alpha=0.0, random_state = 0, hidden_layer_sizes = 50, verbose = 1)
'''
best_err = 10
best_rnd = 0
runs = 40
data_predict = pd.DataFrame(data[['cardio', 'smoke', 'alco', 'active']])
for i in range(runs):
    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=i)
    gbt = MLPClassifier(alpha=0.0, random_state = 5, activation = 'relu', hidden_layer_sizes=(50,), verbose = 0)
    clf4 = gbt.fit(X_train, Y_train)
    err_train = np.mean(Y_train != gbt.predict(X_train))
    err_test = np.mean(Y_test != gbt.predict(X_test))
    err_sum = np.mean(Y != gbt.predict(X))
    data_predict["MLP_"+str(i)] = pd.DataFrame(gbt.predict_proba(X)).drop(0,axis=1)
    #joblib.dump(gbt, "training_models/cardio.pkl", compress=1)
    if(err_test<best_err):
        best_err=err_test
        best_rnd = i
    print("random_state = ", i, err_train, err_test, 'err_sum', err_sum)
    #print("gbt score %s" % clf4.score(X_train, Y_train))
print("Best MLP rnd", best_rnd, "err", best_err)
best_err = 10
best_rnd = 0
for i in range(runs):
    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=i)
    gbt = GradientBoostingClassifier(n_estimators=55, random_state=264,min_samples_leaf = 5, subsample = 0.5, verbose=0)
    clf4 = gbt.fit(X_train, Y_train)
    err_train = np.mean(Y_train != gbt.predict(X_train))
    err_test = np.mean(Y_test != gbt.predict(X_test))
    err_sum = np.mean(Y != gbt.predict(X))
    data_predict["GBT_"+str(i)] = pd.DataFrame(gbt.predict_proba(X)).drop(0,axis=1)
    #joblib.dump(gbt, "training_models/cardio.pkl", compress=1)
    if(err_test<best_err):
        best_err=err_test
        best_rnd = i
    print("random_state = ", i, err_train, err_test, 'err_sum', err_sum)
print("Best GBT rnd", best_rnd, "err", best_err)
best_err = 10
best_rnd = 0
for i in range(runs):
    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=i)
    gbt = AdaBoostClassifier(n_estimators=300, random_state=264)
    clf4 = gbt.fit(X_train, Y_train)
    err_train = np.mean(Y_train != gbt.predict(X_train))
    err_test = np.mean(Y_test != gbt.predict(X_test))
    err_sum = np.mean(Y != gbt.predict(X))
    data_predict["ABT_"+str(i)] = pd.DataFrame(gbt.predict_proba(X)).drop(0,axis=1)
    #joblib.dump(gbt, "training_models/cardio.pkl", compress=1)
    if(err_test<best_err):
        best_err=err_test
        best_rnd = i
    print("random_state = ", i, err_train, err_test, 'err_sum', err_sum)
print("Best ABC rnd", best_rnd, "err", best_err)
#
best_err = 10
best_rnd = 0
for i in range(runs):
    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=i)
    gbt = RandomForestClassifier(max_depth=10, n_estimators=700, max_features=1, random_state=i)
    clf4 = gbt.fit(X_train, Y_train)
    err_train = np.mean(Y_train != gbt.predict(X_train))
    err_test = np.mean(Y_test != gbt.predict(X_test))
    err_sum = np.mean(Y != gbt.predict(X))
    data_predict["RFC_"+str(i)] = pd.DataFrame(gbt.predict_proba(X)).drop(0,axis=1)
    #joblib.dump(gbt, "training_models/cardio.pkl", compress=1)
    if(err_test<best_err):
        best_err=err_test
        best_rnd = i
    print("random_state = ", i, err_train, err_test, 'err_sum', err_sum)
print("Best RFC rnd", best_rnd, "err", best_err)

data_predict = data_predict.drop(['cardio'],axis=1)
best_err = 10
best_rnd = 0
best_rnd_model = 0
best_layer = 0
for j_m in range(runs):    
    X_train, X_test, Y_train, Y_test = train_test_split(
        data_predict, Y, test_size=0.3, random_state=j_m)
    for j in range(10, 200, 10):
        gbt = MLPClassifier(alpha=0.0, random_state=j_m,
            activation='relu', hidden_layer_sizes=(j,), verbose=0)
        clf4 = gbt.fit(X_train, Y_train)
        err_train = np.mean(Y_train != gbt.predict(X_train))
        err_test = np.mean(Y_test != gbt.predict(X_test))
        err_sum = np.mean(Y != gbt.predict(data_predict))
        if(err_test<best_err):
            best_err=err_test
            best_rnd = j_m
            best_layer = j
            joblib.dump(gbt, "training_models/cardio_"+str(best_rnd)+"_"+str(best_layer)+".pkl", compress=1)
        print("random_state =", j_m, err_train, err_test, 'err_sum', err_sum, "layer", j)
print("Best final rnd", best_rnd, "err", best_err, "best layer", best_layer)

'''
whith smoke and alco
Best MLP rnd 33 err 0.259238095238
Best GBT rnd 20 err 0.259619047619
Best ABC rnd 20 err 0.260714285714
Best RFC rnd 20 err 0.265142857
whithout smoke and alco
Best MLP rnd 14 err 0.25819047619
Best GBT rnd 20 err 0.258476190476
Best ABC rnd 20 err 0.26219047619
Best RFC rnd 20 err 0.263619047619
Best final rnd 20 err 0.253761904762 best layer 70
'''


'''print()
feature_names = X.columns
importances = gbt.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importances:")
for f, idx in enumerate(indices):
    print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))'''
#0.265285714286 0.265238095238 0.265238095238 
#0.254959183673 0.265714285714 0.254408163265 0.265238095238
#0.260326530612 0.255714285714 e

