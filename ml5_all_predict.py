import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


data = pd.read_csv("f:/PY_Project/ML5/ml5/train.csv", header=0,
                   sep=';', na_values=['None'])
data['ap_hi'] = abs(data['ap_hi'])
data['ap_lo'] = abs(data['ap_lo'])

data = data.fillna(0, axis=0)

data.at[(data['ap_lo'] == 0),'ap_lo'] = data['ap_lo'].mean()
data.at[(data['ap_hi'] == 0),'ap_hi'] = data['ap_hi'].mean()

data.at[(data['ap_lo'] > 200),'ap_lo'] = data[data.ap_lo > 200]['ap_lo'] // 10
data.at[(data['ap_lo'] > 250),'ap_lo'] = data[data.ap_lo > 200]['ap_lo'] // 10

data.at[(data['ap_hi'] > 250),'ap_hi'] = data[data.ap_lo > 250]['ap_hi'] // 10
data.at[(data['ap_hi'] > 250),'ap_hi'] = data[data.ap_lo > 250]['ap_hi'] // 10

while (data[data.ap_hi <= 20]['id'].count()>0):
    data.at[(data['ap_hi'] < 50),'ap_hi'] = data[data.ap_lo < 50]['ap_hi'] * 10
#data.at[(data['ap_hi'] < 50),'ap_hi'] = data[data.ap_lo < 50]['ap_hi'] * 10

while (data[data.ap_lo <= 20]['id'].count()>0):
    data.at[(data['ap_lo'] <= 20),'ap_lo'] = data[data.ap_lo <= 20]['ap_lo'] * 10
#data.at[(data['ap_lo'] <= 20),'ap_lo'] = data[data.ap_lo <= 20]['ap_lo'] * 10

data.at[(data.ap_lo > data.ap_hi),['ap_hi','ap_lo']] = data[data.ap_lo > data.ap_hi][['ap_lo','ap_hi']]

data_bmi = pd.DataFrame(
    {'bmi': round(((data['weight']) / ((data['height'] / 100)**2)), 2)})

data_ap_hi_n = pd.DataFrame({'ap_hi_n': round(
    109 + (0.5 * round(data['age'] / 365, 2)) + (0.1 * data['weight']), 0)})
data_ap_lo_n = pd.DataFrame({'ap_lo_n': round(
    64 + (0.1 * round(data['age'] / 365, 2)) + (0.15 * data['weight']), 0)})

data = pd.concat((data, data_bmi, data_ap_hi_n, data_ap_lo_n), axis=1)
data['ap_hi_n'] = np.sqrt((data['ap_hi'] - data['ap_hi_n'])**2)
data['ap_lo_n'] = np.sqrt((data['ap_lo'] - data['ap_lo_n'])**2)
data.at[(data['ap_hi_n'] >= 20 ), 'ap_hi_c'] = 1.0
#data.at[(data['ap_hi_n'] < 20 ), 'ap_hi_c'] = 0.0
data.at[(data['ap_lo_n'] >= 10 ), 'ap_lo_c'] = 1.0
#data.at[(data['ap_lo_n'] < 10 ), 'ap_lo_c'] = 0.0

data.at[(data['bmi'] <= 16 ), 'bmi_n_1'] = 1 	#Выраженный дефицит массы тела
data.at[(data['bmi'] <= 18.5 ), 'bmi_n_2'] = 1 	#Недостаточная (дефицит) масса тела
data.at[(data['bmi'] >= 25 ), 'bmi_n_4'] = 1 	#Избыточная масса тела (предожирение)
data.at[(data['bmi'] >= 30 ), 'bmi_n_5'] = 1 	#Ожирение первой степени
data.at[(data['bmi'] >= 35 ), 'bmi_n_6'] = 1 	#Ожирение второй степени
data.at[(data['bmi'] >= 40 ), 'bmi_n_7'] = 1 	#Ожирение третьей степени (морбидное)

data.at[(data['bmi'] < 18.5 ), 'bmi_r_1'] = 1 	#Risk of developing problems such as nutritional deficiency and osteoporosis	under 18.5
data.at[(data['bmi'] >= 23 ), 'bmi_r_3'] = 1 	    #Moderate risk of developing heart disease, high blood pressure, stroke, diabetes	23 to 27.5
data.at[(data['bmi'] >= 27.5 ), 'bmi_r_4'] = 1 	#High risk of developing heart disease, high blood pressure, stroke, diabetes	over 27.5
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

data = data.fillna(0, axis=0)
#age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio
X = data.drop(['cardio','id'], axis=1)  # Выбрасываем столбец 'class'.
Y = data['cardio']
#Предсказание курение алкоголь активность
X_saa = data.drop(['cardio','id','smoke','alco','active'], axis=1)  # Выбрасываем столбец 'class'.

Y_smoke = data['smoke']
Y_alco = data['alco']
Y_active = data['active']
Y_saa = data[['smoke','alco','active']]

from sklearn.model_selection import train_test_split
from sklearn import ensemble

X_train, X_test, Y_train, Y_test = train_test_split(X_saa, Y_smoke, test_size = 0.3, random_state = 11)
print("start Y_smoke")
gbt_smoke = ensemble.GradientBoostingClassifier(n_estimators=300, random_state=264, min_samples_leaf=300,min_samples_split=150)
clf4 = gbt_smoke.fit(X_train, Y_train)
err_train = np.mean(Y_train != gbt_smoke.predict(X_train))
err_test = np.mean(Y_test != gbt_smoke.predict(X_test))
print(err_train, err_test)
print('Y_smoke score %s' % clf4.score(X_train, Y_train))

print("start Y_alco")
gbt_alko = ensemble.GradientBoostingClassifier(n_estimators=300, random_state=264, min_samples_leaf=300,min_samples_split=150)
clf4 = gbt_alko.fit(X_train, Y_train)
err_train = np.mean(Y_train != gbt_alko.predict(X_train))
err_test = np.mean(Y_test != gbt_alko.predict(X_test))
print(err_train, err_test)
print('Y_alco score %s' % clf4.score(X_train, Y_train))

#Предсказание сердце
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 11)

N_train, _ = X_train.shape 
N_test,  _ = X_test.shape 
print(N_train, N_test)
print("start gbt")
gbt = ensemble.GradientBoostingClassifier(n_estimators=300, random_state=264, min_samples_leaf=300,min_samples_split=150)
clf4 = gbt.fit(X_train, Y_train)
err_train = np.mean(Y_train != gbt.predict(X_train))
err_test = np.mean(Y_test != gbt.predict(X_test))
print(err_train, err_test)
print('gbt score %s' % clf4.score(X_train, Y_train))
#RandomForestClassifier AdaBoostClassifier 
print("start RandomForestClassifier")
rfc = ensemble.RandomForestClassifier(n_estimators=300, random_state=264, min_samples_leaf=900,min_samples_split=150)
clf4 = rfc.fit(X_train, Y_train)
err_train = np.mean(Y_train != rfc.predict(X_train))
err_test = np.mean(Y_test != rfc.predict(X_test))
print(err_train, err_test)
print('RandomForestClassifier score %s' % clf4.score(X_train, Y_train))

print("start AdaBoostClassifier")
abc = ensemble.AdaBoostClassifier(n_estimators=300, random_state=264)
clf4 = abc.fit(X_train, Y_train)
err_train = np.mean(Y_train != abc.predict(X_train))
err_test = np.mean(Y_test != abc.predict(X_test))
print(err_train, err_test)
print('AdaBoostClassifier score %s' % clf4.score(X_train, Y_train))

#criterion="entropy"
print("start criterion=mse gbt")
gbt = ensemble.GradientBoostingClassifier(n_estimators=300, random_state=264, min_samples_leaf=300,min_samples_split=150, criterion="mse")
clf4 = gbt.fit(X_train, Y_train)
err_train = np.mean(Y_train != gbt.predict(X_train))
err_test = np.mean(Y_test != gbt.predict(X_test))
print(err_train, err_test)
print('gbt criterion=mse score %s' % clf4.score(X_train, Y_train))

print("start criterion=mae gbt")
gbt = ensemble.GradientBoostingClassifier(n_estimators=300, random_state=264, min_samples_leaf=150,min_samples_split=75, criterion="mae")
clf4 = gbt.fit(X_train, Y_train)
err_train = np.mean(Y_train != gbt.predict(X_train))
err_test = np.mean(Y_test != gbt.predict(X_test))
print(err_train, err_test)
print('gbt criterion=mae score %s' % clf4.score(X_train, Y_train))
#RandomForestClassifier AdaBoostClassifier 
print("start criterion=entropy RandomForestClassifier")
rfc = ensemble.RandomForestClassifier(n_estimators=300, random_state=264, min_samples_leaf=900,min_samples_split=150, criterion="entropy")
clf4 = rfc.fit(X_train, Y_train)
err_train = np.mean(Y_train != rfc.predict(X_train))
err_test = np.mean(Y_test != rfc.predict(X_test))
print(err_train, err_test)
print('RandomForestClassifier criterion=entropy score %s' % clf4.score(X_train, Y_train))

print("start criterion=entropy AdaBoostClassifier")
abc = ensemble.AdaBoostClassifier(n_estimators=300, random_state=264, criterion="entropy")
clf4 = abc.fit(X_train, Y_train)
err_train = np.mean(Y_train != abc.predict(X_train))
err_test = np.mean(Y_test != abc.predict(X_test))
print(err_train, err_test)
print('AdaBoostClassifier criterion=entropy score %s' % clf4.score(X_train, Y_train))
#end criterion="entropy"
print("start SVC")
from sklearn.svm import SVC
svc = SVC(kernel='rbf', random_state=264)
clf1 = svc.fit(X_train, Y_train)
err_train = np.mean(Y_train != svc.predict(X_train))
err_test  = np.mean(Y_test  != svc.predict(X_test))
print(err_train, err_test)
print('svc rbf score %s' % clf1.score(X_train, Y_train))
#
print("start dtc")
from sklearn import tree
dtc = tree.DecisionTreeClassifier(max_features="auto",random_state=264, min_samples_leaf=300,min_samples_split=150, max_depth=7000)
clf5 = dtc.fit(X_train, Y_train)
err_train = np.mean(Y_train != dtc.predict(X_train))
err_test = np.mean(Y_test != dtc.predict(X_test))
print(err_train, err_test)
print('dtc score %s' % clf5.score(X_train, Y_train))

##Predict

data_p = pd.read_csv("f:/PY_Project/ML5/ml5/test.csv", header=0,
                   sep=';', na_values=['None'])
data_p['ap_hi'] = abs(data_p['ap_hi'])
data_p['ap_lo'] = abs(data_p['ap_lo'])
#df[1].fillna(0, inplace=True)
data_p['ap_lo'] = data_p['ap_lo'].fillna(0)
data_p['ap_hi'] = data_p['ap_hi'].fillna(0)

#data_p = data_p.fillna(0, axis=0)

data_p.at[(data_p['ap_lo'] == 0),'ap_lo'] = data_p['ap_lo'].mean()
data_p.at[(data_p['ap_hi'] == 0),'ap_hi'] = data_p['ap_hi'].mean()

data_p.at[(data_p['ap_lo'] > 200),'ap_lo'] = data_p[data_p.ap_lo > 200]['ap_lo'] // 10
data_p.at[(data_p['ap_lo'] > 250),'ap_lo'] = data_p[data_p.ap_lo > 200]['ap_lo'] // 10

data_p.at[(data_p['ap_hi'] > 250),'ap_hi'] = data_p[data_p.ap_lo > 250]['ap_hi'] // 10
data_p.at[(data_p['ap_hi'] > 250),'ap_hi'] = data_p[data_p.ap_lo > 250]['ap_hi'] // 10

data_p.at[(data_p['ap_hi'] < 50),'ap_hi'] = data_p[data_p.ap_lo < 50]['ap_hi'] * 10
data_p.at[(data_p['ap_hi'] < 50),'ap_hi'] = data_p[data_p.ap_lo < 50]['ap_hi'] * 10
data_p.at[(data_p['ap_lo'] <= 20),'ap_lo'] = data_p[data_p.ap_lo <= 20]['ap_lo'] * 10
data_p.at[(data_p['ap_lo'] <= 20),'ap_lo'] = data_p[data_p.ap_lo <= 20]['ap_lo'] * 10

data_p.at[(data_p.ap_lo > data_p.ap_hi),['ap_hi','ap_lo']] = data_p[data_p.ap_lo > data_p.ap_hi][['ap_lo','ap_hi']]

data_p_bmi = pd.DataFrame(
    {'bmi': round(((data_p['weight']) / ((data_p['height'] / 100)**2)), 2)})

data_p_ap_hi_n = pd.DataFrame({'ap_hi_n': round(
    109 + (0.5 * round(data_p['age'] / 365, 2)) + (0.1 * data_p['weight']), 0)})
data_p_ap_lo_n = pd.DataFrame({'ap_lo_n': round(
    64 + (0.1 * round(data_p['age'] / 365, 2)) + (0.15 * data_p['weight']), 0)})

data_p = pd.concat((data_p, data_p_bmi, data_p_ap_hi_n, data_p_ap_lo_n), axis=1)

data_p['ap_hi_n'] = data_p['ap_hi_n'].fillna(0)
data_p['ap_lo_n'] = data_p['ap_lo_n'].fillna(0)

data_p['ap_hi_n'] = np.sqrt((data_p['ap_hi'] - data_p['ap_hi_n'])**2)
data_p['ap_lo_n'] = np.sqrt((data_p['ap_lo'] - data_p['ap_lo_n'])**2)
data_p.at[(data_p['ap_hi_n'] >= 20 ), 'ap_hi_c'] = 1.0
data_p.at[(data_p['ap_lo_n'] >= 10 ), 'ap_lo_c'] = 1.0

data_p['ap_hi_c'] = data_p['ap_hi_c'].fillna(0)
data_p['ap_lo_c'] = data_p['ap_lo_c'].fillna(0)

data_p.at[(data_p['bmi'] <= 16 ), 'bmi_n_1'] = 1 	#Выраженный дефицит массы тела
data_p.at[(data_p['bmi'] <= 18.5 ), 'bmi_n_2'] = 1 	#Недостаточная (дефицит) масса тела
data_p.at[(data_p['bmi'] >= 25 ), 'bmi_n_4'] = 1 	#Избыточная масса тела (предожирение)
data_p.at[(data_p['bmi'] >= 30 ), 'bmi_n_5'] = 1 	#Ожирение первой степени
data_p.at[(data_p['bmi'] >= 35 ), 'bmi_n_6'] = 1 	#Ожирение второй степени
data_p.at[(data_p['bmi'] >= 40 ), 'bmi_n_7'] = 1 	#Ожирение третьей степени (морбидное)

data_p['bmi_n_1'] = data_p['bmi_n_1'].fillna(0)
data_p['bmi_n_2'] = data_p['bmi_n_2'].fillna(0)
data_p['bmi_n_4'] = data_p['bmi_n_4'].fillna(0)
data_p['bmi_n_5'] = data_p['bmi_n_5'].fillna(0)
data_p['bmi_n_6'] = data_p['bmi_n_6'].fillna(0)
data_p['bmi_n_7'] = data_p['bmi_n_7'].fillna(0)
#bmi
data_p.at[(data_p['bmi'] < 18.5 ), 'bmi_r_1'] = 1 	#Risk of developing problems such as nutritional deficiency and osteoporosis	under 18.5
data_p.at[(data_p['bmi'] >= 23 ), 'bmi_r_3'] = 1 	    #Moderate risk of developing heart disease, high blood pressure, stroke, diabetes	23 to 27.5
data_p.at[(data_p['bmi'] >= 27.5 ), 'bmi_r_4'] = 1 	#High risk of developing heart disease, high blood pressure, stroke, diabetes	over 27.5

data_p['bmi_r_1'] = data_p['bmi_r_1'].fillna(0)
data_p['bmi_r_3'] = data_p['bmi_r_3'].fillna(0)
data_p['bmi'] = data_p['bmi'].fillna(0)

#cholesterol
data_p.at[(data_p['cholesterol'] == 1 ), 'cholesterol_1'] = 1 	#High risk of developing heart disease, high blood pressure, stroke, diabetes	over 27.5
data_p.at[(data_p['cholesterol'] == 2 ), 'cholesterol_2'] = 1 	#High risk of developing heart disease, high blood pressure, stroke, diabetes	over 27.5
data_p.at[(data_p['cholesterol'] == 3 ), 'cholesterol_3'] = 1 	#High risk of developing heart disease, high blood pressure, stroke, diabetes	over 27.5
data_p['cholesterol_1'] = data_p['cholesterol_1'].fillna(0)
data_p['cholesterol_2'] = data_p['cholesterol_2'].fillna(0)
data_p['cholesterol_3'] = data_p['cholesterol_3'].fillna(0)
#gluc
data_p.at[(data_p['gluc'] == 1 ), 'gluc_1'] = 1
data_p.at[(data_p['gluc'] == 2 ), 'gluc_2'] = 1
data_p.at[(data_p['gluc'] == 3 ), 'gluc_3'] = 1
data_p['gluc_1'] = data_p['gluc_1'].fillna(0)
data_p['gluc_2'] = data_p['gluc_2'].fillna(0)
data_p['gluc_3'] = data_p['gluc_3'].fillna(0)

#Заполняем предсказанными данными 'alco'
data_na = data_p.loc[:]
data_na_t = data_p.drop(['id','smoke','alco','active'], axis=1)
data_na_t = data_na_t.fillna(0, axis=0)
#data_na_t = data_na_t.fillna(0, axis=0)
data_na['p_smoke'] = gbt_smoke.predict(data_na_t)
#Заполняем предсказанными данными 'smoke'
data_na['p_alko'] = gbt_alko.predict(data_na_t)

data_na.loc[(pd.isnull(data_na['smoke'])),'smoke'] = data_na.loc[(pd.isnull(data_na['smoke'])),'p_smoke']
data_na.loc[(pd.isnull(data_na['alco'])),'alco'] = data_na.loc[(pd.isnull(data_na['alco'])),'p_alko']
#print('data_na',data_na[['smoke','alco']].describe())
data_p['alco'] = data_na.loc[:,'alco']
data_p['smoke'] = data_na.loc[:,'smoke']
#print('data_p',data_p[['smoke','alco']].describe())
data_p = data_p.fillna(0, axis=0)

data_p = data_p.drop(['id','p_smoke','p_alko'], axis=1)  # Выбрасываем столбец 'class'.

data_p_s1 = pd.DataFrame(gbt.predict_proba(data_p))
data_p_s2 = pd.DataFrame(gbt.predict_log_proba(data_p))
data_p_s1 = data_p_s1.drop([0], axis=1)
data_p_s2 = data_p_s2.drop([0], axis=1)

data_p = pd.concat((data_p, data_p_s1, data_p_s2), axis=1)

data_p_s1.to_csv("f:/PY_Project/ML5/ml5/result/test_predict1.csv", sep=';', index=False)
data_p.to_csv("f:/PY_Project/ML5/ml5/result/test_predict.csv", sep=';', index=False)
print("File is save")