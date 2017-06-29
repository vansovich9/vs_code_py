from processing_data import LoadFile

from sklearn.externals import joblib
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.svm import SVC

data = pd.DataFrame(LoadFile("ml5/train.csv"))
#Нормализация данных
data_n = data[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n']]

data_n = (data_n - data_n.mean()) / data_n.std()

data[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc','bmi', 'ap_hi_n', 'ap_lo_n']] = data_n[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc','bmi', 'ap_hi_n', 'ap_lo_n']]

data = data.fillna(0, axis=0)
#age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio
#Предсказание курение алкоголь активность
X = data.drop(["cardio","id","smoke","alco","active"], axis=1)  # Выбрасываем столбец "class".
Y_smoke = data["smoke"]
Y_alco = data["alco"]
Y_active = data["active"]

#Предсказание курения
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_smoke, test_size = 0.3, random_state = 11)

print("start gbt Y_smoke")
gbt_smoke = ensemble.GradientBoostingClassifier(n_estimators=300, random_state=264, min_samples_leaf=300,min_samples_split=150)
clf4 = gbt_smoke.fit(X_train, Y_train)
err_train = np.mean(Y_train != gbt_smoke.predict(X_train))
err_test = np.mean(Y_test != gbt_smoke.predict(X_test))
joblib.dump(gbt_smoke, "training_models/gbt_smoke.pkl", compress=1)
print(err_train, err_test)
print("Y_smoke gbt score %s" % clf4.score(X_train, Y_train))

print("start Y_smoke svc_l")
svc_l = SVC(kernel="linear", C=0.025)
clf4 = svc_l.fit(X_train, Y_train)
err_train = np.mean(Y_train != svc_l.predict(X_train))
err_test = np.mean(Y_test != svc_l.predict(X_test))
joblib.dump(svc_l, "training_models/svc_l.pkl", compress=1)
print(err_train, err_test)
print("Y_smoke svc_l score %s" % clf4.score(X_train, Y_train))

print("start Y_smoke svc_r_g")
svc_r_g = SVC(gamma=200, C=1)
clf4 = svc_r_g.fit(X_train, Y_train)
err_train = np.mean(Y_train != svc_r_g.predict(X_train))
err_test = np.mean(Y_test != svc_r_g.predict(X_test))
joblib.dump(svc_r_g, "training_models/svc_r_g.pkl", compress=1)
print(err_train, err_test)
print("Y_smoke svc_r_g score %s" % clf4.score(X_train, Y_train))
#Предсказание алкоголя
print("start Y_alco")
gbt_alko = ensemble.GradientBoostingClassifier(n_estimators=300, random_state=264, min_samples_leaf=300,min_samples_split=150)
clf4 = gbt_alko.fit(X_train, Y_train)
err_train = np.mean(Y_train != gbt_alko.predict(X_train))
err_test = np.mean(Y_test != gbt_alko.predict(X_test))
joblib.dump(gbt_alko, "training_models/gbt_alko.pkl", compress=1)
print(err_train, err_test)
print("Y_alco score %s" % clf4.score(X_train, Y_train))