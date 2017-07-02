import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from processing_data import LoadFile

data = pd.DataFrame(LoadFile("ml5/train.csv"))
# Нормализация данных
data_n = data[['age', 'height', 'weight', 'ap_hi', 'ap_lo',
               'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n']]

data_n = (data_n - data_n.mean()) / data_n.std()

data[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n']
     ] = data_n[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n']]

#data = data.fillna(0, axis=0)
# age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio
# Предсказание курение алкоголь

X = data.drop(["cardio", "id", "smoke", "alco", "active","gluc_3" ,"bmi_n_4" ,"gluc_1" ,
    "gluc_2" ,"bmi_n_2" ,"bmi_r_4" ,"bmi_n_1" ,"bmi_r_1" ,"bmi_n_7" ,"bmi_n_6" ,
    "bmi_n_5" ,"ap_lo_c" ,"bmi_r_3"], axis=1)
Y_smoke = data["smoke"]
Y_alco = data["alco"]

# Предсказание курения
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_smoke, test_size=0.3, random_state=11)

print("start gbt Y_smoke")
gbt_smoke = RandomForestClassifier(
    n_estimators=300, random_state=264, min_samples_leaf=300, min_samples_split=150)
clf4 = gbt_smoke.fit(X_train, Y_train)
err_train = np.mean(Y_train != gbt_smoke.predict(X_train))
err_test = np.mean(Y_test != gbt_smoke.predict(X_test))
joblib.dump(gbt_smoke, "training_models/smoke.pkl", compress=1)
print(err_train, err_test)
print("Y_smoke gbt score %s" % clf4.score(X_train, Y_train))

# Предсказание алкоголя
print("start Y_alco")
gbt_alko = RandomForestClassifier(
    n_estimators=300, random_state=264, min_samples_leaf=300, min_samples_split=150)
clf4 = gbt_alko.fit(X_train, Y_train)
err_train = np.mean(Y_train != gbt_alko.predict(X_train))
err_test = np.mean(Y_test != gbt_alko.predict(X_test))
joblib.dump(gbt_alko, "training_models/alko.pkl", compress=1)
print(err_train, err_test)
print("Y_alco score %s" % clf4.score(X_train, Y_train))

'''
last score
start gbt Y_smoke
0.0877346938776 0.0881904761905
Y_smoke gbt score 0.912265306122
start Y_alco
0.0877346938776 0.0881904761905
Y_alco score 0.912265306122

start gbt Y_smoke
0.0877551020408 0.0880476190476
Y_smoke gbt score 0.912244897959
start Y_alco
0.0877551020408 0.0880476190476
Y_alco score 0.912244897959

'''