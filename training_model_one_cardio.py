import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
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
data_n = pd.DataFrame(StandardScaler().fit_transform(data[['age', 'height', 'weight', 'ap_hi', 'ap_lo',
               'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c']]))
data[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n', 'weight_o', 'weight_nfg_o', 'weight_nfg_o_с', 'weight_o_c']
     ] = data_n

X = data.drop(['cardio','id'], axis=1)  # Выбрасываем столбец 'class'.
Y = data['cardio']

# Предсказание курения
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=11)
X1=X[:]
print("start gbt Y_smoke")
#abc = ensemble.AdaBoostClassifier(n_estimators=300, random_state=264)
#gbt = ensemble.GradientBoostingClassifier(n_estimators=300, random_state=264)
X_train, X_test, Y_train, Y_test = train_test_split(
    X1, Y, test_size=0.3, random_state=11)
gbt = ensemble.AdaBoostClassifier(n_estimators=300, random_state=264)
clf4 = gbt.fit(X_train, Y_train)

err_train = np.mean(Y_train != gbt.predict(X_train))
err_test = np.mean(Y_test != gbt.predict(X_test))
joblib.dump(gbt, "training_models/cardio.pkl", compress=1)

print(err_train, err_test)
print("Y_smoke gbt score %s" % clf4.score(X_train, Y_train))
