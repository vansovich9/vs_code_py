import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from processing_data import LoadFile

data = pd.DataFrame(LoadFile("ml5/train.csv"))

from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Предсказание курения
print("start ap_hi")
X = data.drop(['cardio','id','ap_hi', 'weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c','alco','bmi_r_4','bmi_n_7','bmi_r_1','bmi_n_2','bmi_n_1'], axis=1)  # Выбрасываем столбец 'class'.
Y =  np.ravel(data['ap_hi'])

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=11)

gbt = MLPClassifier(alpha=0.0, random_state = 0, activation = 'relu', hidden_layer_sizes=(50,), verbose = 0)

from sklearn.multioutput import MultiOutputClassifier
#gbt = RandomForestClassifier(n_estimators=300, random_state=264, min_samples_leaf=300, min_samples_split=150)
#MultiOutputClassifier(gbt,n_job=-1).fit(X_train, Y_train)
mor = MultiOutputClassifier(gbt,n_jobs=-1)
clf4 = mor.fit(X_train, Y_train)

err_train = np.mean(Y_train != mor.predict(X_train))
err_test = np.mean(Y_test != mor.predict(X_test))
err_sum = np.mean(Y != mor.predict(X))
joblib.dump(mor, "training_models/cardio_hi.pkl", compress=1)

print("gender 0",err_train, err_test, 'err_sum', err_sum)
print("gbt score %s" % clf4.score(X_train, Y_train))

print()
print("start ap_lo")
X = data.drop(['cardio','ap_lo','id','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c','alco','bmi_r_4','bmi_n_7','bmi_r_1','bmi_n_2','bmi_n_1'], axis=1)  # Выбрасываем столбец 'class'.
Y = data['ap_lo']



X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=11)
gbt = MLPClassifier(alpha=0.0, random_state = 0, activation = 'relu', hidden_layer_sizes=(120,), verbose = 0)
clf4 = gbt.fit(X_train, Y_train)

err_train = np.mean(Y_train != gbt.predict(X_train))
err_test = np.mean(Y_test != gbt.predict(X_test))
err_sum = np.mean(Y != gbt.predict(X))
joblib.dump(gbt, "training_models/cardio_lo.pkl", compress=1)

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