'''import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC'''

from processing_data import LoadFile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", '''"Linear SVM", "Sigmoid SVM", "Poly SVM",''' "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA","GradientBoostingClassifier"]

classifiers = [
    KNeighborsClassifier(3),
    #SVC(kernel="linear", C=0.025),
    #SVC(kernel="sigmoid", C=0.025),
    #SVC(kernel="poly", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    GradientBoostingClassifier()
    ]

#data = pd.DataFrame(LoadFile("ml5/train.csv"))  c:/Python/train.csv
data = pd.DataFrame(LoadFile("ml5/train.csv"))
# Нормализация данных
#data_n = (data_n - data_n.mean()) / data_n.std()
data_n = pd.DataFrame(StandardScaler().fit_transform(data[['age', 'height', 'weight', 'ap_hi', 'ap_lo',
               'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c']]))
data[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n', 'weight_o', 'weight_nfg_o', 'weight_nfg_o_с', 'weight_o_c']
     ] = data_n
# age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio
# Предсказание курение алкоголь активность

X = data.drop(["cardio", "id", "smoke", "alco", "active","gluc_3" ,"bmi_n_4" ,"gluc_1" ,
    "gluc_2" ,"bmi_n_2" ,"bmi_r_4" ,"bmi_n_1" ,"bmi_r_1" ,"bmi_n_7" ,"bmi_n_6" ,
    "bmi_n_5" ,"ap_lo_c" ,"bmi_r_3"], axis=1)
Y_smoke = data["smoke"]
Y_alco = data["alco"]

# Предсказание курения
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_smoke, test_size=0.3, random_state=11)
print("start faind")
 # iterate over classifiers
for name, clf in zip(names, classifiers):
    clf4 = clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    err_train = np.mean(Y_train != clf.predict(X_train))
    err_test = np.mean(Y_test != clf.predict(X_test))
    print(name,"err_train=", err_train, "err_test=",err_test, "score %s" % clf4.score(X_train, Y_train))
print("stop faind")
'''
print("start gbt Y_smoke")
gbt_smoke = ensemble.GradientBoostingClassifier(
    n_estimators=300, random_state=264, min_samples_leaf=300, min_samples_split=150)
clf4 = gbt_smoke.fit(X_train, Y_train)
err_train = np.mean(Y_train != gbt_smoke.predict(X_train))
err_test = np.mean(Y_test != gbt_smoke.predict(X_test))
joblib.dump(gbt_smoke, "training_models/gbt_smoke.pkl", compress=1)

importances = gbt_smoke.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns
print("Feature importances:")
for f, idx in enumerate(indices):
    print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))

print(err_train, err_test)
print("Y_smoke gbt score %s" % clf4.score(X_train, Y_train))

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
# Предсказание алкоголя
print("start Y_alco")
gbt_alko = ensemble.GradientBoostingClassifier(
    n_estimators=300, random_state=264, min_samples_leaf=300, min_samples_split=150)
clf4 = gbt_alko.fit(X_train, Y_train)
err_train = np.mean(Y_train != gbt_alko.predict(X_train))
err_test = np.mean(Y_test != gbt_alko.predict(X_test))
joblib.dump(gbt_alko, "training_models/gbt_alko.pkl", compress=1)
print(err_train, err_test)
print("Y_alco score %s" % clf4.score(X_train, Y_train))'''
