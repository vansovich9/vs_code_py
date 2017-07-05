from processing_data import LoadFile

import pandas as pd
import numpy as np
import datetime

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

names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA","GradientBoostingClassifier"]

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=700, max_features=1, random_state=None),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    GradientBoostingClassifier()
    ]

data = pd.DataFrame(LoadFile("ml5/train.csv"))
# Нормализация данных
text_file = open("logs/best_cardio.log", "a+")
text_file.write("Start date: %s \n" % datetime.datetime.now())

data_n = pd.DataFrame(StandardScaler().fit_transform(data[['age', 'height', 'weight', 'ap_hi', 'ap_lo',
               'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c']]))
data[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n', 'weight_o', 'weight_nfg_o', 'weight_nfg_o_с', 'weight_o_c']
     ] = data_n
# age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio
# Предсказание курение алкоголь активность

X = data.drop(['cardio','id'], axis=1)  # Выбрасываем столбец 'class'.
Y = data['cardio']

# Предсказание курения
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=11)
print("start faind")
 # iterate over classifiers
for name, clf in zip(names, classifiers):
    clf4 = clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    err_train = np.mean(Y_train != clf.predict(X_train))
    err_test = np.mean(Y_test != clf.predict(X_test))
    log_string = name + "err_train = " + str(err_train) + " err_test = " + str(err_test) + " score " + str(clf4.score(X_train, Y_train))
    print(log_string)
    text_file.write(log_string + "\n" )
print("stop faind")
text_file.write("End date: %s \n" % datetime.datetime.now())
text_file.close()

'''

'''
