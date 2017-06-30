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

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA","GradientBoostingClassifier"]

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=700, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    GradientBoostingClassifier()
    ]

data = pd.DataFrame(LoadFile("ml5/train.csv"))

text_file = open("logs/best_cardio.log", "a+")
text_file.write("Start date: %s \n" % datetime.datetime.now())
# Нормализация данных
data_n = pd.DataFrame(StandardScaler().fit_transform(data[['age', 'height', 'weight', 'ap_hi', 'ap_lo',
               'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c']]))
data[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'bmi', 'ap_hi_n', 'ap_lo_n', 'weight_o', 'weight_nfg_o', 'weight_nfg_o_с', 'weight_o_c']
     ] = data_n
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
    log_string = name + "err_train = " + err_train + " err_test = " + err_test + " score %s" % clf4.score(X_train, Y_train)
    print(log_string)
    text_file.write(log_string + "\n" )
    
print("stop faind")
text_file.write("End date: %s \n" % datetime.datetime.now())
text_file.close()

'''
Nearest Neighbors err_train= 0.0718163265306 err_test= 0.108761904762 score 0.928183673469
Decision Tree err_train= 0.0798979591837 err_test= 0.0928571428571 score 0.920102040816
Random Forest err_train= 0.0878367346939 err_test= 0.088 score 0.912163265306
Neural Net err_train= 0.0881836734694 err_test= 0.088 score 0.911816326531
AdaBoost err_train= 0.0881224489796 err_test= 0.0882857142857 score 0.91187755102
Naive Bayes err_train= 0.170897959184 err_test= 0.17119047619 score 0.829102040816
QDA err_train= 0.296163265306 err_test= 0.295761904762 score 0.703836734694
GradientBoostingClassifier err_train= 0.087612244898 err_test= 0.088 score 0.912387755102

'''
