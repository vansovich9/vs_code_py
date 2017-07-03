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

#X = data.drop(['cardio','id'], axis=1)  # Выбрасываем столбец 'class'.
X = data.drop(['cardio','id','weight_o','weight_nfg_o','weight_nfg_o_с','weight_o_c','alco','bmi_r_4','bmi_n_7','bmi_r_1','bmi_n_2','bmi_n_1'], axis=1)  # Выбрасываем столбец 'class'.
Y = data['cardio']

# Предсказание курения
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=11)
print("start faind")
'''
class sklearn.ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
 criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,
  min_impurity_split=1e-07, init=None, max_features=None,
  max_leaf_nodes=None)

'''


 # iterate over classifiers
from sklearn.model_selection import GridSearchCV
n_estimators = [10,50,100,150,250,300,450,800]
criterion = ['gini','entropy']
max_features = ['auto', 'log2', None, 0.8]
max_depth = [None, 100, 700, 7000, 10000]
min_samples_split = [2, 10, 100, 0.01, 0.1, 0.3]
min_samples_leaf = [1, 10, 30, 0.1, 0.01, 0.3]
min_weight_fraction_leaf = [0.0, 0.01,0.1,0.2,0.3]
max_leaf_nodes = [None,100,300,600,1000,1500]
verbose = [0,10,20,30,40]

abc = GradientBoostingClassifier(random_state=264)
grid = GridSearchCV(abc, param_grid={
    'n_estimators' : n_estimators
})
grid.fit(X_train, Y_train)

best_cv_err = 1 - grid.best_score_
best_n_neighbors = grid.best_estimator_.n_estimators
print(best_cv_err, best_n_neighbors)
print("best params")
print(grid.best_params_)

print("Grid scores on development set:")
print()
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

print("stop faind")
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
