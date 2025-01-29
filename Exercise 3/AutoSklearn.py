import autosklearn.classification
import autosklearn.regression

from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

import numpy as np
import pandas as pd

iris = datasets.load_iris()
iris_data = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
iris_data['target'] = iris_data['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

df_voting = pd.read_csv('data/CongressionalVotingID.shuf.lrn.csv')

df_airfoil = pd.read_csv("data/airfoil_noise_data.csv")

url='./data/abalone.csv'
column_names = ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
df_abalone = pd.read_csv(url, header=0, names=column_names)
df_abalone = df_abalone[df_abalone.Height != 0]

X_iris = iris_data.drop(['target'], axis=1)
y_iris = iris_data['target']

X_train_iris, X_temp, y_train_iris, y_temp = train_test_split(X_iris, y_iris, test_size=0.6, random_state=42)
X_val_iris, X_test_iris, y_val_iris, y_test_iris = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

cls = autosklearn.classification.AutoSklearnClassifier()
cls.fit(X_train_iris, y_train_iris)
predictions = cls.predict(X_test_iris)

print("Accuracy score", accuracy_score(y_test_iris, predictions))
print(cls.show_models())
print(cls.sprint_statistics())
print(cls.cv_results_)

pd.set_option('future.no_silent_downcasting', True)
df_voting = df_voting.replace({"democrat": 0,"republican": 1,"n": 0,"y": 1,"unknown": np.nan})
df_voting = df_voting.drop(columns=['ID'])

imp = IterativeImputer(max_iter=10, random_state=0)
df_voting = pd.DataFrame(imp.fit_transform(df_voting), columns=df_voting.columns)

X_voting = df_voting.drop(['class'], axis=1)
y_voting = df_voting['class']

X_train_voting, X_temp, y_train_voting, y_temp = train_test_split(X_voting, y_voting, test_size=0.6, random_state=42)
X_val_voting, X_test_voting, y_val_voting, y_test_voting = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

cls.fit(X_train_voting, y_train_voting)
predictions = cls.predict(X_test_voting)

print("Accuracy score", accuracy_score(y_test_voting, predictions))
print(cls.show_models())
print(cls.sprint_statistics())
print(cls.cv_results_)

df_abalone = df_abalone[df_abalone.Height != 0]

X_airfoil = df_airfoil.drop(['y'], axis=1)
y_airfoil = df_airfoil['y']

X_train_airfoil, X_temp, y_train_airfoil, y_temp = train_test_split(X_airfoil, y_airfoil, test_size=0.6, random_state=42)
X_val_airfoil, X_test_airfoil, y_val_airfoil, y_test_airfoil = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

reg = autosklearn.regression.AutoSklearnRegressor()
reg.fit(X_train_airfoil, y_train_airfoil)
predictions = reg.predict(X_test_airfoil)

print("Leaderboard", reg.leaderboard(detailed = True,ensemble_only=False,sort_order="descending"))
print("MAE test score:", mean_absolute_error(y_test_airfoil, predictions))
print("MSE test score:", mean_squared_error(y_test_airfoil, predictions))
print(reg.show_models())
print(reg.sprint_statistics())
print(reg.cv_results_)

X_abalone_reg = df_abalone.drop(['Rings'], axis=1)
y_abalone_reg = df_abalone['Rings']

X_train_abalone_reg, X_temp_reg, y_train_abalone_reg, y_temp_reg = train_test_split(X_abalone_reg, y_abalone_reg, test_size=0.6, random_state=42)
X_val_abalone_reg, X_test_abalone_reg, y_val_abalone_reg, y_test_abalone_reg = train_test_split(X_temp_reg, y_temp_reg, test_size=0.5, random_state=42)

reg.fit(X_train_abalone_reg, y_train_abalone_reg)
predictions = reg.predict(X_test_abalone_reg)

print("Leaderboard", reg.leaderboard(detailed = True,ensemble_only=False,sort_order="descending"))
print("MAE test score:", mean_absolute_error(y_test_abalone_reg, predictions))
print("MSE test score:", mean_squared_error(y_test_abalone_reg, predictions))
print(reg.show_models())
print(reg.sprint_statistics())
print(reg.cv_results_)