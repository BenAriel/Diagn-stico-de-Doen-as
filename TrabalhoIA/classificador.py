import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)
df = pd.read_csv('dados.csv')
print(len(df))


class_counts = df['prognosis'].value_counts()
print(class_counts)

pd.set_option('display.max_rows', None)
print(df.isna().sum())

X = df.drop('prognosis', axis=1)
y = df['prognosis']



# Definindo os classificadores
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
et = ExtraTreesClassifier(random_state=42)
knn = KNeighborsClassifier()

# Criando pipelines
pipeline_dt = Pipeline([('decisiontree', dt)])
pipeline_rf = Pipeline([('randomforest', rf)])
pipeline_et = Pipeline([('extratrees', et)])
pipeline_knn = Pipeline([('kneighbors', knn)])

# Definindo os grids de parâmetros
param_grid_dt = {
    'decisiontree__max_depth': [None, 10, 20, 30],
    'decisiontree__min_samples_split': [2, 10, 20],
    'decisiontree__min_samples_leaf': [1, 5, 10]
}

param_grid_rf = {
    'randomforest__n_estimators': [20, 30, 50],
    'randomforest__max_depth': [None, 10, 20, 30],
    'randomforest__min_samples_split': [2, 10, 20],
    'randomforest__min_samples_leaf': [1, 5, 10]
}

param_grid_et = {
    'extratrees__n_estimators': [20, 30, 50],
    'extratrees__max_depth': [None, 10, 20, 30],
    'extratrees__min_samples_split': [2, 10, 20],
    'extratrees__min_samples_leaf': [1, 5, 10]
}

param_grid_knn = {
    'kneighbors__n_neighbors': range(1, 100, 2),
    'kneighbors__weights': ['uniform', 'distance'],
    'kneighbors__p': [1, 2]
}

# Realizando a busca em grade
grid_dt = GridSearchCV(pipeline_dt, param_grid_dt, cv=10, scoring='accuracy')
grid_dt.fit(X, y)

grid_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=10, scoring='accuracy')
grid_rf.fit(X, y)
grid_et = GridSearchCV(pipeline_et, param_grid_et, cv=10, scoring='accuracy')
grid_et.fit(X, y)

grid_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=10, scoring='accuracy')
grid_knn.fit(X, y)

print('Decision Tree:')
print('Melhor parâmetro:', grid_dt.best_params_)
print('Melhor resultado:', grid_dt.best_score_)

print('Random Forest:')
print('Melhor parâmetro:', grid_rf.best_params_)
print('Melhor resultado:', grid_rf.best_score_)

print('Extra Trees:')
print('Melhor parâmetro:', grid_et.best_params_)
print('Melhor resultado:', grid_et.best_score_)

print('KNN:')
print('Melhor parâmetro:', grid_knn.best_params_)
print('Melhor resultado:', grid_knn.best_score_)