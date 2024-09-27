import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Configurações de exibição do pandas para mostrar todas as colunas
pd.set_option('display.max_columns', None)

# Carregar os dados dos sintomas e doenças
df = pd.read_csv('dados.csv')
print(f"Número de instâncias: {len(df)}")

# Verificar a distribuição das classes
class_counts = df['prognosis'].value_counts()
print("Distribuição das classes:")
print(class_counts)

# Renomear a coluna 'spotting_ urination' para 'spotting_urination'
df = df.rename(columns={'spotting_ urination': 'spotting_urination'})

# Verificar valores ausentes
print("Valores ausentes por coluna:")
pd.set_option('display.max_rows', None)
print(df.isna().sum())

# Separar variáveis independentes (sintomas) e dependente (doença)
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Carregar os pesos dos sintomas
pesos_df = pd.read_csv('Symptom-severity.csv')
pesos_df = pesos_df.drop('prognosis', axis=0)

# Garantir que os sintomas de X estão alinhados com os sintomas do arquivo de pesos
pesos_df = pesos_df.drop_duplicates(subset=['Symptom'])  # Remover duplicatas se houver
pesos = pesos_df.set_index('Symptom')['weight']

# Sintomas presentes em X, mas faltando em pesos_df
sintomas_faltando_pesos = set(X.columns) - set(pesos.index)
if sintomas_faltando_pesos:
    print(f"Sintomas faltando no arquivo de pesos: {sintomas_faltando_pesos}")

# Sintomas presentes em pesos_df, mas faltando em X
sintomas_excedentes_pesos = set(pesos.index) - set(X.columns)
if sintomas_excedentes_pesos:
    print(f"Sintomas excedentes no arquivo de pesos (não usados): {sintomas_excedentes_pesos}")

# Aplicar os pesos apenas aos sintomas que existem em ambas as tabelas
for sintoma in X.columns:
    if sintoma in pesos.index:
        df[sintoma] = X[sintoma] * pesos[sintoma]
    else:
        print(f"Sintoma {sintoma} não encontrado nos pesos, mantendo sem peso.")

# Usar LabelEncoder para transformar os valores categóricos em valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Definir os classificadores
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
et = ExtraTreesClassifier(random_state=42)
knn = KNeighborsClassifier()

# Criar pipelines com PCA e normalização
pipeline_dt = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('decisiontree', dt)])
pipeline_rf = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('randomforest', rf)])
pipeline_et = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('extratrees', et)])
pipeline_knn = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('kneighbors', knn)])

# Definir os grids de parâmetros
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

# Realizar a busca em grade para cada modelo
grid_dt = GridSearchCV(pipeline_dt, param_grid_dt, cv=10, scoring='accuracy')
grid_dt.fit(X, y)

grid_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=10, scoring='accuracy')
grid_rf.fit(X, y)

grid_et = GridSearchCV(pipeline_et, param_grid_et, cv=10, scoring='accuracy')
grid_et.fit(X, y)

grid_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=10, scoring='accuracy')
grid_knn.fit(X, y)

# Exibir os melhores resultados
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
