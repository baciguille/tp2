import pandas as pd
import gc
from urllib.parse import unquote
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Función para clasificar plataforma
def clasificar_plataforma(plat: str) -> str:
    if pd.isnull(plat):
        return 'unknown'
    plat = plat.lower()
    if any(x in plat for x in ['android', 'ios', 'iphone', 'ipad', 'mobile', 'phone']):
        return 'movil'
    elif any(x in plat for x in ['windows', 'mac', 'web', 'desktop']):
        return 'pc'
    elif any(x in plat for x in ['tv', 'xbox', 'ps', 'console']):
        return 'tv/consola'
    else:
        return 'otro'

# ----------- CARGA Y PROCESAMIENTO DE TRAINING DATA -----------
data = pd.read_csv("competition_data.csv")
data.drop(columns=['spotify_track_uri', 'username'], inplace=True)

# Procesamiento de ts
data['ts'] = pd.to_datetime(data['ts'], errors='coerce') # convierto ts con pandas para poder operar sobre ella
data['hora'] = data['ts'].dt.hour
data['dia_semana'] = data['ts'].dt.dayofweek
data['es_fin_de_semana'] = data['dia_semana'].isin([5, 6]).astype(int)
data['mes'] = data['ts'].dt.month
data['anio'] = data['ts'].dt.year
#data.drop(columns=['dia_semana'], inplace=True)
data.drop(columns=['ts'], inplace=True) # borro ts

# Procesamiento de platform
data['tipo_dispositivo'] = data['platform'].apply(clasificar_plataforma)
# One-hot encoding para tipo_dispositivo
data = pd.get_dummies(data, columns=['tipo_dispositivo']) # identifico los tipos de dispositivos
# One-hot encoding para reason_start
data = pd.get_dummies(data, columns=['reason_start'], prefix='start', drop_first=True)
# One-hot encoding para conn_country
data = pd.get_dummies(data, columns=['conn_country'], prefix='pais', drop_first=True)

# Separar los datos de entrenamiento/ validación
split_idx = int(len(data) * 0.8)
data = data.select_dtypes(include=['number', 'bool'])
data = data.drop(columns=['id'])

x_train = data[:split_idx].drop(columns=['TARGET'])
x_val = data[split_idx:].drop(columns=['TARGET'])

y_train = data[:split_idx]["TARGET"]
y_val = data[split_idx:]["TARGET"]

X_columns = x_train.columns # Guardo las columnas para usar en eval

# Separo los datos en training y validación, 80% y 20% respectivamente
#x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------- CARGA Y PROCESAMIENTO DE EVALUATION DATA -----------
eval_data = pd.read_csv("submission.csv")
eval_ids = eval_data["id"]
eval_data.drop(columns=['spotify_track_uri', 'username'], inplace=True)

# Procesamiento de ts
eval_data['ts'] = pd.to_datetime(eval_data['ts'], errors='coerce')
eval_data['hora'] = eval_data['ts'].dt.hour
eval_data['dia_semana'] = eval_data['ts'].dt.dayofweek
eval_data['es_fin_de_semana'] = eval_data['dia_semana'].isin([5, 6]).astype(int)
eval_data['mes'] = eval_data['ts'].dt.month
eval_data['anio'] = eval_data['ts'].dt.year
eval_data.drop(columns=['ts'], inplace=True)

# Procesamiento de platform
eval_data['tipo_dispositivo'] = eval_data['platform'].apply(clasificar_plataforma)
# One-hot encoding para tipo_dispositivo
eval_data = pd.get_dummies(eval_data, columns=['tipo_dispositivo'])
# One-hot encoding para reason_start
eval_data = pd.get_dummies(eval_data, columns=['reason_start'], prefix='start', drop_first=True)
# One-hot encoding para conn_country
eval_data = pd.get_dummies(eval_data, columns=['conn_country'], prefix='pais', drop_first=True)

eval_data = eval_data.select_dtypes(include=['number', 'bool'])

# Asegurar que tenga las mismas columnas que X
for col in X_columns:
    if col not in eval_data.columns:
        eval_data[col] = 0
eval_data = eval_data[X_columns]  

# ----------- ENTRENAMIENTO DEL MODELO -----------
param_grid = {
    'n_estimators': [100, 200],           # cantidad de árboles
    'max_depth': [10, 30],                 # profundidad máxima del árbol
    'learning_rate': [0.01, 0.05],        # tasa de aprendizaje
    'subsample': [0.4, 0.6],              # proporción de datos usados para cada árbol
    'colsample_bytree': [0.6, 0.8],       # proporción de columnas usadas en cada árbol
    'min_child_weight': [1, 3],             # peso mínimo de los nodos hoja
    'gamma': [0.1, 0.3],                    # regularización: ganancia mínima para hacer split
    'reg_alpha': [0.1, 0.8],                  # regularización L1 (lasso)
    'reg_lambda': [1.5, 2],                 # regularización L2 (ridge)
    'scale_pos_weight': [1, 3]              # peso de la clase positiva (útil si hay desbalance)
}

print("por entrenar")
modelo = XGBClassifier(eval_metric='auc', random_state=42)
grid = GridSearchCV(modelo, param_grid, cv=5, scoring='roc_auc', n_jobs=1)
grid.fit(x_train, y_train)

print("Mejores hiperparámetros:")
print(grid.best_params_)

modelo = grid.best_estimator_

# Me fijo el AUC-ROC para los conjuntos de train y test para chequear
# AUC sobre el conjunto de entrenamiento
y_train_preds = modelo.predict_proba(x_train)[:, 1]
auc_train = roc_auc_score(y_train, y_train_preds)
print(f"AUC-ROC (Train): {auc_train:.4f}")

# AUC sobre el conjunto de validación
y_val_preds = modelo.predict_proba(x_val)[:, 1]
auc_val = roc_auc_score(y_val, y_val_preds)
print(f"AUC-ROC (Validación): {auc_val:.4f}")

# Evaluación del modelo con eval_data
y_preds = modelo.predict_proba(x_val)[:, 1]
auc = roc_auc_score(y_val, y_preds)
print(f"AUC-ROC: {auc:.4f}")

# Predicción final
y_preds_eval = modelo.predict_proba(eval_data)[:, 1]

# Guardar resultado
submission_df = pd.DataFrame({
    "ID": eval_ids.astype(int),
    "TARGET": y_preds_eval
})
submission_df.to_csv("modelo_final.csv", sep=",", index=False)
