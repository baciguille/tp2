import pandas as pd
import gc
from urllib.parse import unquote
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

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
data['ts'] = pd.to_datetime(data['ts'], errors='coerce')

data['hora'] = data['ts'].dt.hour
data['dia_semana'] = data['ts'].dt.dayofweek
data['es_fin_de_semana'] = data['dia_semana'].isin([5, 6]).astype(int)
data['mes'] = data['ts'].dt.month
data['anio'] = data['ts'].dt.year

data.drop(columns=['ts'], inplace=True)
data['tipo_dispositivo'] = data['platform'].apply(clasificar_plataforma)

# One-hot encoding para tipo_dispositivo
data = pd.get_dummies(data, columns=['tipo_dispositivo'])
# One-hot encoding para reason_start
data = pd.get_dummies(data, columns=['reason_start'], prefix='start', drop_first=True)

# Separar variables
y = data["TARGET"]
X = data.drop(columns=["TARGET"])
X = X.select_dtypes(include=['number', 'bool'])

# Train/val split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------- CARGA Y PROCESAMIENTO DE EVALUATION DATA -----------
eval_data = pd.read_csv("submission.csv")
eval_ids = eval_data["id"]

# Procesamiento de ts 
eval_data['ts'] = pd.to_datetime(eval_data['ts'], errors='coerce')

eval_data['hora'] = eval_data['ts'].dt.hour
eval_data['dia_semana'] = eval_data['ts'].dt.dayofweek
eval_data['es_fin_de_semana'] = eval_data['dia_semana'].isin([5, 6]).astype(int)
eval_data['mes'] = eval_data['ts'].dt.month
eval_data['anio'] = eval_data['ts'].dt.year

eval_data.drop(columns=['ts'], inplace=True)

eval_data['tipo_dispositivo'] = eval_data['platform'].apply(clasificar_plataforma)
# One-hot encoding para tipo_dispositivo
eval_data = pd.get_dummies(eval_data, columns=['tipo_dispositivo'])
# One-hot encoding para reason_start
eval_data = pd.get_dummies(eval_data, columns=['reason_start'], prefix='start', drop_first=True)


# Asegurar que eval_data tenga todas las columnas de entrenamiento
for col in x_train.columns:
    if col not in eval_data.columns:
        eval_data[col] = 0

# Quitar columnas que no están en x_train
eval_data = eval_data[x_train.columns]

# ----------- ENTRENAMIENTO DEL MODELO -----------
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

modelo = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid = GridSearchCV(modelo, param_grid, cv=5, scoring='roc_auc', n_jobs=1)
grid.fit(x_train, y_train)

print("Mejores hiperparámetros:")
print(grid.best_params_)

modelo = grid.best_estimator_

# Evaluación
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
submission_df.to_csv("xgboost2.csv", sep=",", index=False)
