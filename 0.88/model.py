import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV 
from urllib.parse import unquote


# Mostrar todas las columnas y filas en pantalla
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Cargar datos 
data = pd.read_csv("competition_data.csv")

# PROCESAMIENTO DE DATOS 
data.drop(columns=['spotify_track_uri', 'username'], inplace=True)

# decodificamos los caracteres especiales para 'platform'
data['user_agent_decrypted'] = data['platform'].apply(lambda x: unquote(x) if pd.notnull(x) else x) 
data['user_agent_decrypted'] = data['platform'].str.lower()

def clasificar_plataforma(plat:str) -> str:
    if pd.isnull(plat):
        return 'unknown'
    if any(x in plat for x in ['android', 'ios', 'iphone', 'ipad', 'mobile', 'phone']):
        return 'movil'
    elif any(x in plat for x in ['windows', 'mac', 'web', 'desktop']):
        return 'pc'
    elif any(x in plat for x in ['tv', 'xbox', 'ps', 'console']):
        return 'tv/consola'
    else:
        return 'otro'

data['tipo_dispositivo'] = data['platform'].apply(clasificar_plataforma)

# Eliminar columnas innecesarias
data.drop(columns=[
    'platform',
    'user_agent_decrypted',
    'master_metadata_track_name',
    'master_metadata_album_artist_name',
    'master_metadata_album_album_name',
    'ts'
], inplace=True)


# TRAIN SET
train_data = data.sample(frac=0.8, random_state=42)
train_data = train_data.sample(frac=6/10)
y_train = train_data["TARGET"]
x_train = train_data.drop(columns=["TARGET"])
x_train = x_train.select_dtypes(include=['number', 'bool'])

# VAL SET
val_data = data.drop(train_data.index)
y_val = val_data["TARGET"]
x_val = val_data.drop(columns=["TARGET"])
x_val = x_val.select_dtypes(include=['number', 'bool'])

# Cargar datos de evaluación
eval_data = pd.read_csv("submission.csv")
eval_ids = eval_data["id"]  # guardamos los IDs antes de perderlos
eval_data = eval_data[x_train.columns]  # nos quedamos solo con columnas numéricas y booleanas

# Liberar memoria
del train_data
gc.collect()

# Definir y entrenar modelo con GridSearch
param_grid = {
    'splitter': ['best'],
    'class_weight': ['balanced'],
    'max_depth': [8, 10, 12, 14],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [5, 7, 9]
}

# Definir el modelo
modelo = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 12, 14],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [3, 5],
    'class_weight': ['balanced']
}

grid = GridSearchCV(modelo, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(x_train, y_train)

print("Mejores hiperparámetros:")
print(grid.best_params_)

modelo = grid.best_estimator_

# Métrica en validation
y_preds = modelo.predict_proba(x_val)[:, modelo.classes_ == 1].squeeze()
auc = roc_auc_score(y_val, y_preds)
print(f"AUC-ROC: {auc:.4f}")

# Predicción final
y_preds_eval = modelo.predict_proba(eval_data)[:, modelo.classes_ == 1].squeeze()

# Crear archivo de salida con formato correcto
submission_df = pd.DataFrame({
    "ID": eval_ids.astype(int),
    "TARGET": y_preds_eval
})
submission_df.to_csv("random.csv", sep=",", index=False)
