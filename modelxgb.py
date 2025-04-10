import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline

# Configuraciones de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Cargar datos
train_data = pd.read_csv("/Users/guille/Desktop/ditella/td6/tp2/spotify-skipped-track/competition_data.csv")
eval_data = pd.read_csv("/Users/guille/Desktop/ditella/td6/tp2/spotify-skipped-track/submission.csv")

# Separar features y variable objetivo
y_train = train_data["TARGET"]
X_train = train_data.drop(columns=["TARGET", "id"])
X_train = X_train.select_dtypes(include='number')

X_eval = eval_data.drop(columns=["id"])
X_eval = X_eval.select_dtypes(include='number')

# Limpiar memoria
del train_data
gc.collect()

# Modelo con XGBoost
cls = make_pipeline(
    SimpleImputer(),
    XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=2345)
)
cls.fit(X_train, y_train)

# Predicción
y_preds = cls.predict_proba(X_eval)[:, cls.classes_ == 1].squeeze()

# Crear archivo de submission
submission_df = pd.DataFrame({
    "ID": eval_data["id"],
    "TARGET": y_preds
})
submission_df["ID"] = submission_df["ID"].astype(int)
submission_df.to_csv("xgboost_model.csv", sep=",", index=False)
