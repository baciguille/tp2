import pandas as pd
import gc
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

# Configuración de pandas
pd.set_option('display.max_columns', None)

# Cargar datos
train_data = pd.read_csv("spotify-skipped-track/competition_data.csv")
eval_data = pd.read_csv("spotify-skipped-track/submission.csv")

# Preparar variables
y_train = train_data["TARGET"]
X_train = train_data.drop(columns=["TARGET", "id"]).select_dtypes(include="number")
X_eval = eval_data.drop(columns=["id"]).select_dtypes(include="number")

# Limpiar memoria
del train_data
gc.collect()

# Modelo XGBoost con hiperparámetros mejorados
xgb_model = make_pipeline(
    SimpleImputer(),
    XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc",
        use_label_encoder=False
    )
)

# Validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"AUC scores en CV: {scores}")
print(f"AUC promedio: {scores.mean():.4f}")

# Entrenar modelo en todo el dataset
xgb_model.fit(X_train, y_train)

# Predicciones
y_preds = xgb_model.predict_proba(X_eval)[:, 1]

# Crear archivo de submission
submission = pd.DataFrame({
    "ID": eval_data["id"].astype(int),
    "TARGET": y_preds
})
submission.to_csv("xgboost_submission.csv", index=False)
