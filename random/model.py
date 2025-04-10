import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# Configuraciones de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Cargar datos de entrenamiento
train_data = pd.read_csv("/Users/guille/Desktop/ditella/td6/tp2/spotify-skipped-track/competition_data.csv")
eval_data = pd.read_csv("/Users/guille/Desktop/ditella/td6/tp2/spotify-skipped-track/submission.csv")

# Separar target y features
y_train = train_data["TARGET"]
X_train = train_data.drop(columns=["TARGET", "id"])
X_train = X_train.select_dtypes(include='number')

# Modelo con imputación + Random Forest
cls = make_pipeline(
    SimpleImputer(),
    RandomForestClassifier(n_estimators=300, max_depth=15, random_state=2345)
)

# Validación cruzada con AUC
auc_scores = cross_val_score(cls, X_train, y_train, cv=5, scoring="roc_auc")
print("AUC scores en CV:", auc_scores)
print("AUC promedio:", auc_scores.mean())

# ENTRENAMIENTO FINAL con todo el training set
cls.fit(X_train, y_train)

# Predecir en el set de evaluación (¡con columnas consistentes!)
columnas_esperadas = X_train.columns
X_eval = eval_data.drop(columns=["id"])
X_eval = X_eval[columnas_esperadas]

y_preds = cls.predict_proba(X_eval)[:, cls.classes_ == 1].squeeze()

# Crear archivo de submission
submission_df = pd.DataFrame({
    "ID": eval_data["id"],
    "TARGET": y_preds
})
submission_df["ID"] = submission_df["ID"].astype(int)
submission_df.to_csv("randomforest_cv.csv", sep=",", index=False)