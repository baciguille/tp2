import pandas as pd
import gc
from urllib.parse import unquote
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

data = pd.read_csv("competition_data.csv")
#print(data['conn_country'].nunique())
#print(data['conn_country'].value_counts())

#print(data['master_metadata_album_artist_name'].nunique())
#print(data['master_metadata_album_artist_name'].value_counts().head(10))

print(data.isnull().sum())
