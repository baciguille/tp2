import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("competition_data.csv")

print(df.head())
# print(df.describe(include='all'))
# print(df['TARGET'].value_counts(normalize=True))

# #grafico proporcion de TARGET
# plt.figure(figsize=(10, 5))
# df['TARGET'].value_counts(normalize=True).plot(kind='bar')
# plt.title('Proporción de TARGET')
# plt.xlabel('TARGET')
# plt.ylabel('Proporción')
# plt.xticks(rotation=0)
# # plt.show()


# # grafico proporcion de conn_country con la cantidad de observaciones por categoría
# plt.figure(figsize=(10, 5))
# df['conn_country'].value_counts(normalize=True).plot(kind='bar')
# plt.title('Proporción de conn_country')
# plt.xlabel('País')
# plt.ylabel('Proporción')
# plt.xticks(rotation=90)
# # plt.show()
# # ahora vemos la proporcion de 'AR' en la variable
# print(f"Proporción de 'AR': {sum(df['conn_country'] == 'AR') / len(df) * 100}%\n")
# ## la mayoria de las observaciones son de 'AR'. Vemos cuántos exactamente

# # veo la propocion de target según conn_country=AR  
# df_ar = df[df['conn_country'] == 'AR']
# df_no_ar = df[df['conn_country'] != 'AR']
# print(f"Proporción de TARGET en AR: {df_ar['TARGET'].value_counts(normalize=True)}\n")
# print(f"Proporción de TARGET fuera de AR: {df_no_ar['TARGET'].value_counts(normalize=True)}\n")
# # vemos que estan balanceadas las proporciones según el país. Pareciera que el atributo no aporta mucha información. El 97% de las obs son de un tipo y además TARGET está balanceado dentro de las categorías.

# # grafico proporción de TARGET por conn_country
# pd.crosstab(df['conn_country'], df['TARGET'], normalize='index').plot(kind='bar')
# plt.title('Proporción de TARGET por país')
# plt.xlabel('país')	
# plt.ylabel('Proporción')
# plt.xticks(rotation=0)
# plt.legend(title='TARGET')
# plt.show()

# # hago OHE de conn_country para ver correlacion con las demas categorías
df_ohe = pd.get_dummies(df, columns=['conn_country'], prefix='country')
# # veo correlacion entre conn_country y TARGET
# # veo df ohe
# print(f"\n{df_ohe.head()}\n")


# # proporción de TARGET por shuffle
# print(f"Proporción de shuffle: {df['shuffle'].value_counts(normalize=True)}\n")
# pd.crosstab(df['shuffle'], df['TARGET'], normalize='index').plot(kind='bar')
# plt.title('Proporción de TARGET por shuffle')
# plt.xlabel('shuffle')
# plt.ylabel('Proporción')
# plt.xticks(rotation=0)
# plt.legend(title='TARGET')
# plt.show()


# top 10 artistas mas esKcuchados
#print(f"10 mas escuchados: {df['master_metadata_album_artist_name'].value_counts().head(10)}\n")

# top 10 artistas con más skipps
#print(f"10 con más skips: {df[df['TARGET'] == 1]['master_metadata_album_artist_name'].value_counts().head(10)}\n")

# top 10 artistas con más skips por shuffle
#print(f"10 con más skips por shuffle: {df[df['TARGET'] == 1].groupby(['shuffle', 'master_metadata_album_artist_name']).size().reset_index(name='counts').sort_values(['shuffle', 'counts'], ascending=[True, False]).groupby('shuffle').head(10)}\n")

# veo correlacion entre shuffle y TARGET
#print(df[['shuffle', 'TARGET']].corr(method='pearson'))

# quiero ver otras correlaciones
# hago un heatmap con los atributos numéricos
# primero preparo el dataframe, voy a incluir ademas de ohe de paises, ohe de top artistas escuchados y reason_start. Elimino tambien las columnas que no me interesan
df_ohe =pd.get_dummies(df_ohe, columns=['reason_start'], prefix='artist')
# df_ohe = df.drop(columns=['username', 'id'], axis=1) 
plt.figure(figsize=(10, 5))
numerical_df = df_ohe.select_dtypes(include=['int64', 'bool'])
correlation = numerical_df.corr()
plt.imshow(correlation, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title('Heatmap de correlaciones')
plt.show()

# hago grafico de correlacion para los artistas mas escuchados 
# me quedo con las observaciones de los 10 artistas mas escuchados
top_artists = df['master_metadata_album_artist_name'].value_counts().head(20).index
df_top_artistas = df[df['master_metadata_album_artist_name'].isin(top_artists)]
df_top_artistas_ohe = pd.get_dummies(df_top_artistas, columns=['master_metadata_album_artist_name'], prefix='artist')
plt.figure(figsize=(10, 5))
numerical_df = df_top_artistas_ohe.select_dtypes(include=['int64', 'bool'])
correlation = numerical_df.corr()
plt.imshow(correlation, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title('Heatmap de correlaciones')
plt.show()

# veo que algunos artistas tienen mas correlacion con la variable target, mas proba de que skippeen.
# voy a ver si la cantidad de observaciones de aquellos artistas es significativa
# veo la cantidad de observaciones por artista, según el grafico son: 'Talking Heads', 'Polyphia', 'Childish Gambino', 'Santana', 'Pescado Rabioso'
# con esto estoy viendo si hay artistas con más chance de skippear. Hoy en día hay artistas que se viralizan por partes de canciones a partir de trends, y la gente suele escuchar solo las partes que se viralizan, skippeando el resto de la canción.
artistas_corr = ['Talking Heads', 'Polyphia', 'Childish Gambino', 'Santana', 'Pescado Rabioso']
for x in artistas_corr:
    print(f"Proporción de {x}: {sum(df['master_metadata_album_artist_name'] == x) / len(df) }%\n") 
# Notamos como Talking Heads y Childish Gambino juntan mas del 20% de los artistas 

# veo 'correlacion entre las varibales categoricas con target
categorical_df = df.select_dtypes(include=['object'])