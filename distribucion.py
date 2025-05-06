import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset (reemplazar por tu path si estás trabajando localmente)
df = pd.read_csv("competition_data.csv")  # Cambiar por el nombre real del archivo

# Estilo de gráfico
sns.set(style="whitegrid")

# GRAFICO PROPORCION TARGET 
# Conteo de clases
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="TARGET", data=df, palette=["#5DADE2", "#E74C3C"])
plt.title("Distribución de la variable objetivo (TARGET)")
plt.xlabel("TARGET (0 = No salta, 1 = Salta)")
plt.ylabel("Cantidad de observaciones")
plt.xticks([0, 1], ['No Salta', 'Salta'])

# Mostrar porcentaje arriba de cada barra
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 5,
            f'{height / len(df):.1%}',
            ha="center")

plt.tight_layout()
#plt.show()


# GRAFICO PROPORCION CONN_COUNTRY 
# Calcular proporciones
country_counts = df['conn_country'].value_counts()
country_proportions = country_counts / len(df)

# Crear gráfico
plt.figure(figsize=(10, 5))
ax = sns.barplot(x=country_proportions.index, y=country_proportions.values, palette="pastel")
plt.title("Proporción de conexiones por país (conn_country)")
plt.xlabel("País")
plt.ylabel("Proporción")
plt.xticks(rotation=90)

# Mostrar porcentaje arriba de cada barra
for p, val in zip(ax.patches, country_proportions.values):
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.005,
            f'{val:.1%}',
            ha="center", fontsize=9)

# Resaltar 'AR'
for i, label in enumerate(country_proportions.index):
    if label == 'AR':
        ax.patches[i].set_color("#E74C3C")  # Rojo
        ax.patches[i].set_edgecolor("black")
        ax.patches[i].set_linewidth(1.5)

plt.tight_layout()
#plt.show()

# GRAFICO PROPORCION CONN_COUNTRY SEGUN TARGET
# Filtrar por país AR
df_ar = df[df['conn_country'] == 'AR']

# Crear gráfico de conteo de TARGET dentro de AR
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="TARGET", data=df_ar, palette=["#5DADE2", "#E74C3C"])
plt.title("Distribución de TARGET para conn_country = 'AR'")
plt.xlabel("TARGET (0 = No salta, 1 = Salta)")
plt.ylabel("Cantidad de observaciones")
plt.xticks([0, 1], ['No Salta', 'Salta'])

# Mostrar porcentaje arriba de cada barra
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 5,
            f'{height / len(df_ar):.1%}',
            ha="center", fontsize=9)

plt.tight_layout()
#plt.show()

#GRAFICO PROPORCION SEGUN TARGET PARA TODOS LOS COUNTRYS

# Crear tabla de proporciones
prop_df = pd.crosstab(df['conn_country'], df['TARGET'], normalize='index')

# Convertir a formato largo (para Seaborn)
prop_long = prop_df.reset_index().melt(id_vars='conn_country', value_vars=[0, 1], var_name='TARGET', value_name='Proporción')

# Crear gráfico
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='conn_country', y='Proporción', hue='TARGET', data=prop_long, palette=["#5DADE2", "#E74C3C"])

# Título y ejes
plt.title('Proporción de TARGET por país (conn_country)')
plt.xlabel('País')
plt.ylabel('Proporción')
plt.xticks(rotation=90)
plt.legend(title='TARGET', labels=['No Salta', 'Salta'])

# Agregar porcentajes arriba de cada barra
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(p.get_x() + p.get_width() / 2.,
                height + 0.01,
                f'{height:.1%}',
                ha="center", fontsize=8)

plt.tight_layout()
#plt.show()

#GRAFICO PROPORCION TARGET SEGUN SHUFFLE 
# Tabla de proporciones por shuffle
prop_df = pd.crosstab(df['shuffle'], df['TARGET'], normalize='index')

# Convertir a formato largo
prop_long = prop_df.reset_index().melt(id_vars='shuffle', value_vars=[0, 1],
                                       var_name='TARGET', value_name='Proporción')

# Crear gráfico
plt.figure(figsize=(6, 4))
ax = sns.barplot(x='shuffle', y='Proporción', hue='TARGET', data=prop_long,
                 palette=["#5DADE2", "#E74C3C"])

# Títulos y ejes
plt.title('Proporción de TARGET por uso de shuffle')
plt.xlabel('Shuffle (0 = No, 1 = Sí)')
plt.ylabel('Proporción')
plt.legend(title='TARGET', labels=['No Salta', 'Salta'])

# Mostrar porcentajes arriba de cada barra
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(p.get_x() + p.get_width() / 2.,
                height + 0.01,
                f'{height:.1%}',
                ha="center", fontsize=9)

plt.tight_layout()
#plt.show()


#GRAFICO
# Eliminar la columna 'username' si está presente
df_ohe = pd.get_dummies(df, columns=['conn_country'], prefix='country')
if 'username' in df_ohe.columns:
    df_ohe = df_ohe.drop(columns=['username'])

# Seleccionar variables numéricas o booleanas
numerical_df = df_ohe.select_dtypes(include=['int64', 'bool'])

# Calcular la matriz de correlación
correlation = numerical_df.corr()

# Crear heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(correlation,
                 cmap='coolwarm',
                 annot=False,
                 fmt=".2f",
                 linewidths=0.5,
                 cbar_kws={"shrink": 0.8},
                 square=True)

plt.title("Heatmap de correlaciones entre variables numéricas", fontsize=14)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
#plt.show()

#GRAFICO 
# Seleccionar los 20 artistas más escuchados
top_artists = df['master_metadata_album_artist_name'].value_counts().head(20).index
df_top_artistas = df[df['master_metadata_album_artist_name'].isin(top_artists)]

# One-hot encoding de artistas
df_top_artistas_ohe = pd.get_dummies(df_top_artistas, columns=['master_metadata_album_artist_name'], prefix='artist')

# Eliminar 'username' si existe
if 'username' in df_top_artistas_ohe.columns:
    df_top_artistas_ohe = df_top_artistas_ohe.drop(columns=['username'])

# Seleccionar variables numéricas
numerical_df = df_top_artistas_ohe.select_dtypes(include=['int64', 'bool'])

# Calcular correlación
correlation = numerical_df.corr()

# Crear heatmap
plt.figure(figsize=(14, 10))
ax = sns.heatmap(correlation,
                 cmap='coolwarm',
                 annot=False,
                 fmt=".2f",
                 linewidths=0.5,
                 cbar_kws={"shrink": 0.8})

plt.title("Heatmap de correlaciones entre los 20 artistas más escuchados y otras variables", fontsize=14)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
# plt.show()








