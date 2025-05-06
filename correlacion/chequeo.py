import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset
data = pd.read_csv("competition_data.csv")

# Ver cuántos artistas únicos hay
print(f"Cantidad de artistas únicos: {data['master_metadata_album_artist_name'].nunique()}")

# Agrupar por artista y calcular la tasa promedio de TARGET=1 por artista
conversion_por_artista = data.groupby('master_metadata_album_artist_name')['TARGET'].mean().sort_values(ascending=False)

# Mostrar los 10 artistas con mayor y menor tasa de TARGET=1
print("\nTop 10 artistas con mayor tasa de TARGET=1:")
print(conversion_por_artista.head(10))

print("\nTop 10 artistas con menor tasa de TARGET=1:")
print(conversion_por_artista.tail(10))

# Calcular la varianza de las tasas
varianza_conversion = conversion_por_artista.var()
print(f"\nVarianza de tasa TARGET por artista: {varianza_conversion:.5f}")

# Histograma de la distribución de tasas por artista
plt.figure(figsize=(10, 6))
conversion_por_artista.hist(bins=30, color='skyblue', edgecolor='black')
plt.title("Distribución de tasas TARGET=1 por artista")
plt.xlabel("Tasa de conversión TARGET")
plt.ylabel("Cantidad de artistas")
plt.grid(False)
plt.tight_layout()
plt.show()
