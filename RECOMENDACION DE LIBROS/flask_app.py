from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)

# Cargar los datos del CSV
libros_df = pd.read_csv('libros.csv', sep=';')

# Convertir los géneros en índices numéricos para el algoritmo de agrupación
libros_df['genero'] = libros_df['genero'].astype('category')
libros_df['categoría_id'] = libros_df['genero'].cat.codes

# Configuración del algoritmo KMeans
kmeans = KMeans(n_clusters=len(libros_df['genero'].unique()), random_state=0)
kmeans.fit(libros_df[['categoría_id']])

# Asociar cada libro a un cluster
libros_df['cluster'] = kmeans.labels_

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recomendar', methods=['POST'])
def recomendar():
    libro_titulo = request.form['titulo']
    
    # Obtener el género del libro ingresado
    libro = libros_df[libros_df['titulo'].str.contains(libro_titulo, case=False)]
    
    if libro.empty:
        return 'No se encontró el libro ingresado.', 404
    
    cluster_id = libro['cluster'].values[0]
    
    # Obtener recomendaciones basadas en el mismo cluster
    recomendaciones = libros_df[libros_df['cluster'] == cluster_id]
    
    # Excluir el libro ingresado de la lista de recomendaciones
    recomendaciones = recomendaciones[recomendaciones['titulo'] != libro_titulo]
    
    return render_template('recomendaciones.html', recomendaciones=recomendaciones)

if __name__ == '__main__':
    app.run(debug=True)
