import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


print(tf.config.list_physical_devices('GPU'))
import sys
import os
build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
sys.path.append(build_path)


import node2vec_cpp

df = pd.read_csv('datasets/arxiv/raw/edge.csv.gz',header=None,names=['source', 'target']) 
aristas_ordenadas = np.sort(df[['source', 'target']].values, axis=1)
df_filtrado = pd.DataFrame(aristas_ordenadas, columns=['source', 'target']).drop_duplicates()
dff = df_filtrado.reset_index(drop=True)



nodos_totales = list(set(dff['source']).union(set(dff['target'])))


edges_train, edges_test = train_test_split(dff.values, test_size=0.4, random_state=42)

print(f"Total nodos únicos: {len(nodos_totales)}")
print(f"Enlaces Train: {len(edges_train)} | Enlaces Test: {len(edges_test)}")

graph = node2vec_cpp.Graph()
print("metiendo nodos")
for nodo in nodos_totales:
    graph.add_vertex(str(nodo), [])

print("metiendo edges")
for row in dff.itertuples(index=False):
    graph.add_edge(str(row.source), str(row.target), 1.0)
    
print("completado")
EMBEDDING_DIM = 128
sg = node2vec_cpp.SkipGram(N=EMBEDDING_DIM)
nodes = graph.get_nodes()
degrees = graph.get_degrees()
degrees = [x + 1 for x in degrees]
print("vocab")
sg.build_vocab(nodes,degrees)
print("Entrenando Node2Vec...")
sg.train(
    graph=graph, 
    epochs=2, 
    walk_length=80, 
    p=1.0, 
    q=1.0, 
    K=10, 
    C=10, 
    starting_alpha=0.025, 
    verbose=True,
    batch_size=1024,
    tol=1e-4,
    patience=3
)

def generate_negative_edges(nodes_list, all_true_edges, num_edges):
    negative_edges = []
    # Usamos un set con TODOS los enlaces (train+test) para asegurarnos 
    # de no generar como negativo un enlace que realmente existe en Test.
    true_edges_set = set(map(tuple, all_true_edges)) 
    
    while len(negative_edges) < num_edges:
        u, v = random.sample(nodes_list, 2)
        if u != v and (u, v) not in true_edges_set and (v, u) not in true_edges_set:
            negative_edges.append((u, v))
            true_edges_set.add((u, v)) # Evita duplicados en los propios negativos
    return negative_edges

print("Generando enlaces falsos (negativos)...")
# Generamos la misma cantidad de negativos que de positivos para tener clases balanceadas (50/50)
neg_train = generate_negative_edges(nodos_totales, df.values, len(edges_train))
neg_test = generate_negative_edges(nodos_totales, df.values, len(edges_test))




def get_edge_embeddings(edge_list, skipgram_model):
    embeddings = []
    for u, v in edge_list:
        try:
            # Obtenemos el embedding de cada nodo
            emb_u = np.array(skipgram_model.get_embedding(str(u)))
            emb_v = np.array(skipgram_model.get_embedding(str(v)))
            
            # Combinamos ambos nodos multiplicándolos elemento a elemento (Hadamard)
            edge_emb = emb_u * emb_v 
            embeddings.append(edge_emb)
        except Exception:
            # Si por algún motivo extremo falla (ej. el nodo devolvió error en C++),
            # devolvemos un vector de ceros para no descuadrar el array
            embeddings.append(np.zeros(EMBEDDING_DIM))
            
    return np.array(embeddings)

print("Calculando embeddings de los enlaces...")
X_train_pos = get_edge_embeddings(edges_train, sg)
X_train_neg = get_edge_embeddings(neg_train, sg)
# Apilamos positivos y negativos
X_train = np.vstack((X_train_pos, X_train_neg))
# Creamos las etiquetas: 1 para positivo, 0 para negativo
y_train = np.hstack((np.ones(len(X_train_pos)), np.zeros(len(X_train_neg))))

X_test_pos = get_edge_embeddings(edges_test, sg)
X_test_neg = get_edge_embeddings(neg_test, sg)
X_test = np.vstack((X_test_pos, X_test_neg))
y_test = np.hstack((np.ones(len(X_test_pos)), np.zeros(len(X_test_neg))))

# Mezclar aleatoriamente el set de entrenamiento
indices_train = np.arange(len(X_train))
np.random.shuffle(indices_train)
X_train = X_train[indices_train]
y_train = y_train[indices_train]



print("Construyendo y entrenando la Red Neuronal...")

model = Sequential([
    # Capa de entrada (128 dimensiones por el tamaño del embedding)
    Dense(64, activation='relu', input_shape=(EMBEDDING_DIM,)),
    Dropout(0.3),
    
    # Capa oculta
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    # Capa de salida: Clasificación binaria (0 o 1)
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy', 'AUC']
)

# Entrenar modelo
history = model.fit(
    X_train, 
    y_train, 
    epochs=15, 
    batch_size=64, 
    validation_split=0.1, 
    verbose=1
)


print("\nEvaluando en el set de Test...")
# Keras devuelve probabilidades continuas entre 0 y 1
y_pred_proba = model.predict(X_test).ravel() 
# Convertimos las probabilidades a 0 o 1 (usando 0.5 como umbral de decisión)
y_pred = (y_pred_proba > 0.5).astype(int) 

auc_score = roc_auc_score(y_test, y_pred_proba)
acc_score = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"ROC AUC Score : {auc_score:.4f}")
print(f"Accuracy      : {acc_score:.4f}")
print("-" * 30)

