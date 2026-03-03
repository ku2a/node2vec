import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
sys.path.append(build_path)
import node2vec_cpp

df_edges = pd.read_csv('datasets/arxiv/raw/edge.csv.gz', header=None, names=['source', 'target'])

nodos_unicos = set(df_edges['source']).union(set(df_edges['target']))

mi_grafo = node2vec_cpp.Graph()

start_load = time.time()
print("Cargando grafo...")

for nodo in nodos_unicos:
    mi_grafo.add_vertex(str(nodo), [])

for row in df_edges.itertuples(index=False):
    mi_grafo.add_edge(str(row.source), str(row.target), 1.0)

end_load = time.time()
print(f"Grafo cargado. Nodos: {len(mi_grafo.get_nodes())}. Tiempo: {(end_load - start_load):.4f} segundos")

print("Inicializando SkipGram y construyendo vocabulario...")
modelo_sg = node2vec_cpp.SkipGram(N=128)

nodos = mi_grafo.get_nodes()
grados = mi_grafo.get_degrees()
print("Nodos", len(nodos))
modelo_sg.build_vocab(nodos, grados)

print("Entrenando...")
start_train = time.time()

losses = modelo_sg.train(
    graph=mi_grafo, 
    epochs=3, 
    walk_length=80, 
    p=1.0, 
    q=1.0, 
    K=5, 
    C=10, 
    starting_alpha=0.025, 
    verbose=True,
    tol=1e-3,
    patience=10,
    batch_size=1024
)

end_train = time.time()
plt.figure(figsize=(14, 10))
plt.title("Loss curve")
plt.xlabel("Batch Iterations")
plt.ylabel("Loss")
plt.plot(losses) 
plt.show()
print(losses)
modelo_sg.save_model("models/modelo_arxiv")
modelo_sg.save_embeddings_bin("outputs/embeddings/arxiv_embeddings")
print(f"Tiempo de entrenamiento SkipGram: {(end_train - start_train):.4f} segundos")