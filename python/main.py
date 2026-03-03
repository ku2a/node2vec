import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
sys.path.append(build_path)
import node2vec_cpp

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 

graph = node2vec_cpp.Graph()

countries = [
    "Portugal", "Espana", "Andorra", "Francia", "Belgica", "Paises Bajos", 
    "Luxemburgo", "Alemania", "Suiza", "Italia", "Dinamarca", "Polonia", 
    "Rep. Checa", "Austria", "Eslovaquia", "Hungria", "Eslovenia", "Croacia", 
    "Bosnia y Herzegovina", "Serbia", "Montenegro", "Kosovo", "Albania", 
    "Macedonia del Norte", "Grecia", "Bulgaria", "Rumania", "Moldavia", 
    "Ucrania", "Bielorrusia", "Lituania", "Letonia", "Estonia", "Rusia", 
    "Finlandia", "Suecia", "Noruega", "Reino Unido", "Irlanda"
]


country_to_idx = {country: i for i, country in enumerate(countries)}

for country in countries:
    graph.add_vertex(country, [1])


edges = [
    ("Portugal", "Espana"), ("Espana", "Francia"), ("Espana", "Andorra"), ("Andorra", "Francia"),
    ("Francia", "Belgica"), ("Francia", "Luxemburgo"), ("Francia", "Alemania"), ("Francia", "Suiza"),
    ("Francia", "Italia"), ("Belgica", "Paises Bajos"), ("Belgica", "Luxemburgo"), ("Belgica", "Alemania"),
    ("Paises Bajos", "Alemania"), ("Luxemburgo", "Alemania"), ("Alemania", "Dinamarca"),
    ("Alemania", "Polonia"), ("Alemania", "Rep. Checa"), ("Alemania", "Austria"), ("Alemania", "Suiza"),
    ("Suiza", "Austria"), ("Suiza", "Italia"), ("Italia", "Austria"), ("Italia", "Eslovenia"),
    ("Polonia", "Rusia"), ("Polonia", "Lituania"), ("Polonia", "Bielorrusia"), ("Polonia", "Ucrania"),
    ("Polonia", "Eslovaquia"), ("Polonia", "Rep. Checa"), ("Rep. Checa", "Eslovaquia"),
    ("Rep. Checa", "Austria"), ("Austria", "Eslovaquia"), ("Austria", "Hungria"), ("Austria", "Eslovenia"),
    ("Eslovaquia", "Ucrania"), ("Eslovaquia", "Hungria"), ("Hungria", "Ucrania"), ("Hungria", "Rumania"),
    ("Hungria", "Serbia"), ("Hungria", "Croacia"), ("Hungria", "Eslovenia"), ("Eslovenia", "Croacia"),
    ("Croacia", "Serbia"), ("Croacia", "Bosnia y Herzegovina"), ("Croacia", "Montenegro"),
    ("Bosnia y Herzegovina", "Serbia"), ("Bosnia y Herzegovina", "Montenegro"), ("Serbia", "Rumania"),
    ("Serbia", "Bulgaria"), ("Serbia", "Macedonia del Norte"), ("Serbia", "Kosovo"), ("Serbia", "Montenegro"),
    ("Montenegro", "Kosovo"), ("Montenegro", "Albania"), ("Kosovo", "Macedonia del Norte"),
    ("Kosovo", "Albania"), ("Albania", "Macedonia del Norte"), ("Albania", "Grecia"),
    ("Macedonia del Norte", "Bulgaria"), ("Macedonia del Norte", "Grecia"), ("Bulgaria", "Rumania"),
    ("Bulgaria", "Grecia"), ("Rumania", "Moldavia"), ("Rumania", "Ucrania"), ("Moldavia", "Ucrania"),
    ("Ucrania", "Bielorrusia"), ("Ucrania", "Rusia"), ("Bielorrusia", "Rusia"), ("Bielorrusia", "Letonia"),
    ("Bielorrusia", "Lituania"), ("Lituania", "Letonia"), ("Lituania", "Rusia"), ("Letonia", "Estonia"),
    ("Letonia", "Rusia"), ("Estonia", "Rusia"), ("Rusia", "Finlandia"), ("Rusia", "Noruega"),
    ("Finlandia", "Noruega"), ("Finlandia", "Suecia"), ("Suecia", "Noruega"), ("Reino Unido", "Irlanda")
]


for src, dst in edges:
    graph.add_edge(src, dst, 1.0)


modelo = node2vec_cpp.SkipGram(100, False)
walks = graph.get_walks(15, 10, 1.0, 1.0)
epochs = 50
K = 5
C = 3           
alpha = 0.025 
p= 1.0
q = 1.0
walk_length = 10
modelo.build_vocab(graph.get_nodes(), graph.get_degrees())
losses = modelo.train(graph,epochs, walk_length ,p , q, K, C, alpha, True)
embeddings = pd.DataFrame(modelo.get_embeddings())

pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)

tsne = TSNE(n_components=2, perplexity=10, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)


plt.figure(figsize=(14, 10))
x_coords = embeddings_2d[:, 0]
y_coords = embeddings_2d[:, 1]


for src, dst in edges:
    if src in country_to_idx and dst in country_to_idx:
        i, j = country_to_idx[src], country_to_idx[dst]
        plt.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 
                 color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=1)


plt.scatter(x_coords, y_coords, color='dodgerblue', s=50, edgecolors='black', alpha=0.9, zorder=2)

for i, pais in enumerate(countries):
    plt.annotate(pais, (x_coords[i], y_coords[i]), xytext=(5, 2), 
                 textcoords='offset points', fontsize=9, alpha=0.8, zorder=3)

plt.title("Europe visualization with TSNE", fontsize=16, fontweight='bold')
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(OUTPUT_DIR / "europe_map_tsne.png", dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(14, 10))
x_coords = embeddings_pca[:, 0]
y_coords = embeddings_pca[:, 1]


for src, dst in edges:
    if src in country_to_idx and dst in country_to_idx:
        i, j = country_to_idx[src], country_to_idx[dst]
        plt.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 
                 color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=1)

plt.scatter(x_coords, y_coords, color='dodgerblue', s=50, edgecolors='black', alpha=0.9, zorder=2)

for i, pais in enumerate(countries):
    plt.annotate(pais, (x_coords[i], y_coords[i]), xytext=(5, 2), 
                 textcoords='offset points', fontsize=9, alpha=0.8, zorder=3)

plt.title("Europe visualization with PCA", fontsize=16, fontweight='bold')
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(OUTPUT_DIR / "europe_map_pca.png", dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(14, 10))
plt.title("Loss curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(losses, marker='o') 

plt.show()

modelo.save_model("models/modelo1")

