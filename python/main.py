import sys
import os

build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
sys.path.append(build_path)
import node2vec_cpp
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from  sklearn.manifold import TSNE
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "figures"
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

for country in countries:
    graph.add_vertex(country,[1])
graph.add_edge("Portugal", "Espana", 1.0)
graph.add_edge("Espana", "Francia", 1.0)
graph.add_edge("Espana", "Andorra", 1.0)
graph.add_edge("Andorra", "Francia", 1.0)
graph.add_edge("Francia", "Belgica", 1.0)
graph.add_edge("Francia", "Luxemburgo", 1.0)
graph.add_edge("Francia", "Alemania", 1.0)
graph.add_edge("Francia", "Suiza", 1.0)
graph.add_edge("Francia", "Italia", 1.0)
graph.add_edge("Belgica", "Paises Bajos", 1.0)
graph.add_edge("Belgica", "Luxemburgo", 1.0)
graph.add_edge("Belgica", "Alemania", 1.0)
graph.add_edge("Paises Bajos", "Alemania", 1.0)
graph.add_edge("Luxemburgo", "Alemania", 1.0)
graph.add_edge("Alemania", "Dinamarca", 1.0)
graph.add_edge("Alemania", "Polonia", 1.0)
graph.add_edge("Alemania", "Rep. Checa", 1.0)
graph.add_edge("Alemania", "Austria", 1.0)
graph.add_edge("Alemania", "Suiza", 1.0)
graph.add_edge("Suiza", "Austria", 1.0)
graph.add_edge("Suiza", "Italia", 1.0)
graph.add_edge("Italia", "Austria", 1.0)
graph.add_edge("Italia", "Eslovenia", 1.0)
graph.add_edge("Polonia", "Rusia", 1.0)
graph.add_edge("Polonia", "Lituania", 1.0)
graph.add_edge("Polonia", "Bielorrusia", 1.0)
graph.add_edge("Polonia", "Ucrania", 1.0)
graph.add_edge("Polonia", "Eslovaquia", 1.0)
graph.add_edge("Polonia", "Rep. Checa", 1.0)
graph.add_edge("Rep. Checa", "Eslovaquia", 1.0)
graph.add_edge("Rep. Checa", "Austria", 1.0)
graph.add_edge("Austria", "Eslovaquia", 1.0)
graph.add_edge("Austria", "Hungria", 1.0)
graph.add_edge("Austria", "Eslovenia", 1.0)
graph.add_edge("Eslovaquia", "Ucrania", 1.0)
graph.add_edge("Eslovaquia", "Hungria", 1.0)
graph.add_edge("Hungria", "Ucrania", 1.0)
graph.add_edge("Hungria", "Rumania", 1.0)
graph.add_edge("Hungria", "Serbia", 1.0)
graph.add_edge("Hungria", "Croacia", 1.0)
graph.add_edge("Hungria", "Eslovenia", 1.0)
graph.add_edge("Eslovenia", "Croacia", 1.0)
graph.add_edge("Croacia", "Serbia", 1.0)
graph.add_edge("Croacia", "Bosnia y Herzegovina", 1.0)
graph.add_edge("Croacia", "Montenegro", 1.0)
graph.add_edge("Bosnia y Herzegovina", "Serbia", 1.0)
graph.add_edge("Bosnia y Herzegovina", "Montenegro", 1.0)
graph.add_edge("Serbia", "Rumania", 1.0)
graph.add_edge("Serbia", "Bulgaria", 1.0)
graph.add_edge("Serbia", "Macedonia del Norte", 1.0)
graph.add_edge("Serbia", "Kosovo", 1.0)
graph.add_edge("Serbia", "Montenegro", 1.0)
graph.add_edge("Montenegro", "Kosovo", 1.0)
graph.add_edge("Montenegro", "Albania", 1.0)
graph.add_edge("Kosovo", "Macedonia del Norte", 1.0)
graph.add_edge("Kosovo", "Albania", 1.0)
graph.add_edge("Albania", "Macedonia del Norte", 1.0)
graph.add_edge("Albania", "Grecia", 1.0)
graph.add_edge("Macedonia del Norte", "Bulgaria", 1.0)
graph.add_edge("Macedonia del Norte", "Grecia", 1.0)
graph.add_edge("Bulgaria", "Rumania", 1.0)
graph.add_edge("Bulgaria", "Grecia", 1.0)
graph.add_edge("Rumania", "Moldavia", 1.0)
graph.add_edge("Rumania", "Ucrania", 1.0)
graph.add_edge("Moldavia", "Ucrania", 1.0)
graph.add_edge("Ucrania", "Bielorrusia", 1.0)
graph.add_edge("Ucrania", "Rusia", 1.0)
graph.add_edge("Bielorrusia", "Rusia", 1.0)
graph.add_edge("Bielorrusia", "Letonia", 1.0)
graph.add_edge("Bielorrusia", "Lituania", 1.0)
graph.add_edge("Lituania", "Letonia", 1.0)
graph.add_edge("Lituania", "Rusia", 1.0) 
graph.add_edge("Letonia", "Estonia", 1.0)
graph.add_edge("Letonia", "Rusia", 1.0)
graph.add_edge("Estonia", "Rusia", 1.0)
graph.add_edge("Rusia", "Finlandia", 1.0)
graph.add_edge("Rusia", "Noruega", 1.0)
graph.add_edge("Finlandia", "Noruega", 1.0)
graph.add_edge("Finlandia", "Suecia", 1.0)
graph.add_edge("Suecia", "Noruega", 1.0)
graph.add_edge("Reino Unido", "Irlanda", 1.0)

modelo = node2vec_cpp.SkipGram(100,False)
walks = graph.get_walks(10,15,1.0,1.0)
epochs = 10
K = 5
C = 3           
alpha = 0.025 
modelo.build_vocab(graph.get_nodes(),graph.get_degrees())
losses = modelo.train(walks,epochs,K,C,alpha,True)
embeddings = pd.DataFrame(modelo.get_embeddings())

pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)

tsne = TSNE(n_components=2, perplexity=10, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)


plt.figure(figsize=(14, 10))

x_coords = embeddings_2d[:, 0]
y_coords = embeddings_2d[:, 1]


plt.scatter(x_coords, y_coords, color='dodgerblue', s=50, edgecolors='black', alpha=0.7)


for i, pais in enumerate(countries):
    plt.annotate(
        pais,
        (x_coords[i], y_coords[i]),
        xytext=(5, 2), 
        textcoords='offset points',
        fontsize=9,
        alpha=0.8
    )


plt.title("Europe visualization with TSNE", fontsize=16, fontweight='bold')
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True, linestyle='--', alpha=0.5)
#plt.savefig(OUTPUT_DIR/"europe_map_tsne.png",dpi=300, bbox_inches='tight')

plt.show()
plt.figure(figsize=(14, 10))

x_coords = embeddings_pca[:, 0]
y_coords = embeddings_pca[:, 1]


plt.scatter(x_coords, y_coords, color='dodgerblue', s=50, edgecolors='black', alpha=0.7)


for i, pais in enumerate(countries):
    plt.annotate(
        pais,
        (x_coords[i], y_coords[i]),
        xytext=(5, 2), 
        textcoords='offset points',
        fontsize=9,
        alpha=0.8
    )

plt.title("Europe visualization with PCA", fontsize=16, fontweight='bold')
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True, linestyle='--', alpha=0.5)
#plt.savefig(OUTPUT_DIR/"europe_map_pca.png",dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(14, 10))
plt.title("Loss curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(losses)
plt.savefig(OUTPUT_DIR/"loss_curve",dpi=300, bbox_inches='tight')
plt.show()
