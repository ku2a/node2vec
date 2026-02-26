
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from  sklearn.manifold import TSNE


def plot_embeddings(csv_filename):
    
    print(f"Cargando datos de {csv_filename}...")
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: File not found'{csv_filename}'.")
        return

    paises = df['Nodo'].values
    embeddings = df.drop('Nodo', axis=1).values


 
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)


    plt.figure(figsize=(14, 10))

    x_coords = embeddings_2d[:, 0]
    y_coords = embeddings_2d[:, 1]


    plt.scatter(x_coords, y_coords, color='dodgerblue', s=50, edgecolors='black', alpha=0.7)

   
    for i, pais in enumerate(paises):
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
    
 
    plt.savefig("../outputs/figures/europe_map_node2vec_tsne.png", dpi=300, bbox_inches='tight')

    print("Saving tsne map")
    
    plt.show()
    plt.figure(figsize=(14, 10))

    x_coords = embeddings_pca[:, 0]
    y_coords = embeddings_pca[:, 1]


    plt.scatter(x_coords, y_coords, color='dodgerblue', s=50, edgecolors='black', alpha=0.7)


    for i, pais in enumerate(paises):
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
    

    plt.savefig("../outputs/figures/europe_amp_node2vec_pca.png", dpi=300, bbox_inches='tight')
    
    print("Saving pca map")
    
    plt.show()

if __name__ == "__main__":
    plot_embeddings("../outputs/embeddings/embeddings_europa.csv")