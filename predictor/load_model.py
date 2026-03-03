import sys
import os
import time
import pandas as pd

build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
sys.path.append(build_path)
import node2vec_cpp

modelo = node2vec_cpp.SkipGram()
modelo.load_model("models/modelo_arxiv")
print(len(modelo.get_embeddings()))
modelo.save_embeddings_bin("outputs/embeddings/arxiv_embeddings")