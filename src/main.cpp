#include "graph.hpp"
#include "skipGram.hpp"
#include <iostream>
#include <fstream>

void load_europe(Graph<std::string, std::vector<int>>& graph) {
    std::vector<std::string> countries = {
        "Portugal", "Espana", "Andorra", "Francia", "Belgica", "Paises Bajos", 
        "Luxemburgo", "Alemania", "Suiza", "Italia", "Dinamarca", "Polonia", 
        "Rep. Checa", "Austria", "Eslovaquia", "Hungria", "Eslovenia", "Croacia", 
        "Bosnia y Herzegovina", "Serbia", "Montenegro", "Kosovo", "Albania", 
        "Macedonia del Norte", "Grecia", "Bulgaria", "Rumania", "Moldavia", 
        "Ucrania", "Bielorrusia", "Lituania", "Letonia", "Estonia", "Rusia", 
        "Finlandia", "Suecia", "Noruega", "Reino Unido", "Irlanda"
    };

    for (const auto& country : countries) {
        graph.add_vertex(country, {});
    }

    graph.add_edge("Portugal", "Espana", 1.0f);
    graph.add_edge("Espana", "Francia", 1.0f);
    graph.add_edge("Espana", "Andorra", 1.0f);
    graph.add_edge("Andorra", "Francia", 1.0f);
    graph.add_edge("Francia", "Belgica", 1.0f);
    graph.add_edge("Francia", "Luxemburgo", 1.0f);
    graph.add_edge("Francia", "Alemania", 1.0f);
    graph.add_edge("Francia", "Suiza", 1.0f);
    graph.add_edge("Francia", "Italia", 1.0f);
    graph.add_edge("Belgica", "Paises Bajos", 1.0f);
    graph.add_edge("Belgica", "Luxemburgo", 1.0f);
    graph.add_edge("Belgica", "Alemania", 1.0f);
    graph.add_edge("Paises Bajos", "Alemania", 1.0f);
    graph.add_edge("Luxemburgo", "Alemania", 1.0f);
    graph.add_edge("Alemania", "Dinamarca", 1.0f);
    graph.add_edge("Alemania", "Polonia", 1.0f);
    graph.add_edge("Alemania", "Rep. Checa", 1.0f);
    graph.add_edge("Alemania", "Austria", 1.0f);
    graph.add_edge("Alemania", "Suiza", 1.0f);
    graph.add_edge("Suiza", "Austria", 1.0f);
    graph.add_edge("Suiza", "Italia", 1.0f);
    graph.add_edge("Italia", "Austria", 1.0f);
    graph.add_edge("Italia", "Eslovenia", 1.0f);
    graph.add_edge("Polonia", "Rusia", 1.0f);
    graph.add_edge("Polonia", "Lituania", 1.0f);
    graph.add_edge("Polonia", "Bielorrusia", 1.0f);
    graph.add_edge("Polonia", "Ucrania", 1.0f);
    graph.add_edge("Polonia", "Eslovaquia", 1.0f);
    graph.add_edge("Polonia", "Rep. Checa", 1.0f);
    graph.add_edge("Rep. Checa", "Eslovaquia", 1.0f);
    graph.add_edge("Rep. Checa", "Austria", 1.0f);
    graph.add_edge("Austria", "Eslovaquia", 1.0f);
    graph.add_edge("Austria", "Hungria", 1.0f);
    graph.add_edge("Austria", "Eslovenia", 1.0f);
    graph.add_edge("Eslovaquia", "Ucrania", 1.0f);
    graph.add_edge("Eslovaquia", "Hungria", 1.0f);
    graph.add_edge("Hungria", "Ucrania", 1.0f);
    graph.add_edge("Hungria", "Rumania", 1.0f);
    graph.add_edge("Hungria", "Serbia", 1.0f);
    graph.add_edge("Hungria", "Croacia", 1.0f);
    graph.add_edge("Hungria", "Eslovenia", 1.0f);
    graph.add_edge("Eslovenia", "Croacia", 1.0f);
    graph.add_edge("Croacia", "Serbia", 1.0f);
    graph.add_edge("Croacia", "Bosnia y Herzegovina", 1.0f);
    graph.add_edge("Croacia", "Montenegro", 1.0f);
    graph.add_edge("Bosnia y Herzegovina", "Serbia", 1.0f);
    graph.add_edge("Bosnia y Herzegovina", "Montenegro", 1.0f);
    graph.add_edge("Serbia", "Rumania", 1.0f);
    graph.add_edge("Serbia", "Bulgaria", 1.0f);
    graph.add_edge("Serbia", "Macedonia del Norte", 1.0f);
    graph.add_edge("Serbia", "Kosovo", 1.0f);
    graph.add_edge("Serbia", "Montenegro", 1.0f);
    graph.add_edge("Montenegro", "Kosovo", 1.0f);
    graph.add_edge("Montenegro", "Albania", 1.0f);
    graph.add_edge("Kosovo", "Macedonia del Norte", 1.0f);
    graph.add_edge("Kosovo", "Albania", 1.0f);
    graph.add_edge("Albania", "Macedonia del Norte", 1.0f);
    graph.add_edge("Albania", "Grecia", 1.0f);
    graph.add_edge("Macedonia del Norte", "Bulgaria", 1.0f);
    graph.add_edge("Macedonia del Norte", "Grecia", 1.0f);
    graph.add_edge("Bulgaria", "Rumania", 1.0f);
    graph.add_edge("Bulgaria", "Grecia", 1.0f);
    graph.add_edge("Rumania", "Moldavia", 1.0f);
    graph.add_edge("Rumania", "Ucrania", 1.0f);
    graph.add_edge("Moldavia", "Ucrania", 1.0f);
    graph.add_edge("Ucrania", "Bielorrusia", 1.0f);
    graph.add_edge("Ucrania", "Rusia", 1.0f);
    graph.add_edge("Bielorrusia", "Rusia", 1.0f);
    graph.add_edge("Bielorrusia", "Letonia", 1.0f);
    graph.add_edge("Bielorrusia", "Lituania", 1.0f);
    graph.add_edge("Lituania", "Letonia", 1.0f);
    graph.add_edge("Lituania", "Rusia", 1.0f); 
    graph.add_edge("Letonia", "Estonia", 1.0f);
    graph.add_edge("Letonia", "Rusia", 1.0f);
    graph.add_edge("Estonia", "Rusia", 1.0f);
    graph.add_edge("Rusia", "Finlandia", 1.0f);
    graph.add_edge("Rusia", "Noruega", 1.0f);
    graph.add_edge("Finlandia", "Noruega", 1.0f);
    graph.add_edge("Finlandia", "Suecia", 1.0f);
    graph.add_edge("Suecia", "Noruega", 1.0f);
    graph.add_edge("Reino Unido", "Irlanda", 1.0f);
}

int main() {
    Graph<std::string, std::vector<int>> graph;
    load_europe(graph);

    std::cout << "Generating Random Walks..." << std::endl;
    
    int num_walks_per_node = 10;
    int walk_length = 15;      
    std::vector<std::vector<std::string>> all_walks = graph.get_walks(num_walks_per_node, walk_length, 1.0f, 1.0f);

    std::vector<std::string> flat_corpus;
    for (const auto& walk : all_walks) {
        for (const auto& node : walk) {
            flat_corpus.push_back(node);
        }
    }

    int vector_dimension = 16;  
    bool use_subsampling = false;

    SkipGram<std::string> model(vector_dimension, use_subsampling);
    model.build_vocab(flat_corpus);

    int epochs = 10;      
    int K = 5;            
    int C = 3;            
    float alpha = 0.025f; 

    std::cout << "Starting training..." << std::endl;
    model.train(graph, epochs,15,1.0f,1.0f, K, C, alpha, true);

    std::vector<float> emb_spain = model.get_embedding("Espana");
    
    std::cout << "\nResulting embedding for Espana:" << std::endl;
    for (float val : emb_spain) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nTop 3 most similar countries to 'Alemania':" << std::endl;
    auto neighbors_germany = model.most_similar("Alemania", 3);
    for (const auto& pair : neighbors_germany) {
        std::cout << "- " << pair.first << " (Similarity: " << pair.second << ")" << std::endl;
    }
    
    std::string filename = "../outputs/embeddings/embeddings_europa.csv";
    std::cout << "\nSaving embeddings to " << filename << "..." << std::endl;
    
    std::ofstream out_file(filename);
    if (out_file.is_open()) {
        out_file << "Node";
        for (int i = 0; i < vector_dimension; ++i) {
            out_file << ",D" << (i + 1);
        }
        out_file << "\n";

        for (const auto& node : graph.get_nodes()) {
            std::vector<float> emb = model.get_embedding(node);
            
            if (!emb.empty()) {
                out_file << node;
                for (float val : emb) {
                    out_file << "," << val;
                }
                out_file << "\n";
            }
        }
        out_file.close();
        std::cout << "File saved." << std::endl;
    } else {
        std::cerr << "Error opening file " << filename << " for writing." << std::endl;
    }

    return 0;
}