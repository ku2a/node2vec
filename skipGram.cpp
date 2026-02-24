#pragma once
#include <vector>
#include <random>
#include "graph.hpp"
template <typename IDType>
class SkipGram{
    public:

        SkipGram(int N, int M, int V) : 
        W1(N * M),  
        W2(N * M ),
        dim_N(N),
        dim_M(M),
        dim_V(V)
        {
            /* PRE: 
                    - N: Vocabulary length
                    - V: Desired  dimensions for embedding
                    - neurons: number of neurons on the hidden layer. Reccomendations: 
                        · 50-100: Small corpuses, avoid overfitting.
                        · 300: Standard size for big corpuses.
                        · 300-500: Massive corpuses or rich vocabularies.

            */
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-0.5f / M, 0.5f / M);

            for (int i = 0; i < N*M; ++i) {
                    W1[i] = dis(gen);
            }
        }

        void set_vocab(const std::vector<IDtype>& vocab){
            Vocab.reserve(vocab.size());
            for (int i = 0; i<vocab.size(); i++ ){
                IDType word = vocab[i];
                if (Vocab.find(word) == Vocab.end()){
                    Vocab[word] = i;
                }
            }
        }

        

    private:
        //We use flattened matrix's for better optimization 
        std::vector<float> W1; // N x M Matrix
        std::vector<float> W2; // V x M Matrix
        int dim_N; // Vocabulary
        int dim_M; // Hidden layer length
        int dim_V; // Embedded vector dimensions
        std::unordered_map<IDType, int> Vocab;
        std::vector<std::vector<float>> embeddings;

        std::vector<float> forward(IDType word) const {
            auto it = vocab.find(word);
            if (it == vocab.end()){
                std::cout << "Word not found in vocabulary: " << word << std::endl;
                return {};
            }

            std::vector<float> out;
            out.reserve(dim_V);
            int index = it->second;

            for (int v = 0; v < dim_V; v++) {
                float sum = 0.0f;
                for (int m = 0; m < dim_M; m++) {
                    sum += W1[index * dim_M + m] * W2[v * dim_M + m];
                }
                out.push_back(sum);
                }

            return out;
            
        }

        void backward() {}
        void loss(){}




};
