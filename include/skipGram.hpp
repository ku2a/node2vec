#pragma once
#include <vector>
#include <random>
#include "graph.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>



template <typename IDType>
class SkipGram{
    public:

        SkipGram(int N = 200, bool subsampling = false) : 
            dim_N(N),
            dim_V(0),    
            subsampling(subsampling),
            Iters(0)
        {}

        void save_model(const std::string& filename) const {
            std::ofstream out(filename, std::ios::binary);
            if (!out) {
                throw std::runtime_error("Error: No se pudo abrir el archivo para escribir: " + filename);
            }


            out.write(reinterpret_cast<const char*>(&dim_V), sizeof(int));
            out.write(reinterpret_cast<const char*>(&dim_N), sizeof(int));
            out.write(reinterpret_cast<const char*>(&Iters), sizeof(int));
            out.write(reinterpret_cast<const char*>(&subsampling), sizeof(bool));


            out.write(reinterpret_cast<const char*>(W1.data()), W1.size() * sizeof(float));
            out.write(reinterpret_cast<const char*>(W2.data()), W2.size() * sizeof(float));


            out.write(reinterpret_cast<const char*>(Frecs.data()), Frecs.size() * sizeof(int));


            size_t vocab_size = Vocab.size();
            out.write(reinterpret_cast<const char*>(&vocab_size), sizeof(size_t));
            
            for (const auto& pair : Vocab) {
               
                size_t str_len = pair.first.size();
                out.write(reinterpret_cast<const char*>(&str_len), sizeof(size_t)); 
                out.write(pair.first.data(), str_len);                               
                out.write(reinterpret_cast<const char*>(&pair.second), sizeof(int)); 
            }


            
            out.close();
        }

      
        void load_model(const std::string& filename) {
            std::ifstream in(filename, std::ios::binary);
            if (!in) {
                throw std::runtime_error("Error: No se pudo abrir el archivo para leer: " + filename);
            }


            in.read(reinterpret_cast<char*>(&dim_V), sizeof(int));
            in.read(reinterpret_cast<char*>(&dim_N), sizeof(int));
            in.read(reinterpret_cast<char*>(&Iters), sizeof(int));
            in.read(reinterpret_cast<char*>(&subsampling), sizeof(bool));

            W1.resize(dim_V * dim_N);
            in.read(reinterpret_cast<char*>(W1.data()), W1.size() * sizeof(float));

            W2.resize(dim_N * dim_V);
            in.read(reinterpret_cast<char*>(W2.data()), W2.size() * sizeof(float));

       
            Frecs.resize(dim_V);
            in.read(reinterpret_cast<char*>(Frecs.data()), Frecs.size() * sizeof(int));

            size_t vocab_size;
            in.read(reinterpret_cast<char*>(&vocab_size), sizeof(size_t));
            Vocab.clear();
            
            for (size_t i = 0; i < vocab_size; ++i) {
                size_t str_len;
                in.read(reinterpret_cast<char*>(&str_len), sizeof(size_t));
                
                std::string key(str_len, '\0'); 
                in.read(&key[0], str_len);      
                
                int value;
                in.read(reinterpret_cast<char*>(&value), sizeof(int));
                
                Vocab[key] = value;
            }


            Embeddings.assign(dim_V, std::vector<float>(dim_N, 0.0f));
            for (int i = 0; i < dim_V; ++i) {
                for (int j = 0; j < dim_N; ++j) {
                    Embeddings[i][j] = W1[i * dim_N + j];
                }
            }

            in.close();
        }

        float cosine_similarity(IDType word1, IDType word2) const {
            std::vector<float> emb1 = get_embedding(word1);
            std::vector<float> emb2 = get_embedding(word2);

            if (emb1.empty() || emb2.empty()) {
                return -2.0f; 
            }

            float dot_product = 0.0f;
            float norm1 = 0.0f;
            float norm2 = 0.0f;

            for (size_t i = 0; i < emb1.size(); ++i) {
                dot_product += emb1[i] * emb2[i];
                norm1 += emb1[i] * emb1[i];
                norm2 += emb2[i] * emb2[i];
            }

            if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;

            return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
        }

        std::vector<std::pair<IDType, float>> most_similar(IDType word, int top_k = 5) const {
            std::vector<float> target_emb = get_embedding(word);
            if (target_emb.empty()) return {};

            std::vector<std::pair<IDType, float>> similarities;
            similarities.reserve(Vocab.size());

            for (const auto& pair : Vocab) {
                if (pair.first == word) continue; 
                
                float sim = cosine_similarity(word, pair.first);
                similarities.push_back({pair.first, sim});
            }

            
            std::sort(similarities.begin(), similarities.end(), 
                [](const std::pair<IDType, float>& a, const std::pair<IDType, float>& b) {
                    return a.second > b.second;
                });

            
            if (similarities.size() > static_cast<size_t>(top_k)) {
                similarities.resize(top_k);
            }

            return similarities;
        }
        void build_vocab(const std::vector<IDType>& corpus) {
            Vocab.clear();
            Frecs.clear();
            int current_idx = 0;

            for (const auto& element : corpus) {
                if (Vocab.find(element) == Vocab.end()) {
                    Vocab[element] = current_idx++;
                    Frecs.push_back(1); 
                } else {
                    Frecs[Vocab[element]]++;
                }
            }
            
            update_network_size(Vocab.size());
        }

        void build_vocab(const std::vector<IDType>& vocab, const std::vector<int>& frecs) {
            if (vocab.size() != frecs.size()) {
                throw std::invalid_argument("Vocab length and frecs length must match");
            }

            Vocab.clear();
            Frecs.clear();
            Frecs.reserve(vocab.size());

            for (size_t i = 0; i < vocab.size(); ++i) {
                if (Vocab.find(vocab[i]) == Vocab.end()) {
                    Vocab[vocab[i]] = i;
                    Frecs.push_back(frecs[i]);
                }
            }

            update_network_size(Vocab.size());
        }
       


    std::vector<float> train(std::vector<std::vector<IDType>>& walks, int epochs, int K, int C, float starting_alpha, bool verbose) {
        std::vector<float> mean_losses;
        std::uniform_int_distribution<int> window_dis(1, C);
        std::uniform_real_distribution<float> prob_dis(0.0f, 1.0f);
        std::uniform_int_distribution<int> unigram_dis(0, table_size - 1);
        std::uniform_int_distribution<int> fallback_dis(0, dim_V - 1);

        long long total_words = 0;
        for (const auto& walk : walks) {
            total_words += walk.size();
        }
        long long total_iters = epochs * total_words;
        long long current_iter = 0;

        std::vector<float> p_discard(dim_V, 0.0f);
        if (subsampling && total_words > 0) {
            const float t = 1e-5f;
            for (int i = 0; i < dim_V; i++) {
                float fw = static_cast<float>(Frecs[i]) / static_cast<float>(total_words);
                float p_keep = (std::sqrt(fw / t) + 1.0f) * (t / fw);
                p_discard[i] = std::max(0.0f, 1.0f - p_keep);
            }
        }

        std::vector<float> out(K + 1, 0.0f);
        std::vector<int> negatives(K + 1, 0);
        std::vector<float> diff_W1(dim_N, 0.0f);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            
            std::shuffle(walks.begin(), walks.end(), gen); 

            float epoch_loss = 0.0f;
            int iter_count = 0;

            for (const auto& current_walk : walks) {
                int walk_size = static_cast<int>(current_walk.size());

                for (int i = 0; i < walk_size; i++) {
                    
                    float alpha = starting_alpha * (1.0f - static_cast<float>(current_iter) / total_iters);
                    if (alpha < starting_alpha * 0.0001f) alpha = starting_alpha * 0.0001f;
                    current_iter++;

                    auto it_i = Vocab.find(current_walk[i]);
                    if (it_i == Vocab.end()) continue; 
                    int word_idx = it_i->second;

                    if (subsampling && prob_dis(gen) < p_discard[word_idx]) {
                        continue; 
                    }

                    int R = window_dis(gen); 
                    int left = std::max(0, i - R);
                    int right = std::min(walk_size, i + R + 1);

                    for (int j = left; j < right; j++) {
                        if (j == i) continue;

                        auto it_j = Vocab.find(current_walk[j]);
                        if (it_j == Vocab.end()) continue;
                        int target_idx = it_j->second;

                        negatives[0] = target_idx; 
                        for (int k = 1; k <= K; ++k) {
                            int random_idx = unigram_dis(gen);
                            int negative_word = unigram_table[random_idx];
                            if (negative_word == target_idx) {
                                negative_word = fallback_dis(gen);
                            }
                            negatives[k] = negative_word;
                        }

                        for (int n = 0; n <= K; ++n) {
                            int target = negatives[n];
                            float sum = 0.0f;
                            for (int d = 0; d < dim_N; ++d) {
                                sum += W1[word_idx * dim_N + d] * W2[target * dim_N + d];
                            }
                            out[n] = 1.0f / (1.0f + std::exp(-sum)); 
                        }

                        float epsilon = 1e-7f;
                        epoch_loss -= std::log(std::max(out[0], epsilon)); 
                        for (int k = 1; k <= K; ++k) {
                            epoch_loss -= std::log(std::max(1.0f - out[k], epsilon));
                        }
                        iter_count++;
                        Iters++;

                        std::fill(diff_W1.begin(), diff_W1.end(), 0.0f);
                        
                        for (int n = 0; n <= K; ++n) {
                            int target = negatives[n];
                            float err = (n == 0) ? (out[n] - 1.0f) : out[n];
                            
                            for (int d = 0; d < dim_N; ++d) {
                                diff_W1[d] += err * W2[target * dim_N + d];
                                W2[target * dim_N + d] -= alpha * err * W1[word_idx * dim_N + d];
                            }
                        }

                        for (int d = 0; d < dim_N; ++d) {
                            W1[word_idx * dim_N + d] -= alpha * diff_W1[d];
                        }
                    }
                }
            } 
            
            if (iter_count > 0) {
                float avg = epoch_loss / static_cast<float>(iter_count);
                mean_losses.push_back(avg);
                if (verbose) {
                    std::cout << "Epoch " << (epoch + 1) << "/" << epochs << " loss: " << avg << std::endl;
                }
            }
        } 

        return mean_losses;
    }

        void clear(){
            /* Clear the current weights */
            if (dim_V == 0) return; 

            std::fill(W2.begin(), W2.end(), 0.0f);
            
            std::uniform_real_distribution<float> current_dis(-0.5f / dim_N, 0.5f / dim_N);
            for (int i = 0; i < dim_V * dim_N; ++i) {
                W1[i] = current_dis(gen);
            }
        }

        std::vector<float> get_embedding(IDType word) const {
            auto it = Vocab.find(word);


            if (it == Vocab.end()) {
                return {}; 
            }
            int word_idx = it->second;
            auto start = W1.begin() + (word_idx * dim_N);
            auto end = start + dim_N;
            return std::vector<float>(start, end);
        }

        std::vector<std::vector<float>> get_embeddings() const {
            std::vector<std::vector<float>> all_embeddings;
            all_embeddings.reserve(dim_V);
            for (int i = 0; i < dim_V; ++i) {
                auto start = W1.begin() + (i * dim_N);
                auto end = start + dim_N;
                all_embeddings.push_back(std::vector<float>(start, end));
            }
            return all_embeddings;
        }
        
    private:
        //We use flattened matrix's for better optimization 
        std::vector<float> W1; 
        std::vector<float> W2; 
        int dim_N; 
        int dim_V; 
        std::unordered_map<IDType, int> Vocab;
        std::vector<int> Frecs;
        std::vector<std::vector<float>> Embeddings;
        int Iters;
        bool subsampling;
        mutable std::mt19937 gen{std::random_device{}()};

        
        std::vector<int> unigram_table;
        const int table_size = 1e7;
        
        void init_unigram_table() {
            unigram_table.resize(table_size);
            double train_words_pow = 0.0;
            const double power = 0.75;

            
            for (int frec : Frecs) {
                train_words_pow += std::pow(frec, power);
            }

           
            int vocab_idx = 0;
            double d1 = std::pow(Frecs[vocab_idx], power) / train_words_pow;
            
            for (int a = 0; a < table_size; a++) {
                unigram_table[a] = vocab_idx;
                

                if (a / (double)table_size > d1) {
                    vocab_idx++;

                    if (vocab_idx >= dim_V) {
                        vocab_idx = dim_V - 1; 
                    }
                    d1 += std::pow(Frecs[vocab_idx], power) / train_words_pow;
                }
            }
        }

        void update_network_size(int new_V) {
            dim_V = new_V;
            
            
            W1.assign(dim_V * dim_N, 0.0f);
            W2.assign(dim_V * dim_N, 0.0f);
            
            
            std::uniform_real_distribution<float> current_dis(-0.5f / dim_N, 0.5f / dim_N);
            for (int i = 0; i < dim_V * dim_N; ++i) {
                W1[i] = current_dis(gen);
            }

            if (dim_V > 0) {
                init_unigram_table();
            }

        }
        
        std::vector<float> forward(int word, const std::vector<int>& target_words) const {
            /*
            For the loss calculations: target_words[0] will be the positive word while the rest will be the negative.
            */
                       
            std::vector<float> out;
            out.reserve(target_words.size());
            for (const int target: target_words){
                float sum = 0;
                for(int i = 0; i<dim_N; i++){
                    sum += W1[word*dim_N + i] * W2[target*dim_N +i]; 
                }
                out.push_back(sigmoid(sum));
            }
            return out;
        }

        float sigmoid(float num) const{
            return (1.0f / (1.0f + std::exp(-num)));
        }

        void backward(const std::vector<float>& out, int word, const std::vector<int>&  target_words, float alpha){
            std::vector<float> diff_W1( dim_N, 0.0f);

            //update W2 and calculate W1_diff
            for (size_t i = 0; i<target_words.size(); i++){
                int target = target_words[i];
                float err = (i==0) ? (out[i] - 1.0f) : (out[i]);
                
                for (int j = 0; j < dim_N; j++){
                    diff_W1[j] += err*W2[target*dim_N + j];
                    W2[target*dim_N + j] -= alpha * err * W1[word*dim_N + j];
                }

            }

            for (int i = 0; i<dim_N; i++){
                W1[word*dim_N+ i] -= alpha * diff_W1[i];  
            }

        }


        float loss(const std::vector<float>& out) const {
            float epsilon = 1e-7f; 
            float loss = 0.0f;

            float p_pos = std::max(out[0], epsilon);
            loss -= std::log(p_pos);

            for (size_t i = 1; i < out.size(); i++) {
                float p_neg = std::max(1.0f - out[i], epsilon);
                loss -= std::log(p_neg);
            }

            return loss;
        }


        std::vector<int> get_negatives(IDType target, int K) const {
            std::vector<int> words;
            words.reserve(K + 1);
            
            int target_idx = Vocab.at(target);
            words.push_back(target_idx); 

            std::uniform_int_distribution<int> unigram_dis(0, table_size - 1);

            for (int i = 0; i < K; ++i) {
                int random_idx = unigram_dis(gen);
                int negative_word = unigram_table[random_idx];


                if (negative_word == target_idx) {

                    std::uniform_int_distribution<int> fallback_dis(0, dim_V - 1);
                    negative_word = fallback_dis(gen);
                }
                
                words.push_back(negative_word);
            }

            return words;
        }
        

};
