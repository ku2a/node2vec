#pragma once
#include <vector>
#include <random>
#include "graph.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <omp.h>
#include <atomic>
#include <limits>

template <typename IDType>
class SkipGram{
    public:
        void save_embeddings_bin(const std::string& filename) const {
            std::ofstream out(filename, std::ios::binary);
            if (!out) {
                throw std::runtime_error("No se pudo abrir el archivo: " + filename);
            }
            out.write(reinterpret_cast<const char*>(&dim_V), sizeof(int));
            out.write(reinterpret_cast<const char*>(&dim_N), sizeof(int));
            out.write(reinterpret_cast<const char*>(W1.data()), W1.size() * sizeof(float));

            out.close();
        }
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
            auto it1 = Vocab.find(word1);
            auto it2 = Vocab.find(word2);

            if (it1 == Vocab.end() || it2 == Vocab.end()) {
                return -2.0f; 
            }

            int idx1 = it1->second * dim_N;
            int idx2 = it2->second * dim_N;

            float dot_product = 0.0f;
            float norm1 = 0.0f;
            float norm2 = 0.0f;

            for (int i = 0; i < dim_N; ++i) {
                float v1 = W1[idx1 + i];
                float v2 = W1[idx2 + i];
                dot_product += v1 * v2;
                norm1 += v1 * v1;
                norm2 += v2 * v2;
            }

            if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;

            return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
        }

        std::vector<std::pair<IDType, float>> most_similar(IDType word, int top_k = 5) const {
            auto it_target = Vocab.find(word);
            if (it_target == Vocab.end()) return {};
            
            int target_idx = it_target->second * dim_N;
            float norm_target = 0.0f;
            
            for (int i = 0; i < dim_N; ++i) {
                norm_target += W1[target_idx + i] * W1[target_idx + i];
            }
            
            if (norm_target == 0.0f) return {};
            
            float inv_norm_target = 1.0f / std::sqrt(norm_target);
            std::vector<std::pair<IDType, float>> similarities;
            similarities.reserve(Vocab.size());

            for (const auto& pair : Vocab) {
                if (pair.second == it_target->second) continue;

                int w_idx = pair.second * dim_N;
                float dot_product = 0.0f;
                float norm_w = 0.0f;

                for (int i = 0; i < dim_N; ++i) {
                    float val = W1[w_idx + i];
                    dot_product += W1[target_idx + i] * val;
                    norm_w += val * val;
                }

                if (norm_w > 0.0f) {
                    float sim = (dot_product * inv_norm_target) / std::sqrt(norm_w);
                    similarities.push_back({pair.first, sim});
                }
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

        template <typename ContentType>
        std::vector<float> train(Graph<IDType, ContentType>& graph, int epochs, int walk_length, float p, float q, int K, int C, float starting_alpha, bool verbose, int batch_size = 1024, float tol = 1e-4f, int patience = 10) {
            
            if (dim_V <= 0) {
                throw std::runtime_error("Error: Vocabulario vacio.");
            }

            std::vector<float> mean_losses;
            long long num_nodes = static_cast<long long>(graph.get_nodes().size());
            long long total_words = num_nodes * epochs * walk_length;
            std::atomic<long long> shared_iter{0};

            std::vector<float> p_discard(dim_V, 0.0f);
            if (subsampling && total_words > 0) {
                const float t = 1e-5f;
                long long corpus_words = 0;
                for(int f : Frecs) corpus_words += f;
                for (int i = 0; i < dim_V; i++) {
                    float fw = static_cast<float>(Frecs[i]) / static_cast<float>(std::max(1LL, corpus_words));
                    float p_keep = (std::sqrt(fw / t) + 1.0f) * (t / fw);
                    p_discard[i] = std::max(0.0f, 1.0f - p_keep);
                }
            }

            float best_loss = std::numeric_limits<float>::infinity();
            int no_improve_batches = 0;
            bool early_stop = false;

            for (int epoch = 0; epoch < epochs; ++epoch) {
                if (early_stop) break;

                auto iter = graph.get_walks_iter(1, walk_length, p, q);
                long long processed_walks = 0;

                while (true) {
                    std::vector<IDType> flat_walks;
                    try {
                        flat_walks = iter.next_batch(batch_size);
                    } catch (const std::runtime_error&) {
                        break;
                    }

                    int num_walks = flat_walks.size() / walk_length;
                    float batch_loss_accum = 0.0f;
                    int batch_iter_count = 0;

                    #pragma omp parallel reduction(+:batch_loss_accum, batch_iter_count)
                    {
                        std::mt19937 local_gen(std::random_device{}() ^ omp_get_thread_num());
                        std::uniform_int_distribution<int> window_dis(1, std::max(1, C));
                        std::uniform_real_distribution<float> prob_dis(0.0f, 1.0f);
                        std::uniform_int_distribution<int> unigram_dis(0, table_size - 1);
                        std::uniform_int_distribution<int> fallback_dis(0, std::max(0, dim_V - 1));

                        std::vector<float> out(K + 1, 0.0f);
                        std::vector<int> negatives(K + 1, 0);
                        std::vector<float> diff_W1(dim_N, 0.0f);

                        #pragma omp for schedule(dynamic)
                        for (int w = 0; w < num_walks; ++w) {
                            int walk_start = w * walk_length;
                            long long local_iter_increment = 0;

                            for (int i = 0; i < walk_length; i++) {
                                long long current_iter_val = shared_iter.load(std::memory_order_relaxed);
                                float alpha = starting_alpha * (1.0f - static_cast<float>(current_iter_val) / total_words);
                                if (alpha < starting_alpha * 0.0001f) alpha = starting_alpha * 0.0001f;
                                
                                local_iter_increment++;

                                IDType current_word = flat_walks[walk_start + i];
                                auto it_i = Vocab.find(current_word);
                                if (it_i == Vocab.end()) continue; 
                                int word_idx = it_i->second;

                                if (subsampling && prob_dis(local_gen) < p_discard[word_idx]) {
                                    continue; 
                                }

                                int R = window_dis(local_gen); 
                                int left = std::max(0, i - R);
                                int right = std::min(walk_length, i + R + 1);

                                for (int j = left; j < right; j++) {
                                    if (j == i) continue;

                                    auto it_j = Vocab.find(flat_walks[walk_start + j]);
                                    if (it_j == Vocab.end()) continue;
                                    int target_idx = it_j->second;

                                    negatives[0] = target_idx; 
                                    for (int k = 1; k <= K; ++k) {
                                        int random_idx = unigram_dis(local_gen);
                                        int negative_word = unigram_table[random_idx];
                                        if (negative_word == target_idx) {
                                            negative_word = fallback_dis(local_gen);
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
                                    batch_loss_accum -= std::log(std::max(out[0], epsilon)); 
                                    for (int k = 1; k <= K; ++k) {
                                        batch_loss_accum -= std::log(std::max(1.0f - out[k], epsilon));
                                    }
                                    batch_iter_count++;

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
                            shared_iter.fetch_add(local_iter_increment, std::memory_order_relaxed);
                        }
                    } 

                    if (batch_iter_count > 0) {
                        float current_batch_loss = batch_loss_accum / static_cast<float>(batch_iter_count);
                        
                        // Guardamos la pérdida de este lote específico
                        mean_losses.push_back(current_batch_loss);
                        Iters += batch_iter_count;

                        // Lógica de Early Stopping
                        if (current_batch_loss < best_loss - tol) {
                            best_loss = current_batch_loss;
                            no_improve_batches = 0;
                        } else {
                            no_improve_batches++;
                        }

                        if (verbose) {
                            processed_walks += num_walks;
                            float progress = (static_cast<float>(processed_walks) / static_cast<float>(num_nodes)) * 100.0f;
                            std::printf("\rEpoch %d/%d | Progreso: %.2f%% | Batch Loss: %.6f", epoch + 1, epochs, progress, current_batch_loss);
                            std::fflush(stdout);
                        }

                        if (no_improve_batches >= patience) {
                            if (verbose) std::printf("\nEarly stopping en lote %zu\n", mean_losses.size());
                            early_stop = true;
                            break;
                        }
                    }
                }
                if (verbose && !early_stop) std::printf("\n");
            }
            
            return mean_losses;
        }

        void clear(){
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
            double current_prob = std::pow(Frecs[vocab_idx], power) / train_words_pow;
            
            for (int a = 0; a < table_size; a++) {
                unigram_table[a] = vocab_idx;
                
                double target = (double)(a + 1) / table_size;
                
                while (target > current_prob && vocab_idx < dim_V - 1) {
                    vocab_idx++;
                    current_prob += std::pow(Frecs[vocab_idx], power) / train_words_pow;
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