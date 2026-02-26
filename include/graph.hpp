#pragma once
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>


struct Edge {
  float weight;
  int dest;
  bool operator==(const Edge &otro) const {
    return (dest == otro.dest) && (weight == otro.weight);
  }
};

struct AliasTable {
  std::vector<int> J;
  std::vector<float> q;
};

template <typename IDType, typename ContentType> class Graph {
public:
  std::vector<IDType> get_adyacent(IDType ID) const {
    auto it = IDs.find(ID);
    if (it == IDs.end()) {
      printf("ID not found in graph\n");
      return {};
    }
    int index = it->second;
    const std::vector<Edge> &vecinos = Adyacencias[index];
    std::vector<IDType> punts;
    punts.reserve(vecinos.size());
    for (const Edge &arista : vecinos) {
      punts.push_back(ReverseIDs[arista.dest]);
    }
    
    return punts;
  }

  void add_vertex(IDType ID, ContentType content) {
    if (IDs.find(ID) != IDs.end()) {
      printf("Vertex already in graph");
      return;
    }
    int indice;
    if (free.empty()) {
      indice = Adyacencias.size();
      Adyacencias.push_back({});
      Contents.push_back(content);
      ReverseIDs.push_back(ID);
    } else {
      indice = free.back();
      free.pop_back();
      Adyacencias[indice] = {};
      Contents[indice] = content;
      ReverseIDs[indice] = ID;
    }
    IDs[ID] = indice;
  }

  void add_edge(IDType vertex1, IDType vertex2, float weight) {
    if (IDs.find(vertex1) == IDs.end()) {
      printf("Vertex1 not foud in graph");
      return;
    }
    if (IDs.find(vertex2) == IDs.end()) {
      printf("Vertex2 not found in graph");
      return;
    }
    int id1 = IDs[vertex1];
    int id2 = IDs[vertex2];
    Edge edge1{weight, id1};
    Edge edge2{weight, id2};
    if (std::find(Adyacencias[id1].begin(), Adyacencias[id1].end(), edge2) != Adyacencias[id1].end()) {
      printf("Edge already in graph");
      return;
    }
    Adyacencias[id1].push_back(edge2);
    Adyacencias[id2].push_back(edge1);
  }

  void remove_vertex(IDType ID) {
    if (IDs.find(ID) == IDs.end()) {
      printf("ID not found in graph");
      return;
    }
    int index = IDs[ID];
    
    for (const Edge& edge : Adyacencias[index]) {
      int vecino = edge.dest;
      for (size_t i = 0; i < Adyacencias[vecino].size(); ++i) {
        if (Adyacencias[vecino][i].dest == index) {
          Adyacencias[vecino].erase(Adyacencias[vecino].begin() + i);
          break;
        }
      }
    }

    Adyacencias[index].clear();
    Contents[index] = ContentType{};
    ReverseIDs[index] = IDType{};
    IDs.erase(ID);
    free.push_back(index);
  }

  void remove_edge(IDType vertex1, IDType vertex2) {
    if (IDs.find(vertex1) == IDs.end() || IDs.find(vertex2) == IDs.end()) {
      printf("Vertex not found in graph");
      return;
    }
    int id1 = IDs[vertex1];
    int id2 = IDs[vertex2];
    bool found = false;
    for (int i = 0; i < Adyacencias[id1].size(); i++) {
      if (Adyacencias[id1][i].dest == id2) {
        Adyacencias[id1].erase(Adyacencias[id1].begin() + i);
        found = true;
        break;
      }
    }
    for (int i = 0; i < Adyacencias[id2].size(); i++) {
      if (Adyacencias[id2][i].dest == id1) {
        Adyacencias[id2].erase(Adyacencias[id2].begin() + i);
        found = true;
        break;
      }
    }
    if (!found) {
      printf("Edge not found in graph");
    }
  }

  std::vector<std::vector<IDType>> get_walks(int num_walks, int num_steps, float p, float q) {
    preprocess_transition_probs(p, q);

    std::vector<int> nodes;
    int nodeCount = Contents.size(); 
    nodes.reserve(nodeCount);

    for (int i = 0; i < nodeCount; i++) {
        if (std::find(free.begin(), free.end(), i) == free.end()) {
            nodes.push_back(i);
        }
    }

    std::vector<std::vector<IDType>> walk_list;
    walk_list.reserve(nodes.size() * num_walks);

    for (int walk_iter = 0; walk_iter < num_walks; ++walk_iter) {
        std::shuffle(nodes.begin(), nodes.end(), gen);
        
        for (int node : nodes) {
            walk_list.push_back(get_random_walk(node, num_steps));
        }
    }

    return walk_list;
  }

  std::vector<IDType> get_nodes() const {
    return ReverseIDs;
  } 

  std::vector<int> get_degrees() const {
    std::vector<int> degrees;
    degrees.reserve(ReverseIDs.size());
    for (const std::vector<Edge>& edgeList : Adyacencias) {
      degrees.push_back(edgeList.size());
    }
    return degrees;
  }
  
private:
  std::unordered_map<IDType, int> IDs;
  std::vector<std::vector<Edge>> Adyacencias;
  std::vector<ContentType> Contents;
  std::vector<int> free;
  std::vector<IDType> ReverseIDs;
  mutable std::mt19937 gen{std::random_device{}()};

  std::vector<AliasTable> node_alias;
  std::vector<std::vector<AliasTable>> edge_alias;

  AliasTable build_alias_table(const std::vector<float>& probs) const {
    int K = probs.size();
    AliasTable table;
    table.J.resize(K, 0);
    table.q.resize(K, 0.0f);

    std::vector<int> smaller;
    std::vector<int> larger;
    smaller.reserve(K);
    larger.reserve(K);

    float sum = 0.0f;
    for (float p : probs) sum += p;

    std::vector<float> scaled_probs(K);
    for (int i = 0; i < K; ++i) {
        scaled_probs[i] = (probs[i] / sum) * K;
        if (scaled_probs[i] < 1.0f) {
            smaller.push_back(i);
        } else {
            larger.push_back(i);
        }
    }

    while (!smaller.empty() && !larger.empty()) {
        int small = smaller.back(); smaller.pop_back();
        int large = larger.back(); larger.pop_back();

        table.J[small] = large;
        table.q[small] = scaled_probs[small];

        scaled_probs[large] = scaled_probs[large] + scaled_probs[small] - 1.0f;
        if (scaled_probs[large] < 1.0f) {
            smaller.push_back(large);
        } else {
            larger.push_back(large);
        }
    }

    while (!larger.empty()) {
        table.q[larger.back()] = 1.0f;
        larger.pop_back();
    }
    while (!smaller.empty()) {
        table.q[smaller.back()] = 1.0f;
        smaller.pop_back();
    }

    return table;
  }

  void preprocess_transition_probs(float p, float q) {
    int num_nodes = Adyacencias.size();
    node_alias.resize(num_nodes);
    edge_alias.resize(num_nodes);

    for (int i = 0; i < num_nodes; ++i) {
        if (std::find(free.begin(), free.end(), i) != free.end()) continue;

        const auto& neighbors = Adyacencias[i];
        int degree = neighbors.size();
        if (degree == 0) continue;

        std::vector<float> unigram_probs(degree);
        for (int j = 0; j < degree; ++j) {
            unigram_probs[j] = neighbors[j].weight;
        }
        node_alias[i] = build_alias_table(unigram_probs);

        edge_alias[i].resize(degree);
        for (int j = 0; j < degree; ++j) {
            int prev = neighbors[j].dest;
            std::vector<float> edge_probs(degree);
            
            for (int k = 0; k < degree; ++k) {
                int next = neighbors[k].dest;
                float weight = neighbors[k].weight;

                if (next == prev) {
                    edge_probs[k] = weight / p;
                } else if (are_connected(prev, next)) {
                    edge_probs[k] = weight;
                } else {
                    edge_probs[k] = weight / q;
                }
            }
            edge_alias[i][j] = build_alias_table(edge_probs);
        }
    }
  }

  std::vector<IDType> get_random_walk(int node, int num_steps) const {
    if (Adyacencias[node].empty()) return {};

    std::vector<IDType> walk;
    walk.reserve(num_steps);
    int pos = node;
    int prev_idx = -1; 

    std::uniform_int_distribution<int> int_dist(0, 1000000000);
    std::uniform_real_distribution<float> float_dist(0.0f, 1.0f);

    for (int iter = 0; iter < num_steps; iter++) {
      walk.push_back(ReverseIDs[pos]);
      const std::vector<Edge>& adyacentes = Adyacencias[pos];
      if (adyacentes.empty()) break;

      int next_idx;
      if (iter == 0 || prev_idx == -1) {
          const AliasTable& alias = node_alias[pos];
          int K = alias.J.size();
          int k = int_dist(gen) % K;
          float r = float_dist(gen);
          next_idx = (r < alias.q[k]) ? k : alias.J[k];
      } else {
          const AliasTable& alias = edge_alias[pos][prev_idx];
          int K = alias.J.size();
          int k = int_dist(gen) % K;
          float r = float_dist(gen);
          next_idx = (r < alias.q[k]) ? k : alias.J[k];
      }

      int next_node = adyacentes[next_idx].dest;

      int next_prev_idx = -1;
      const auto& next_neighbors = Adyacencias[next_node];
      for(int i = 0; i < next_neighbors.size(); ++i) {
          if(next_neighbors[i].dest == pos) {
              next_prev_idx = i;
              break;
          }
      }

      pos = next_node;
      prev_idx = next_prev_idx;
    }
    return walk;
  }

  bool are_connected(int node1, int node2) const {
    if (node1 == -1 || node2 == -1) return false;
    for (const Edge& ed : Adyacencias[node1]) {
      if (ed.dest == node2) return true;
    }
    return false;
  }
};