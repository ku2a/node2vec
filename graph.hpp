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
    // check in graph
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
    // check in graph
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
    // check edge in graph
    if (std::find(Adyacencias[id1].begin(), Adyacencias[id1].end(), edge2) !=
        Adyacencias[id1].end()) {
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
    Adyacencias[index].clear();
    Contents[index] = ContentType{};
    ReverseIDs[index] = IDType{};
    IDs.erase(ID);
    free.push_back(index);
    // remove adyacents that have this vertex as destination
    for (std::vector<Edge> &node_adyacents : Adyacencias) {
      for (int i = 0; i < node_adyacents.size(); i++) {
        if (node_adyacents[i].dest == index) {
          node_adyacents.erase(node_adyacents.begin() + i);
          break;
        }
      }
    }
  }

  void remove_edge(IDType vertex1, IDType vertex2) {
    // check vertex in graph
    if (IDs.find(vertex1) == IDs.end()) {
      printf("Vertex2 not found in graph");
      return;
    }
    if (IDs.find(vertex2) == IDs.end()) {
      printf("Vertex2 not found in graph");
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

  std::vector<std::vector<IDType>> get_walks(int num_steps, float p, float q) const{
    std::vector<int> nodes;

    int nodeCount = Contents.size(); 
    nodes.reserve(nodeCount);
    std::vector<std::vector<IDType>> walk_list;
    walk_list.reserve(nodeCount);

    for (int i = 0; i<nodeCount; i++){
      if (std::find(free.begin(), free.end(), i) == free.end())
        nodes.push_back(i);
    }
    std::shuffle(nodes.begin(), nodes.end(), gen);
    
    for (int node : nodes){
      walk_list.push_back(get_random_walk(node, num_steps, p, q));
    }

    return walk_list;
  }
  
private:
  std::unordered_map<IDType, int> IDs;
  std::vector<std::vector<Edge>> Adyacencias;
  std::vector<ContentType> Contents;
  std::vector<int> free;
  std::vector<IDType> ReverseIDs;
  mutable std::mt19937 gen{std::random_device{}()};


  std::vector<IDType> get_random_walk(int node, int num_steps, float p, float q) const{
    if (Adyacencias[node].size() == 0){
      return {};
    }
    std::vector<IDType> walk;
    walk.reserve(num_steps);
    int pos = node;
    int prev = -1; 
    std::vector<float> weights;
    for (int iter=0; iter<num_steps; iter++){

      walk.push_back(ReverseIDs[pos]);
      const std::vector<Edge>& adyacentes = get_adyacent_edges(pos);
      
      weights.reserve(adyacentes.size());
      
      for (const Edge& edge : adyacentes ){
        if (edge.dest == prev){
          weights.push_back( edge.weight * (1.0/ p));
        } else if  ( are_connected(prev, edge.dest)){
          weights.push_back( edge.weight);
        } else{
          weights.push_back(edge.weight * (1.0 / q));
        }
      }

      std::discrete_distribution<int> dist(weights.begin(), weights.end());
      int next = adyacentes[dist(gen)].dest; 
      prev = pos;
      pos = next;
      weights.clear();
    }
    return walk;

  }


  const std::vector<Edge>& get_adyacent_edges(int node) const{
    return  Adyacencias[node];
  }

  bool are_connected(int node1, int node2) const{
    if (node1 == -1 || node2==-1){
      return false;
    }
    if (node1 >= Contents.size()){
      printf("Node 1 not in graph");
      return false;
    }
    if (node2 >= Contents.size()){
      printf("Node 2 not in graph");
      return false;
    }
    for (const Edge& ed : Adyacencias[node1]){
      if (ed.dest == node2){
        return true;
      }
    }
    return false;
  }
};

