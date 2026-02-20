#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
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

private:
  std::unordered_map<IDType, int> IDs;
  std::vector<std::vector<Edge>> Adyacencias;
  std::vector<ContentType> Contents;
  std::vector<int> free;
  std::vector<IDType> ReverseIDs;
};

int main() {
  Graph<std::string, std::vector<int>> grafo;
  grafo.add_vertex("Paris", {1, 2, 3});
  grafo.add_vertex("Madrid", {4, 5, 6});
  grafo.add_edge("Paris", "Madrid", 2.0f);
  grafo.add_vertex("Rome", {3, 4, 1, 2});
  grafo.add_edge("Rome", "Paris", 3);
  std::vector<std::string> ad = grafo.get_adyacent("Paris");
  for (std::string &a : ad) {
    std::cout << a << std::endl;
  }
  printf("0\n");
  return 0;
}