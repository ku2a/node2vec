#include "graph.hpp"


void load_europe(Graph<std::string, std::vector<int>>& grafo) {
    std::vector<std::string> paises = {
        "Portugal", "Espana", "Andorra", "Francia", "Belgica", "Paises Bajos", 
        "Luxemburgo", "Alemania", "Suiza", "Italia", "Dinamarca", "Polonia", 
        "Rep. Checa", "Austria", "Eslovaquia", "Hungria", "Eslovenia", "Croacia", 
        "Bosnia y Herzegovina", "Serbia", "Montenegro", "Kosovo", "Albania", 
        "Macedonia del Norte", "Grecia", "Bulgaria", "Rumania", "Moldavia", 
        "Ucrania", "Bielorrusia", "Lituania", "Letonia", "Estonia", "Rusia", 
        "Finlandia", "Suecia", "Noruega", "Reino Unido", "Irlanda"
    };

    for (const auto& pais : paises) {
        grafo.add_vertex(pais, {});
    }

    grafo.add_edge("Portugal", "Espana", 1.0f);

    grafo.add_edge("Espana", "Francia", 1.0f);
    grafo.add_edge("Espana", "Andorra", 1.0f);

    grafo.add_edge("Andorra", "Francia", 1.0f);

    grafo.add_edge("Francia", "Belgica", 1.0f);
    grafo.add_edge("Francia", "Luxemburgo", 1.0f);
    grafo.add_edge("Francia", "Alemania", 1.0f);
    grafo.add_edge("Francia", "Suiza", 1.0f);
    grafo.add_edge("Francia", "Italia", 1.0f);

    grafo.add_edge("Belgica", "Paises Bajos", 1.0f);
    grafo.add_edge("Belgica", "Luxemburgo", 1.0f);
    grafo.add_edge("Belgica", "Alemania", 1.0f);

    grafo.add_edge("Paises Bajos", "Alemania", 1.0f);

    grafo.add_edge("Luxemburgo", "Alemania", 1.0f);

    grafo.add_edge("Alemania", "Dinamarca", 1.0f);
    grafo.add_edge("Alemania", "Polonia", 1.0f);
    grafo.add_edge("Alemania", "Rep. Checa", 1.0f);
    grafo.add_edge("Alemania", "Austria", 1.0f);
    grafo.add_edge("Alemania", "Suiza", 1.0f);

    grafo.add_edge("Suiza", "Austria", 1.0f);
    grafo.add_edge("Suiza", "Italia", 1.0f);

    grafo.add_edge("Italia", "Austria", 1.0f);
    grafo.add_edge("Italia", "Eslovenia", 1.0f);

    grafo.add_edge("Polonia", "Rusia", 1.0f);
    grafo.add_edge("Polonia", "Lituania", 1.0f);
    grafo.add_edge("Polonia", "Bielorrusia", 1.0f);
    grafo.add_edge("Polonia", "Ucrania", 1.0f);
    grafo.add_edge("Polonia", "Eslovaquia", 1.0f);
    grafo.add_edge("Polonia", "Rep. Checa", 1.0f);

    grafo.add_edge("Rep. Checa", "Eslovaquia", 1.0f);
    grafo.add_edge("Rep. Checa", "Austria", 1.0f);

    grafo.add_edge("Austria", "Eslovaquia", 1.0f);
    grafo.add_edge("Austria", "Hungria", 1.0f);
    grafo.add_edge("Austria", "Eslovenia", 1.0f);

    grafo.add_edge("Eslovaquia", "Ucrania", 1.0f);
    grafo.add_edge("Eslovaquia", "Hungria", 1.0f);

    grafo.add_edge("Hungria", "Ucrania", 1.0f);
    grafo.add_edge("Hungria", "Rumania", 1.0f);
    grafo.add_edge("Hungria", "Serbia", 1.0f);
    grafo.add_edge("Hungria", "Croacia", 1.0f);
    grafo.add_edge("Hungria", "Eslovenia", 1.0f);

    grafo.add_edge("Eslovenia", "Croacia", 1.0f);

    grafo.add_edge("Croacia", "Serbia", 1.0f);
    grafo.add_edge("Croacia", "Bosnia y Herzegovina", 1.0f);
    grafo.add_edge("Croacia", "Montenegro", 1.0f);

    grafo.add_edge("Bosnia y Herzegovina", "Serbia", 1.0f);
    grafo.add_edge("Bosnia y Herzegovina", "Montenegro", 1.0f);

    grafo.add_edge("Serbia", "Rumania", 1.0f);
    grafo.add_edge("Serbia", "Bulgaria", 1.0f);
    grafo.add_edge("Serbia", "Macedonia del Norte", 1.0f);
    grafo.add_edge("Serbia", "Kosovo", 1.0f);
    grafo.add_edge("Serbia", "Montenegro", 1.0f);

    grafo.add_edge("Montenegro", "Kosovo", 1.0f);
    grafo.add_edge("Montenegro", "Albania", 1.0f);

    grafo.add_edge("Kosovo", "Macedonia del Norte", 1.0f);
    grafo.add_edge("Kosovo", "Albania", 1.0f);

    grafo.add_edge("Albania", "Macedonia del Norte", 1.0f);
    grafo.add_edge("Albania", "Grecia", 1.0f);

    grafo.add_edge("Macedonia del Norte", "Bulgaria", 1.0f);
    grafo.add_edge("Macedonia del Norte", "Grecia", 1.0f);

    grafo.add_edge("Bulgaria", "Rumania", 1.0f);
    grafo.add_edge("Bulgaria", "Grecia", 1.0f);

    grafo.add_edge("Rumania", "Moldavia", 1.0f);
    grafo.add_edge("Rumania", "Ucrania", 1.0f);

    grafo.add_edge("Moldavia", "Ucrania", 1.0f);

    grafo.add_edge("Ucrania", "Bielorrusia", 1.0f);
    grafo.add_edge("Ucrania", "Rusia", 1.0f);

    grafo.add_edge("Bielorrusia", "Rusia", 1.0f);
    grafo.add_edge("Bielorrusia", "Letonia", 1.0f);
    grafo.add_edge("Bielorrusia", "Lituania", 1.0f);

    grafo.add_edge("Lituania", "Letonia", 1.0f);
    grafo.add_edge("Lituania", "Rusia", 1.0f); 

    grafo.add_edge("Letonia", "Estonia", 1.0f);
    grafo.add_edge("Letonia", "Rusia", 1.0f);

    grafo.add_edge("Estonia", "Rusia", 1.0f);

    grafo.add_edge("Rusia", "Finlandia", 1.0f);
    grafo.add_edge("Rusia", "Noruega", 1.0f);

    grafo.add_edge("Finlandia", "Noruega", 1.0f);
    grafo.add_edge("Finlandia", "Suecia", 1.0f);

    grafo.add_edge("Suecia", "Noruega", 1.0f);

    grafo.add_edge("Reino Unido", "Irlanda", 1.0f);
}

int main(){
  Graph<std::string, std::vector<int>> grafo;
  load_europe(grafo);
  std::vector<std::vector<std::string>> caminos = grafo.get_walks(10,1,1);
  std::vector<std::vector<std::string>> caminos2 = grafo.get_walks(10,1,1);
  for (auto& camino: caminos2){
    for (auto& paso : camino){
      std::cout << paso;
      std::cout << " " ;
    } 
    std::cout << std::endl;
  }

}