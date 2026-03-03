#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include "graph.hpp"
#include "skipGram.hpp"

namespace py = pybind11;

using MyGraph = Graph<std::string, std::vector<float>>;
using MySkipGram = SkipGram<std::string>;

PYBIND11_MODULE(node2vec_cpp, m) {

    py::class_<WalkGenerator<std::string, std::vector<float>>>(m, "WalkGenerator")
        .def(py::init<Graph<std::string, std::vector<float>>&, int, int, float, float>())
        .def("__iter__", [](WalkGenerator<std::string, std::vector<float>>& it) -> WalkGenerator<std::string, std::vector<float>>& { 
            return it; 
        })
        .def("__next__", [](WalkGenerator<std::string, std::vector<float>>& it) {
            try {
                return it.next_batch(1024);
            } catch (const std::runtime_error& e) {
                throw py::stop_iteration();
            }
        })
        .def("next_batch", &WalkGenerator<std::string, std::vector<float>>::next_batch);

    py::class_<MyGraph>(m, "Graph")
        .def(py::init<>())
        .def("get_adyacent", &MyGraph::get_adyacent, py::arg("ID"))
        .def("add_vertex", &MyGraph::add_vertex, py::arg("ID"), py::arg("content"))
        .def("add_edge", &MyGraph::add_edge, py::arg("vertex1"), py::arg("vertex2"), py::arg("weight"))
        .def("remove_vertex", &MyGraph::remove_vertex, py::arg("ID"))
        .def("remove_edge", &MyGraph::remove_edge, py::arg("vertex1"), py::arg("vertex2"))
        .def("get_walks", &MyGraph::get_walks, py::arg("num_walks"), py::arg("num_steps"), py::arg("p"), py::arg("q"))
        .def("get_walks_iter", &MyGraph::get_walks_iter, py::keep_alive<0, 1>(), py::arg("num_walks"), py::arg("num_steps"), py::arg("p"), py::arg("q"))
        .def("get_nodes", &MyGraph::get_nodes)
        .def("get_degrees", &MyGraph::get_degrees);

    py::class_<MySkipGram>(m, "SkipGram")
        .def(py::init<int, bool>(), py::arg("N") = 300, py::arg("subsampling") = false)
        .def("build_vocab", py::overload_cast<const std::vector<std::string>&>(&MySkipGram::build_vocab), py::arg("corpus"))
        .def("build_vocab", py::overload_cast<const std::vector<std::string>&, const std::vector<int>&>(&MySkipGram::build_vocab), py::arg("vocab"), py::arg("frecs"))
        .def("train", &MySkipGram::train<std::vector<float>>, py::arg("graph"), py::arg("epochs"), py::arg("walk_length"), py::arg("p"), py::arg("q"), py::arg("K"), py::arg("C"), py::arg("starting_alpha"), py::arg("verbose"), py::arg("batch_size") = 1024)
        .def("clear", &MySkipGram::clear)
        .def("get_embedding", &MySkipGram::get_embedding, py::arg("word"))
        .def("get_embeddings", &MySkipGram::get_embeddings)
        .def("cosine_similarity", &MySkipGram::cosine_similarity, py::arg("word1"), py::arg("word2"))
        .def("save_model", &MySkipGram::save_model, py::arg("filename"))
        .def("load_model", &MySkipGram::load_model, py::arg("filename"))
        .def("most_similar", &MySkipGram::most_similar, py::arg("word"), py::arg("top_k") = 5);
}