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
    py::class_<MyGraph>(m, "Graph")
        .def(py::init<>())
        .def("get_adyacent", &MyGraph::get_adyacent)
        .def("add_vertex", &MyGraph::add_vertex)
        .def("add_edge", &MyGraph::add_edge)
        .def("remove_vertex", &MyGraph::remove_vertex)
        .def("remove_edge", &MyGraph::remove_edge)
        .def("get_walks", &MyGraph::get_walks)
        .def("get_nodes", &MyGraph::get_nodes)
        .def("get_degrees", &MyGraph::get_degrees);

    py::class_<MySkipGram>(m, "SkipGram")
        .def(py::init<int, bool>())
        .def("build_vocab", py::overload_cast<const std::vector<std::string>&>(&MySkipGram::build_vocab))
        .def("build_vocab", py::overload_cast<const std::vector<std::string>&, const std::vector<int>&>(&MySkipGram::build_vocab))
        .def("train", &MySkipGram::train)
        .def("clear", &MySkipGram::clear)
        .def("get_embedding", &MySkipGram::get_embedding)
        .def("get_embeddings", &MySkipGram::get_embeddings)
        .def("cosine_similarity", &MySkipGram::cosine_similarity)
        .def("most_similar", &MySkipGram::most_similar);
}