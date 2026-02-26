cmake_minimum_required(VERSION 3.12)
project(Node2VecProject)

set(CMAKE_CXX_STANDARD 17)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED PATHS ${pybind11_DIR} NO_DEFAULT_PATH)

include_directories(include)

pybind11_add_module(node2vec_cpp bindings/pybind_module.cpp)

add_executable(main_cpp src/main.cpp)