cmake_minimum_required(VERSION 3.12)
project(Node2VecProject)

set(CMAKE_CXX_STANDARD 17)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(OpenMP REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED PATHS ${pybind11_DIR} NO_DEFAULT_PATH)

include_directories(include)

pybind11_add_module(node2vec_cpp bindings/pybind_module.cpp)
target_link_libraries(node2vec_cpp PRIVATE OpenMP::OpenMP_CXX)

add_executable(main_cpp src/main.cpp)
target_link_libraries(main_cpp PRIVATE OpenMP::OpenMP_CXX)

add_custom_command(TARGET node2vec_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=$<TARGET_FILE_DIR:node2vec_cpp>"
            "${Python_EXECUTABLE}" -m pybind11_stubgen node2vec_cpp -o "${CMAKE_SOURCE_DIR}/typings"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating stub for vscode "
)