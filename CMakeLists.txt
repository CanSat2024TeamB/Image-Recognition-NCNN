cmake_minimum_required(VERSION 3.21)
project(cone_detector CXX)

add_subdirectory(pybind11)

pybind11_add_module(${PROJECT_NAME} src/detect.h src/detect.cpp)

set(ncnn_DIR "I:/Program Files/ncnn/x64/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")

find_package(ncnn REQUIRED)

target_include_directories(${PROJECT_NAME} PRIVATE "I:/Program Files/ncnn/x64/include")

target_link_libraries(${PROJECT_NAME} PRIVATE ncnn)