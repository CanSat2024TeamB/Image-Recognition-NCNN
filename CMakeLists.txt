cmake_minimum_required(VERSION 3.21)
project(cone_detector CXX)

add_subdirectory(pybind11)

pybind11_add_module(${PROJECT_NAME} src/detect.h src/detect.cpp)

set(ncnn_DIR "I:/Program Files/ncnn/x64/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
set(OpenCV_DIR "I:/Program Files/opencv/build/x64/vc16/lib")

find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)

target_include_directories(${PROJECT_NAME} PRIVATE "I:/Program Files/ncnn/x64/include")
target_include_directories(${PROJECT_NAME} PRIVATE "I:/Program Files/opencv/build/include")

target_link_libraries(${PROJECT_NAME} PRIVATE ncnn)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBRARIES})