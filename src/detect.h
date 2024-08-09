#pragma once
#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

const int TARGET_CLS = 0;
const float PROB_THRESHOLD = 0.1;
const float IOU_THRESHOLD = 0.3;

void set_camera_resolution(int width, int height);
void load_model(std::string model_path, int imgsz);
std::vector<std::array<float, 5>> nms(std::vector<std::array<float, 5>>& bboxes, float iou_threshold);
std::vector<std::array<float, 5>> detect(pybind11::array_t<uint8_t>& image, float prob_threshold);
std::pair<std::array<float, 4>, float> get_data(pybind11::array_t<uint8_t>& image, float prob_threshold = PROB_THRESHOLD);
std::array<float, 2> get_pos(pybind11::array_t<uint8_t>& image, float prob_threshold = PROB_THRESHOLD);