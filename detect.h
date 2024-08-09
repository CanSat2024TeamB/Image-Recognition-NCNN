#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

const int TARGET_CLS = 0;
const float PROB_THRESHOLD = 0.1;
const float IOU_THRESHOLD = 0.3;

void set_camera_resolution(int width, int height);
void load_model(std::string model_path, int imgsz);
std::vector<std::array<float, 5>> nms(std::vector<std::array<float, 5>>& bboxes, float iou_threshold);
std::vector<std::array<float, 5>> detect(const cv::Mat& image, float prob_threshold);
std::pair<std::array<float, 4>, float> get_data(cv::Mat& image, float prob_threshold = PROB_THRESHOLD);
std::array<float, 2> get_pos(cv::Mat & image, float prob_threshold = PROB_THRESHOLD); #pragma once
