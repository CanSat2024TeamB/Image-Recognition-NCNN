#include <iostream>
#include <chrono>
#include <ncnn/net.h>
#include "detect.h"

int IMGSZ = 640;
int CAMERA_WIDTH = 640;
int CAMERA_HEIGHT = 480;

ncnn::Net net;

void set_camera_resolution(int width, int height) {
	CAMERA_WIDTH = width;
	CAMERA_HEIGHT = height;
}

void load_model(std::string model_path, int imgsz) {
	net.load_param((model_path + "/model.ncnn.param").c_str());
	net.load_model((model_path + "/model.ncnn.bin").c_str());
	IMGSZ = imgsz;
}

std::vector<std::array<float, 5>> nms(std::vector<std::array<float, 5>>& bboxes, float iou_threshold) {
	std::sort(bboxes.begin(), bboxes.end(), [](std::array<float, 5> arr1, std::array<float, 5> arr2) { return arr1[4] > arr2[4]; });
	std::vector<std::array<float, 5>> result;

	while (bboxes.size() > 0) {
		std::array<float, 5> base = bboxes[bboxes.size() - 1];
		float base_area = (base[2] - base[0]) * (base[3] - base[1]);
		result.push_back(base);
		bboxes.pop_back();

		for (int i = bboxes.size() - 1; i >= 0; i--) {
			std::array<float, 5> target = bboxes[i];
			float target_area = (target[2] - target[0]) * (target[3] - target[1]);

			float w = std::max(0.0f, std::min(base[2], target[2]) - std::max(base[0], target[0]));
			float h = std::max(0.0f, std::min(base[3], target[3]) - std::max(base[1], target[1]));

			float overlap_area = w * h;
			float iou = overlap_area / (base_area + target_area - overlap_area);
			if (iou > iou_threshold) {
				bboxes.erase(bboxes.begin() + i);
			}
		}
	}

	return result;
}

std::vector<std::array<float, 5>> detect(pybind11::array_t<uint8_t>& image, float prob_threshold) {
	int image_width = image.shape(1);
	int image_height = image.shape(0);

	ncnn::Mat in = ncnn::Mat::from_pixels_resize((unsigned char*)image.data(), ncnn::Mat::PIXEL_BGR, image_width, image_height, IMGSZ, IMGSZ);
	const float norm_values[3] = { 1 / 255.0f , 1 / 255.0f, 1 / 255.0f };
	in.substract_mean_normalize(0, norm_values);

	ncnn::Extractor extractor = net.create_extractor();
	extractor.input("in0", in);

	ncnn::Mat out;
	extractor.extract("out0", out);

	std::vector<std::array<float, 5>> out_vec;

	float* target_probs = out.row(4 + TARGET_CLS);
	for (int i = 0; i < out.w; i++) {
		float prob = target_probs[i];
		if (prob < prob_threshold) {
			continue;
		}
		int x = out.row(0)[i];
		int y = out.row(1)[i];
		int w = out.row(2)[i];
		int h = out.row(3)[i];
		int x1 = (x - w / 2) * image_width / IMGSZ;
		int y1 = (y - h / 2) * image_height / IMGSZ;
		int x2 = (x + w / 2) * image_width / IMGSZ;
		int y2 = (y + h / 2) * image_height / IMGSZ;
		std::array<float, 5> row = { x1, y1, x2, y2, prob };
		out_vec.push_back(row);
	}

	return nms(out_vec, IOU_THRESHOLD);
}

std::pair<std::array<float, 4>, float> get_data(pybind11::array_t<uint8_t>& image, float prob_threshold) {
	std::vector<std::array<float, 5>> result = detect(image, prob_threshold);
	if (result.size() > 0) {
		return { {result[0][0], result[0][1], result[0][2], result[0][3]}, result[0][4] };
	}
	else {
		return { {-1, -1, -1, -1}, -1 };
	}
}

std::array<float, 2> get_pos(pybind11::array_t<uint8_t>& image, float prob_threshold) {
	std::vector<std::array<float, 5>> result = detect(image, prob_threshold);
	if (result.size() > 0) {
		float x = (result[0][0] + result[0][2]) / 2;
		float y = (result[0][1] + result[0][3]) / 2;
		float norm_x = 2 * x / CAMERA_WIDTH - 1;
		float norm_y = 1 - 2 * y / CAMERA_HEIGHT;
		return { norm_x, norm_y };
	}
	else {
		return { -2, -2 };
	}
}

//pybind11
PYBIND11_MODULE(cone_detector, m) {
	m.def("set_camera_resolution", &set_camera_resolution, pybind11::arg("width"), pybind11::arg("height"));
	m.def("load_model", &load_model, pybind11::arg("model_path"), pybind11::arg("imgsz"));
	m.def("get_data", &get_data, pybind11::arg("image"), pybind11::arg("prob_threshold"));
	m.def("get_pos", &get_pos, pybind11::arg("image"), pybind11::arg("prob_threshold"));
}