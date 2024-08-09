#include <iostream>
#include "detect.h"

cv::VideoCapture camera(0);

cv::Mat read_camera() {
	if (!camera.isOpened()) {
		std::cout << "Error: Could not open camera" << std::endl;
		return cv::Mat();
	}
	cv::Mat frame;
	camera >> frame;
	return frame;
}

int main() {
	load_model("model/cone_ncnn_model_v9_320", 320);

	auto start = std::chrono::system_clock::now();
	while (true) {
		cv::Mat image = read_camera();
		//std::array<float, 2> pos = get_pos(image);
		//std::cout << pos[0] << " " << pos[1] << std::endl;

		auto now = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed = now - start;
		if (elapsed.count() > 10) {
			break;
		}
	}

	return 0;
}