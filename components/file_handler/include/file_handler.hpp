#pragma once
#include <opencv2/imgcodecs.hpp>

namespace files_handler {
    void load_img(const char* filepath, cv::Mat& image);
    bool save_image(const char* output_path, const cv::Mat& outimage);
    bool save_image_debug(const char* output_path, const uint8_t* data, int rows, int cols);
}
