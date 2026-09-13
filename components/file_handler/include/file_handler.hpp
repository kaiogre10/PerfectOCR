#pragma once
#include <opencv2/imgcodecs.hpp>

namespace files_handler {
    void load_img(const char* filepath, cv::Mat& image);
    bool save_image(const char* output_path, const cv::Mat& outimage);
}
