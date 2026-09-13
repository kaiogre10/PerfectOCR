#pragma once
#include <opencv2/core.hpp>

namespace image_utils {

    void decolorate(cv::Mat& image);

    void make_contiguous(cv::Mat& image);

    bool validate_image(cv::Mat& image);

    void normalize_image(cv::Mat& image);
}