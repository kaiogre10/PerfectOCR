#include "file_handler.hpp"
#include <opencv2/imgcodecs.hpp>

namespace files_handler {
    void load_img(const char* filepath, cv::Mat& image) {
        if (!filepath) {
            return;
        }
        // 1. Carga multiformato sin alterar canales originales (IMREAD_UNCHANGED)
        image = cv::imread(filepath, cv::IMREAD_UNCHANGED);
    }
    bool save_image_debug(const char* output_path, const uint8_t* data, int rows, int cols) {
    if (!output_path || !data || rows <= 0 || cols <= 0) {
        return false;
    }
    cv::Mat image(rows, cols, CV_8UC1, const_cast<uint8_t*>(data));
    return cv::imwrite(output_path, image);
}

    bool save_image(const char* output_path, const cv::Mat& outimage) {
        return cv::imwrite(output_path, outimage);
    }
}