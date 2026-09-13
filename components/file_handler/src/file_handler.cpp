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

    bool save_image(const char* output_path, const cv::Mat& outimage) {
        return cv::imwrite(output_path, outimage);
    }
}