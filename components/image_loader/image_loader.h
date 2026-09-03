#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct ProcessedImage {
    std::vector<uint8_t> data;
    int width;
    int height;
    bool success;
};

ProcessedImage load_and_clean_image(const std::string& filepath, int color_threshold);