#pragma once
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <cstdint>

extern "C" {
    void load_image(const char* filepath);
}

struct FullImage {
    std::vector<uint8_t> data;
    int width;
    int height;
    bool success;
};
