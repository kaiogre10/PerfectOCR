#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct FullImage {
    std::vector<uint8_t> data;
    int width;
    int height;
    bool success;
};

extern "C" {
    FullImage load_and_clean_image(const std::string& filepath);
}
