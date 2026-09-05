#pragma once
#include <vector>
#include <cstdint>

struct FullImage {
    std::vector<uint8_t> data;
    int width;
    int height;
    bool success;
};

extern "C" {
    void load_image(const char* filepath);
}
