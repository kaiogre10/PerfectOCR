#pragma once
#include <cstdint>
#include <memory>

struct FullImage {
    std::unique_ptr<uint8_t[]> data;
    size_t size;
    int widith;
    int height;
    int channels;

};