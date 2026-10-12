#ifndef IMAGE_LOADER_HPP
#define IMAGE_LOADER_HPP
#pragma once
#include "image_container.hpp"

namespace image_buffer_creator {
    Image* get_image_buffer(int width, int height, int channels);
}

Image* load_image(const char* filepath);

#endif