#ifndef POLY_GONE_HPP
#define POLY_GONE_HPP
#pragma once
#include <ippi.h>
#include "buffer_container.hpp"

Buffer* calculate_geometry(Buffer* bboxes_ptr, Buffer* img, const int padding);
Buffer* get_bboxes(Buffer* bboxes_ptr, Buffer* img, int padding);
Buffer* get_centroids(int32_t* bboxes_array, size_t bbox_count);

#endif
