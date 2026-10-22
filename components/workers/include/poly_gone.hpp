#ifndef POLY_GONE_HPP
#define POLY_GONE_HPP
#pragma once
#include "buffer_container.hpp"

Buffer* get_bboxes(int32_t* bboxes_array, int bbox_count, int img_rows, int img_cols, int padding);
Buffer* get_centroids(int32_t* bboxes_array, int bbox_count);

#endif
