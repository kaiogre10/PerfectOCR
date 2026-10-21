#ifndef POLY_GONE_HPP
#define POLY_GONE_HPP
#pragma once
#include "image_container.hpp"

Image* get_bboxes(int32_t* bboxes_array, int bbox_count, int img_rows, int img_cols, int padding);
Image* get_centroids(int32_t* bboxes_array, int bbox_count);

#endif
