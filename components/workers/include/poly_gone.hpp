#ifndef POLY_GONE_HPP
#define POLY_GONE_HPP
#pragma once
#include "buffer_container.hpp"

// Buffer* calculate_geometry(Buffer* bboxes_ptr, Buffer* img, const int padding);
Buffer* get_bboxes(Buffer* bboxes_ptr, Buffer* img, const int32_t padding, size_t* total_buffers_ref);

#endif
