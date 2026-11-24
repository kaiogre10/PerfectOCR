#ifndef POLY_GONE_HPP
#define POLY_GONE_HPP
#pragma once
#include "buffer_container.hpp"

Buffer* get_bboxes(Buffer* bboxes_ptr, Buffer* img, int padding, size_t* total_buffers_ref);

#endif