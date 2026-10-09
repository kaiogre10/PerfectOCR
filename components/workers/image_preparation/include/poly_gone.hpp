#ifndef POLY_GONE_HPP
#define POLY_GONE_HPP
#pragma once
#include <cstdint>

CroppedImage* crop_polygons(uint8_t* img_data, void* arr_data, int ndim, int width, int height, size_t itemsize);

#endif POLY_GONE_HPP;
