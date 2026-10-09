#include "poly_gone.hpp"
#include "image_container.hpp"

using BBoxArr = Image

BBoxArr* crop_polygons(uint8_t* img_data, void* arr_data, int ndim, int width, int height, size_t itemsize) {
    if (!img_data || !arr_data) {
        return nullptr;
    }
    BBoxArr* bboxes_arr = new BBoxArr(width, height, ndim);
    size_t total_bytes = size * itemsize;
    
    if (total_bytes != bboxes_arr->size()) {
        delete bboxes_arr;
        return nullptr;
    }
    memcpy(bboxes_arr->data(), arr_data, total_bytes);
    arr_data.release();   // Liberar imagen original inmediatamente
    return bboxes_arr;
}