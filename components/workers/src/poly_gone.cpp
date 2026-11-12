#include "poly_gone.hpp"
#include "buffer_container.hpp"
#include <algorithm>
#include <cstdint>
#include <stdio.h>
#include <cstring>

static size_t bboxes_type = sizeof(int32_t);
static size_t centroids_type = sizeof(float);
static size_t crops_type = sizeof(uint8_t);

static int geometry_channels = 1;
static int centroid_cols = 2;
static int bboxes_cols = 4;
static int32_t zero = 0;

inline static void compute_bbox(size_t first_index, const int32_t* bboxes_array, const int img_rows, const int img_cols, const int32_t padding, int32_t& x1, int32_t& y1, int32_t& x2, int32_t& y2) {

    int32_t min_x = bboxes_array[first_index];
    int32_t max_x = min_x;
    int32_t min_y = bboxes_array[first_index + 1];
    int32_t max_y = min_y;

    for (int point = 1; point < 4; ++point) {
        const int index = first_index + point * 2;
        min_x = std::min(min_x, bboxes_array[index]);
        max_x = std::max(max_x, bboxes_array[index]);
        min_y = std::min(min_y, bboxes_array[index + 1]);
        max_y = std::max(max_y, bboxes_array[index + 1]);
    }

    x1 = std::max(zero, min_x - padding);
    y1 = std::max(zero, min_y - padding);
    x2 = std::min(img_cols, max_x + padding);
    y2 = std::min(img_rows, max_y + padding);
}

Buffer* get_bboxes(Buffer* bboxes_ptr, Buffer* img, const int32_t padding, size_t* total_buffers_ref) {
    const int img_rows = img->rows();
    const int img_cols = img->cols();
    const size_t img_rows_sized =  static_cast<size_t>(img_rows);
    const size_t img_cols_sized = static_cast<size_t>(img_cols);
    
    int bbox_count = bboxes_ptr->rows();
    
    Buffer valid_idx = Buffer(bbox_count, 1, geometry_channels, bboxes_type);
    int32_t* idx_data = valid_idx.get_int32_buffer();
    
    int32_t* bboxes_array = bboxes_ptr->get_int32_buffer();
    const uint8_t* img_data_const = static_cast<uint8_t*>(img->get_uint8_buffer());

    int32_t x1, y1, x2, y2;
    int valid = 0;
    size_t total_crops_size = 0;

    for (int bbox = 0; bbox < bbox_count; ++bbox) {
        const size_t first_index = static_cast<size_t>(bbox) * 8;
        compute_bbox(first_index, bboxes_array, img_rows, img_cols, padding, x1, y1, x2, y2);

        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        const int w = x2 - x1;
        const int h = y2 - y1;

        if (w < 1 || h < 1) {
            continue;
        }

        double sum = 0.0;
        for (int row = y1; row < y2; ++row) {
            const uint8_t* row_data =
                img_data_const +
                static_cast<size_t>(row) * img_cols_sized +
                static_cast<size_t>(x1);
            for (int col = 0; col < w; ++col) {
                sum += row_data[col];
            }
        }

        size_t crop_size = static_cast<size_t>(w) * static_cast<size_t>(h);
        const double mean = sum / (static_cast<double>(crop_size));

        if (mean < 7.0 || mean > 251.0) {
            continue;
        }

        total_crops_size += crop_size;
        idx_data[valid] = bbox;
        ++valid;
    }
    
    Buffer* cropped_raw = new Buffer(total_crops_size, 1, geometry_channels, crops_type);
    uint8_t* cropped_img = cropped_raw->get_uint8_buffer();

    int total_valid = valid + 3;
    total_buffers_ref[0] = static_cast<size_t>(total_valid);
    
    Buffer* cropped_polygons = new Buffer(total_valid);
    
    Buffer* bboxes = new Buffer(valid, bboxes_cols, geometry_channels, bboxes_type);
    int32_t* output = bboxes->get_int32_buffer();

    Buffer* centroids = new Buffer(valid, centroid_cols, geometry_channels, centroids_type);
    float* centroid_data = centroids->get_float32_buffer();

    size_t current_offset = 0;

    for (int k = 0; k < valid; ++k) {
        const int bbox = idx_data[k];
        const size_t first_index = static_cast<size_t>(bbox) * 8;
        compute_bbox(first_index, bboxes_array, img_rows, img_cols, padding, x1, y1, x2, y2);
        
        output[k * 4 + 0] = x1;
        output[k * 4 + 1] = y1;
        output[k * 4 + 2] = x2;
        output[k * 4 + 3] = y2;
        
        int32_t sum_x = 0;
        int32_t sum_y = 0;

        for (int point = 0; point < 4; ++point) {
            const size_t index = first_index + static_cast<size_t>(point) * 2;
            sum_x += bboxes_array[index];
            sum_y += bboxes_array[index + 1];
        }

        centroid_data[k * 2] = static_cast<float>(sum_x) / 4.0f;
        centroid_data[k * 2 + 1] = static_cast<float>(sum_y) / 4.0f;

        const int rows = x2 - x1; // width
        const int cols = y2 - y1; // height

        size_t size_rows = static_cast<size_t>(rows);

        size_t size_crop = size_rows * static_cast<size_t>(cols);

        cropped_polygons->set_struct_at(k, rows, cols, 1, size_crop, &cropped_img[current_offset]);

        for (int row = y1; row < y2; ++row) {
            const uint8_t* source_offset =
                img_data_const +
                static_cast<size_t>(row) * img_cols_sized +
                static_cast<size_t>(x1);

            const size_t destination_offset =
                current_offset +
                static_cast<size_t>(row - y1) * size_rows;

            memcpy(cropped_img + destination_offset, source_offset, size_rows);
        }
        current_offset += size_crop;
    }

    cropped_polygons->group_structs(valid, cropped_raw);
    cropped_polygons->group_structs(valid + 1, bboxes);
    cropped_polygons->group_structs(valid + 2, centroids);
    
    return cropped_polygons;
}
