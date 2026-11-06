#include "poly_gone.hpp"
#include "buffer_container.hpp"
#include <algorithm>
#include <cstdint>

static size_t bboxes_type = sizeof(int32_t);
static size_t centroids_type = sizeof(float);

static size_t geometry_channels = 1;
static size_t centroid_cols = 2;
static size_t bboxes_cols = 4;
static int32_t zero = 0;

inline static void compute_bbox(const int32_t* bboxes_array, size_t bbox, const size_t img_rows, const int img_cols, const int padding, int32_t& x1, int32_t& y1, int32_t& x2, int32_t& y2) {
    const size_t first_index = bbox * 8;

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

Buffer* calculate_geometry(Buffer* bboxes_ptr, Buffer* img, const int padding) {
    const int img_rows = img->rows();
    const int img_cols = img->cols();

    const uint8_t* data_const = static_cast<const uint8_t*>(img->get_uint8_buffer());
    int32_t* bboxes_array = bboxes_ptr->get_int32_buffer();
    
    Buffer* valid_idx = new Buffer(bbox_count, 1, geometry_channels, bboxes_type);
    int32_t* idx_data = valid_idx->get_int32_buffer();
}

Buffer* get_bboxes(Buffer* bboxes_ptr, Buffer* img, const int padding) {
    const int img_rows = img->rows();
    const int img_cols = img->cols();
    int bbox_count = bboxes_ptr->rows();
    
    Buffer* valid_idx = new Buffer(bbox_count, 1, geometry_channels, bboxes_type);
    int32_t* idx_data = valid_idx->get_int32_buffer();
    
    int32_t* bboxes_array = bboxes_ptr->get_int32_buffer();
    const uint8_t* data_const = img->get_uint8_buffer();

    int32_t x1, y1, x2, y2;
    int valid = 0;

    for (int bbox = 0; bbox < bbox_count; ++bbox) {
        compute_bbox(bboxes_array, bbox, img_rows, img_cols, padding, x1, y1, x2, y2);

        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        const int32_t w = x2 - x1;
        const int32_t h = y2 - y1;

        if (w < 1 || h < 1) {
            continue;
        }

        Ipp64f mean = 0.0;
        const Ipp8u* roi = data_const + (int32_t)y1 * img_cols + x1;
        IppStatus st = ippiMean_8u_C1R(roi, img_cols, (IppiSize){w, h}, &mean);
        if (st != ippStsNoErr) {
            continue;
        }
        
        if (mean < 7.0 || mean > 251.0) {
            continue;
        }

        idx_data[valid] = bbox;
        ++valid;
    }

    // if (valid == bbox_count) {
    //     sizes = valid_idx;
    //     sizes_out = idx_data;
    // } else {    
    //     delete valid_idx;
    //     sizes = new Buffer(valid, 1, geometry_channels, bboxes_type);
    //     sizes_out = static_cast<int32_t*>(sizes->data());
    // }
    
    Buffer* bboxes = new Buffer(valid, bboxes_cols, geometry_channels, bboxes_type);
    int32_t* output = static_cast<int32_t*>(bboxes->data());

    for (int k = 0; k < valid; ++k) {
        const int bbox = idx_data[k];
        compute_bbox(bboxes_array, bbox, img_rows, img_cols, padding, x1, y1, x2, y2);

        const int32_t w = x2 - x1;
        const int32_t h = y2 - y1;

        output[k * 4 + 0] = x1;
        output[k * 4 + 1] = y1;
        output[k * 4 + 2] = x2;
        output[k * 4 + 3] = y2;
    }

    delete bboxes_array;
    return bboxes
}

Buffer* get_centroids(int32_t* bboxes_array, size_t bbox_count) {
    // shape: (bbox_count, 2)
    Buffer* centroids = new Buffer(bbox_count, centroid_cols, geometry_channels, centroids_type);
    float* centroid_data = static_cast<float*>(centroids->data());

    for (size_t bbox = 0; bbox < bbox_count; ++bbox) {
        int32_t sum_x = 0;
        int32_t sum_y = 0;

        for (int point = 0; point < 4; ++point) {
            const int index = (bbox * 4 + point) * 2;
            sum_x += bboxes_array[index];
            sum_y += bboxes_array[index + 1];
        }

        centroid_data[bbox * 2] = static_cast<float>(sum_x) / 4.0f;
        centroid_data[bbox * 2 + 1] = static_cast<float>(sum_y) / 4.0f;
    }

    return centroids;
}
