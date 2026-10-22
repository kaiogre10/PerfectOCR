#include "poly_gone.hpp"
#include "buffer_container.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>

static size_t bboxes_type = sizeof(int32_t);
static int geometry_channels = 1;
static size_t centroids_type = sizeof(float);
static int centroid_cols = 2;
static int bboxes_rows = 4;
static int32_t zero = 0;

Buffer* get_centroids(int32_t* bboxes_array, int bbox_count) {
    // shape: (bbox_count, 2)
    Buffer* centroids = new Buffer(bbox_count, centroid_cols, geometry_channels, centroids_type);
    float* centroid_data = static_cast<float*>(centroids->data());

    for (int bbox = 0; bbox < bbox_count; ++bbox) {
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

inline void compute_bbox(const int32_t* bboxes_array, int bbox, int img_rows, int img_cols, int padding, int32_t& x1, int32_t& y1, int32_t& x2, int32_t& y2) {
    const int first_index = bbox * 8;

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

Buffer* get_bboxes(int32_t* bboxes_array, int bbox_count, int img_rows, int img_cols, int padding) {
    // shape: (bbox_count, 1)
    Buffer* valid_idx = new Buffer(bbox_count, 1, geometry_channels, bboxes_type);
    int32_t* idx_data = static_cast<int32_t*>(valid_idx->data());

    int32_t x1, y1, x2, y2;
    int valid = 0;

    for (int bbox = 0; bbox < bbox_count; ++bbox) {
        compute_bbox(bboxes_array, bbox, img_rows, img_cols, padding, x1, y1, x2, y2);
        if (x2 > x1 && y2 > y1) {
            idx_data[valid] = bbox;
            ++valid;
        }
    }

    // shape: (valid, 4)
    if (valid > bbox_count){
        return nullptr;
    }
    
    Buffer* bboxes = new Buffer(valid, bboxes_rows, geometry_channels, bboxes_type);
    int32_t* output = static_cast<int32_t*>(bboxes->data());

    for (int k = 0; k < valid; ++k) {
        const int bbox = idx_data[k];
        compute_bbox(bboxes_array, bbox, img_rows, img_cols, padding, x1, y1, x2, y2);

        output[k * 4 + 0] = x1;
        output[k * 4 + 1] = y1;
        output[k * 4 + 2] = x2;
        output[k * 4 + 3] = y2;
    }
    return bboxes;
}
