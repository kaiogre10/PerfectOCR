#ifndef IMAGE_CONTAINER_CCP
#define IMAGE_CONTAINER_CCP
#include "image_container.hpp"

extern "C" {
    typedef struct Image Image;
    Image* create_image_c(int rows, int cols, int channels, size_t dtype) {
        try {
            return new Image(rows, cols, channels, dtype);
        } catch (...) {
            return nullptr;
        }
    }

    void destroy_image_c(Image* img) {
        delete img;
    }

    void* image_get_data_c(Image* img) {
        return img->data();
    }

    const void* image_get_data_const_c(const Image* img) {
        return img->data();
    }

    int image_get_rows_c(const Image* img) {
        return img->rows();
    }

    int image_get_cols_c(const Image* img) {
        return img->cols();
    }

    int image_get_channels_c(const Image* img) {
        return img->channels();
    }

    size_t image_get_size_c(const Image* img) {
        return img->size();
    }

} /* extern "C" */
#endif