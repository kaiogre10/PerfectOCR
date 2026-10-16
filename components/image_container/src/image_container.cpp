#ifndef IMAGE_CONTAINER_CCP
#define IMAGE_CONTAINER_CCP
#include "image_container.hpp"

extern "C" {
    typedef struct Image Image;
    Image* create_image_c(int width, int height, int channels, size_t dtype) {
        try {
            return new Image(width, height, channels, dtype);
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

    int image_get_width_c(const Image* img) {
        return img->width();
    }

    int image_get_height_c(const Image* img) {
        return img->height();
    }

    int image_get_channels_c(const Image* img) {
        return img->channels();
    }

    size_t image_get_size_c(const Image* img) {
        return img->size();
    }

} /* extern "C" */
#endif