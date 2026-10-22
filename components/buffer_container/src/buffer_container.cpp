#ifndef BUFFER_CONTAINER_CPP
#define BUFFER_CONTAINER_CPP
#include "buffer_container.hpp"

extern "C" {
    typedef struct Buffer Buffer;
    Buffer* create_buffer_c(int rows, int cols, int channels, size_t dtype) {
        try {
            return new Buffer(rows, cols, channels, dtype);
        } catch (...) {
            return nullptr;
        }
    }

    void destroy_buffer_c(Buffer* buffer) {
        delete buffer;
    }

    void* buffer_get_data_c(Buffer* buffer) {
        return buffer->data();
    }

    const void* buffer_get_data_const_c(const Buffer* buffer) {
        return buffer->data();
    }

    int buffer_get_rows_c(const Buffer* buffer) {
        return buffer->rows();
    }

    int buffer_get_cols_c(const Buffer* buffer) {
        return buffer->cols();
    }

    int buffer_get_channels_c(const Buffer* buffer) {
        return buffer->channels();
    }

    size_t buffer_get_size_c(const Buffer* buffer) {
        return buffer->size();
    }

} /* extern "C" */
#endif