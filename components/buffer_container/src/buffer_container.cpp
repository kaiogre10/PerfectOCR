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

    const void* buffer_get_read_only_data_const_c(const Buffer* buffer) {
        return buffer->read_only_data();
    }

    int buffer_get_rows_c(Buffer* buffer) {
        return buffer->rows();
    }

    int buffer_get_cols_c(Buffer* buffer) {
        return buffer->cols();
    }

    int buffer_get_channels_c(Buffer* buffer) {
        return buffer->channels();
    }

    size_t buffer_get_size_c(Buffer* buffer) {
        return buffer->size();
    }
    uint8_t* get_uint8_buffer_c(Buffer* buffer) {
        return buffer->get_uint8_buffer();
    }
    int32_t* get_int32_buffer_c(Buffer* buffer) {
        return buffer->get_int32_buffer();
    }
    float* get_float32_buffer_c(Buffer* buffer) {
        return buffer->get_float32_buffer();
    }

} /* extern "C" */
#endif