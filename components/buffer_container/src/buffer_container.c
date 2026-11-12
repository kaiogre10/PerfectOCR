#include "buffer_container.h"
#include <stdlib.h>
#include <stddef.h>

// Definición PRIVADA del struct. Nadie más sabe cómo está ordenado en memoria.
struct BufferContainer {
    int rows;    // width
    int cols;    // height
    int channels;
    size_t total_bytes;
    void* data;
};

static const size_t container_size = sizeof(BufferContainer);

BufferContainer* create_array_of_structs(int total_objects) {
    if (total_objects == 0) {
        return NULL;
    }
    return calloc((size_t)total_objects, container_size);
}

BufferContainer* create_struct() {
    return (BufferContainer*)malloc(container_size);
}

BufferContainer* create_buffer(int rows, int cols, int channels, size_t dtype) {
    BufferContainer* buffer = create_struct();
    if (!buffer) {
        return NULL;
    }

    buffer->rows = rows;
    buffer->cols = cols;
    buffer->channels = channels;
    buffer->total_bytes = (size_t)rows * (size_t)cols * (size_t)channels * dtype;
    buffer->data = malloc(buffer->total_bytes);

    if (!buffer->data) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

void delete_buffer(BufferContainer* buffer) {
    if (buffer) {
        // El orden de liberación es CRÍTICO.
        if (buffer->data) {
            free(buffer->data);   // 1. Liberar los bytes
            buffer->data = NULL;  // 2. (Opcional) Marcar como nulo para evitar doble free
        }
        free(buffer);             // 3. Liberar el contenedor
    }
}
// EXPOSICIÓN TOTAL de los datos.
// Cualquiera con el puntero puede escribir donde quiera.
void* buffer_get_data(BufferContainer* buffer) {
    return buffer->data;
}
const void* buffer_get_data_const(const BufferContainer* buffer) {
    return buffer->data;
}
int buffer_get_rows(const BufferContainer* buffer) {
    return buffer->rows;
}
int buffer_get_cols(const BufferContainer* buffer) {
    return buffer->cols;
}
int buffer_get_channels(const BufferContainer* buffer) {
    return buffer->channels;
}
size_t buffer_get_size(const BufferContainer* buffer) {
    return buffer->total_bytes;
}
uint8_t* get_uint8_data(BufferContainer* buffer) {
    return (uint8_t*)buffer->data;
}
int32_t* get_int32_data(BufferContainer* buffer) {
    return (int32_t*)buffer->data;
}
float* get_float32_data(BufferContainer* buffer) {
    return (float*)buffer->data;
}
void set_struct_at(BufferContainer* array, size_t index, int rows, int cols, int channels, size_t total_bytes, void* data) {
    set_struct(&array[index], rows, cols, channels, total_bytes, data);
}
void group_array_structs(BufferContainer* array, size_t index, const BufferContainer* buffer, void* groped_object) {
    set_struct(&array[index], buffer->rows, buffer->cols, buffer->channels, buffer->total_bytes, groped_object);
}
void* take_struct_data(BufferContainer* array, size_t index) {
    void* data = array[index].data;
    array[index].data = NULL;
    return data;
}
void set_struct(BufferContainer* buffer, int rows, int cols, int channels, size_t total_bytes, void* data) {
    buffer->rows = rows;
    buffer->cols = cols;
    buffer->channels = channels;
    buffer->total_bytes = total_bytes;
    buffer->data = data;
}
BufferContainer* array_at(BufferContainer* array, size_t index) {
    if (!array) {
        return NULL;
    } 
    return &array[index];
}