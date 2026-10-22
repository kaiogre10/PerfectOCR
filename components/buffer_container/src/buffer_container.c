#include "buffer_container.h"
#include <stdlib.h>
#include <stdio.h>

// Definición PRIVADA del struct. Nadie más sabe cómo está ordenado en memoria.
struct BufferContainer {
    int rows;    // width
    int cols;    // height
    int channels;
    size_t total_bytes;
    void* data;
};

BufferContainer* create_buffer(int rows, int cols, int channels, size_t dtype) {
    BufferContainer* buffer = (BufferContainer*)malloc(sizeof(BufferContainer));
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
