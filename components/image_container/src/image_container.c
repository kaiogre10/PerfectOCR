#include "image_container.h"
#include <stdlib.h>
#include <stdio.h>

// Definición PRIVADA del struct. Nadie más sabe cómo está ordenado en memoria.
struct ImageContainer {
    int rows;    // width
    int cols;    // height
    int channels;
    size_t total_bytes;
    void* data;
};

ImageContainer* create_img_buffer(int rows, int cols, int channels, size_t dtype) {
    ImageContainer* img = (ImageContainer*)malloc(sizeof(ImageContainer));
    if (!img) {
        return NULL;
    }

    img->rows = rows;
    img->cols = cols;
    img->channels = channels;
    img->total_bytes = (size_t)rows * (size_t)cols * (size_t)channels * dtype;
    img->data = malloc(img->total_bytes);

    if (!img->data) {
        free(img);
        return NULL;
    }
    return img;
}

void delete_image(ImageContainer* img) {
    if (img) {
        // El orden de liberación es CRÍTICO.
        if (img->data) {
            free(img->data);   // 1. Liberar los píxeles
            img->data = NULL;  // 2. (Opcional) Marcar como nulo para evitar doble free
        }
        free(img);             // 3. Liberar el contenedor
    }
}

// EXPOSICIÓN TOTAL de los datos.
// Cualquiera con el puntero puede escribir donde quiera.
void* image_get_data(ImageContainer* img) {
    return img->data;
}
const void* image_get_data_const(const ImageContainer* img) {
    return img->data;
}
int image_get_rows(const ImageContainer* img) {
    return img->rows;
}
int image_get_cols(const ImageContainer* img) {
    return img->cols;
}
int image_get_channels(const ImageContainer* img) {
    return img->channels;
}
size_t image_get_size(const ImageContainer* img) {
    return img->total_bytes;
}
