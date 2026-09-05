#include "image_container.h"
#include <stdlib.h>

// Definición PRIVADA del struct. Nadie más sabe cómo está ordenado en memoria.
struct ImageContainer {
    int width;
    int height;
    int channels;
    size_t total_bytes;
    uint8_t* data;  // <- El puntero a los píxeles
};

ImageContainer* create_img_buffer(int width, int height, int channels) {
    ImageContainer* img = (ImageContainer*)malloc(sizeof(ImageContainer));
    if (!img) {
        return NULL;
    }

    img->width = width;
    img->height = height;
    img->channels = channels;
    img->total_bytes = (size_t)width * height * channels;
    img->data = (uint8_t*)malloc(img->total_bytes);

    if (!img->data) {
        free(img);
        return NULL;
    }
    return img;
}

void delete_image(ImageContainer* img) {
    if (img) {
        // El orden de liberación es CRÍTICO. TÚ lo controlas.
        if (img->data) {
            free(img->data);   // 1. Liberas los píxeles
            img->data = NULL;  // 2. (Opcional) Marcas como nulo para evitar doble free
        }
        free(img);             // 3. Liberas el contenedor
    }
}

// EXPOSICIÓN TOTAL de los datos.
// Cualquiera con el puntero puede escribir donde quiera.
uint8_t* image_get_data(ImageContainer* img) {
    return img->data;
}
const uint8_t* image_get_data_const(const ImageContainer* img) {
    return img->data;
}
int image_get_width(const ImageContainer* img) {
    return img->width;
}
int image_get_height(const ImageContainer* img) {
    return img->height;
}
int image_get_channels(const ImageContainer* img) {
    return img->channels;
}
size_t image_get_size(const ImageContainer* img) {
    return img->total_bytes;
}