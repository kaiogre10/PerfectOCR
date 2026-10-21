#ifndef IMAGE_CONTAINER_H
#define IMAGE_CONTAINER_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
    #endif
    // 1. Estructura OPACA. El usuario solo ve un puntero incompleto.
    typedef struct ImageContainer ImageContainer;

    // 2. FUNCIONES DE CICLO DE VIDA
    ImageContainer* create_img_buffer(int rows, int cols, int channels, size_t dtype);
    void delete_image(ImageContainer* img);

    // 3. EXPOSICIÓN DE MEMORIA (Acceso directo a los bytes)
    //    Cualquiera con el puntero puede leer/escribir aquí.
    void* image_get_data(ImageContainer* img);
    const void* image_get_data_const(const ImageContainer* img);

    // 4. METADATOS (Solo lectura)
    int image_get_rows(const ImageContainer* img);
    int image_get_cols(const ImageContainer* img);
    int image_get_channels(const ImageContainer* img);
    size_t image_get_size(const ImageContainer* img);
    #ifdef __cplusplus
}
#endif
#endif
