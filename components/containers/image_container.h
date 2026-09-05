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
    ImageContainer* create_img_buffer(int width, int height, int channels);
    void delete_image(ImageContainer* img);

    // 3. EXPOSICIÓN DE MEMORIA (Acceso directo a los bytes)
    //    Cualquiera con el puntero puede leer/escribir aquí.
    uint8_t* image_get_data(ImageContainer* img);
    const uint8_t* image_get_data_const(const ImageContainer* img);

    // 4. METADATOS (Solo lectura)
    int image_get_width(const ImageContainer* img);
    int image_get_height(const ImageContainer* img);
    int image_get_channels(const ImageContainer* img);
    size_t image_get_size(const ImageContainer* img);
    #ifdef __cplusplus
}
#endif
#endif  // IMAGE_CONTAINER_H