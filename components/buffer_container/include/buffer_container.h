#ifndef BUFFER_CONTAINER_H
#define BUFFER_CONTAINER_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
    #endif
    // 1. Estructura OPACA. El usuario solo ve un puntero incompleto.
    typedef struct BufferContainer BufferContainer;

    // 2. FUNCIONES DE CICLO DE VIDA
    BufferContainer* create_buffer(int rows, int cols, int channels, size_t dtype);
    void delete_buffer(BufferContainer* buffer);

    // 3. EXPOSICIÓN DE MEMORIA (Acceso directo a los bytes)
    //    Cualquiera con el puntero puede leer/escribir aquí.
    void* buffer_get_data(BufferContainer* buffer);
    const void* buffer_get_data_const(const BufferContainer* buffer);

    // 4. METADATOS (Solo lectura)
    int buffer_get_rows(const BufferContainer* buffer);
    int buffer_get_cols(const BufferContainer* buffer);
    int buffer_get_channels(const BufferContainer* buffer);
    size_t buffer_get_size(const BufferContainer* buffer);
    #ifdef __cplusplus
}
#endif
#endif
