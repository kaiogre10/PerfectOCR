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

    // 4. METADATOS Solo lectura
    int buffer_get_rows(const BufferContainer* buffer);
    int buffer_get_cols(const BufferContainer* buffer);
    int buffer_get_channels(const BufferContainer* buffer);
    size_t buffer_get_size(const BufferContainer* buffer);
    // Tipos Dinamicos
    uint8_t* get_uint8_data(BufferContainer* buffer);
    int32_t* get_int32_data(BufferContainer* buffer);
    float* get_float32_data(BufferContainer* buffer);
    BufferContainer* create_struct();
    BufferContainer* create_array_of_structs(int total_objects);
    void set_struct_at(BufferContainer* array, size_t index, int rows, int cols, int channels, size_t total_bytes, void* data);
    void group_array_structs(BufferContainer* array, size_t index, const BufferContainer* buffer, void* grouped_object);
    void* take_struct_data(BufferContainer* array, size_t index);
    void set_struct(BufferContainer* buffer, int rows, int cols, int channels, size_t total_bytes, void* data);
    BufferContainer* array_at(BufferContainer* array, size_t index);
    #ifdef __cplusplus
}
#endif
#endif
