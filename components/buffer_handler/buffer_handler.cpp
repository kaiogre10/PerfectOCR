#include "buffer_handler.h"
#include "../containers/containers.h"
#include <cstdlib>
#include <deque>
#include <vector>

// Declaración de la función puente (definida en Contenedor.cpp)
void push(std::vector<std::vector<uint8_t>>&& struct_payload);

extern "C" void storage_batch_flat(const uint8_t* plain_data,
                                   const size_t* len_list, 
                                   size_t total_bytes) {
    
    if (!plain_data || !len_list || total_bytes == 0) return;

    // El productor procesa y reconstruye de manera privada
    std::vector<std::vector<uint8_t>> struct_payload;
    struct_payload.resize(total_bytes);

    size_t offset_view = 0;

    for (size_t i = 0; i < total_bytes; ++i) {
        size_t actual_size = len_list[i];

        if (actual_size > 0) {
            struct_payload[i].reserve(actual_size);
            struct_payload[i].assign(plain_data + offset_view,
                plain_data + offset_view + actual_size);
            
            offset_view += actual_size;
        }
    }
    push(std::move(struct_payload));
}
