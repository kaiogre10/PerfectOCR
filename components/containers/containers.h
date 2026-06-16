#pragma once
#include <cstddef>

struct PayloadContainer {
    char*   arena;     // bloque contiguo dueño de los datos
    size_t* offsets;   // offsets[i] = inicio del elemento i en arena
    size_t  count;
    size_t  total;     // bytes totales del arena
};

PayloadContainer* container_create(const char** strings,
                                   const size_t* sizes,
                                   size_t count);
void              container_destroy(PayloadContainer* c);
