#pragma once
#include <cstddef>

extern "C" {
    // Retorna puntero base al arena y llena offsets[]
    // offsets debe tener espacio para (count + 1) elementos
    void* storage_reserve(const char** strings,
                          const size_t* sizes,
                          size_t count,
                          size_t* offsets_out);

    void storage_free(void* ptr);
}