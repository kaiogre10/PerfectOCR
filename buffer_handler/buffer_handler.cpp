#include "buffer_handler.h"
#include "../components/containers/container.h"
#include <cstdlib>

// Container vivo mientras el bloque esté en uso
static PayloadContainer* g_container = nullptr;

extern "C" {

void* storage_reserve(const char** strings,
                      const size_t* sizes,
                      size_t count,
                      size_t* offsets_out) {
    if (g_container) {
        container_destroy(g_container);
        g_container = nullptr;
    }

    g_container = container_create(strings, sizes, count);
    if (!g_container) return nullptr;

    // Copiar offsets al caller
    for (size_t i = 0; i <= count; i++)
        offsets_out[i] = g_container->offsets[i];

    return g_container->arena;
}

void storage_free(void* ptr) {
    (void)ptr; // arena pertenece al container
    container_destroy(g_container);
    g_container = nullptr;
}

}