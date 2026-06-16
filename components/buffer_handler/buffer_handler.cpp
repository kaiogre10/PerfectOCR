#include "buffer_handler.h"
#include "../containers/containers.h"
#include <cstdlib>

// Mitigación de concurrencia: Cada hilo del ciclo de ejecución posee su propia instancia aislada
static thread_local PayloadContainer* g_container = nullptr;

extern "C" {

    void* storage_reserve(const char** strings,
        const size_t* sizes,
        size_t count,
        size_t* offsets_out) {
        // Garantizar la liberación de recursos asignados previamente en el mismo hilo
        if (g_container) {
            container_destroy(g_container);
            g_container = nullptr;
        }

        g_container = container_create(strings, sizes, count);
        if (!g_container) return nullptr;

        // Transferencia de offsets (Se asume invariante de diseño: offsets tiene tamaño count + 1)
        for (size_t i = 0; i <= count; i++) {
            offsets_out[i] = g_container->offsets[i];
        }

        return g_container->arena;
    }

    void storage_free(void* ptr) {
        // Monitorear que el puntero liberado corresponda efectivamente a la arena activa del hilo
        if (g_container && g_container->arena == ptr) {
            container_destroy(g_container);
            g_container = nullptr;
        }
    }

}
