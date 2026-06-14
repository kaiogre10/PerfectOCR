#ifndef BUFFER_HANDLER_H
#define BUFFER_HANDLER_H

#include <cstddef>

// Configuración dinámica de exportación según el Sistema Operativo
#if defined(_WIN32)
    #define EXPORT_API __declspec(dllexport)
#else
    #define EXPORT_API __attribute__((visibility("default")))
#endif

extern "C" {
    EXPORT_API void* storage_reserve(size_t size);
    EXPORT_API void storage_commit(void* ptr, size_t size);
    EXPORT_API void storage_free(void* ptr);
}

#endif // BUFFER_HANDLER_H
