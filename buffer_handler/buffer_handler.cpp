// El include debe reflejar el nuevo nombre del archivo
#include "buffer_handler.h"
#include <cstdlib>

extern "C" {

    void* storage_reserve(size_t size) {
        if (size == 0) return nullptr;
        return std::malloc(size);
    }

    void storage_commit(void* ptr, size_t size) {
        if (!ptr) return;
        (void)ptr;
        (void)size;
    }

    void storage_free(void* ptr) {
        if (ptr) {
            std::free(ptr);
        }
    }
}