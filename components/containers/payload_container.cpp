#include "containers.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

PayloadContainer* container_create(const char** strings,
                                   const size_t* sizes,
                                   size_t count) {
    if (!strings || !sizes || count == 0) {
        printf("container_create: validacion fallida strings=%p sizes=%p count=%zu\n", 
               (void*)strings, (void*)sizes, count);
        return nullptr;
    }

    size_t total = 0;
    for (size_t i = 0; i < count; i++) total += sizes[i];

    auto* c = static_cast<PayloadContainer*>(std::malloc(sizeof(PayloadContainer)));
    if (!c) return nullptr;

    c->arena   = static_cast<char*>(std::malloc(total));
    c->offsets = static_cast<size_t*>(std::malloc((count + 1) * sizeof(size_t)));
    c->count   = count;
    c->total   = total;

    if (!c->arena || !c->offsets) {
        container_destroy(c);
        return nullptr;
    }

    size_t off = 0;
    for (size_t i = 0; i < count; i++) {
        c->offsets[i] = off;
        std::memcpy(c->arena + off, strings[i], sizes[i]);
        off += sizes[i];
    }
    c->offsets[count] = total; // sentinel

    return c;
}

void container_destroy(PayloadContainer* c) {
    if (!c) return;
    std::free(c->arena);
    std::free(c->offsets);
    std::free(c);
}