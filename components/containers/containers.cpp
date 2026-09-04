#include "containers.h"
#include <vector>
#include <queue>
#include <cstdint>
#include <mutex>

struct PayloadContainer {
    std::queue<std::vector<uint8_t>> payload_container;
    std::mutex mtx;
};

static PayloadContainer* g_canal = nullptr;

extern "C" {
    void container_create(const int trigger) {
        if (trigger > 0 && g_canal == nullptr) 
            g_canal = new PayloadContainer(); // El contenedor nace vacío en el heap
    };
    uint8_t* create_storage_image()
}

void push(std::vector<uint8_t>&& plain_payload) {
    if (!g_canal) return;
    std::lock_guard<std::mutex> lock(g_canal->mtx);
    g_canal->payload_container.push(std::move(plain_payload));
}

std::queue<std::vector<uint8_t>> drain() {
    std::queue<std::vector<uint8_t>> local;
    {
        std::lock_guard<std::mutex> lock(g_canal->mtx);
        // El consumidor se lleva TODOS los lotes acumulados hasta el momento de golpe
        g_canal->payload_container.swap(local);
    }
    return local;
}
