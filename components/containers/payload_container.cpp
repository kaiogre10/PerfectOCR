#include "containers.h"
#include <vector>
#include <deque>
#include <cstdint>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <fstream>

struct PayloadContainer {
    std::deque<std::vector<std::vector<uint8_t>>> payload_container;
    std::mutex mtx;
};

static PayloadContainer* g_canal = nullptr;

extern "C" void container_create(int trigger) {
    if (trigger != 0 && g_canal == nullptr) {
        g_canal = new PayloadContainer(); // El contenedor nace vacío en el heap
    }
}

void push(std::vector<std::vector<uint8_t>>&& struct_payload) {
    if (!g_canal) return;

    std::lock_guard<std::mutex> lock(g_canal->mtx);
    std::ofstream archivo_log("verificacion_init.log", std::ios::app);

    if(archivo_log.is_open()) {
        for (const auto& fila : struct_payload) {
            // Equivale exactamente a: "".join(fila) de Python
            archivo_log << std::string(fila.begin(), fila.end()) << "\n";
        }
        archivo_log.close();
    }
    g_canal->payload_container.push_back(std::move(struct_payload));
}

std::deque<std::vector<std::vector<uint8_t>>> drain() {
    std::deque<std::vector<std::vector<uint8_t>>> local;
    {
        std::lock_guard<std::mutex> lock(g_canal->mtx);
        // El consumidor se lleva TODOS los lotes acumulados hasta el momento de golpe
        g_canal->payload_container.swap(local); 
    }
    return local;
}