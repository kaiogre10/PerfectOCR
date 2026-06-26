#pragma once
#include <vector>
#include <cstdint>
#include <queue>

extern "C" {
    // La señal mínima en las primeras líneas de main que acciona todo
    void container_create(int trigger);
    //void load_config(const char* config_path);
}

void push(std::vector<uint8_t>&& plain_payload);
std::queue<std::vector<uint8_t>> drain();
