#pragma once
#include <vector>
#include <cstdint>
#include <queue>

extern "C" {
    // La señal mínima en las primeras líneas de main que acciona todo
    void container_create(int trigger);
}

void push(std::vector<std::vector<uint8_t>>&& struct_payload);
std::queue<std::vector<std::vector<uint8_t>>> drain();
