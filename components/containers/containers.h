#pragma once
#include <vector>
#include <cstdint>
#include <queue>

extern "C" {
    void container_create(int trigger);
}

void push(std::vector<uint8_t>&& plain_payload);
std::queue<std::vector<uint8_t>> drain();
