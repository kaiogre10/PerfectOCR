#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {
    uint8_t* reserve_buffer(size_t len_bytes);
    void commit_buffer(int signal);
}
