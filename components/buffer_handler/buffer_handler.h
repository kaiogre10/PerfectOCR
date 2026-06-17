#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {
    void storage_batch_flat(const uint8_t* plain_data,
                            const size_t* len_list, 
                            size_t total_bytes);
}
