#include "buffer_handler.h"
#include "../containers/containers.h"
#include <vector>
#include <cstdint>
#include <mutex>
#include <iostream>

uint8_t* buffer_ptr = nullptr;
size_t buffer_size = 0;

namespace {
    void storage_batch_flat(uint8_t* buffer_ptr, size_t buffer_size) {
        if (!buffer_ptr || !buffer_size) return;
        printf("[BUFFER HANDLER LOG] ptr: %p, size: %zu\n", buffer_ptr, buffer_size);
        std::vector<uint8_t> plain_payload(buffer_ptr, buffer_ptr + buffer_size);
        for (uint8_t byte : plain_payload) {
            std::cout << (char)byte;
        }
        std::cout << std::flush;
        std::cout << "\nSize: " << plain_payload.size() << "\n";
        push(std::move(plain_payload));
    }
}

// namespace Send {
//     void restruct_final_payloads(std::queue<std::vector<uint8_t>> payloads, HANDLE pipe_handle) {
//         while (!payloads.empty()) {
//             std::vector<uint8_t> lote = std::move(payloads.front());
//             payloads.pop();
//
//             uint32_t len = static_cast<uint32_t>(lote.size());
//             WriteFile(pipe_handle, &len, sizeof(len), nullptr, nullptr);
//             WriteFile(pipe_handle, lote.data(), len, nullptr, nullptr);
//         }
//     }
// }

extern "C" {
    uint8_t* reserve_buffer(size_t len_bytes) {
        buffer_ptr = new uint8_t[len_bytes];
        buffer_size = len_bytes;
        return buffer_ptr;
    }
    void commit_buffer(int signal) {
        if (signal > 0) {
            try {
                storage_batch_flat(buffer_ptr, buffer_size);
                delete[] buffer_ptr;
            }
            catch (...) {
                delete[] buffer_ptr;
            }
            buffer_ptr = nullptr;
            buffer_size = 0;
        }
    }
    // void send_payloads(int trigger) {
    //     if (trigger > 0) {
    //         try {
    //             std::queue<std::vector<uint8_t>> payloads = drain();
    //             Send::restruct_final_payloads(std::move(payloads));
    //         }
    //         catch (...) {
    //             return;
    //         }
    //     }
    // }
}

// size_t offset_view = 0;

// for (size_t i = 0; i < total_cols; ++i) {
//     size_t actual_size = buffer_size[i];
//
//     if (actual_size > 0) {
// plain_payload.reserve(buffer_size);
// plain_payload.assign(buffer_ptr + buffer_size),
//             buffer_ptr + offset_view + actual_size);
//
//         offset_view += actual_size;
// }
// }