#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
    void create_deque();
    uint8_t* reserve_buffer(size_t len_bytes);
    void commit_buffer();
    // void send_payloads(int trigger);
#ifdef __cplusplus
}
#endif