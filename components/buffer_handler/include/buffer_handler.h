#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "../image_container/image.hpp"

#ifdef __cplusplus
extern "C" {
#endif
    void create_deque();
    uint8_t* reserve_buffer(size_t len_bytes);
    void commit_buffer();
    bool release_image(ImageContainer* image_ptr);
    // void send_payloads(int trigger);
#ifdef __cplusplus
}
#endif