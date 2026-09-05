#include "image_loader.h"
#include <opencv2/core.hpp>
#include "../c_utils/c_utils.hpp"
#include <cstdint>
#include <cstdlib>
#include <vector>

extern "C" {
    void load_image(const char* filepath) {
        if (!filepath) {
            return;
        }

        // 1. Carga multiformato sin alterar canales originales (IMREAD_UNCHANGED)
        cv::Mat image = cv::imread(filepath, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            return;
        }
        // 2. Normalización según espacio de color de entrada
        image_utils::normalize_image(image);

        // En este punto la imagen ya está en escala de grises normalizada a uint8
        size_t total_bytes = image.step[0] * image.rows;
        uint8_t* heap_data = static_cast<uint8_t*>(std::malloc(total_bytes));
        // size_t total_bytes = continuous_mat.total() * sizeof(uint8_t);
        if (!heap_data || heap_data == nullptr) {
            return;
        }
        // std::memcpy(heap_data, continuous_mat.data, total_bytes);

        // 6. Empaquetar descriptor de imagen
        // FullImage* result = static_cast<FullImage*>(std::malloc(sizeof(FullImage)));
        // if (!result) {
        //     std::free(heap_data);
        //     return;
        // }
        return;
    }
    void free_image_buffer(FullImage* buf) {
        if (buf) {
            if (buf->data) {
                std::free(buf->data);
                buf->data = nullptr;
            }
            std::free(buf);
        }
    };

}
