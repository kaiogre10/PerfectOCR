#include "image_loader.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

extern "C" {
    void load_image(const char* filepath) {
        if (!filepath) return;

        // 1. Carga multiformato sin alterar canales originales (IMREAD_UNCHANGED)
        cv::Mat raw = cv::imread(filepath, cv::IMREAD_UNCHANGED);
        if (raw.empty()) return;

        cv::Mat normalized_gray;

        // 2. Normalización según espacio de color de entrada
        int channels = raw.channels();

        if (channels == 1) {
            // Ya es monocromática / escala de grises
            normalized_gray = raw;
        }
        else if (channels == 3) {
            // Imagen BGR estándar: Aquí ejecutas correcciones cromáticas previas si aplica
            // cv::Mat color_corrected = ...
            cv::cvtColor(raw, normalized_gray, cv::COLOR_BGR2GRAY);
        }
        else if (channels == 4) {
            // Imagen con canal Alpha (BGRA) -> se descarta transparencia hacia escala de grises
            cv::cvtColor(raw, normalized_gray, cv::COLOR_BGRA2GRAY);
        }
        else {
            // Formatos atípicos o no soportados
            return;
        }

        // 3. Asegurar profundidad de 8 bits (uint8)
        if (normalized_gray.depth() != CV_8U) {
            normalized_gray.convertTo(normalized_gray, CV_8U);
        }

        // 4. Garantizar memoria continua (C-contiguous)
        cv::Mat continuous_mat;
        if (!normalized_gray.isContinuous()) {
            continuous_mat = normalized_gray.clone();
        } else {
            continuous_mat = normalized_gray;
        }

        // 5. Asignación manual del búfer Heap en C
        size_t total_bytes = continuous_mat.total() * sizeof(uint8_t);
        uint8_t* heap_data = static_cast<uint8_t*>(std::malloc(total_bytes));
        if (!heap_data) return;

        std::memcpy(heap_data, continuous_mat.data, total_bytes);

        // 6. Empaquetar descriptor de imagen
        FullImage* result = static_cast<FullImage*>(std::malloc(sizeof(FullImage)));
        if (!result) {
            std::free(heap_data);
            return;
        }

        result->data = heap_data;
        result->width = continuous_mat.cols;
        result->height = continuous_mat.rows;
        result->channels = 1;

        return;
    };

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
