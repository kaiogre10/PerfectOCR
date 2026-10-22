#include "image_loader.hpp"
#include <opencv2/core.hpp>
#include "c_utils.hpp"
#include "file_handler.hpp"
#include "buffer_container.hpp"

static const uint8_t dtype = sizeof(uint8_t);

Buffer* load_image(const char* filepath) {
    if (!filepath) {
        return nullptr;
    }
    // 1. Carga multiformato sin alterar canales originales (IMREAD_UNCHANGED)
    cv::Mat image_temp;
    files_handler::load_img(filepath, image_temp);
    if (image_temp.empty()) {
        return nullptr;
    }
    // 2. Normalización según espacio de color de entrada
    image_utils::normalize_image(image_temp);
    const int channels = image_temp.channels();

    if (channels != 1) {
        return nullptr;
    }
    // En este punto la imagen ya está en escala de grises normalizada a uint8
    const int cols = image_temp.cols;  
    const int rows = image_temp.rows;

    Buffer* image = new Buffer(rows, cols, channels, dtype);

    const size_t total_bytes = image_temp.total() * image_temp.elemSize();
    
    if (total_bytes != image->size()) {
        delete image;
        return nullptr;
    }

    memcpy(image->data(), image_temp.data, total_bytes);
    image_temp.release(); // Liberar imagen original inmediatamente
    return image;
}
