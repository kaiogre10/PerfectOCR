#include "image_loader.hpp"
#include <opencv2/core.hpp>
#include "../c_utils/c_utils.hpp"
#include "../file_handler/file_handler.hpp"
#include "../image_container/image.hpp"

Image* load_image(const char* filepath) {
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

    if (image_temp.channels() != 1) {
        return nullptr;
    }
    // En este punto la imagen ya está en escala de grises normalizada a uint8
    const int width = image_temp.cols;    // Número de columnas = ancho
    const int height = image_temp.rows;   // Número de filas = alto

    Image* image = new Image(image_temp.cols, image_temp.rows, 1);

    const size_t total_bytes = image_temp.total() * image_temp.elemSize();
    if (total_bytes != image->size()) {
        delete image;
        return nullptr;
    }

    memcpy(image->data(), image_temp.data, total_bytes);
    image_temp.release();   // Liberar imagen original inmediatamente
    return image;
}

