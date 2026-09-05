#pragma once
#include <opencv2/imgproc.hpp>
// #include <opencv2/core.hpp>

namespace image_utils {
    // 1. Sin parámetros extra: Estiramiento lineal de contraste a rango completo [0, 255]
    void decolorate(cv::Mat& image);

    // 2. Sin parámetros extra: Inversión de color in-place (ej. fondo negro con texto blanco)
    void make_contiguous(cv::Mat& image);

    // 3. Con parámetros específicos: Reducción de ruido gaussiano leve
    bool validate_image(cv::Mat& image);

    // 4. Con parámetros específicos: Corrección Gamma (curva de iluminación no lineal)
    void normalize_image(cv::Mat& image);
}