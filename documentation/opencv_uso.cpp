#include <opencv2/core.hpp>       // Estructuras base: cv::Mat, cv::Point, cv::Scalar
#include <opencv2/imgproc.hpp>    // Procesamiento: cv::resize, cv::cvtColor, cv::threshold
#include <opencv2/imgcodecs.hpp>   // Entrada/Salida: cv::imread, cv::imdecode

// Ejemplo de función dentro de tu librería
void procesar_frame(unsigned char* data, int width, int height) {
    // Creación de una matriz OpenCV envolviendo el buffer sin copiar memoria
    cv::Mat imagen(height, width, CV_8UC3, data);

    // Operación nativa optimizada con AVX2
    cv::Mat gris;
    cv::cvtColor(imagen, gris, cv::COLOR_BGR2GRAY);
}