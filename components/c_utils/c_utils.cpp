#include "c_utils.hpp"
#include <opencv2/imgproc.hpp>
#include <vector>

namespace {
    const int black_thr = 160;
    const int white_thr = 180;
    const inline cv::Scalar three_channels = cv::Scalar(255, 255, 255);
    const inline cv::Scalar four_channels = cv::Scalar(255, 255, 255, 255);
};

namespace image_utils {
    void decolorate(cv::Mat& image) {
        if (image.empty()) {
            return;
        }

        int channels = image.channels();
        if (channels < 3 || channels > 4) {
            image.release();
            return; // Solo soporta BGR o BGRA
        }
        // Crear máscaras para píxeles negros y blancos
        std::vector<cv::Mat> bgr_planes;
        cv::split(image, bgr_planes);

        // Para BGR: bgr_planes[0]=B, [1]=G, [2]=R
        cv::Mat black_condition = (bgr_planes[0] < black_thr) & (bgr_planes[1] < black_thr) & (bgr_planes[2] < black_thr);
        cv::Mat white_condition = (bgr_planes[0] > white_thr) & (bgr_planes[1] > white_thr) & (bgr_planes[2] > white_thr);
        cv::Mat mask_valid = black_condition | white_condition;

        // Rellenar píxeles no válidos con blanco
        if (channels == 3) {
                image.setTo(three_channels, ~mask_valid);
        } else { // channels == 4
                image.setTo(four_channels, ~mask_valid);
        }
            
        if (!validate_image(image)) {
            image.release();
        };
    }

    void make_contiguous(cv::Mat& image) {
        if (!image.isContinuous()) {
            cv::Mat contiguous_image = image.clone();
            image = contiguous_image;
        };
    }

    bool validate_image(cv::Mat& image) {
        if (image.empty() || image.channels() != 1) {
            return false;
        }

        cv::Scalar mean_val = cv::mean(image);
        double avg_brightness = mean_val[0];

        return (avg_brightness > 7.0 && avg_brightness < 251.0);
    }

    void normalize_image(cv::Mat& image) {
        // Verificar si la imagen es válida
        if (image.empty()) {
            return;
        }

        int channels = image.channels();
        int depth = image.depth();

        // Si la imagen tiene 3 o 4 canales, decolorar primero
        if (channels == 3 || channels == 4) {
            // Aplicar decolorate antes de convertir a gris
            decolorate(image);

            if (image.empty()) {  // decolorate la vació si falló validate_image
                return;
            }
            // Convertir a gris
            cv::Mat gray_image;

            if (channels == 3) {
                cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
            }
            else { // channels == 4
                cv::cvtColor(image, gray_image, cv::COLOR_BGRA2GRAY);
            }
            image = gray_image;
        }

        else if (channels == 2) {
            std::vector<cv::Mat> planes;
            cv::extractChannel(image, image, 0); // más directo, sin copiar el canal alpha
        }

        else if (channels != 1) {
            // Si no es 1, 2, 3 o 4 canales, 
            image.release();
            return;
        }

        // Ahora image debería tener 1 canal (gris)
        if (image.channels() != 1) {
            image.release();
            return;
        }

        // Convertir a uint8 si es necesario
        if (depth == CV_32F || depth == CV_64F) {
            double min_val, max_val;
            cv::minMaxLoc(image, &min_val, &max_val);

            if (max_val <= 1.0) {
                // Escalar de [0,1] a [0,255]
                image.convertTo(image, CV_8UC1, 255.0);
            }
            else {
                // Convertir a uint8 con clipping
                image.convertTo(image, CV_8UC1);
            }
        }
        else if (depth != CV_8UC1) {
            // Convertir cualquier otro tipo a uint8
            image.convertTo(image, CV_8UC1);
        }

        if (!validate_image(image)) {
            image.release();
            return;
        }
        make_contiguous(image);
    }
}