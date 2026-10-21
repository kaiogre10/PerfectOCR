#include "angle_corrector.hpp"
#include "image_container.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

const cv::Scalar WHITE_COLOR = cv::Scalar(255);

Image* rotate_image(Image* source, double angle, double center_x, double center_y) {
    const int rows = source->rows();
    const int cols = source->cols();
    const int channels = source->channels();

    const cv::Point2f center(static_cast<float>(center_x), static_cast<float>(center_y));
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);

    const double cos_value = std::abs(rotation_matrix.at<double>(0, 0));
    const double sin_value = std::abs(rotation_matrix.at<double>(0, 1));

    const int new_rows = static_cast<int>((cols * sin_value) + (rows * cos_value));
    const int new_cols = static_cast<int>((cols* cos_value) + (rows * sin_value));

    if (new_rows <= 0 || new_cols <= 0) {
        return nullptr;
    }

    rotation_matrix.at<double>(0, 2) += (new_cols / 2.0) - center_x;
    rotation_matrix.at<double>(1, 2) += (new_rows / 2.0) - center_y;
    
    const uint8_t dtype = sizeof(uint8_t);
    Image* rotated = new Image(new_rows, new_cols, channels, dtype);
    static int cv_type = CV_MAKETYPE(CV_8UC1, channels);
    cv::Mat source_view(rows, cols, cv_type, source->data());
    cv::Mat destination_view(new_rows, new_cols, cv_type, rotated->data());

    cv::warpAffine(source_view, destination_view, rotation_matrix, cv::Size(new_cols, new_rows), cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, WHITE_COLOR);
    return rotated;
}
