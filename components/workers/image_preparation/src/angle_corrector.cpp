#include "angle_corrector.hpp"
#include "image_container.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

const cv::Scalar WHITE_COLOR = cv::Scalar(255);

Image* rotate_image(Image* source, double angle, double center_x, double center_y) {
    const int width = source->width();
    const int height = source->height();
    const int channels = source->channels();

    const cv::Point2f center(static_cast<float>(center_x), static_cast<float>(center_y));
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);

    const double cos_value = std::abs(rotation_matrix.at<double>(0, 0));
    const double sin_value = std::abs(rotation_matrix.at<double>(0, 1));

    const int new_width = static_cast<int>((height * sin_value) + (width * cos_value));
    const int new_height = static_cast<int>((height * cos_value) + (width * sin_value));

    if (new_width <= 0 || new_height <= 0) {
        return nullptr;
    }

    rotation_matrix.at<double>(0, 2) += (new_width / 2.0) - center_x;
    rotation_matrix.at<double>(1, 2) += (new_height / 2.0) - center_y;
    const uint8_t dims = 1;

    Image* rotated = new Image(new_width, new_height, channels, dims);
    static int cv_type = CV_MAKETYPE(CV_8UC1, channels);
    cv::Mat source_view(height, width, cv_type, source->data());
    cv::Mat destination_view(new_height, new_width, cv_type, rotated->data());

    cv::warpAffine(source_view, destination_view, rotation_matrix, cv::Size(new_width, new_height), cv::INTER_CUBIC, cv::BORDER_CONSTANT, WHITE_COLOR);
    return rotated;
}
