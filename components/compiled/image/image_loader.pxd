# image_loader.pxd
from image cimport Image
from libc.stdint cimport int32_t

cdef extern from "image_loader.hpp":
    Image* load_image(const char* filepath) except +

cdef extern from "angle_corrector.hpp":
    Image* rotate_image(Image* source, double angle, double center_x, double center_y) except +

cdef extern from "poly_gone.hpp":
    Image* get_centroids(int32_t* bboxes_array, int bbox_count) except +
    Image* get_bboxes(int32_t* bboxes_array, int bbox_count, int img_rows, int img_cols, int padding) except +