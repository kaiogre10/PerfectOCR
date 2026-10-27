# bidings/container/workers.pxd
from container cimport Buffer
from libc.stdint cimport int32_t

cdef extern from "image_loader.hpp":
    Buffer* load_image(const char* filepath) except +

cdef extern from "angle_corrector.hpp":
    Buffer* rotate_image(Buffer* source, double angle, double center_x, double center_y) except +

cdef extern from "poly_gone.hpp":
    Buffer* get_centroids(int32_t* bboxes_array, int bbox_count) except +
    Buffer* get_bboxes(int32_t* bboxes_array, int bbox_count, int img_rows, int img_cols, int padding) except +