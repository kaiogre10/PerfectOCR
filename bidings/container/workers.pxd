from .container cimport Buffer
from libc.stddef cimport size_t

cdef extern from "image_loader.hpp" nogil:
    Buffer* load_image(const char* filepath) except +

cdef extern from "angle_corrector.hpp" nogil:
    Buffer* rotate_image(Buffer* source, double angle, double center_x, double center_y) except +

cdef extern from "poly_gone.hpp" nogil:
    Buffer* get_bboxes(Buffer* bboxes_ptr, Buffer* img, int padding, size_t* total_buffers_ref) except +