# image_loader.pxd
from image cimport Image

cdef extern from "image_loader.hpp":
    Image* load_image(const char* filepath) except +

cdef extern from "angle_corrector.hpp":
    Image* rotate_image(Image* source, double angle, double center_x, double center_y) except +