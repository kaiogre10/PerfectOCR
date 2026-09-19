# image_loader.pxd
from image_container cimport Image

cdef extern from "image_loader.hpp":
    Image* load_image(const char* filepath) except +