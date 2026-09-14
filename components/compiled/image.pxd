# image.pxd
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t

cdef extern from "image.hpp":
    cppclass Image:
        Image(int width, int height, int channels) except +
        uint8_t* data()
        const uint8_t* data() const
        int width() const
        int height() const
        int channels() const
        size_t size() const

    void destroy_image(Image* img) noexcept