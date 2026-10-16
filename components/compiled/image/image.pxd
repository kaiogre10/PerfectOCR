# image.pxd
from libc.stddef cimport size_t

cdef extern from "image_container.hpp":
    cppclass Image:
        Image(int width, int height, int channels, size_t dtype) except +
        void* data()
        const void* data() const
        int width() const
        int height() const
        int channels() const
        size_t size() const

    cdef void destroy_image(Image* img) noexcept

cdef inline Image* new_buffer(int width, int height, int channels, size_t dtype) except +:
    return new Image(width, height, channels, dtype)

cdef inline Image* get_image_ptr(size_t ptr_addr) noexcept:
    if ptr_addr == 0:
        return <Image*>0
    return <Image*>ptr_addr
