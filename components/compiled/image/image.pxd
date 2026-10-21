# image.pxd
from libc.stddef cimport size_t

cdef extern from "image_container.hpp":
    cppclass Image:
        Image(int rows, int cols, int channels, size_t dtype) except +
        void* data()
        const void* data() const
        int rows() const
        int cols() const
        int channels() const
        size_t size() const

    cdef void destroy_image(Image* img) noexcept

cdef inline Image* new_buffer(int rows, int cols, int channels, size_t dtype) except +:
    return new Image(rows, cols, channels, dtype)
    
cdef inline Image* get_image_ptr(size_t ptr_addr) except +:
    if ptr_addr == 0:
        return <Image*>0
    return <Image*>ptr_addr

cdef inline int get_rows(Image* img_ptr) except +:
    return img_ptr.rows()
    
cdef inline int get_cols(Image* img_ptr) except +:
    return img_ptr.cols()

cdef inline int get_channels(Image* img_ptr) except +:
    return img_ptr.channels()

cdef inline size_t get_size(Image* img_ptr) except +:
    return img_ptr.size()

cdef inline void* get_data(Image* img_ptr) except +:
    return img_ptr.data()