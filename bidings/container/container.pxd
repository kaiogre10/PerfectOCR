# bidings/container/container.pxd
from libc.stddef cimport size_t

cdef extern from "buffer_container.hpp":
    cppclass Buffer:
        Buffer(int rows, int cols, int channels, size_t dtype) noexcept
        void* data()
        const void* data() const
        int rows() const
        int cols() const
        int channels() const
        size_t size() const

    cdef inline void destroy_buffer(Buffer* buffer) noexcept   # NO BORRAR ESTA LINEA
    
cdef inline bint check_null_ptr(Buffer* ptr_addr)
 
cdef inline Buffer* get_buffer_ptr(size_t ptr_addr)

cdef inline void free_buffer(size_t ptr_addr) 

cdef inline int get_rows(Buffer* buff_ptr)
    
cdef inline int get_cols(Buffer* buff_ptr)

cdef inline int get_channels(Buffer* buff_ptr)

cdef inline size_t get_size(Buffer* buff_ptr)

cdef inline void* get_data(Buffer* buff_ptr)
        
cdef inline size_t create_new_buffer(int rows, int cols, int channels, const char* dtype)

cdef inline object get_array_view(size_t arr_ptr, const char* dtype, int dims)