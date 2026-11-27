# container.pxd
from libc.stddef cimport size_t
from libcpp cimport bool
from ..compiled_utils.compiled_utils cimport get_arr_dtype_size, array_view, validate_arr_vals

cdef extern from "buffer_container.hpp" nogil:
    cppclass Buffer:        
        Buffer(int rows, int cols, int channels, size_t dtype) noexcept
        void* data()
        const void* read_only_data() const
        int rows()
        int cols()
        int channels()
        size_t size()
        size_t* get_sizet_buffer()

        Buffer* recover_struct(size_t index)
        void* take_struct_data(size_t index)

        Buffer* buffer_addr(size_t i)
        size_t buffer_addr_sized(size_t i)

    void destroy_buffer(Buffer* buffer) noexcept
    size_t buffer_container_size() noexcept

cdef inline bool check_null_ptr(Buffer* ptr_addr) noexcept:
    return ptr_addr == NULL or <size_t>ptr_addr < 1 # Devuelve True si es invalido

cdef inline Buffer* get_buffer_ptr(size_t ptr_addr) noexcept:
    if ptr_addr < 1:
        return NULL
    return <Buffer*>ptr_addr

cdef inline void free_buffer(size_t ptr_addr) noexcept:
    cdef Buffer* buff = get_buffer_ptr(ptr_addr)
    if not check_null_ptr(buff):
        destroy_buffer(buff)

cdef inline void free_buffer_class(Buffer* buff) noexcept:
    if not check_null_ptr(buff):
        destroy_buffer(buff)

cdef inline Buffer* _create_memmory_buffer(int rows, int cols, int channels, size_t bad_dtype) noexcept:
    if not validate_arr_vals(rows, cols, channels):
        return NULL

    cdef Buffer* b = new Buffer(rows, cols, channels, bad_dtype)
    return b

cdef inline int get_rows(Buffer* buff_ptr) noexcept:
    return buff_ptr.rows()

cdef inline int get_cols(Buffer* buff_ptr) noexcept:
    return buff_ptr.cols()

cdef inline int get_channels(Buffer* buff_ptr) noexcept:
    return buff_ptr.channels()

cdef inline size_t get_size(Buffer* buff_ptr) noexcept:
    return buff_ptr.size()

cdef inline void* get_data(Buffer* buff_ptr) noexcept:
    return buff_ptr.data()

cdef inline size_t* get_data_sizet(Buffer* buff_ptr) noexcept:
    return buff_ptr.get_sizet_buffer()

cdef inline size_t create_new_buffer(int rows, int cols, int channels, const char* dtype_raw) noexcept:
    cdef size_t bad_dtype = get_arr_dtype_size(dtype_raw)
    if bad_dtype < 1:
        return 0

    cdef Buffer* buff = _create_memmory_buffer(rows, cols, channels, bad_dtype)
    if check_null_ptr(buff):
        return 0
    
    return <size_t>buff

cdef inline object get_array_view(size_t arr_ptr, const char* dtype, int dims) noexcept:
    cdef Buffer* new_array = get_buffer_ptr(arr_ptr)
    if check_null_ptr(new_array):
        return 0

    cdef int rows = get_rows(new_array)
    cdef int cols = get_cols(new_array)
    cdef int channels = get_channels(new_array)

    if not validate_arr_vals(rows, cols, channels):
        return 0

    cdef bool tridimentional_view = True

    if dims == 0 or dims == 2:
        tridimentional_view = False

    cdef void* data = get_data(new_array)
    return array_view(data, rows, cols, channels, dtype, tridimentional_view)

cdef inline Buffer* recover_class(size_t index, Buffer* cropped_polygons) noexcept:
    if check_null_ptr(cropped_polygons):
        return NULL

    return cropped_polygons.recover_struct(index)

cdef inline size_t* get_all_buffer_addrs_size(Buffer* aos, size_t n) noexcept:
    cdef size_t i
    cdef size_t buffer_size = buffer_container_size()
    cdef Buffer* index = _create_memmory_buffer(n, 1, 1, buffer_size)  # Buffer para punteros

    if check_null_ptr(index):
        return NULL
    cdef size_t* result = get_data_sizet(index)

    for i in range(n):
        result[i] = aos.buffer_addr_sized(i)

    return result

cdef inline void free_crops(size_t cropped_buffer_ptr, size_t addr_aos, size_t total_crops) noexcept:
    cdef Buffer* aos = get_buffer_ptr(addr_aos)  # Crops
    cdef size_t* buffer_list_ptr = get_all_buffer_addrs_size(aos, total_crops)
    list_ptr = [buffer_list_ptr[i] for i in range(total_crops)]   # LIsta de punteros para python
    
    cdef Buffer* crops = get_buffer_ptr(cropped_buffer_ptr)
    cdef size_t i = 0
    # 1. Desconectar las vistas (aritmética de punteros) en el AoS y limpiar la lista
    if not check_null_ptr(aos):
        for i in range(total_crops):
            aos.take_struct_data(i)
            list_ptr[i] = 0
    
    # 2. Liberar el bloque contiguo de píxeles completo de un solo golpe
    if not check_null_ptr(crops):
        free_buffer_class(crops)