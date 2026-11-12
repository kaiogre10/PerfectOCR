#ifndef BUFFER_CONTAINER_HPP
#define BUFFER_CONTAINER_HPP
#include "buffer_container.h"
#include <memory>
#include <vector>
#include <stdexcept>
#include <utility>
#include <cstdint>
#include <cstddef>

class Buffer {
private:
    struct Deleter {
        void operator()(BufferContainer* p) const {
            if (p) delete_buffer(p);
        }
    };

    std::unique_ptr<BufferContainer, Deleter> ptr;

    // NUEVO: estructura de llaves, solo para AoS
    std::vector<std::unique_ptr<Buffer>> elements_;
    bool is_aos_ = false;

    // envuelve un BufferContainer existente en un Buffer
    struct WrapTag {};
        explicit Buffer(BufferContainer* existing, WrapTag)
            : ptr(existing) {}

        // crea la llave de un índice (llamado por el AoS) 
        void make_element(size_t i) {
            BufferContainer* s = array_at(ptr.get(), i);
            elements_.emplace_back(new Buffer(s, WrapTag{}));
        }

    public:
        explicit Buffer(int rows, int cols, int channels, size_t dtype)
            : ptr(create_buffer(rows, cols, channels, dtype)) {
            if (!ptr) {
                throw std::runtime_error("Error creando Buffer");
            }
        }
        explicit Buffer(int total_objects)
            : ptr(create_array_of_structs(total_objects)), is_aos_(true) {
            if (!ptr) {
                throw std::runtime_error("Error creando Array of Structs");
            }
            elements_.reserve(static_cast<size_t>(total_objects));
            for (int i = 0; i < total_objects; ++i) {
                make_element(static_cast<size_t>(i));
            }
        }
        // === DELETE DE COPIA (¡PROHIBIDO!) ===
        // Esto evita que alguien haga: Buffer buff2 = buff1;
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;

        // === MOVE SEMANTICS (Yo decido ceder el ownership) ===
        // Esto permite: Buffer buff2 = std::move(buff1);
        Buffer(Buffer&& other) noexcept : ptr(std::move(other.ptr)) {}

        Buffer& operator=(Buffer&& other) noexcept {
            if (this != &other) {
                ptr = std::move(other.ptr);
            }
            return *this;
        }

        // === EXPOSICIÓN DE DATOS (Acceso directo a los bytes) cualquiera puede escribir en esta memoria.
        void* data() {
            return buffer_get_data(ptr.get());
        }
        // Se castea a diferentes tipos, al delegar al caller la responsabilidad de la congruencia de sus tipos esto es seguro ya que siempre se reservan con el mismo sizeof
        const void* read_only_data() const {
            return buffer_get_data_const(ptr.get());
        }
        int rows() {
            return buffer_get_rows(ptr.get());
        }
        int cols() {
            return buffer_get_cols(ptr.get());
        }
        int channels() {
            return buffer_get_channels(ptr.get());
        }
        size_t size() {
            return buffer_get_size(ptr.get());
        }
        uint8_t* get_uint8_buffer() {
            return get_uint8_data(ptr.get());
        }
        int32_t* get_int32_buffer() {
            return get_int32_data(ptr.get());
        }
        float* get_float32_buffer() {
            return get_float32_data(ptr.get());
        }
        void set_struct_at(size_t index, int rows, int cols, int channels, size_t total_bytes, void* data) {
            ::set_struct_at(ptr.get(), index, rows, cols, channels, total_bytes, data);
        }
        void group_structs(size_t index, Buffer* buffer) {
            ::group_array_structs(ptr.get(), index, buffer->ptr.get(), static_cast<void*>(buffer));
        }
        Buffer* recover_struct(size_t index) {
            return static_cast<Buffer*>(take_struct_data(index));
        }
        void* take_struct_data(size_t index) {
           return ::take_struct_data(ptr.get(), index);
        }
        Buffer* at(size_t i) {
            return elements_[i].get();
        }
        size_t buffer_addr(size_t i) {
            return reinterpret_cast<size_t>(elements_[i].get());
        }
};

inline void destroy_buffer(Buffer* buffer) noexcept {
    delete buffer;
}
#endif
