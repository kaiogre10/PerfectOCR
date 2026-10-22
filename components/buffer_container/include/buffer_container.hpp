#ifndef BUFFER_CONTAINER_HPP
#define BUFFER_CONTAINER_HPP
#include "buffer_container.h"
#include <memory>
#include <stdexcept>
#include <utility>
#include <cstdint>
#include <cstddef>

class Buffer {
    private:
        struct Deleter {
            void operator()(BufferContainer* p) const {
                if (p) {
                    delete_buffer(p);
                }
            }
        };
        // Usamos unique_ptr para garantizar que se llame al destructor.
        // Pero NUNCA exponemos el unique_ptr directamente.
        std::unique_ptr<BufferContainer, Deleter> ptr;

    public:
        explicit Buffer(int rows, int cols, int channels, size_t dtype)
            : ptr(create_buffer(rows, cols, channels, dtype)) {
            if (!ptr) {
                throw std::runtime_error("Error creando Buffer");
            }
        }
        // === DELETE DE COPIA (¡PROHIBIDO!) ===
        // Esto evita que alguien haga: Buffer img2 = img1;
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;

        // === MOVE SEMANTICS (Yo decido ceder el ownership) ===
        // Esto permite: Buffer img2 = std::move(img1);
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
        const void* data() const {
            return buffer_get_data_const(ptr.get());
        }
        int rows() const {
            return buffer_get_rows(ptr.get());
        }
        int cols() const {
            return buffer_get_cols(ptr.get());
        }
        int channels() const {
            return buffer_get_channels(ptr.get());
        }
        size_t size() const {
            return buffer_get_size(ptr.get());
        }
};

inline void destroy_buffer(Buffer* buffer) noexcept {
    delete buffer;
}
#endif
