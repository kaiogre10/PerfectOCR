#ifndef IMAGE_HPP
#define IMAGE_HPP
#include "image_container.h"
#include <cstdint>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

class Image {
private:
    // Deleter personalizado (llama a image_destruir)
    struct Deleter {
        void operator()(ImageContainer* p) const {
            if (p) {
                delete_image(p);
            }
        }
    };

    // Usamos unique_ptr para garantizar que se llame al destructor.
    // Pero NUNCA exponemos el unique_ptr directamente.
    std::unique_ptr<ImageContainer, Deleter> ptr;

public:
    // === CONSTRUCTORES ===
    explicit Image(int width, int height, int channels)
        : ptr(create_img_buffer(width, height, channels)) {
        if (!ptr) {
            throw std::runtime_error("Error creando Image");
        }
    }

    // === DELETE DE COPIA (¡PROHIBIDO!) ===
    // Esto evita que alguien haga: Image img2 = img1;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    // === MOVE SEMANTICS (TÚ decides ceder el ownership) ===
    // Esto permite: Image img2 = std::move(img1);
    Image(Image&& other) noexcept: ptr(std::move(other.ptr)) {}

    Image& operator=(Image&& other) noexcept {
        if (this != &other) {
            ptr = std::move(other.ptr);
        }
        return *this;
    }

    // === EXPOSICIÓN DE DATOS (Acceso directo a los bytes) ===
    // Cualquiera puede escribir en esta memoria.
    uint8_t* data() {
        return image_get_data(ptr.get());
    }

    const uint8_t* data() const {
        return image_get_data_const(ptr.get());
    }
    // === METADATOS ===
    int width() const {
        return image_get_width(ptr.get());
    }
    int height() const {
        return image_get_height(ptr.get());
    }
    int channels() const {
        return image_get_channels(ptr.get());
    }
    size_t size() const {
        return image_get_size(ptr.get());
    }
};
#endif