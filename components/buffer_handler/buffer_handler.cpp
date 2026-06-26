#include "buffer_handler.h"
#include "../containers/containers.h"
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <iostream>

uint8_t* buffer_ptr = nullptr;
size_t buffer_size = 0;

namespace {
    void storage_batch_flat(uint8_t* buffer_ptr,
                                size_t buffer_size) {
        if (!buffer_ptr || !buffer_size) return;
        printf("ptr: %p, size: %zu\n", buffer_ptr, buffer_size);
        std::vector<uint8_t> plain_payload(buffer_ptr, buffer_ptr + buffer_size);
        for (uint8_t byte : plain_payload) {
            std::cout << (char)byte;
        }
        std::cout << std::flush;
        push(std::move(plain_payload));
        std::cout << "\nSize: " << plain_payload.size() << "\n";
    }
}

extern "C" {
    uint8_t* reserve_buffer(size_t len_bytes) {
        buffer_ptr = new uint8_t[len_bytes];
        buffer_size = len_bytes;
        return buffer_ptr;
    }
    void commit_buffer(int signal) {
        try {
            storage_batch_flat(buffer_ptr, buffer_size);
            delete[] buffer_ptr;
        }
        catch (...) {
            delete[] buffer_ptr;
        }
        buffer_ptr = nullptr;
        buffer_size = 0;
    }
}


//extern "C" {
//    // La señal mínima en las primeras líneas de main que acciona todo
//    void container_create(int trigger);
//}
//#include <mutex>
//
//std::mutex mtx;
//CallbackDatosNativos InvocadorReceptor = nullptr;
//
//// Función interna para conectar el puente
//void ConfigurarCallbackEmisario(CallbackDatosNativos callback) {
//    InvocadorReceptor = callback;
//}
//
//void Emisario_EjecutarEnvio() {
//    // 1. Protección de alcance automática
//    std::lock_guard<std::mutex> guard(mtx);
//
//    // 2. Va por la info en vivo y la aplana en UTF-16
//    const wchar_t* bytesCrudosNativos = ObtenerDatosEnVivoNativos();
//
//    // 3. EL ENLACE FUERTE (El Backend llama al Frontend):
//    // Si el receptor está registrado, se le entregan los bytes en su propia mano
//    if (InvocadorReceptor != nullptr) {
//        InvocadorReceptor(bytesCrudosNativos);
//    }
//
//    // 4. Limpieza de información nativa inmediata
//    LimpiarMemoriaNativa();
//
//} // <-- Fin del scope: El compilador destruye 'guard' y libera el mutex automáticamente.
//  // El Frontend ya terminó de procesar porque la llamada es síncrona.

// size_t offset_view = 0;

// for (size_t i = 0; i < total_cols; ++i) {
//     size_t actual_size = buffer_size[i];
//
//     if (actual_size > 0) {
// plain_payload.reserve(buffer_size);
// plain_payload.assign(buffer_ptr + buffer_size),
//             buffer_ptr + offset_view + actual_size);
//
//         offset_view += actual_size;
// }
// }