#include "buffer_handler.h"
#include "../containers/containers.h"
#include <vector>
#include <cstdint>
#include <mutex>

extern "C" void storage_batch_flat(const uint8_t* plain_data,
                                   const size_t* len_list,
                                   size_t total_elements) {

    if (!plain_data || !len_list || total_elements == 0) return;

    // El productor procesa y reconstruye de manera privada
    std::vector<std::vector<uint8_t>> struct_payload;
    struct_payload.resize(total_elements);

    size_t offset_view = 0;

    for (size_t i = 0; i < total_elements; ++i) {
        size_t actual_size = len_list[i];

        if (actual_size > 0) {
            struct_payload[i].reserve(actual_size);
            struct_payload[i].assign(plain_data + offset_view,
                plain_data + offset_view + actual_size);

            offset_view += actual_size;
        }
    }
    push(std::move(struct_payload));
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
//// Tu algoritmo original adaptado al empuje de datos
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