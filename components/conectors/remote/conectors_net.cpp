#include <vector>
#include <list>
#include <deque>
#include "remote_conectors.h"
#include "../../containers/containers.h"

class NetConector {
private:
    // Guardamos una referencia al canal para saber de dónde extraer datos
    BufferEnlaceFIFO& canal_datos;

    // Método privado para procesar los datos ya extraídos (Fuera del mutex)
    void procesarPaquete(const std::vector<uint8_t>& paquete) {
        // Aquí va tu lógica de negocio con los bytes recibidos
        // Por ejemplo: enviarlos a la red, decodificarlos, etc.
    }

public:
    // El constructor recibe el canal por referencia y lo enlaza
    ConsumidorCiego(BufferEnlaceFIFO& canal) : canal_datos(canal) {}

    // El nivel superior llamará a este método cuando reciba la señal de hacer el swap
    void ejecutarVaciado() {
        // 1. Ejecuta el drain ciegamente del canal seguro
        std::deque<std::vector<uint8_t>> rafaga_local = canal_datos.drain();

        // 2. Procesa la ráfaga de forma secuencial en orden FIFO
        while (!rafaga_local.empty()) {
            // Pasamos el paquete del frente al procesador interno
            procesarPaquete(rafaga_local.front());

            // Lo eliminamos para avanzar en la cola FIFO
            rafaga_local.pop_front();
        }

        std::cout << "[Clase Consumidora] Vaciado y procesamiento FIFO completado.\n";
    }
};
