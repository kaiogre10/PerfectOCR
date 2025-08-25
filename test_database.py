# test_database.py - Ejemplo de uso del DatabaseService
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.database_service import DatabaseService
from datetime import datetime

def test_database():
    """Prueba básica del servicio de base de datos"""
    
    print("🔧 Iniciando prueba de base de datos...")
    
    # Crear servicio de BD
    db_service = DatabaseService("base_datos/test_perfectocr.db")
    
    print("✅ Base de datos creada/conectada")
    
    # Insertar un registro de prueba
    test_data = {
        'tipo_documento': 'factura',
        'numero_documento': 'F001-123',
        'fecha_emision': '2025-08-24',
        'rfc_proveedor': 'ABC123456789',
        'nombre_proveedor': 'Proveedor de Prueba S.A.',
        'total': 1500.50,
        'archivo_original': 'AAAA.0006.png'
    }
    
    registro_id = db_service.insert_registro_compra(test_data)
    print(f"✅ Registro insertado con ID: {registro_id}")
    
    # Obtener registros
    registros = db_service.get_registros_compra(5)
    print(f"✅ Encontrados {len(registros)} registros:")
    
    for registro in registros:
        print(f"  - ID: {registro['IDRegistro']}, Proveedor: {registro['NombreProveedor']}, Total: ${registro['Total']}")
    
    print("🎉 Prueba completada!")

if __name__ == "__main__":
    test_database()
