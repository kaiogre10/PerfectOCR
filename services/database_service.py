# services/database_service.py
import sqlite3
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseService:
    """Servicio para manejar la base de datos SQLite del sistema PerfectOCR"""
    
    def __init__(self, db_path: str = "base_datos/perfectocr.db"):
        """
        Inicializa el servicio de base de datos
        
        Args:
            db_path: Ruta al archivo de base de datos SQLite
        """
        self.db_path = db_path
        self.ensure_database_exists()
        
    def ensure_database_exists(self):
        """Crea la base de datos y las tablas si no existen"""
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Crear tablas
            self.create_tables()
            logger.info(f"Base de datos inicializada en: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}")
            raise
    
    def get_connection(self) -> sqlite3.Connection:
        """Obtiene una conexión a la base de datos"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Para acceder a columnas por nombre
        return conn
    
    def create_tables(self):
        """Crea todas las tablas necesarias basadas en metadatos.csv"""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tabla RegistrosCompra (documento principal)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS RegistrosCompra (
                    IDRegistro INTEGER PRIMARY KEY AUTOINCREMENT,
                    TipoDocumento TEXT NOT NULL,
                    NumeroDocumento TEXT,
                    FechaEmision DATE,
                    FechaVencimiento DATE,
                    IDProveedor INTEGER,
                    RFCProveedor TEXT,
                    NombreProveedor TEXT,
                    Subtotal REAL,
                    IVA REAL,
                    Total REAL,
                    Moneda TEXT DEFAULT 'MXN',
                    TipoCambio REAL DEFAULT 1.0,
                    EstadoPago TEXT DEFAULT 'pendiente',
                    FechaProcesamiento DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ArchivoOriginal TEXT,
                    IDClienteConsultoria INTEGER,
                    NotasGenerales TEXT,
                    RequiereRevisionManual BOOLEAN DEFAULT 0,
                    FOREIGN KEY (IDProveedor) REFERENCES Proveedores(IDProveedor),
                    FOREIGN KEY (IDClienteConsultoria) REFERENCES Clientes(IDCliente)
                )
            ''')
            
            # Tabla DetallesCompra (líneas de productos/servicios)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS DetallesCompra (
                    IDDetalle INTEGER PRIMARY KEY AUTOINCREMENT,
                    IDRegistro INTEGER NOT NULL,
                    NumeroLinea INTEGER,
                    SKU TEXT,
                    DescripcionProducto TEXT,
                    Cantidad REAL,
                    UnidadMedida TEXT,
                    PrecioUnitario REAL,
                    Descuento REAL DEFAULT 0,
                    Subtotal REAL,
                    IVATasa REAL,
                    IVAImporte REAL,
                    TotalLinea REAL,
                    CodigoIEPS TEXT,
                    IEPSImporte REAL DEFAULT 0,
                    CategoriaProducto TEXT,
                    NotasDetalle TEXT,
                    FOREIGN KEY (IDRegistro) REFERENCES RegistrosCompra(IDRegistro),
                    FOREIGN KEY (SKU) REFERENCES Productos(SKU)
                )
            ''')
            
            # Tabla Productos (catálogo maestro)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Productos (
                    SKU TEXT PRIMARY KEY,
                    NombreProducto TEXT NOT NULL,
                    Descripcion TEXT,
                    Categoria TEXT,
                    UnidadMedida TEXT,
                    PrecioReferencia REAL,
                    ProveedorPrincipal INTEGER,
                    FechaCreacion DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FechaActualizacion DATETIME DEFAULT CURRENT_TIMESTAMP,
                    Activo BOOLEAN DEFAULT 1,
                    FOREIGN KEY (ProveedorPrincipal) REFERENCES Proveedores(IDProveedor)
                )
            ''')
            
            # Tabla Proveedores (catálogo maestro)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Proveedores (
                    IDProveedor INTEGER PRIMARY KEY AUTOINCREMENT,
                    RFC TEXT UNIQUE,
                    RazonSocial TEXT NOT NULL,
                    NombreComercial TEXT,
                    Direccion TEXT,
                    Telefono TEXT,
                    Email TEXT,
                    SitioWeb TEXT,
                    ContactoPrincipal TEXT,
                    TerminosPago TEXT,
                    FechaRegistro DATETIME DEFAULT CURRENT_TIMESTAMP,
                    Activo BOOLEAN DEFAULT 1,
                    NotasProveedor TEXT
                )
            ''')
            
            # Tabla Clientes (para consultoría)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Clientes (
                    IDCliente INTEGER PRIMARY KEY AUTOINCREMENT,
                    RFC TEXT UNIQUE,
                    RazonSocial TEXT NOT NULL,
                    NombreComercial TEXT,
                    Direccion TEXT,
                    Telefono TEXT,
                    Email TEXT,
                    ContactoPrincipal TEXT,
                    FechaRegistro DATETIME DEFAULT CURRENT_TIMESTAMP,
                    Activo BOOLEAN DEFAULT 1,
                    NotasCliente TEXT
                )
            ''')
            
            # Tabla TransaccionesCompra (pagos y movimientos financieros)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS TransaccionesCompra (
                    IDTransaccion INTEGER PRIMARY KEY AUTOINCREMENT,
                    IDRegistro INTEGER NOT NULL,
                    TipoTransaccion TEXT NOT NULL,
                    FechaTransaccion DATE,
                    Monto REAL,
                    MetodoPago TEXT,
                    ReferenciaPago TEXT,
                    CuentaBancaria TEXT,
                    EstadoTransaccion TEXT DEFAULT 'completada',
                    NotasTransaccion TEXT,
                    FechaRegistro DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (IDRegistro) REFERENCES RegistrosCompra(IDRegistro)
                )
            ''')
            
            conn.commit()
            logger.debug("Tablas creadas correctamente")
    
    def insert_registro_compra(self, data: Dict[str, Any]) -> Optional[int]:
        """
        Inserta un nuevo registro de compra
        
        Args:
            data: Diccionario con los datos del registro
            
        Returns:
            ID del registro insertado o None si hay error
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO RegistrosCompra (
                        TipoDocumento, NumeroDocumento, FechaEmision, 
                        RFCProveedor, NombreProveedor, Total, ArchivoOriginal
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('tipo_documento', 'ticket'),
                    data.get('numero_documento'),
                    data.get('fecha_emision'),
                    data.get('rfc_proveedor'),
                    data.get('nombre_proveedor'),
                    data.get('total'),
                    data.get('archivo_original')
                ))
                
                registro_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Registro de compra insertado con ID: {registro_id}")
                return registro_id
                
        except Exception as e:
            logger.error(f"Error insertando registro de compra: {e}")
            return None
    
    def get_registros_compra(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene los registros de compra más recientes"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM RegistrosCompra 
                    ORDER BY FechaProcesamiento DESC 
                    LIMIT ?
                ''', (limit,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error obteniendo registros: {e}")
            return []
    
    def close(self):
        """Cierra la conexión a la base de datos"""
        # SQLite se cierra automáticamente con context manager
        logger.debug("Servicio de base de datos cerrado")
