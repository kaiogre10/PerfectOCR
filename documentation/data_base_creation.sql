-- 4. ENCABEZADOS DE TRANSACCIONES (REGISTROS)
CREATE TABLE registros_compra (
    id_registro VARCHAR(80) PRIMARY KEY,
    id_cliente VARCHAR(10) REFERENCES clientes(id_cliente),
    folio_doc VARCHAR(50),
    date_doc TEXT,
    id_proveedor VARCHAR(10) REFERENCES proveedores(id_proveedor),
    total_doc DECIMAL(12, 2), -- Total impreso en el documento
    total_cal DECIMAL(12, 2) NOT NULL,
    total_art DECIMAL(12, 2),
    art_cal DECIMAL(12, 2) NOT NULL,
    subtotal DECIMAL(12, 2),
    monto_iva DECIMAL(12, 2),
    auditate BOOLEAN DEFAULT FALSE,
    fecha_captura TEXT NOT NULL,
    reason TEXT
);

-- 5. DETALLES DE COMPRA (TRANSACCIONES ATÓMICAS)
CREATE TABLE detalles_compra (
    idd_detalle SERIAL PRIMARY KEY,
    id_registro VARCHAR(80) REFERENCES registros_compra(id_registro),
    id_producto INTEGER REFERENCES productos(id_producto),
    cantidad_art DECIMAL(12, 3) NOT NULL,
    producto_norm TEXT NOT NULL,
    precio_unitario DECIMAL(12, 4) NOT NULL,
    costo_tran DECIMAL(12, 2) NOT NULL,
    auditate BOOLEAN DEFAULT FALSE,
    reason VARCHAR(20),
    sku_prov TEXT
);
-- 1. MAESTRO DE CLIENTES
CREATE TABLE clientes (
    id_cliente VARCHAR(10) PRIMARY KEY,
    nombre_cliente VARCHAR(255) NOT NULL,
    rfc_cliente VARCHAR(13),
    giro VARCHAR(100),
    id_homologo_prov VARCHAR(10), -- Referencia manual al ID de Proveedor si existe
    c_activo BOOLEAN DEFAULT TRUE,
    notas_cliente TEXT
);

-- 2. MAESTRO DE PROVEEDORES
CREATE TABLE proveedores (
    id_proveedor VARCHAR(10) PRIMARY KEY,
    proveedor_norm VARCHAR(255) NOT NULL,
    rfc_prov VARCHAR(13),
    id_homologo_cli VARCHAR(10), -- Referencia manual al ID de Cliente si existe
    notas_proveedor TEXT
);

-- 3. CATÁLOGO GLOBAL DE PRODUCTOS
CREATE TABLE productos (
    id_producto SERIAL PRIMARY KEY, -- Soporte para Hexadecimal
    producto_norm VARCHAR(255) UNIQUE,
    producto_raw TEXT,
    marca VARCHAR(255),
    umd VARCHAR(20), -- Unidad de Medida (Pz, Cj, Kg)
    categoria VARCHAR(50)
);

-- 6. VÍNCULOS DE HOMÓLOGOS (RESTRICTIVOS)
ALTER TABLE clientes ADD CONSTRAINT fk_homologo_prov FOREIGN KEY (id_homologo_prov) REFERENCES proveedores(id_proveedor);
ALTER TABLE proveedores ADD CONSTRAINT fk_homologo_cli FOREIGN KEY (id_homologo_cli) REFERENCES clientes(id_cliente);