-- 1. MAESTRO DE CLIENTES
CREATE TABLE clientes (
    id_cliente VARCHAR(10) PRIMARY KEY,
    nombre_cliente VARCHAR(255) NOT NULL,
    rfc_cliente VARCHAR(13),
    giro VARCHAR(100) NOT NULL,
    zona VARCHAR(100),
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
    id_producto VARCHAR(12) PRIMARY KEY, -- Soporte para Hexadecimal
    producto_norm VARCHAR(255) UNIQUE,
    producto_raw TEXT NOT NULL,
    marca VARCHAR(255),
    umd VARCHAR(20), -- Unidad de Medida (Pz, Cj, Kg)
    categoria VARCHAR(50)
);

-- 4. ENCABEZADOS DE TRANSACCIONES (REGISTROS)
CREATE TABLE registros_compra (
    id_registro VARCHAR(80) PRIMARY KEY,
    id_cliente VARCHAR(10) REFERENCES clientes(id_cliente),
    id_proveedor VARCHAR(10) REFERENCES proveedores(id_proveedor),
    folio_doc VARCHAR(50),
    date_doc DATE,
    total_doc DECIMAL(12, 2) NOT NULL, -- Total impreso en el documento
    total_art DECIMAL(12, 2), -- Suma total de cantidades en el ticket
    auditate BOOLEAN DEFAULT FALSE,
    fecha_captura TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- 5. DETALLES DE COMPRA (TRANSACCIONES ATÓMICAS)
CREATE TABLE detalles_compra (
    idd_detalle SERIAL PRIMARY KEY,
    id_registro VARCHAR(80) REFERENCES registros_compra(id_registro) ON DELETE CASCADE,
    id_producto VARCHAR(12) REFERENCES productos(id_producto),
    cantidad_art DECIMAL(12, 3) NOT NULL,
    precio_unitario DECIMAL(12, 4) NOT NULL,
    importe_doc DECIMAL(12, 2) NOT NULL, -- Valor impreso por línea (el del OCR)
    importe_cal DECIMAL(12, 2) NOT NULL, -- Valor calculado (Cant * PU)
    sku_prov TEXT -- Código impreso en el ticket si existe
);

CREATE TABLE detalles_compra (
    idd_detalle SERIAL PRIMARY KEY,
    id_registro VARCHAR(80) REFERENCES registros_compra(id_registro),
    id_producto VARCHAR(12) REFERENCES productos(id_producto),
    cantidad_art DECIMAL(12, 3) NOT NULL,
    precio_unitario DECIMAL(12, 4) NOT NULL,
    importe_doc DECIMAL(12, 2) NOT NULL,
    importe_cal DECIMAL(12, 2) NOT NULL,
    auditate BOOLEAN DEFAULT FALSE,
    reason VARCHAR(20), -- Los códigos definidos arriba
    sku_prov TEXT
);

-- 6. VÍNCULOS DE HOMÓLOGOS (RESTRICTIVOS)
ALTER TABLE clientes ADD CONSTRAINT fk_homologo_prov FOREIGN KEY (id_homologo_prov) REFERENCES proveedores(id_proveedor);
ALTER TABLE proveedores ADD CONSTRAINT fk_homologo_cli FOREIGN KEY (id_homologo_cli) REFERENCES clientes(id_cliente);