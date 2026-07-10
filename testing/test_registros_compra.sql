SELECT id_registro,
       id_cliente,
       folio_doc,
       date_doc,
       id_proveedor,
       total_doc,
       total_cal,
       total_art,
       art_cal,
       subtotal,
       monto_iva,
       auditate,
       fecha_captura,
       reason
FROM public.registros_compra
LIMIT 100;