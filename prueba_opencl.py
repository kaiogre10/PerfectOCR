import cv2
import numpy as np

# Habilitar el uso de OpenCL para la GPU Intel Iris Xe
has_opencl = cv2.ocl.haveOpenCL()

print(f"Soporte para OpenCL disponible en este build de OpenCV: {has_opencl}")

if has_opencl:
    # Si estuviera disponible, así se consultaría si está habilitado
    is_enabled = cv2.ocl.useOpenCL()
    print(f"OpenCL está habilitado por defecto: {is_enabled}")
# (Opcional) Verificar que está funcionando
if cv2.ocl.haveOpenCL():
    print("OpenCL está habilitado y listo para usarse en la GPU.")
else:
    print("Advertencia: OpenCL no está disponible.")

# ... el resto de su código de procesamiento de imágenes ...