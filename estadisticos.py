import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, uniform, beta, chi2, t
# --- Parámetros de la distribución ---
mu = 12  # Media 
sigma = 2.5  # Desviación estándar 
# --- Configuración del gráfico ---
# Crear un rango de valores para el eje X
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
# Calcular la función de densidad de probabilidad (PDF) para cada valor de X
y = norm.pdf(x, mu, sigma)
# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 6))
# Dibujar la curva de la distribución normal
ax.plot(x, y, 'b-', linewidth=2, label='PDF Normal (μ=12, σ=2.5)')
# --- Resaltar el área entre 11 y 14 días ---
x_fill = np.linspace(11, 14, 100)
y_fill = norm.pdf(x_fill, mu, sigma)
ax.fill_between(x_fill, y_fill, 0, alpha=0.5, color='orange', label='Área entre 11 y 14 días (44.35%)')
# --- Estética del gráfico ---
ax.set_title('Distribución Normal del Tiempo hasta Falla', fontsize=16)
ax.set_xlabel('Tiempo hasta Falla (Días)', fontsize=12)
ax.set_ylabel('Densidad de Probabilidad', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
plt.show()

# --- Parámetros de la distribución ---
a = 60  # Límite inferior [cite: 7]
b = 75  # Límite superior [cite: 7]
loc = a
scale = b - a
# --- Configuración del gráfico ---
# Crear un rango de valores para el eje X
x = np.linspace(loc - 10, loc + scale + 10, 1000)
# Calcular la PDF
y = uniform.pdf(x, loc=loc, scale=scale)
# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 6))
# Dibujar la línea de la distribución uniforme
ax.plot(x, y, 'g-', linewidth=2, label=f'PDF Uniforme [60, 75]')
# Rellenar el área de probabilidad
ax.fill_between(x, y, 0, where=(x >= loc) & (x <= loc + scale), color='green', alpha=0.3)
# --- Estética del gráfico ---
ax.set_title('Distribución Uniforme de la Temperatura del Sistema', fontsize=16)
ax.set_xlabel('Temperatura (°C)', fontsize=12)
ax.set_ylabel('Densidad de Probabilidad', fontsize=12)
ax.set_ylim(0, uniform.pdf(loc, loc=loc, scale=scale) * 1.2) # Ajustar el eje Y
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
plt.show()

# --- Parámetros de la distribución ---
alpha_param = 2  # Parámetro alpha [cite: 6]
beta_param = 5   # Parámetro beta [cite: 6]

# --- Configuración del gráfico ---
# El rango de la distribución Beta es [0, 1]
x = np.linspace(0, 1, 1000)
# Calcular la PDF
y = beta.pdf(x, alpha_param, beta_param)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 6))

# Dibujar la curva de la distribución Beta
ax.plot(x, y, 'r-', linewidth=2, label=f'PDF Beta (α={alpha_param}, β={beta_param})')

# Rellenar el área bajo la curva
ax.fill_between(x, y, 0, alpha=0.3, color='red')

# --- Estética del gráfico ---
ax.set_title('Distribución Beta del Porcentaje de Carga del Motor', fontsize=16)
ax.set_xlabel('Porcentaje de Carga (0 a 1)', fontsize=12)
ax.set_ylabel('Densidad de Probabilidad', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
plt.show()

# --- Parámetros de la distribución ---
df = 6  # Grados de libertad (k) 

# --- Configuración del gráfico ---
# Crear un rango de valores para el eje X
x = np.linspace(0, df + 15, 1000)
# Calcular la PDF
y = chi2.pdf(x, df)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 6))

# Dibujar la curva de la distribución Chi-Cuadrado
ax.plot(x, y, 'm-', linewidth=2, label=f'PDF Chi-Cuadrado (df={df})')

# Rellenar el área bajo la curva
ax.fill_between(x, y, 0, alpha=0.3, color='purple')

# --- Estética del gráfico ---
ax.set_title(f'Distribución $\chi^2$ con {df} Grados de Libertad', fontsize=16)
ax.set_xlabel('Valor Chi-Cuadrado', fontsize=12)
ax.set_ylabel('Densidad de Probabilidad', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
plt.show()

# --- Parámetros de la prueba ---
df = 9  # Grados de libertad (n-1 = 10-1) [cite: 27, 28]
alpha = 0.05  # Nivel de significancia [cite: 27]
t_stat = -2.3715 # Estadístico calculado
t_critical = t.ppf(1 - alpha / 2, df) # Valor crítico positivo (+2.262) [cite: 27]

# --- Configuración del gráfico ---
# Crear un rango de valores para el eje X
x = np.linspace(-4, 4, 1000)
# Calcular la PDF
y = t.pdf(x, df)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 6))

# Dibujar la curva de la distribución t
ax.plot(x, y, 'k-', linewidth=2, label=f'PDF t-Student (df={df})')

# --- Sombrear las zonas de rechazo ---
# Zona de rechazo izquierda
x_fill_left = np.linspace(-4, -t_critical, 100)
y_fill_left = t.pdf(x_fill_left, df)
ax.fill_between(x_fill_left, y_fill_left, 0, color='red', alpha=0.5, label=f'Zona de Rechazo (α={alpha})')

# Zona de rechazo derecha
x_fill_right = np.linspace(t_critical, 4, 100)
y_fill_right = t.pdf(x_fill_right, df)
ax.fill_between(x_fill_right, y_fill_right, 0, color='red', alpha=0.5)

# --- Marcar el estadístico calculado ---
ax.axvline(t_stat, color='cyan', linestyle='--', linewidth=2, label=f't-estadístico = {t_stat:.4f}')

# --- Estética del gráfico ---
ax.set_title('Prueba t de Student Bilateral', fontsize=16)
ax.set_xlabel('Valor t', fontsize=12)
ax.set_ylabel('Densidad de Probabilidad', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
plt.show()