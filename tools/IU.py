# launcher_gui.py
import tkinter as Tk
import subprocess
import threading
import os
import sys

Tk.set_appearance_mode("System")  # Puede ser "Dark", "Light"
Tk.set_default_color_theme("blue")

class AppLauncher(Tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PerfectOCR Launcher")
        self.geometry("800x600")

        # --- Layout de la ventana ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1) # El área de texto se expandirá

        # --- Frame para los controles ---
        self.controls_frame = AppLauncher(Tk.Tk)
        self.controls_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        self.controls_frame.grid_columnconfigure(0, weight=1)

        # --- Selección de Archivos de Entrada ---
        self.input_label = Tk.Label(self.controls_frame, text="Archivos de Entrada:")
        self.input_label.grid(row=0, column=0, padx=10, pady=10)
        
        self.input_entry = Tk.Entry(self.controls_frame, placeholder_text="Selecciona una o más imágenes...")
        self.input_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.input_button = Tk.Button(self.controls_frame, text="Examinar...", command=self.select_input_files)
        self.input_button.grid(row=0, column=2, padx=10, pady=10)

        # --- Selección de Carpeta de Salida ---
        self.output_label = Tk.Label(self.controls_frame, text="Carpeta de Salida:")
        self.output_label.grid(row=1, column=0, padx=10, pady=10)
        
        self.output_entry = yTk.Entry(self.controls_frame, placeholder_text="Selecciona una carpeta de salida...")
        self.output_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        self.output_button = Tk.Button(self.controls_frame, text="Examinar...", command=self.select_output_folder)
        self.output_button.grid(row=1, column=2, padx=10, pady=10)

        # --- Botón de Ejecución ---
        self.run_button = ctk.CTkButton(self, text="▶️ Iniciar Procesamiento", height=40, font=ctk.CTkFont(size=16, weight="bold"), command=self.start_processing_thread)
        self.run_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        # --- Área de Texto para mostrar la salida del script ---
        self.output_textbox = ctk.CTkTextbox(self, state="disabled")
        self.output_textbox.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")

        # --- Variables para guardar las rutas ---
        self.input_file_paths = []
        self.output_folder_path = ""

    def select_input_files(self):
        """Abre un diálogo para seleccionar múltiples archivos de imagen."""
        files = filedialog.askopenfilenames(
            title="Selecciona Imágenes",
            filetypes=(("Archivos de Imagen", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("Todos los archivos", "*.*"))
        )
        if files:
            self.input_file_paths = list(files)
            # Muestra los archivos seleccionados en la entrada de texto
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, f"{len(self.input_file_paths)} archivo(s) seleccionado(s)")

    def select_output_folder(self):
        """Abre un diálogo para seleccionar una carpeta."""
        folder = filedialog.askdirectory(title="Selecciona la Carpeta de Salida")
        if folder:
            self.output_folder_path = folder
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, self.output_folder_path)

    def start_processing_thread(self):
        """
        Inicia el procesamiento en un hilo separado para no congelar la GUI.
        """
        # Deshabilita el botón para evitar ejecuciones múltiples
        self.run_button.configure(state="disabled", text="Procesando...")
        self.output_textbox.configure(state="normal")
        self.output_textbox.delete("1.0", "end")
        self.output_textbox.configure(state="disabled")

        # Crea y lanza el hilo
        thread = threading.Thread(target=self.run_main_script)
        thread.start()

    def run_main_script(self):
        """
        Construye el comando y ejecuta main.py usando subprocess,
        capturando su salida en tiempo real.
        """
        if not self.input_file_paths:
            self.update_output("Error: Debes seleccionar al menos un archivo de entrada.\n")
            self.processing_finished()
            return
            
        # 1. Construir el comando que ejecutarías en la terminal
        command = [sys.executable, "main.py"]  # 'sys.executable' es la ruta al intérprete de Python actual
        command.extend(self.input_file_paths) # Añade las rutas de los archivos de entrada
        
        # Añade la carpeta de salida si se especificó
        if self.output_folder_path:
            command.extend(["--output", self.output_folder_path])

        self.update_output(f"Ejecutando comando: {' '.join(command)}\n" + "="*50 + "\n")
        
        try:
            # 2. Ejecutar el script como un proceso secundario
            #    `subprocess.PIPE` captura la salida
            #    `text=True` y `encoding='utf-8'` para manejar texto
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Redirige errores al mismo stream
                text=True,
                encoding='utf-8',
                bufsize=1, # Lee línea por línea
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0 # Opcional: para no abrir una ventana de terminal en Windows
            )

            # 3. Leer la salida del proceso línea por línea en tiempo real
            if process.stdout is not None:
                for line in iter(process.stdout.readline, ''):
                    self.update_output(line)
                process.stdout.close()
            process.wait() # Espera a que el proceso termine
            self.update_output("\n" + "="*50 + "\nProceso finalizado.\n")

        except FileNotFoundError:
            self.update_output("\nError: No se encontró 'main.py'. Asegúrate de que este lanzador está en la misma carpeta que main.py.\n")
        except Exception as e:
            self.update_output(f"\nOcurrió un error inesperado: {e}\n")
        
        # 4. Cuando todo termina, reactiva el botón
        self.processing_finished()

    def update_output(self, text):
        """Actualiza el área de texto de forma segura desde el hilo."""
        self.output_textbox.configure(state="normal")
        self.output_textbox.insert("end", text)
        self.output_textbox.see("end") # Auto-scroll hacia el final
        self.output_textbox.configure(state="disabled")

    def processing_finished(self):
        """Reactiva el botón de 'run'."""
        self.run_button.configure(state="normal", text="▶️ Iniciar Procesamiento")

if __name__ == "__main__":
    app = AppLauncher()
    app.mainloop()