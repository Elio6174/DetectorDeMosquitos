import sys
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

from tensorflow.keras.utils import load_img,img_to_array
from keras.models import load_model

from PIL import Image, ImageTk

# Bloqueamos errores de sistema
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D

class MosquitoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clasificador UPEMOR 8-C")
        self.geometry("500x750")
        self.configure(bg="#121212")
        
        # Construir el modelo desde cero con la arquitectura conocida
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.modelo_path = os.path.join(script_dir, "modelo_MosquitosROBUSTO.keras")
        
        try:
            self.model = self._build_and_load_model()
            if self.model:
                print(f"Modelo cargado con éxito desde: {self.modelo_path}")
            else:
                print("No se pudo cargar el modelo")
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            self.model = None

        self.init_ui()

    def _build_and_load_model(self):
        """Construye la arquitectura del modelo y carga los pesos"""
        try:
            # Intentar cargar el modelo directamente primero
            model = keras.models.load_model(self.modelo_path, compile=False)
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
            return model
        except:
            pass
        
        try:
            # Si falla, construir la arquitectura manualmente
            altura, anchura = 50, 50
            kernels1 = 16
            kernels2 = 32
            kernel1_size = (3, 3)
            kernel2_size = (3, 3)
            size_pooling = (2, 2)
            
            model = Sequential()
            model.add(Conv2D(kernels1, kernel1_size, padding="same", 
                                   input_shape=(altura, anchura, 3), activation="relu"))
            model.add(MaxPooling2D(pool_size=size_pooling))
            model.add(Conv2D(kernels2, kernel2_size, padding="same", activation="relu"))
            model.add(MaxPooling2D(pool_size=size_pooling))
            model.add(Flatten())
            model.add(Dense(1000, activation="relu"))
            model.add(Dense(1000, activation="relu"))
            model.add(Dropout(0.5))
            model.add(Dense(3, activation="softmax"))  # 3 clases: Aedes Aegypti, Anopheles, Culex
            
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
            
            # Cargar pesos si existen
            try:
                weights_path = self.modelo_path.replace('.keras', '_weights.h5')
                if os.path.exists(weights_path):
                    model.load_weights(weights_path)
            except:
                pass
            
            return model
        except Exception as e:
            print(f"Error construyendo modelo: {e}")
            return None

    def init_ui(self):
        # Frame principal
        main_frame = tk.Frame(self, bg="#121212")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = tk.Label(main_frame, text="Detector de Mosquitos", 
                               font=("Arial", 20, "bold"), bg="#121212", fg="white")
        title_label.pack(pady=10)

        # Frame para la imagen
        img_frame = tk.Frame(main_frame, bg="#1e1e1e", relief=tk.SOLID, bd=2)
        img_frame.pack(pady=10)
        
        self.lbl_img = tk.Label(img_frame, text="Inserta la foto aquí", 
                               bg="#1e1e1e", fg="white", width=38, height=19, 
                               font=("Arial", 10))
        self.lbl_img.pack(padx=5, pady=5)
        self.img_photo = None

        # Botón para seleccionar imagen
        btn = tk.Button(main_frame, text="SELECCIONAR IMAGEN", command=self.abrir,
                       bg="#0078d4", fg="white", font=("Arial", 12, "bold"),
                       height=2, cursor="hand2")
        btn.pack(fill=tk.X, pady=10)

        # Etiqueta para resultado
        self.lbl_res = tk.Label(main_frame, text="Esperando imagen...", 
                               font=("Arial", 16, "bold"), bg="#121212", fg="white")
        self.lbl_res.pack(pady=20)

    def abrir(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.png *.jpeg"), ("Todos", "*.*")]
        )
        if path:
            try:
                # Cargar y mostrar la imagen
                img = Image.open(path)
                img.thumbnail((380, 380))
                self.img_photo = ImageTk.PhotoImage(img)
                self.lbl_img.config(image=self.img_photo, text="")
                self.predecir(path)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")

    def predecir(self, path):
        if not self.model: return
        try:
            altura,anchura = 50,50
            modelo = "modelo_MosquitosROBUSTO.keras"

            cnn = load_model(modelo)
            imagen_clasificar = load_img(path,target_size=(altura,anchura))
            imagen_clasificar = img_to_array(imagen_clasificar)
            imagen_clasificar =imagen_clasificar / 255.0
            imagen_clasificar = np.expand_dims(imagen_clasificar,axis =0)



            prediccion = cnn.predict(imagen_clasificar)

            print(f"Probabilidades por clase: {prediccion}")

            arg_max = np.argmax(prediccion)


            
            if arg_max == 0:
                  resultado_texto = "🟢 NO PELIGROSO"
                  color = "#00ff00"  # Verde
            elif(arg_max==1):
                  resultado_texto = "🔴 PELIGROSO"
                  color = "#ff0000"  # Rojo
            # Mostrar resultado
            #if es_peligroso:
            #    resultado_texto = "🔴 PELIGROSO"
            #    color = "#ff0000"  # Rojo
            #else:
            #    resultado_texto = "🟢 NO PELIGROSO"
            #    color = "#00ff00"  # Verde
            
            self.lbl_res.config(text=resultado_texto, fg=color)
        except Exception as e:
            print(f"Error: {e}")
            self.lbl_res.config(text=f"Error en predicción: {str(e)}", fg="#ff0000")
            messagebox.showerror("Error", f"Error en predicción: {str(e)}")



if __name__ == "__main__":
    app = MosquitoApp()
    app.mainloop()