import sys
import os
import numpy as np

# Bloqueamos errores de sistema
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Convolution2D, MaxPooling2D
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

class MosquitoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clasificador UPEMOR 8-C")
        self.setGeometry(100, 100, 450, 650)
        self.setStyleSheet("background-color: #121212; color: white;")
        
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
            model.add(Convolution2D(kernels1, kernel1_size, padding="same", 
                                   input_shape=(altura, anchura, 3), activation="relu"))
            model.add(MaxPooling2D(pool_size=size_pooling))
            model.add(Convolution2D(kernels2, kernel2_size, padding="same", activation="relu"))
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
        layout = QVBoxLayout()
        lbl = QLabel("Detector de Mosquitos")
        lbl.setFont(QFont("Arial", 20, QFont.Bold))
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)

        self.lbl_img = QLabel("Inserta la foto aquí")
        self.lbl_img.setFixedSize(380, 380)
        self.lbl_img.setStyleSheet("border: 2px solid #0078d4; background: #1e1e1e;")
        self.lbl_img.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_img, alignment=Qt.AlignCenter)

        btn = QPushButton("SELECCIONAR IMAGEN")
        btn.setFixedHeight(50)
        btn.setStyleSheet("background: #0078d4; font-weight: bold; border-radius: 8px;")
        btn.clicked.connect(self.abrir)
        layout.addWidget(btn)

        self.lbl_res = QLabel("Esperando imagen...")
        self.lbl_res.setFont(QFont("Arial", 16))
        self.lbl_res.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_res)
        self.setLayout(layout)

    def abrir(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir", "", "Imágenes (*.jpg *.png *.jpeg)")
        if path:
            self.lbl_img.setPixmap(QPixmap(path).scaled(380, 380, Qt.KeepAspectRatio))
            self.predecir(path)

    def predecir(self, path):
        if not self.model: return
        try:
            # Usar el tamaño correcto del modelo: 50x50
            img = keras.utils.load_img(path, target_size=(50, 50))
            x = keras.utils.img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)

            preds = self.model.predict(x, verbose=0)
            print(preds)
            idx = np.argmax(preds[0])
            
            clases = ["Aedes Aegypti", "Anopheles", "Culex"]
            nombre_mosquito = clases[idx]
            
            # Obtener información de peligrosidad
            info_peligro = self._get_mosquito_danger_info(nombre_mosquito)
            
            # Mostrar solo el nivel de peligro y enfermedades
            resultado_texto = info_peligro['nivel'] + "\n\n"
            resultado_texto += f"Enfermedades:\n{info_peligro['enfermedades']}"
            
            self.lbl_res.setText(resultado_texto)
            self.lbl_res.setStyleSheet(f"color: {info_peligro['color']}; font-weight: bold;")
        except Exception as e:
            print(f"Error: {e}")
            self.lbl_res.setText(f"Error en predicción: {str(e)}")
            self.lbl_res.setStyleSheet("color: #ff0000; font-weight: bold;")

    def _get_mosquito_danger_info(self, nombre_mosquito):
        """Retorna información de peligrosidad y enfermedades para cada tipo de mosquito"""
        mosquito_info = {
            "Aedes Aegypti": {
                "nivel": "⚠️ MUY PELIGROSO ⚠️",
                "enfermedades": "• Dengue\n• Zika\n• Chikungunya\n• Fiebre Amarilla",
                "color": "#ff0000"  # Rojo
            },
            "Anopheles": {
                "nivel": "⚠️ MUY PELIGROSO ⚠️",
                "enfermedades": "• Malaria\n• Fiebre Paratifoidea",
                "color": "#ff0000"  # Rojo
            },
            "Culex": {
                "nivel": "⚠️ PELIGROSO ⚠️",
                "enfermedades": "• Virus del Nilo Occidental\n• Encefalitis Japonesa",
                "color": "#ff9900"  # Naranja
            }
        }
        
        return mosquito_info.get(nombre_mosquito, {
            "nivel": "DESCONOCIDO",
            "enfermedades": "No disponible",
            "color": "#ffffff"
        })

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MosquitoApp()
    ex.show()
    sys.exit(app.exec_())