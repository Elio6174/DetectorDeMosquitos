import sys
import os
import numpy as np

# Bloqueamos errores de sistema
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
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
        
        # Carga del modelo usando el motor de legado
        self.modelo_path = "modelo_MosquitosROBUSTO.keras"
        try:
            # tf_keras lee modelos viejos sin el error de conv2d
            self.model = keras.models.load_model(self.modelo_path, compile=False)
            print("✅ Modelo cargado con éxito.")
        except Exception as e:
            print(f"❌ Error al cargar: {e}")
            self.model = None

        self.init_ui()

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
            img = keras.utils.load_img(path, target_size=(224, 224))
            x = keras.utils.img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)

            preds = self.model.predict(x, verbose=0)
            idx = np.argmax(preds[0])
            
            clases = ["Aedes Aegypti", "Anopheles", "Culex"]
            self.lbl_res.setText(f"RESULTADO: {clases[idx]}")
            self.lbl_res.setStyleSheet("color: #00ff00; font-weight: bold;")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MosquitoApp()
    ex.show()
    sys.exit(app.exec_())