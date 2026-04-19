#Evaluación del modelo de red neuronal convolucional entrenado

import numpy as np
from tensorflow.keras.utils import load_img,img_to_array
from keras.models import load_model
import os.path
from pathlib import Path

#Leer la imagen a evaluar

#imagen = "archive/train/cats/cat_2.jpg"
#imagen = "archive/train/dogs/dog_0.jpg"

#Recortar la imagen

altura,anchura = 50,50
#Leer el modelo entrenado 
modelo = "modelo_CatDog.keras"


#Cargar el modelo entrenado
cnn = load_model(modelo)


for imagen in Path('archive/train/dogs/').iterdir():
  #Transformar la imagen a clasificar
  imagen_clasificar = load_img(imagen,target_size=(altura,anchura))
  imagen_clasificar = img_to_array(imagen_clasificar)
  imagen_clasificar =imagen_clasificar / 255.0
  imagen_clasificar = np.expand_dims(imagen_clasificar,axis =0)

  #Evaluar la imagen
  prediccion = cnn.predict(imagen_clasificar)
  print(f"Probabilidades por clase: {prediccion}")
  arg_max = np.argmax(prediccion)
  print(f"argMax:{arg_max}")

  if arg_max == 0:
    print("Gato")
  elif(arg_max==1):
    print("Perro")