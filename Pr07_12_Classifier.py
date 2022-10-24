import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from tensorflow.keras.activations import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers.experimental.preprocessing import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.optimizers.schedules import *
#import albumentations as A

#from itertools import islice
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image

class_names = [ 
  'Приора', #0
  'Ford Focus', #1
  'Самара', #2
  'ВАЗ-2110', #3
  'Жигули', #4
  'Нива', #5
  'Калина', #6
  'ВАЗ-2109', #7
  'Volkswagen Passat', #8
  'ВАЗ-21099' #9
]

print('File:',sys.argv[1])
# '/content/drive/MyDrive/DS/Pr07_FvF/niva.jpg'
image_size = (224,224)
image = np.array(Image.open(sys.argv[1]).convert('RGB').resize(image_size))
#image = image.resize(image_size)
# plt.imshow(image)
# plt.show() 

model = Sequential([
  EfficientNetB0(weights='imagenet', input_shape=(*image_size, 3), include_top=False), #предобученная нейросеть из модуля keras.applications
  GlobalAveragePooling2D(),
  Dense(256,activation='relu', bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
  BatchNormalization(),
  Dropout(0.25),
  Dense(10)
])

model.compile(
    loss=CategoricalCrossentropy(from_logits=True),
    #0.96 - лучшаяя сходимость 
    #0.97 - возможен уникальный выдающийся результата
    optimizer=Adam(ExponentialDecay(1e-3, 100, 0.96)),  
    metrics='accuracy'
)

image = image[None, ...]

model.load_weights('/content/drive/MyDrive/DS/Pr07_FvF/Models/ENB095914.hdf5')

# получаем батч предсказаний и берем нулевой элемент
pred = model.predict(image)[0]

# берем индекс класса с максимальным значением
class_idx = pred.argmax()

# получаем название
print('Predicted class:', class_idx, ',', class_names[class_idx])
