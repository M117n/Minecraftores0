import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from yolov4.tf import YOLOv4
import tempfile
import matplotlib.pyplot as plt

# Configuración del dataset
data_dir = r"C:\Users\maart\Desktop\Martin\Minecraft proyect\i minecraft ores dataset"
categories = ['coal_ores', 'copper_ores', 'diamond_ores', 'esmerald_ores', 'gold_ores', 'iron_ores', 'lapis_ores', 'netherita_ores', 'quartz_ore', 'redstone_ores']
img_size = 64

# Funciones para cargar y preparar datos
def load_data(data_dir, categories, img_size):
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_resized = cv2.resize(img_array, (img_size, img_size))
                data.append([img_resized, class_num])
            except Exception as e:
                pass
    return data

def prepare_image(image_path, img_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.array(img).reshape(-1, img_size, img_size, 3)
    return img

# Cargar y procesar datos
data = load_data(data_dir, categories, img_size)
np.random.shuffle(data)

X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 3)
y = np.array(y)

X = X / 255.0

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Crear y entrenar el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

datagen_train = ImageDataGenerator()
datagen_val = ImageDataGenerator()

train_generator = datagen_train.flow(X_train, y_train)
validation_generator = datagen_val.flow(X_val, y_val)

model.fit(train_generator, validation_data=validation_generator, epochs=15, verbose=1)

# Guardar el modelo entrenado
model.save('minecraft_mineral_detection_model.h5')

# Cargar el modelo entrenado
model.load_weights("minecraft_mineral_detection_model.h5")

def predict_image(image_path, model, categories, img_size):
    prepared_image = prepare_image(image_path, img_size)
    prediction = model.predict(prepared_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = categories[predicted_class_index]
    return predicted_class

# Reemplaza esto con la ruta de tu imagen de prueba
test_image_path = r"C:\Users\maart\Desktop\Martin\Minecraft proyect\gold.png"

predicted_class = predict_image(test_image_path, model, categories, img_size)
print("La imagen fue clasificada como:", predicted_class)
