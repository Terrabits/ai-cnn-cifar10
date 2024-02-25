import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Dummy dataset for illustration
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0

# Define the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the dummy dataset
model.fit(x_train, y_train, epochs=5, batch_size=64)
