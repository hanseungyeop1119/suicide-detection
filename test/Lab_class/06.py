import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('../models/mymodel.h5')

class_names = ['ready', 'suicide', 'trying']

image = np.random.rand(600, 760, 3)
resize_image = cv2.resize(image, (600, 760))
print(resize_image.shape)

data = np.array([resize_image])
print(data.shape)

predict = model.predict(data)
print(predict)

index = np.argmax(predict)
print(index)

print(class_names[index])