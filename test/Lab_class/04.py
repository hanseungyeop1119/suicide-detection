import tensorflow as tf
import os

model = tf.keras.applications.MobileNet(
    input_shape=(600, 760, 3),
    include_top=False,
    weights='imagenet'
)

model.trainable = False

model = tf.keras.Sequential([
    model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(3),
    tf.keras.layers.Softmax()
])

if not os.path.exists('../models'):
    os.mkdir('../models')

model.save('../models/mymodel.h5')