import tensorflow as tf

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

model.summary()
