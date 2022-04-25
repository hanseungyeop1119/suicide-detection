import tensorflow as tf
# 훈련을 위한 데이터
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
   '../classification_data/',
    image_size=(600, 760),
    label_mode='categorical'
)

data = train_dataset.take(1)
for image, labels in data:
    # image : 문제, labels : 정답
    print(image, labels)
print(train_dataset.class_names)