import cv2

file_name = 'IMG_3493.jpg'
image = cv2.imread('../data/images/' + file_name)

print(image)
print(image.shape)
print(type(image))