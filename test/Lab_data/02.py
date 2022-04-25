import cv2

file_name = 'IMG_3493.jpg'
image = cv2.imread('../data/images/Read/' + file_name)

# 띄우고 나서 바로 꺼짐

resize_image = cv2.resize(image, (600, 760))

cv2.imshow('resize image', resize_image)
cv2.waitKey(0)
