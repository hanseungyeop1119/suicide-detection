import cv2
import os

path = '../data/videos/IMG_3489.mov'
cap = cv2.VideoCapture(path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

if not os.path.exists('../classification_data/images'):
    os.mkdir('../classification_data/images')
cnt = 0



while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break
    resize_image = cv2.resize(image, (600, 760))
    cv2.imwrite('../classification_data/images/image' + str(cnt) + '.jpg', resize_image)
    cnt += 1


cap.release()


