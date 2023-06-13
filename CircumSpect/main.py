from image_captioning import caption_image
import time
import cv2

cap = cv2.VideoCapture(0)
time.sleep(2)

while True:
    _, frame = cap.read()
    cv2.imwrite("image.png", frame)
    cv2.putText(frame, str(caption_image('image.png')[0]), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 5)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)