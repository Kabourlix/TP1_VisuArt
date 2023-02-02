import cv2
import numpy as np

test = np.zeros((100, 100, 3), np.uint8)
test[0:50, 0:50] = (255, 0, 0)
test[50:100, 50:100] = (0, 255, 0)

cv2.imshow("test", test)
cv2.waitKey(0)