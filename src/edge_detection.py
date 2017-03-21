import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('../examples/curved-lane.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

abs_sobelx = np.absolute(sobelx)
abs_sobely = np.absolute(sobely)

scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))

# Apply mask
min_threshold = 20
max_threshold = 100

sxbinary = np.zeros_like(scaled_sobelx)
sxbinary[(scaled_sobelx >= min_threshold) & (scaled_sobelx <= max_threshold)] = 1
plt.imshow(sxbinary, cmap='gray')
plt.show()


sybinary = np.zeros_like(scaled_sobely)
sybinary[(scaled_sobely >= min_threshold) & (scaled_sobely <= max_threshold)] = 1
plt.imshow(sybinary, cmap='gray')
plt.show()

syscombine = np.sqrt(sxbinary**2 + sybinary**2)
plt.imshow(syscombine, cmap='gray')
plt.show()
