import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nx = 10
ny = 6

filename = './snippets/chessboard.png'

img = cv2.imread(filename)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

if ret == True:
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)




