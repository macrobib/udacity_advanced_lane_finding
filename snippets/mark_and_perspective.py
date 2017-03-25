import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def markRegion(img):
    """Marks a trapezoidal region for perspective transform."""
    image = np.copy(img)
    width = image.shape[1]
    height = image.shape[0]

    mask = [0.078, 0.458, 0.53, 0.932, 0.58]
    x1_1 = mask[0] * width
    x1_2 = mask[1] * width
    x2_1 = mask[2] * width
    x2_2 = mask[3] * width
    y = mask[4] * height
    vertices = np.array([[(x1_1, width), (x1_2, y), (x2_1, y), (x2_2, width)]])
    pass

def doPerspetiveTransform(img):
    """Do a perspetive transform on the given image."""
    pass

def findCurvature(img):
    """Calculate and return the curvature of the image."""
    pass


def main():
    image = mpimg.imread('../examples/solidYellowCurve.jpg')
    dst = markRegion(image)

if __name__ == '__main__':
    main()