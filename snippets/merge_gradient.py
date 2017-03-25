import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


image = mpimg.imread('../examples/test6.jpg', 'r')


def pipeline(img, s_threshold=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    image_v = hsv[:, :, 2]

    hsl = cv2.cvtColor(img, )

