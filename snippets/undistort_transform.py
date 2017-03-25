import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle


mtx = None
dist = None
def load_data():
    global mtx
    global dist
    dist_pickle = pickle.load(open('../data/cam.p', 'rb'))
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']

img = cv2.imread('../camera_cal/calibration1.jpg')
nx = 9
ny = 6

def corner_unwarp(img, nx, ny, mtx, dist):
    """Unwarp the image and draw borders."""
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(dst, (nx, ny), None)

    if ret:
        cv2.drawChessboardCorners(dst, (nx, ny), corners, ret)

        offset = 100
        img_size = (gray.shape[1], gray.shape[0])


        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                [img_size[0]-offset, img_size[1]-offset], [offset, img_size[1] - offset]])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(dst, M, img_size)

        return warped, M


def main():
    global img
    global nx
    global ny
    global mtx
    global dist
    top_down, perspective_M = corner_unwarp(img, nx, ny, mtx, dist)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('original Image', fontsize=20)
    ax2.imshow(top_down)
    ax2.set_title('unwarped image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()



if __name__ == '__main__':
    main()

