import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('../examples/color-shadow-example.jpg', 'r')


def gradient_and_combine(s_thresh=(170, 255), sx_thresh=(20, 100)):
    """Take gradient of the color channels and combine."""
    global image
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # sobel in hls
    sobelx = cv2.Sobel(image_hls[:, :, 1], cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx/np.max(abs_sobelx))

    # sobel in gray
    sobelGray = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelGray = np.absolute(sobelGray)
    scaled_sobelGray = np.uint8(255 * abs_sobelGray / np.max(abs_sobelGray))

    # Threshold
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel in hls.
    sbinary = np.zeros_like(image_hls[:, :, 2])
    sbinary[(image_hls[:, :, 2] >= s_thresh[0]) & (image_hls[:, :, 2] <= s_thresh[1])]  = 1

    # Threshold color channel in hsv.
    svbinary = np.zeros_like(image_hsv[:, :, 2])
    svbinary[(image_hsv[:, :, 2] >= s_thresh[0]) & (image_hsv[:, :, 2] <= s_thresh[1])] = 1

    sobelxv = cv2.Sobel(image_hsv[:, :, 2], cv2.CV_64F, 1, 0)
    abs_sobelxv = np.absolute(sobelxv)
    scaled_sobelxv = np.uint8(255 * abs_sobelxv/np.max(abs_sobelxv))

    sxvbinary = np.zeros_like(scaled_sobelxv)
    sxvbinary[(scaled_sobelxv >= s_thresh[0])&( scaled_sobelxv <= s_thresh[1])] = 1

    labels = ['hls-sobel', 'hls-masking', 'hsv-masking']
    plot_multiple(sxbinary, sbinary, svbinary, labels)

    # Merge binary.
    mergeBinary = np.zeros_like(sxbinary)
    mergeBinary[(sxbinary == 1) | (svbinary == 1) | (sbinary == 1)] = 1
    plt.imshow(mergeBinary, cmap='gray')
    plt.show()
    color_binary =  np.dstack((np.zeros_like(sxbinary), sxbinary, sbinary))
    return color_binary


def plot_multiple(image1, image2, image3, labels):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    f.tight_layout()
    ax1.imshow(image1, cmap='gray')
    ax1.set_title(labels[0], fontsize=20)
    ax2.imshow(image2, cmap='gray')
    ax2.set_title(labels[1], fontsize=20)
    ax3.imshow(image3, cmap='gray')
    ax3.set_title(labels[2], fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def visualize_colorspace():
    global image
    thres = (150, 255)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary_image = np.zeros_like(gray)
    binary_image[(gray > thres[0]) & (gray < thres[1])] = 1
    plt.imshow(binary_image, cmap='gray')
    plt.show()

    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    f.tight_layout()
    ax1.imshow(image_hsv[:, :, 0], cmap='gray')
    ax1.set_title('H', fontsize=20)
    ax2.imshow(image_hsv[:, :, 1], cmap='gray')
    ax2.set_title('S', fontsize=20)
    ax3.imshow(image_hsv[:, :, 2], cmap='gray')
    ax3.set_title('V', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    f.tight_layout()
    ax1.imshow(image_hls[:, :, 0], cmap='gray')
    ax1.set_title('H', fontsize=20)
    ax2.imshow(image_hls[:, :, 1], cmap='gray')
    ax2.set_title('L', fontsize=20)
    ax3.imshow(image_hls[:, :, 2], cmap='gray')
    ax3.set_title('S', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

    # Threshold S channel.
    thres = (50, 70)
    image_h = image_hsv[:, :, 2]
    print("Max and min..",np.max(image_h), np.min(image_h))
    plt.imshow(image_h, cmap='gray')
    plt.show()

def main():
    image = None
    # image = gradient_and_combine()
    visualize_colorspace()
    # plt.imshow(image)
    # plt.show()

if __name__ == '__main__':
    main()