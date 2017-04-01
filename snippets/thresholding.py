import numpy as np
import cv2
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = cv2.imread('../examples/color-shadow-example.jpg')
# image = cv2.imread('../examples/solidYellowCurve.jpg')

#Load the calibration paramters.
data = pickle.load(open('../data/cam.p', 'rb'))
mtx = data['mtx']
dist = data['dist']


def gradient_and_combine(s_thresh=(50, 100), sx_thresh=(10, 200)):
    """Take gradient of the color channels and combine."""
    global image
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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


def thresholdImage(image, thres):
    """Apply thresholding to image."""
    mergeBinary = np.zeros_like(image)
    mergeBinary[(image >= thres[0])&(image <= thres[1])] = 1
    return mergeBinary

def sharpen_image(image, condition='edge'):
    """Sharpen given image."""
    kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    kernel_sharpen_2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
    kernel_sharpen_3 = np.array([[-1, -1, -1, -1, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, 2, 8, 2, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 8.0
    if condition == 'sharpen':
        output = cv2.filter2D(image, -1, kernel_sharpen_1)
    elif condition == 'edge':
        output = cv2.filter2D(image, -1, kernel_sharpen_3)
    else:
        output = cv2.filter2D(image, -1, kernel_sharpen_2)
    return output


def sobel_thresh(image, thres=(170, 255), axis='x', kernel_size=3, apply_gray=False):
    """Do sobel threholding along specific axis."""
    interm = np.copy(image)
    ax = [1, 0]
    if apply_gray:
        interm = cv2.cvtColor(interm, cv2.COLOR_BGR2GRAY)
    if axis == 'x':
        sobel = cv2.Sobel(interm, cv2.CV_64F, 1, 0, kernel_size)
    elif axis == 'y':
        sobel = cv2.Sobel(interm, cv2.CV_64F, 0, 1, kernel_size)
    else:
        sobel_x = cv2.Sobel(interm, cv2.CV_64F, 1, 0, kernel_size)
        sobel_y = cv2.Sobel(interm, cv2.CV_64F, 0, 1, kernel_size)
        # Magnitude across both axis.
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    mag_binary = thresholdImage(sobel, thres)
    return mag_binary


def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2), apply_gray=False):
    """Threshold an image to a specific range of direction of gradients."""

    if apply_gray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.copy(img)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient directions.
    absgrad = np.arctan2(np.absolute(sobel_x), np.absolute(sobel_y))
    dir_binary = thresholdImage(absgrad, thresh)
    return dir_binary

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
    thres = (180, 255)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plt.imshow(thresholdImage(gray, thres), cmap='gray')
    plt.show()

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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

    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
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


def segment_area_of_interest(img):
    """Segment area of interest."""
    image = np.copy(img)
    print("Image shape: ", image.shape)
    width = image.shape[1]
    height = image.shape[0]

    mask = [0.078, 0.458, 0.56, 0.932, 0.58]
    x1_1 = mask[0] * width
    x1_2 = mask[1] * width
    x2_1 = mask[2] * width
    x2_2 = mask[3] * width
    y = mask[4] * height
    left_bot_outer = (x1_1 + 10, height)
    right_bot_outer = (x2_2, height)
    left_top_outer = (x1_2 + 10, y)
    right_top_outer = (x2_1, y)

    #Remove artifacts from center.
    left_bot_inner = (x1_1 + 100, height)
    right_bot_inner = (x2_2 - 100, height)
    left_top_inner = (x1_2 + 20, y + 50)
    right_top_inner = (x2_1 - 20, y + 50)

    vertices = np.array([[left_bot_outer, left_top_outer, right_top_outer,
                          right_bot_outer, right_bot_inner,
                          right_top_inner, left_top_inner, left_bot_inner ]], dtype=np.int32)
    return vertices


def create_perspective_transform(img):
    """
    Create a perspective transform based on the predefined set of values.
    :return:
    """
    output = np.copy(img)
    plt_img = np.copy(img)
    print("Image shape: ", img.shape)
    width = output.shape[1]
    height = output.shape[0]
    img_size = (output.shape[1], output.shape[0])

    mask = [0.16, 0.462, 0.537, 0.932, 0.62]
    x1_1 = mask[0] * width
    x1_2 = mask[1] * width
    x2_1 = mask[2] * width
    x2_2 = mask[3] * width
    y = mask[4] * height
    left_bot_outer = [x1_1, height]
    right_bot_outer = [x2_2, height]
    left_top_outer = [x1_2, y]
    right_top_outer = [x2_1, y]

    src = np.array([[left_bot_outer, left_top_outer, right_top_outer,
                     right_bot_outer]], dtype=np.int32)
    offset = 400 # Arbitrary offset.
    left_bot_dst = [x1_1 + offset, height]
    right_bot_dst = [x2_2 - offset, height]
    left_top_dst = [x1_1 + offset, y]
    right_top_dst = [x2_2 - offset, y]

    arg_src = np.array([left_top_outer, right_top_outer, right_bot_outer,
                        left_bot_outer], dtype=np.float32)
    print("source coordinates...", arg_src)

    dst = np.float32([[offset, 0], [img_size[0]-offset, 0],
                                     [img_size[0]-offset, img_size[1]],
                                     [offset, img_size[1]]] )

    # # Test the coordinates.
    # cv2.polylines(plt_img, src, True, (153, 255, 51), thickness=1)
    # cv2.imshow('default', plt_img)
    # cv2.waitKey(0)

    M = cv2.getPerspectiveTransform(arg_src, dst)
    warped = cv2.warpPerspective(output, M, (width, height))
    return warped


def mask_region(img):
    """
    Mask the ROI in the image.
    :param img: Input Image.
    :return: Binary image with region mask.
    """
    global image
    temp_image = np.copy(image)
    binary_mask = np.zeros_like(img)
    vertices = segment_area_of_interest(img)
    cv2.polylines(temp_image, vertices, True, (0, 255, 255))
    plt.imshow(temp_image)
    plt.show()
    color = (255,)
    print(vertices)
    cv2.fillPoly(binary_mask, vertices, color)
    masked_image = cv2.bitwise_and(img, binary_mask)
    return masked_image

def undistort_image(img):
    """
    Undistort the image, as per calibration parameters.
    :param img: Input image.
    :return: Undistorted image.
    """
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


def create_merged_binary(image, apply_gray=False):
    """Create a merged binary using magnitude, gradient and direction."""
    thresh_x = (80, 255)
    thresh_y = (70, 255)
    thresh_md = (40, 255)
    thresh_d = (0.6, 1.03)
    binary_x = sobel_thresh(image, thresh_x, axis='x', kernel_size=15, apply_gray=True)
    binary_y = sobel_thresh(image, thresh_y, axis='y', kernel_size=15, apply_gray=True)
    binary_md = sobel_thresh(image, thresh_md, axis='xy', kernel_size=15, apply_gray=True)
    binary_dir = dir_thresh(image, 15, thresh_d, apply_gray=True)
    # Merge the binary image over x,y and magnitude, direction.
    binary_img = np.zeros_like(binary_dir)
    binary_img[((binary_x==1) & (binary_y ==1)) | ((binary_md == 1) & (binary_dir ==1))] = 1

    # Gradient on S channel.
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel_binary = thresholdImage(hls_image[:, :, 2], (120, 255))

    # combine both binary images.
    output = np.zeros_like(binary_img)
    output[(binary_img == 1) | ( s_channel_binary == 1)] = 1
    return output


def detect_lane_lines(img):
    # binary_image = create_merged_binary(img, apply_gray=True)
    binary_image = codsnippet_from_udacity(img)
    output = mask_region(binary_image)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    f.tight_layout()
    ax1.imshow(binary_image, cmap='gray')
    ax1.set_title("Binary", fontsize=20)
    ax2.imshow(output, cmap='gray')
    plt.show()
    return output


def codsnippet_from_udacity(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    # Plotting thresholded images
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.set_title('Stacked thresholds')
    # ax1.imshow(color_binary)

    # ax2.set_title('Combined S channel and gradient thresholds')
    # ax2.imshow(combined_binary, cmap='gray')
    # plt.show()
    return combined_binary


def main():

    global image
    print("mtx", mtx)
    print("dist", dist)
    # image = gradient_and_combine()
    # visualize_colorspace()
    # plt.imshow(image)
    # plt.show()
    output = cv2.GaussianBlur(image, (3, 3), 0)
    dst = undistort_image(image)
    # codsnippet_from_udacity(dst)
    # dst = sharpen_image(dst)
    # output = create_merged_binary(dst, apply_gray=True)
    # plt.imshow(output, cmap='gray')
    # plt.show()
    otp = detect_lane_lines(dst)
    warped = create_perspective_transform(otp)
    plt.imshow(warped, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()