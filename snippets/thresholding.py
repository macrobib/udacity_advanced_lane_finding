import numpy as np
import cv2
import time
import pickle
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from collections import deque
# image = cv2.imread('../examples/color-shadow-example.jpg')
# image = cv2.imread('../examples/solidYellowCurve.jpg')


prev_xLeft_values = deque(maxlen=11)
prev_xRight_values = deque(maxlen=11)
last_left_value = None
last_right_value = None
weights = None
#Load the calibration paramters.
data = pickle.load(open('../data/cam.p', 'rb'))
mtx = data['mtx']
dist = data['dist']
condition = {
    'visualize': False,
    'smoothen' : False,
    'sharpen'  : False
}


def gradient_and_combine(image, s_thresh=(50, 100), sx_thresh=(10, 200), visualize = False):
    """Take gradient of the color channels and combine."""
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
    if visualize:
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


def visualize_colorspace(image):
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
    plt.imshow(image_h, cmap='gray')
    plt.show()


def vertices_area_of_interest(img):
    """Segment area of interest."""
    global y_points
    global weights
    image = np.copy(img)
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
    y_points = np.linspace(0, img.shape[0], 128)
    w1 = np.array([0.8])
    w2 = np.ones([10]) * 0.2/10
    weights = np.append(w1, w2)

    vertices = np.array([[left_bot_outer, left_top_outer, right_top_outer,
                          right_bot_outer, right_bot_inner,
                          right_top_inner, left_top_inner, left_bot_inner ]], dtype=np.int32)
    print("Vertices area of interest: --", vertices.shape)
    return vertices


def create_perspective_transform(img, visualize=False):
    """
    Create a perspective transform based on the predefined set of values.
    :return:
    """
    output = np.copy(img)
    plt_img = np.copy(img)
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

    dst = np.float32([[offset, 0], [img_size[0]-offset, 0],
                                     [img_size[0]-offset, img_size[1]],
                                     [offset, img_size[1]]] )

    # # Test the coordinates.
    if visualize:
        cv2.polylines(plt_img, src, True, (153, 255, 51), thickness=1)
        cv2.imshow('default', plt_img)
        cv2.waitKey(0)

    M = cv2.getPerspectiveTransform(arg_src, dst)
    MInv = cv2.getPerspectiveTransform(dst, arg_src)
    warped = cv2.warpPerspective(output, M, (width, height))
    print("Warped dimension: - ", warped.shape[0])
    return warped, M , MInv


def mask_region(img, visualize= True):
    """
    Mask the ROI in the image.
    :param img: Input Image.
    :return: Binary image with region mask.
    """
    print("Mask region.")
    temp_image = np.copy(img)
    binary_mask = np.zeros_like(img)
    vertices = vertices_area_of_interest(img)
    cv2.polylines(temp_image, vertices, True, (0, 255, 255))
    if visualize:
        print("visualize")
        plt.imshow(temp_image)
        plt.show()
    color = (255,)
    cv2.fillPoly(binary_mask, vertices, color)
    masked_image = cv2.bitwise_and(img, binary_mask)
    plt.imshow(masked_image)
    plt.show()
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


def mark_area_of_interest(img, visualize=False):
    # binary_image = create_merged_binary(img, apply_gray=True)
    binary_image = binarize_image(img)
    output = mask_region(binary_image, True)
    if visualize:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        f.tight_layout()
        ax1.imshow(binary_image, cmap='gray')
        ax1.set_title("Binary", fontsize=20)
        ax2.imshow(output, cmap='gray')
        plt.show()
    return output


def binarize_image(img, visualize=False):
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
    if visualize:
        # Plotting thresholded images
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary)

        ax2.set_title('Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')
        plt.show()

    return combined_binary


def histogram_detect(img, visualize = False):
    """ Find peaks in histogram in the binary image."""
    global y_points
    histogram = np.sum(img[img.shape[0]//2:, :], axis=0)
    # https://gist.github.com/jul/3794506
    histogram = medfilt(histogram, [41]) # Smoothen the histogram values.
    if visualize:
        plt.plot(histogram)
        plt.show()
    out_img = np.dstack((img, img, img)) * 255
    if visualize:
        plt.imshow(out_img)
        plt.show()

    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9
    window_height = np.int(img.shape[0]/nwindows)

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    # Width of the window.
    width = 100
    # Minimum number of pixels found in recenter window.
    minpix = 50

    # Empty lists to recieve left and right indices.
    left_lane_indices = []
    right_lane_indices = []

    # loop through the window
    for window in range(nwindows):
        win_y_low       = img.shape[0] - (window + 1)* window_height
        win_y_high      = img.shape[0] - window * window_height
        win_xleft_low   = leftx_current - width
        win_xleft_high  = leftx_current + width
        win_xright_low   = rightx_current - width
        win_xright_high  = rightx_current + width

        # Draw the windows on the visualization image.
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify nonzero pixels in x and y with the window.
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low)
                          & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                           & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the list.
        left_lane_indices.append(good_left_inds)
        right_lane_indices.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the indices.
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # Extract the left and right indices.
    leftx = nonzerox[left_lane_indices]
    lefty = nonzeroy[left_lane_indices]
    rightx = nonzerox[right_lane_indices]
    righty = nonzeroy[right_lane_indices]

    # Fit the points
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # print("HIstogram search", left_fit, right_fit)
    points = [lefty, leftx, righty, rightx]
    if visualize:
        draw_lane_line(out_img, img.shape, points, (left_fit, right_fit))

    return (points, (left_fit, right_fit))


def draw_lane_line(img, shape, coordinates, fit_points):
    """ Draw the lanes."""
    global y_points
    left_fit_params = fit_points[0]
    right_fit_params = fit_points[1]
    lfx = left_fit_params[0]*y_points**2 + left_fit_params[1]*y_points + left_fit_params[2]
    rfx = right_fit_params[0]*y_points**2 + right_fit_params[1]*y_points + right_fit_params[2]

    img[coordinates[0], coordinates[1]] = [255, 0, 0]
    img[coordinates[2], coordinates[3]] = [0, 0, 255]
    plt.imshow(img)
    plt.plot(lfx, draw_lane_line, color='yellow')
    plt.plot(rfx, draw_lane_line, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


def locality_search(img, fit_points, visualize = False):
    """
    Do sliding window search based on previous frames.
    :return:
    """
    global weights
    global last_left_value
    global last_right_value
    nonzerox = np.array(img.nonzero()[1])
    nonzeroy = np.array(img.nonzero()[0])
    fp_left  = fit_points[0]
    fp_right = fit_points[1]
    # print("Inside locality search: ", fit_points[0])

    left_x_points = fp_left[0]*(nonzeroy**2) + fp_left[1]*(nonzeroy) + fp_left[2]
    right_x_points = fp_right[0]*(nonzeroy**2) + fp_right[1]*(nonzeroy) + fp_right[2]
    margin = 100

    # determine the range to lookup for in the new frame based on the old coordinates.
    left_lane_inds = ((nonzerox > (left_x_points - margin)) & (nonzerox < (left_x_points + margin)))
    right_lane_inds = ((nonzerox > (right_x_points - margin)) & (nonzerox < (right_x_points + margin)))

    # Extract points.
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # fit points.
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # print("Locality search", left_fit, right_fit)

    # Generate new sets of points.
    gen_points = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*(y_points**2) + left_fit[1]*(y_points) + left_fit[2]
    right_fitx = right_fit[0]*(y_points**2) + right_fit[1]*(y_points) + right_fit[2]

    prev_xLeft_values.appendleft(left_fitx)
    prev_xRight_values.appendleft(right_fitx)
    if len(prev_xLeft_values) == 11:
        left_fitx = np.average(np.array(prev_xLeft_values), 0, weights=weights)
    if len(prev_xRight_values) == 11:
        right_fitx = np.average(np.array(prev_xRight_values), 0, weights=weights)
        # print(prev_xRight_values)

    # Weighted averaging:
    print("Shape: ", np.array(prev_xLeft_values).shape)
    # if visualize:
    #     draw_lane_search_area(img, (left_lane_inds, right_lane_inds), (left_fitx, right_fitx), 100)
    return ((leftx, lefty), (rightx, righty), (left_fitx, right_fitx))


def search_lane_points(img, visualize = False):
    """ Search for lane points based on previous frame, in case of multiple frame miss do a new histogram search."""


def draw_lane_search_area(img, lane_point_indices, fit_points, margin):
    """Visualize the range of search for the coordinates in the new image."""
    global y_points
    gen_points = np.linspace(0, img.shape[0] - 1, img.shape[0])
    nonzeroy = np.array(img.nonzero()[0])
    nonzerox = np.array(img.nonzero()[1])

    output = np.dstack((img, img, img))*255
    window_img = np.zeros_like(output)

    # color the left and right coordinates.
    output[nonzeroy[lane_point_indices[0]], nonzerox[lane_point_indices[0]]] = [255, 0, 0]
    output[nonzeroy[lane_point_indices[1]], nonzerox[lane_point_indices[1]]] = [0, 0, 255]

    # Generate point indicating the search area.
    left_line_window_1 = np.array([np.transpose(np.vstack([fit_points[0] - margin, y_points]))])
    left_line_window_2 = np.array([np.flipud(np.transpose(np.vstack([fit_points[0] + margin, y_points])))])
    left_line_pts = np.hstack((left_line_window_1, left_line_window_2))

    right_line_window_1 = np.array([np.transpose(np.vstack([fit_points[1] - margin, y_points]))])
    right_line_window_2 = np.array([np.flipud(np.transpose(np.vstack([fit_points[1] + margin, y_points])))])
    right_line_pts = np.hstack((right_line_window_1, right_line_window_2))

    # Draw the search are on the blank image.
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    merged = cv2.addWeighted(output, 1, window_img, 0.4, 0)
    plt.imshow(merged)
    plt.plot(fit_points[0], y_points, color='yellow')
    plt.plot(fit_points[1], y_points, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


def calculate_curvature_and_center(img, left_points, right_points,fit_points, image_dim):
    """Determine the curvature from lane points"""
    x_m_per_pix  = 3.7/image_dim[1]
    y_m_per_pix  = 30/image_dim[0]
    global y_points
    # x_m_per_pix = 3.7 / 700
    # y_m_per_pix = 30 / 720
    # Calculate Curvature corresponding to centre of image.
    y_centre = img.shape[0] - img.shape[0]/2

    vehicle_center = (fit_points[1][-1] - fit_points[0][-1])//2 # (Max Right fit - Max Left fit)/2
    reference_center = (img.shape[1]//2)
    pixel_deviation = reference_center - vehicle_center # Calculates the deviation of vehicle in terms of pixels.
    deviation_in_m = pixel_deviation * x_m_per_pix

    left_fit_cr  = np.polyfit(left_points[1] * y_m_per_pix, left_points[0] * x_m_per_pix, 2)
    right_fit_cr = np.polyfit(right_points[1] * y_m_per_pix, right_points[0] * x_m_per_pix, 2)

    l_curv = ((1 + (left_fit_cr[0] * y_centre + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    r_curv = ((1 + (right_fit_cr[0] * y_centre + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    return (l_curv, r_curv, deviation_in_m)


def sanity_check_curvature(l_curv, r_curv):
    """ Check if the curvature is in limits."""
    check = True
    # Check similarity of curvature.
    if not -0.5 < l_curv/r_curv < 0.5:
        check = False
    # Check sanity of curvature.
    if l_curv < 400 or r_curv < 400:
        check = False
    return check


def lane_visualize(img, warped, fit_left, fit_right, minv, visualize= False):
    """Visualize the detected lane lines."""
    global y_points
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    print("warped image shape: --", warped.shape)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    print("lane visualization: Left {0} --- Right{1}".format(fit_left.shape, y_points.shape))
    pts_left  = np.array([np.transpose(np.vstack([fit_left, y_points]))])
    pts_middle_left = np.array([np.flipud(np.transpose(np.vstack([(fit_left + fit_right)/2, y_points])))])
    pts_middle_right = np.array([np.transpose(np.vstack([(fit_left + fit_right)/2, y_points]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_right, y_points])))])
    pts_1_halve = np.hstack((pts_left, pts_middle_left))
    pts_2_halve = np.hstack((pts_middle_right, pts_right))
    poly_arg_left = np.array(pts_1_halve)
    poly_arg_right = np.array(pts_2_halve)
    cv2.fillPoly(color_warp, np.int_(poly_arg_left), (102, 255, 178))
    cv2.fillPoly(color_warp, np.int_(poly_arg_right), (0, 204, 102))
    pts_left = pts_left.astype(np.int32)
    pts_middle = pts_middle_right.astype(np.int32)
    pts_right = pts_right.astype(np.int32)
    cv2.polylines(color_warp, pts_left, False, (255, 51, 51), 3, cv2.LINE_AA)
    cv2.polylines(color_warp, pts_middle, False, (204, 204, 0), 3, cv2.LINE_AA)
    cv2.polylines(color_warp, pts_right, False, (255, 51, 51), 3, cv2.LINE_AA)
    if visualize:
        plt.imshow(color_warp)
        plt.show()
    print("Minv shape: {0} -- Color warp shape: {1}".format(minv.shape, color_warp.shape))
    # Warp the image back and merge with stock image.
    new_warp = cv2.warpPerspective(color_warp, minv, (warped.shape[1], warped.shape[0]))
    print("New warped shape: -- ", new_warp.shape)
    if visualize:
        plt.imshow(new_warp)
        plt.show()
    result = cv2.addWeighted(img, 1, new_warp, 0.3, 0)
    if visualize:
        plt.imshow(result)
        plt.show()
    return result


def draw_lane_and_text(image, warped, curvatures, lane_distance, fitpoints, minv):
    """Draw the text and images"""
    width = warped.shape[1]
    height = warped.shape[0]
    delta = 50

    update_img = lane_visualize(image, warped, fitpoints[0], fitpoints[1], minv, False)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    left_curvature = "Left Curvature: " + str(round(curvatures[0], 2))
    right_curvature = "Right Curvature: " + str(round(curvatures[1], 2))
    lane_distance = "Lane Dev " + str(round(lane_distance, 3))
    cv2.putText(update_img,left_curvature, (0 + delta, 0 + delta), font, 0.8, (51, 255, 153), 1, cv2.LINE_AA)
    cv2.putText(update_img, right_curvature, (width - 400 - len(right_curvature), 0 + delta), font, 0.8, (51, 255, 153), 1, cv2.LINE_AA)
    cv2.putText(update_img, lane_distance, (width//2 - len(lane_distance) - 10, height - 150), font, 0.7, (51, 255, 153), 1, cv2.LINE_AA)
    return update_img


def pipeline(img):
    """Coordinate the processing of clip."""
    global condition
    output = img
    print(img.shape)
    plt.imshow(img)
    plt.show()
    if condition['visualize']:
        image = gradient_and_combine(img)
        visualize_colorspace(img)
        plt.imshow(img)
        plt.show()
    if condition['smoothen']:
        output = cv2.GaussianBlur(output, (3, 3), 0)
    dst = undistort_image(output)
    if condition['sharpen']:
        dst = sharpen_image(dst)
    otp = mark_area_of_interest(dst)
    warped, M, MInv = create_perspective_transform(otp)
    if condition['visualize']:
        plt.imshow(warped, cmap='gray')
        plt.show()
    opt = histogram_detect(warped)
    args = locality_search(warped, opt[1], False)
    output = calculate_curvature_and_center(img, args[0], args[1], args[2], img.shape)
    if sanity_check_curvature(output[0], output[1]):
        print("Curves meet the standards.")
    result = draw_lane_and_text(img, warped, (output[0], output[1]), output[2], args[2], MInv) # Draw the lane area and text info.
    plt.imshow(result)
    plt.show()
    return result


def process_video(file):
    """Process video file and output annotated video."""
    output = "../output/project_video.mp4"
    clip = VideoFileClip(file)
    output_video = clip.fl_image(pipeline)
    output_video.write_videofile(output, audio=False)

def main():

    visualize = False
    print("Starting video processing pipeline.")
    process_video("../videos/project_video.mp4")

if __name__ == '__main__':
    main()