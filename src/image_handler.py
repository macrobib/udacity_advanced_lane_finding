import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageHandler:
    """
    Utility class for handling images.
    """
    def __init__(self, image = None):
        self.image = image

    def draw_polygon(self, coordinates, image=None):
        """ Draw a polygon on the image."""
        cv2.polylines(image, coordinates, True, (153, 255, 51), thickness=1)
        cv2.imshow('default', image)
        cv2.waitKey(0)


    def binarize_image(self, img=None):
        """
        Create thresholded binary image.
        :param image: Input Image.
        :return: Binarized image.
        """
        if img == None:
            img = self.image

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

    def plot_image(self, image=None):
        """Display the given image."""
        if image:
            cv2.imshow('window', image)
            cv2.waitKey(0)

    def grid_plot(self, images):
        """Do a grid plot of given images."""
        pass

    def sharpen_image(self, image=None, condition='edge'):
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

    def segment_area_of_interest(self, img = None):
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

        # Remove artifacts from center.
        left_bot_inner = (x1_1 + 100, height)
        right_bot_inner = (x2_2 - 100, height)
        left_top_inner = (x1_2 + 20, y + 50)
        right_top_inner = (x2_1 - 20, y + 50)

        vertices = np.array([[left_bot_outer, left_top_outer, right_top_outer,
                              right_bot_outer, right_bot_inner,
                              right_top_inner, left_top_inner, left_bot_inner]], dtype=np.int32)
        return vertices

    def mask_region(self, image=None):
        """ Mask a specific region of image. """
        if image:
            temp_image = np.copy(image)
            binary_mask = np.zeros_like(image)
            vertices = self.segment_area_of_interest(image)
            poly_vertices = vertices.reshape((-1, 1, 2))
            cv2.polylines(temp_image, vertices, True, (0, 255, 255))
            plt.imshow(temp_image)
            plt.show()
            color = (255,)
            print(vertices)
            cv2.fillPoly(binary_mask, vertices, color)
            masked_image = cv2.bitwise_and(image, binary_mask)
            return masked_image
