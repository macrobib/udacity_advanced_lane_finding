import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageHandler:
    """
    Utility class for handling images.
    """
    def __init__(self):
        self.mask = [0.078, 0.458, 0.56, 0.932, 0.58]
        self.kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.kernel_sharpen_2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
        self.kernel_sharpen_3 = np.array([[-1, -1, -1, -1, -1],
                                     [-1, 2, 2, 2, -1],
                                     [-1, 2, 8, 2, -1],
                                     [-1, 2, 2, 2, -1],
                                     [-1, -1, -1, -1, -1]]) / 8.0
        self.sobel_thres_min = 20
        self.sobel_thres_max = 100
        self.channel_thresh_min = 170
        self.channel_thresh_max = 255
        self.MInv = None
        self.M = None

    @staticmethod
    def blur_image(img):
        cv2.GaussianBlur(img, (3, 3), 0)

    def get_minv(self):
        return self.MInv

    def binarize_image(self, img=None):
        """
        Create thresholded binary image.
        :param image: Input Image.
        :return: Binarized image.
        """
        if img == None:
            raise FileNotFoundError

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
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.sobel_thres_min) & (scaled_sobel <= self.sobel_thres_max)] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.channel_thresh_min) & (s_channel <= self.channel_thresh_max)] = 1

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return combined_binary

    def sharpen_image(self, image=None, condition='edge'):

        if condition == 'sharpen':
            output = cv2.filter2D(image, -1, self.kernel_sharpen_1)
        elif condition == 'edge':
            output = cv2.filter2D(image, -1, self.kernel_sharpen_3)
        else:
            output = cv2.filter2D(image, -1, self.kernel_sharpen_2)
        return output

    def segment_area_of_interest(self, img = None):
        """Segment area of interest."""
        image = np.copy(img)
        print("Image shape: ", image.shape)
        width = image.shape[1]
        height = image.shape[0]

        x1_1 = self.mask[0] * width
        x1_2 = self.mask[1] * width
        x2_1 = self.mask[2] * width
        x2_2 = self.mask[3] * width
        y = self.mask[4] * height
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
        else:
            raise FileNotFoundError

    def mark_area_of_interest(self, img):
        # binary_image = create_merged_binary(img, apply_gray=True)
        binary_image = self.binarize_image(img)
        output = self.mask_region(binary_image)
        return output

    def create_perspective_transform(self, img, visualize=False):
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
        offset = 400  # Arbitrary offset.
        left_bot_dst = [x1_1 + offset, height]
        right_bot_dst = [x2_2 - offset, height]
        left_top_dst = [x1_1 + offset, y]
        right_top_dst = [x2_2 - offset, y]

        arg_src = np.array([left_top_outer, right_top_outer, right_bot_outer,
                            left_bot_outer], dtype=np.float32)

        dst = np.float32([[offset, 0], [img_size[0] - offset, 0],
                          [img_size[0] - offset, img_size[1]],
                          [offset, img_size[1]]])

        # # Test the coordinates.
        if visualize:
            cv2.polylines(plt_img, src, True, (153, 255, 51), thickness=1)
            cv2.imshow('default', plt_img)
            cv2.waitKey(0)

        if self.M is None or self.MInv is None:
            self.M = cv2.getPerspectiveTransform(arg_src, dst)
            self.MInv = cv2.getPerspectiveTransform(dst, arg_src)
        warped = cv2.warpPerspective(output, self.M, (width, height))
        print("Warped dimension: - ", warped.shape[0])
        return warped

    def pipeline(self, img):
        """"""