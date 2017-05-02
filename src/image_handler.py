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
        self.channel_thresh_min = 150
        self.channel_thresh_max = 255
        self.MInv = None
        self.M = None

    @staticmethod
    def blur_image(img):
        cv2.GaussianBlur(img, (3, 3), 0)

    def get_minv(self):
        return self.MInv

    def thresholdImage(self, image, thres):
        """Apply thresholding to image."""
        mergeBinary = np.zeros_like(image)
        mergeBinary[(image >= thres[0]) & (image <= thres[1])] = 1
        return mergeBinary

    def dir_thresh(self, img, sobel_kernel=3, thres=(0, np.pi / 2), apply_gray=False):
        """Threshold an image to a specific range of direction of gradients."""

        if apply_gray:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print("dir_thresh: Convert to grayscale")
        else:
            gray = np.copy(img)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient directions.
        absgrad = np.arctan2(np.absolute(sobel_x), np.absolute(sobel_y))
        dir_binary = self.thresholdImage(absgrad, thres)
        return dir_binary

    def sobel_thresh(self, image, thres=(170, 255), axis='x', kernel_size=3, apply_gray=False):
        """Do sobel threholding along specific axis."""
        interm = np.copy(image)
        ax = [1, 0]
        if apply_gray:
            print("sobel: Convert to grayscale")
            interm = cv2.cvtColor(interm, cv2.COLOR_BGR2GRAY)
        if axis == 'x':
            sobel = cv2.Sobel(interm, cv2.CV_64F, 1, 0, kernel_size)
        elif axis == 'y':
            sobel = cv2.Sobel(interm, cv2.CV_64F, 0, 1, kernel_size)
        else:
            sobel_x = cv2.Sobel(interm, cv2.CV_64F, 1, 0, kernel_size)
            sobel_y = cv2.Sobel(interm, cv2.CV_64F, 0, 1, kernel_size)
            # Magnitude across both axis.
            sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        mag_binary = self.thresholdImage(sobel, thres)
        return mag_binary

    def binarize_v3(self, image, s_thresh=(50, 100), sx_thresh=(10, 200), visualize=False):
        """Take gradient of the color channels and combine."""
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # sobel in hls
        sobelx = cv2.Sobel(image_hls[:, :, 1], cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # sobel in gray
        sobelGray = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobelGray = np.absolute(sobelGray)
        scaled_sobelGray = np.uint8(255 * abs_sobelGray / np.max(abs_sobelGray))

        # Threshold
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel in hls.
        sbinary = np.zeros_like(image_hls[:, :, 2])
        sbinary[(image_hls[:, :, 2] >= s_thresh[0]) & (image_hls[:, :, 2] <= s_thresh[1])] = 1

        # Threshold color channel in hsv.
        svbinary = np.zeros_like(image_hsv[:, :, 2])
        svbinary[(image_hsv[:, :, 2] >= s_thresh[0]) & (image_hsv[:, :, 2] <= s_thresh[1])] = 1

        sobelxv = cv2.Sobel(image_hsv[:, :, 2], cv2.CV_64F, 1, 0)
        abs_sobelxv = np.absolute(sobelxv)
        scaled_sobelxv = np.uint8(255 * abs_sobelxv / np.max(abs_sobelxv))

        sxvbinary = np.zeros_like(scaled_sobelxv)
        sxvbinary[(scaled_sobelxv >= s_thresh[0]) & (scaled_sobelxv <= s_thresh[1])] = 1

        labels = ['hls-sobel', 'hls-masking', 'hsv-masking']
        self.plot_multiple(sxbinary, sbinary, svbinary, labels)

        # Merge binary.
        mergeBinary = np.zeros_like(sxbinary)
        mergeBinary[(sxbinary == 1) | (svbinary == 1) | (sbinary == 1)] = 1
        if visualize:
            plt.imshow(mergeBinary, cmap='gray')
            plt.show()
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, sbinary))
        return color_binary

    def plot_multiple(self, image1, image2, image3, labels):
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

    def binarize_image_v2(self, image, apply_gray=False):
        """Create a merged binary using magnitude, gradient and direction."""
        thresh_x = (80, 255)
        thresh_y = (70, 255)
        thresh_md = (40, 255)
        thresh_d = (0.1, 1.03)
        binary_x = self.sobel_thresh(image, thresh_x, axis='x', kernel_size=15, apply_gray=True)
        binary_y = self.sobel_thresh(image, thresh_y, axis='y', kernel_size=15, apply_gray=True)
        binary_md = self.sobel_thresh(image, thresh_md, axis='xy', kernel_size=15, apply_gray=True)
        binary_dir = self.dir_thresh(image, 15, thresh_d, apply_gray=True)
        # Merge the binary image over x,y and magnitude, direction.
        binary_img = np.zeros_like(binary_dir)
        print("binary_x", binary_x.shape)
        print("binary_y", binary_y.shape)
        print("binary_md", binary_md.shape)
        print("binary_dir", binary_dir.shape)
        binary_img[((binary_x == 1) & (binary_y == 1)) | ((binary_md == 1) & (binary_dir == 1))] = 1

        # Gradient on S channel.
        hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel_binary = self.thresholdImage(hls_image[:, :, 2], (120, 255))

        # combine both binary images.
        output = np.zeros_like(binary_img)
        output[(binary_img == 1) | (s_channel_binary == 1)] = 1
        return output

    def binarize_image(self, img=None):
        """
        Create thresholded binary image.
        :param image: Input Image.
        :return: Binarized image.
        """
        if img == None:
            raise FileNotFoundError

        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        s_channel = hls[:, :, 2]
        v_channel = hsv[:, :, 2]
        s2_channel = hsv[:, :, 1]
        r_channel = img[:, :, 0]
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
        v_binary = np.ones_like(s_channel)
        r_binary = np.ones_like(r_channel)
        s_binary[(s_channel >= self.channel_thresh_min) & (s_channel <= self.channel_thresh_max)] = 1
        v_binary[(v_channel >= 0) & (v_channel <= 220)] = 0

        r_binary[(r_channel >= 0) & (r_channel <= 220)] = 0
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (v_binary == 1) | (r_binary == 1)] = 1
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
        left_bot_inner = (x1_1 + 200, height)
        right_bot_inner = (x2_2 - 200, height)
        left_top_inner = (x1_2 + 20, y + 50)
        right_top_inner = (x2_1 - 20, y + 50)
        vertices = np.array([[left_bot_outer, left_top_outer, right_top_outer,
                              right_bot_outer, right_bot_inner,
                              right_top_inner, left_top_inner, left_bot_inner]], dtype=np.int32)
        return vertices

    def mask_region(self, image):
        """ Mask a specific region of image. """
        temp_image = np.copy(image)
        binary_mask = np.zeros_like(image)
        vertices = self.segment_area_of_interest(image)
        cv2.polylines(temp_image, vertices, True, (0, 255, 255))
        color = (255,)
        cv2.fillPoly(binary_mask, vertices, color)
        masked_image = cv2.bitwise_and(image, binary_mask)
        return masked_image

    def mark_area_of_interest(self, img):
        # binary_image = create_merged_binary(img, apply_gray=True)
        binary_image = self.binarize_image(img)
        plt.imshow(binary_image, cmap='gray')
        plt.show()
        # test = self.binarize_image_v2(img)
        # plt.imshow(test, cmap='gray')
        # plt.show()
        # test = self.binarize_v3(img)
        # plt.imshow(test, cmap='gray')
        # plt.show()
        output = self.mask_region(binary_image)
        plt.imshow(output, cmap='gray')
        plt.show()
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
            plt.imshow( plt_img)
            plt.show()

        if self.M is None or self.MInv is None:
            self.M = cv2.getPerspectiveTransform(arg_src, dst)
            self.MInv = cv2.getPerspectiveTransform(dst, arg_src)
        warped = cv2.warpPerspective(output, self.M, (width, height))
        return warped