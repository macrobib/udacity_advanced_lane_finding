"""Abstract Lane points."""
import cv2
import numpy as np
from src.visualize import visualize as viz
from scipy.signal import medfilt
from collections import deque
import matplotlib.pyplot as plt


class lanes:

    def __init__(self, image_dim = None):
        self.detected = False
        self.dropped_frames = 11 # Initialize with max dropped frames.
        self.image_dim = image_dim
        self.y_centre = self.image_dim[0] - self.image_dim[0] / 2

        self.recent_x_left = deque(maxlen=6)
        self.recent_x_right = deque(maxlen=6)

        self.current_fit_left = None
        self.current_fit_right = None

        self.current_left_lane = None
        self.current_right_lane = None

        self.curvature_right = None
        self.curvature_left = None
        self.curvature_right_prev = None
        self.curvature_left_prev = None

        self.xpoints_left = None
        self.xpoints_right = None

        self.vehicle_offset = 0
        w1 = np.array([0.7])
        w2 = np.ones([5]) * 0.3 / 5
        self.weights = np.append(w1, w2)

        if image_dim:
            self.x_m_per_pix = 3.7 / image_dim[1]
            self.y_m_per_pix = 30 / image_dim[0]
        else:
            self.x_m_per_pix = 3.7 / 700
            self.y_m_per_pix = 30 / 720
        self.y_points = np.linspace(0, image_dim[0], 128) # Points in y axis for curve fitting.

    def _update_lane_points(self, left, right, left_fit, right_fit):
        """Update the new lane points and update average."""
        self.current_fit_left = left_fit
        self.current_fit_right = right_fit

        if len(self.recent_x_left) == 6:
            self.recent_x_left.appendleft(left)
            self.current_left_lane = np.average(np.array(self.recent_x_left), 0, weights=self.weights)
        else:
            self.recent_x_left.appendleft(left)
            self.current_left_lane = left
        if len(self.recent_x_right) == 6:
            self.recent_x_right.appendleft(right)
            self.current_right_lane = np.average(np.array(self.recent_x_right), 0, weights=self.weights)
        else:
            self.recent_x_right.appendleft(right)
            self.current_right_lane = right

    def _update_curvatures(self, l_curv, r_curv):
        if self.curvature_left_prev:
            self.curvature_left_prev = self.curvature_left
        else:
            self.curvature_left_prev = l_curv
        if self.curvature_right_prev:
            self.curvature_right_prev = self.curvature_right
        else:
            self.curvature_right_prev = r_curv
        self.curvature_left = l_curv
        self.curvature_right = r_curv

    def get_curvatures(self):
        """Get left and right curvatures."""
        return (self.curvature_left, self.curvature_right)

    def get_current_lane_points(self):
        """Current lane positions."""
        return (self.current_left_lane, self.current_right_lane)

    def get_vehicle_offset(self):
        return self.vehicle_offset


    def calculate_top_bottom_curv(self, left_x, left_y, right_x, right_y):
        """Compute the slope of top and bottom halves of the lines detected.
        """
        right_len = len(left_y)
        left_len = len(left_x)
        left_top_x = left_x[left_len//2: ]
        right_top_x = right_x[right_len//2: ]
        left_bottom_x = left_x[: left_len//2]
        right_bottom_x = right_x[: right_len//2]
        curvature_sanity = None
        left_fit = None
        right_fit = None

        left_top_y = left_y[left_len // 2:]
        right_top_y = right_y[right_len // 2:]
        left_bottom_y = left_y[: left_len // 2]
        right_bottom_y = right_y[: right_len // 2]
        top_curvatures = self.calculate_curvature((left_top_x, left_top_y), (right_top_x, right_top_y))
        bottom_curvatures = self.calculate_curvature((left_bottom_x, left_bottom_y), (right_bottom_x, right_bottom_y))
        full_curvature = self.calculate_curvature((left_x, left_y), (right_x, right_y))
        # print("Top Curvature:{0} -- Bottom Curvature:{1}".format(top_curvatures, bottom_curvatures))

        if self.sanity_check_curvature(full_curvature[0], full_curvature[1]):
            left_fit = np.polyfit(left_y, left_x, 2)
            right_fit = np.polyfit(right_y, right_x, 2)
            curvature_sanity = True
        elif self.sanity_check_curvature(bottom_curvatures[0], bottom_curvatures[1]):
            left_fit = np.polyfit(left_bottom_y, left_bottom_x, 2)
            right_fit = np.polyfit(right_bottom_y, right_bottom_x, 2)
            curvature_sanity = True
        elif self.sanity_check_curvature(top_curvatures[0], top_curvatures[1]):
            left_fit = np.polyfit(left_top_y, left_top_x, 2)
            right_fit = np.polyfit(right_top_y, right_top_x, 2)
            curvature_sanity = True
        else:
            # TODO: patch fix.
            print("sanity failed..")
            left_fit = np.polyfit(left_y, left_x, 2)
            right_fit = np.polyfit(right_y, right_x, 2)

        new_left_x_points = left_fit[0] * (self.y_points ** 2) + left_fit[1] * (self.y_points) + left_fit[2]
        new_right_x_points = right_fit[0] * (self.y_points ** 2) + right_fit[1] * (self.y_points) + right_fit[2]
        if curvature_sanity:
            self.xpoints_left = new_left_x_points
            self.xpoints_right = new_right_x_points
        if new_left_x_points is None or new_right_x_points is None:
            self.detected = False
            self.dropped_frames += 1
        else:
            self.detected = True
            if curvature_sanity:
                self.calculate_center_offset((new_left_x_points, new_left_x_points))
                self._update_lane_points(new_left_x_points, new_right_x_points, left_fit, right_fit)




    def histogram_detect(self, img, visualize=False):
        """ Find peaks in histogram in the binary image."""
        width = 80 # Width of the window.
        minpix = 50 # Minimum number of pixels found in recenter window.
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        # https://gist.github.com/jul/3794506
        histogram = medfilt(histogram, [41])  # Smoothen the histogram values.
        out_img = np.dstack((img, img, img)) * 255

        if visualize:
            plt.plot(histogram)
            plt.show()
            viz.draw_img(out_img, True)

        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        nwindows = 9
        window_height = np.int(img.shape[0] / nwindows)

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        # Empty lists to recieve left and right indices.
        left_lane_indices = []
        right_lane_indices = []

        # loop through the window
        for window in range(nwindows):
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - width
            win_xleft_high = leftx_current + width
            win_xright_low = rightx_current - width
            win_xright_high = rightx_current + width

            # Draw the windows on the visualization image.
            # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

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
        # fit points.
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        new_left_x_points = left_fit[0] * (self.y_points ** 2) + left_fit[1] * (self.y_points) + left_fit[2]
        new_right_x_points = right_fit[0] * (self.y_points ** 2) + right_fit[1] * (self.y_points) + right_fit[2]
        self.xpoints_left = new_left_x_points
        self.xpoints_right = new_right_x_points
        if new_left_x_points is None or new_right_x_points is None:
            self.detected = False
            self.dropped_frames += 1
        else:
            self.detected = True
            params = self.calculate_curvature((leftx, lefty), (rightx, righty))
            if self.sanity_check_curvature(params[0], params[1]):
                self.calculate_center_offset((new_left_x_points, new_left_x_points))
                self._update_lane_points(new_left_x_points, new_right_x_points, left_fit, right_fit)


    def locality_search(self, img):
        """
        Do sliding window search based on previous frames.
        :return:
        """
        nonzerox = np.array(img.nonzero()[1])
        nonzeroy = np.array(img.nonzero()[0])
        fp_left = self.current_fit_left
        fp_right = self.current_fit_right

        left_x_points = fp_left[0] * (nonzeroy ** 2) + fp_left[1] * (nonzeroy) + fp_left[2]
        right_x_points = fp_right[0] * (nonzeroy ** 2) + fp_right[1] * (nonzeroy) + fp_right[2]
        # right_x_points = right_x_points[::-1]
        margin = 80

        # determine the range to lookup for in the new frame based on the old coordinates.
        left_lane_inds = ((nonzerox > (left_x_points - margin)) & (nonzerox < (left_x_points + margin)))
        right_lane_inds = ((nonzerox > (right_x_points - margin)) & (nonzerox < (right_x_points + margin)))

        # Extract points.
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if leftx is None or rightx is None or lefty is None or righty is None:
            self.detected = False
            self.dropped_frames += 1
            self.xpoints_left = None
            self.xpoints_right = None
        else:
            # Generate new sets of points.
            # fit points.
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            left_fit_x = left_fit[0] * (self.y_points ** 2) + left_fit[1] * (self.y_points) + left_fit[2]
            right_fit_x = right_fit[0] * (self.y_points ** 2) + right_fit[1] * (self.y_points) + right_fit[2]
            self.detected = True
            params = self.calculate_curvature((leftx, lefty), (rightx, righty))
            if self.sanity_check_curvature(params[0], params[1]):
                self.calculate_center_offset((left_fit_x, right_fit_x))
                self._update_lane_points(left_fit_x, right_fit_x, left_fit, right_fit)
        #     draw_lane_search_area(img, (left_lane_inds, right_lane_inds), (left_fitx, right_fitx), 100)

    def calculate_center_offset(self, fit_points):
        """ Determine center deviation of vehicle."""
        vehicle_center = (fit_points[1][-1] - fit_points[0][-1]) // 2  # (Max Right fit - Max Left fit)/2
        reference_center = (self.image_dim[1] // 2)
        pixel_deviation = reference_center - vehicle_center  # Calculates the deviation of vehicle in terms of pixels.
        self.vehicle_offset = pixel_deviation * self.x_m_per_pix

    def calculate_curvature(self, left_points, right_points):
        """Determine the curvature from lane points"""
        # Calculate Curvature corresponding to centre of image.
        l_curve = 0
        r_curve = 0
        if len(left_points[0]) and len(left_points[1]) and len(right_points[0]) and len(right_points[1]):
            left_fit_cr = np.polyfit(left_points[1] * self.y_m_per_pix, left_points[0] * self.x_m_per_pix, 2)
            right_fit_cr = np.polyfit(right_points[1] * self.y_m_per_pix, right_points[0] * self.x_m_per_pix, 2)

            l_curve = ((1 + (left_fit_cr[0] * self.y_centre + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * left_fit_cr[0])
            r_curve = ((1 + (right_fit_cr[0] * self.y_centre + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * left_fit_cr[0])
        output = (l_curve, r_curve)

        return output

    def sanity_check_curvature(self, l_curv, r_curv):
        """ Check if the curvature is in limits."""
        check = True
        curv_diff_l = None
        curv_diff_r = None

        # Check sanity of curvature.
        if l_curv < 600 or r_curv < 600:
            check = False

        if l_curv > 2500 or r_curv > 2500:
            check = False

        if check:
            self._update_curvatures(l_curv, r_curv)

        if self.curvature_left_prev != None and self.curvature_left != None:
            curv_diff_l = (self.curvature_left - self.curvature_left_prev)/self.curvature_left_prev
        if self.curvature_right_prev != None and self.curvature_right != None:
            curv_diff_r = (self.curvature_right - self.curvature_right_prev)/self.curvature_right_prev

        if curv_diff_l != None and curv_diff_r != None:
            if abs(curv_diff_l) > 0.05 or abs(curv_diff_r) > 0.05:
                check = False
        else:
            check = True # First instance check.
        return check

    def pipeline(self, img):
        if self.dropped_frames >= 11:
            self.histogram_detect(img, False)
        else:
            self.locality_search(img)
            if self.dropped_frames >= 11:
                self.histogram_detect(img, False)
        if self.detected:
            self.dropped_frames = 0
