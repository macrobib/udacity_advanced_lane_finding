"""Abstract Lane points."""
import cv2
import numpy as np
from src.visualize import visualize as viz
from scipy.signal import medfilt
from collections import deque


class lanes:

    def __init__(self, image_dim = None):
        self.detected = False
        self.dropped_frames = 6
        self.image_dim = image_dim
        self.y_centre = self.image_dim[0] - self.image_dim[0] / 2

        self.recent_x_left = deque(maxlen=11)
        self.recent_x_right = deque(maxlen=11)

        self.current_fit_left = None
        self.current_fit_right = None

        self.curvature_right = None
        self.curvature_left = None

        self.xpoints_left = None
        self.xpoints_right = None

        self.vehicle_offset = 0
        w1 = np.array([0.8])
        w2 = np.ones([10]) * 0.2 / 10
        self.weights = np.append(w1, w2)

        if image_dim:
            self.x_m_per_pix = 3.7 / image_dim[1]
            self.y_m_per_pix = 30 / image_dim[0]
        else:
            self.x_m_per_pix = 3.7 / 700
            self.y_m_per_pix = 30 / 720
        self.y_points = np.linspace(0, image_dim[0], 128) # Points in y axis for curve fitting.

    def _update_lane_points(self, left, right):
        """Update the new lane points and update average."""
        if len(self.recent_x_left) == 11:
            self.recent_x_left.append(left)
            self.current_fit_left = np.average(np.array(self.recent_x_left), 0, weights=self.weights)
        if len(self.recent_x_right) == 11:
            self.recent_x_right.append(right)
            self.current_fit_right = np.average(np.array(self.recent_x_right), 0, weights=self.weights)

    def _update_curvatures(self, l_curv, r_curv):
        self.curvature_left = l_curv
        self.curvature_right = r_curv

    def get_curvatures(self):
        """Get left and right curvatures."""
        return (self.curvature_left, self.curvature_right)

    def get_vehicle_offset(self):
        return self.vehicle_offset

    def histogram_detect(self, img, visualize=False):
        """ Find peaks in histogram in the binary image."""
        width = 100 # Width of the window.
        minpix = 50 # Minimum number of pixels found in recenter window.

        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        # https://gist.github.com/jul/3794506
        histogram = medfilt(histogram, [41])  # Smoothen the histogram values.
        out_img = np.dstack((img, img, img)) * 255
        if visualize:
            viz.draw_img(histogram)
            viz.draw_img(out_img)

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

        # fit points.
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        xpoints_left = left_fit[0] * (self.y_points ** 2) + left_fit[1] * (self.y_points) + left_fit[2]
        xpoints_right = right_fit[0] * (self.y_points ** 2) + right_fit[1] * (self.y_points) + right_fit[2]

        if xpoints_left is None or xpoints_right is None:
            self.detected = False
            self.dropped_frames += 1
            self.xpoints_left = None
            self.xpoints_right = None
        else:
            self.detected = True
            self.xpoints_left = xpoints_left
            self.xpoints_right = xpoints_right
            params = self.calculate_curvature(img, (leftx, lefty), (rightx, righty))
            if self.sanity_check_curvature(params[0], params[1]):
                self.calculate_center_offset((left_fit, right_fit))
                self._update_lane_points(left_fit, right_fit)

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
        print("Locality search", left_fit, right_fit)

        # Generate new sets of points.
        left_fit_x = left_fit[0] * (self.y_points ** 2) + left_fit[1] * (self.y_points) + left_fit[2]
        right_fit_x = right_fit[0] * (self.y_points ** 2) + right_fit[1] * (self.y_points) + right_fit[2]
        if left_fit_x is None or right_fit_x is None:
            self.detected = False
            self.dropped_frames += 1
            self.xpoints_left = None
            self.xpoints_right = None
        else:
            self.detected = True
            params = self.calculate_curvature((leftx, lefty), (rightx, righty))
            if self.sanity_check_curvature(params[0], params[1]):
                self.calculate_center_offset((left_fit_x, right_fit_x))
                self._update_lane_points(left_fit_x, right_fit_x)
        #     draw_lane_search_area(img, (left_lane_inds, right_lane_inds), (left_fitx, right_fitx), 100)

    def calculate_center_offset(self, fit_points):
        """ Determine center deviation of vehicle."""
        vehicle_center = (fit_points[1][-1] - fit_points[0][-1]) // 2  # (Max Right fit - Max Left fit)/2
        reference_center = (self.image_dim[1] // 2)
        pixel_deviation = reference_center - vehicle_center  # Calculates the deviation of vehicle in terms of pixels.

    def calculate_curvature(self, left_points, right_points):
        """Determine the curvature from lane points"""
        # Calculate Curvature corresponding to centre of image.

        left_fit_cr = np.polyfit(left_points[1] * self.y_m_per_pix, left_points[0] * self.x_m_per_pix, 2)
        right_fit_cr = np.polyfit(right_points[1] * self.y_m_per_pix, right_points[0] * self.x_m_per_pix, 2)

        l_curve = ((1 + (left_fit_cr[0] * self.y_centre + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        r_curve = ((1 + (right_fit_cr[0] * self.y_centre + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        output = (l_curve, r_curve)
        return output

    def sanity_check_curvature(self, l_curv, r_curv):
        """ Check if the curvature is in limits."""
        check = True
        # Check similarity of curvature.
        if not -0.5 < l_curv/r_curv < 0.5:
            check = False
        # Check sanity of curvature.
        if l_curv < 400 or r_curv < 400:
            check = False

        if l_curv > 1500 or r_curv > 1500:
            check = False
        if check:
            self._update_curvatures(l_curv, r_curv)
        return check

    def pipeline(self, img):
        if self.dropped_frames >= 6:
            self.histogram_detect(img)
        else:
            self.locality_search(img)
            if self.dropped_frames >= 6:
                self.histogram_detect(img)
        if self.detected:
            self.dropped_frames = 0