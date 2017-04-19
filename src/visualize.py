"""Lane visualization and tex drawing helper"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


class visualize:

    def __init__(self, image_shape, gmask = None, enable = False):
        """Color parameters initialize."""
        self.image_shape = image_shape # (height, width)
        self.gray_mask =  gmask
 # zero image with gray scale coordinates.
        self.color_mask = np.dstack((self.gray_mask, self.gray_mask ,self.gray_mask))
        self.enable_debug = enable

        # Font
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL # Font family
        self.font_th = 0.8  # Font scale
        self.line_tp = cv2.LINE_AA # Line type.
        self.font_color = (51, 255, 153) # Font color

        # color
        self.left_lane_color = (102, 255, 178) # Left lane section color.
        self.right_lane_color = (0, 204, 102) # Right lane section color.
        self.edge_line_color = (255, 51, 51) # edge bounding lines color.
        self.center_line_color = (204, 204, 0) # center line color.

    def draw_polygon(self, coordinates, image=None):
        """ Draw a polygon on the image."""
        cv2.polylines(image, coordinates, True, (153, 255, 51), thickness=1)
        cv2.imshow('default', image)
        cv2.waitKey(0)

    @staticmethod
    def draw_img(img, grayscale = False):
        """Render the given image."""
        if grayscale:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.show()

    @staticmethod
    def draw_tandem(img1, img2, text1, text2, gray1=False, gray2=False):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title(text1)
        if gray1:
            ax1.imshow(img1, cmap='gray')
        else:
            ax1.imshow(img1)
        ax2.set_title(text2)
        if gray2:
            ax2.imshow(img2, cmap='gray')
        else:
            ax2.imshow(img2)
        plt.show()

    def draw_lane_line(self, img, shape, coordinates, fit_points):
        """ Draw the lanes."""
        global y_points
        left_fit_params = fit_points[0]
        right_fit_params = fit_points[1]
        lfx = left_fit_params[0] * y_points ** 2 + left_fit_params[1] * y_points + left_fit_params[2]
        rfx = right_fit_params[0] * y_points ** 2 + right_fit_params[1] * y_points + right_fit_params[2]

        img[coordinates[0], coordinates[1]] = [255, 0, 0]
        img[coordinates[2], coordinates[3]] = [0, 0, 255]
        plt.imshow(img)
        plt.plot(lfx, y_points, color='yellow')
        plt.plot(rfx, y_points, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    def draw_lane_search_area(self, img, lane_point_indices, fit_points, margin):
        """Visualize the range of search for the coordinates in the new image."""
        global y_points
        gen_points = np.linspace(0, img.shape[0] - 1, img.shape[0])
        nonzeroy = np.array(img.nonzero()[0])
        nonzerox = np.array(img.nonzero()[1])

        output = np.dstack((img, img, img)) * 255
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

    def lane_visualize(self, img, warped, fit_left, fit_right, minv, visualize=False):
        """Visualize the detected lane lines."""
        global y_points
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        print("warped image shape: --", warped.shape)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        print("lane visualization: Left {0} --- Right{1}".format(fit_left.shape, y_points.shape))
        pts_left = np.array([np.transpose(np.vstack([fit_left, y_points]))])
        pts_middle_left = np.array([np.flipud(np.transpose(np.vstack([(fit_left + fit_right) / 2, y_points])))])
        pts_middle_right = np.array([np.transpose(np.vstack([(fit_left + fit_right) / 2, y_points]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_right, y_points])))])
        pts_1_halve = np.hstack((pts_left, pts_middle_left))
        pts_2_halve = np.hstack((pts_middle_right, pts_right))
        poly_arg_left = np.array(pts_1_halve)
        poly_arg_right = np.array(pts_2_halve)
        print("Fill poly shape--", poly_arg_left.shape)
        cv2.fillPoly(color_warp, np.int_(poly_arg_left), self.left_lane_color)
        cv2.fillPoly(color_warp, np.int_(poly_arg_right), self.right_lane_color)
        pts_left = pts_left.astype(np.int32)
        pts_middle = pts_middle_right.astype(np.int32)
        pts_right = pts_right.astype(np.int32)
        print("fit_left shape: ", pts_left, pts_left.shape)
        cv2.polylines(color_warp, pts_left, False, self.edge_line_color, 3, self.line_tp)
        cv2.polylines(color_warp, pts_middle, False, self.center_line_color, 3, self.line_tp)
        cv2.polylines(color_warp, pts_right, False, self.edge_line_color, 3, self.line_tp)
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

    def draw_lane_and_text(self, image, warped, curvatures, lane_distance, lane_points, minv):
        """Draw the text and images"""
        width = self.image_shape[1]
        height = self.image_shape[0]
        delta = 50 # arbitrary value to adjust text position.

        update_img = self.lane_visualize(image, warped, lane_points[0], lane_points[1], minv, True)
        left_curvature_str = "Left Curvature: " + str(curvatures[0])
        right_curvature_str = "Right Curvature: " + str(curvatures[1])
        lane_string = "Delta:" + str(lane_distance)
        cv2.putText(update_img, left_curvature_str, (0 + delta, 0 + delta), self.font, self.font_th, self.font_color, 1, self.line_tp)
        cv2.putText(update_img, right_curvature_str, (width - 400 - len(right_curvature_str), 0 + delta), self.font, self.font_th,
                    self.font_color, 1, self.line_tp)
        cv2.putText(update_img, lane_distance, (width // 2 - len(lane_string), height // 2), self.font, self.font_th,
                    self.font_color, 1, self.line_tp)
        return update_img
