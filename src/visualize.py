"""Lane visualization and tex drawing helper"""
import numpy as np
import matplotlib.pyplot as plt


class visualize:

    def __init__(self, image_shape, lec, lfc, grad_enable, txc, mc, gmask):
        """Color parameters initialize."""
        self.image_shape = image_shape # (height, width)
        self.lane_edge_color = lec # Lane edge color
        self.lane_fill_color = lfc # lane fill
        self.color_gradient_enable = grad_enable # Enable gradient in left and right side of lane fill.
        self.text_color = txc # Text color
        self.mark_center = mc # Draw center line on fill.
        self.gray_mask = gmask # zero image with gray scale coordinates.
        self.color_mask = np.dstack((self.gray_mask, self.gray_mask ,self.gray_mask))


    def draw_lane_area(self):
        pass


    def draw_frame_text(self):
        pass


    def draw_img(self, img):
        """Render the given image."""
        pass
