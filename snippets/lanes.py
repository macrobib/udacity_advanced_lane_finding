import numpy as np
import matplotlib.pyplot as plt

class lanes:
    """Lane point abstraction."""

    def __init__(self, img_size, mask, area_selection):

        self.detected = False

        self.recent_x_fitted_left = []
        self.recent_x_fitted_right = []

        self.bestx_left = None
        self.bestx_right = None

        self.best_fit_left = None
        self.best_fit_right = None

        self.current_fit_left = [np.array([False])]
        self.current_fit_right = [np.array([False])]

        self.radius_of_curvature_left = None
        self.radius_of_curvature_right = None

        self.line_base_pos_left = None
        self.line_base_pos_right = None

        self.dropped_frames = None
        self.image_size = img_size
        self.gray_mask = mask
        self.color_mask = np.dstack((self.gray_mask, self.gray_mask, self.gray_mask))
        self.area_selection = area_selection


    def get_new_fit(self):
        """ Retrieve fit points. """
        pass

    def set_new_lane_points(self):
        """New set of lane points."""
        pass


    def get_rad_curvature(self):
        """Calculate radius of curvature."""
        pass


    def get_center_deviation(self):
        """ Calculate deviation of the vehicle from center base point. """


    def _sanity_check_values(self):
        """ Do sanity check on the detected values."""
        pass


    def _histogram_search(self):
        """Do a full histogram search on image."""
        pass


    def _locality_search(self):
        """ Do locality search on image for lane line."""
        pass


    def _weighted_smoothening(self):
        """Do weighted smoothening of old and new values."""
        pass