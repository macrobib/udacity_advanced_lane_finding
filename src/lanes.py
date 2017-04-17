"""Abstract Lane points."""

class lanes:

    def __init__(self):
        self.detected = False
        self.recent_x_fit_left = []
        self.average_x_left = None
        self.average_fit_left = None
        self.current_fit_left = None
        self.curvature_left = None
        self.recent_x_fit_right = []
        self.average_x_right = None
        self.average_fit_right = None
        self.current_fit_right = None
        self.curvature_right = None
        self.vehicle_offset = None

    def update_lane_points(self, left, right):
        """Update the new lane points and update average."""
        pass

    def get_averaged_lane_points(self):
        """Get average lane points over previous iterations."""
        pass

    def get_averaged_fit_points(self):
        """Get averaged fit points over previous iterations."""
        pass

    def get_curvatures(self):
        """Get left and right curvatures."""
        pass

    def get_vehicle_offset(self):
        pass

    def __average_lane_points(self):
        pass

    def __average_fit_points(self):
        pass

