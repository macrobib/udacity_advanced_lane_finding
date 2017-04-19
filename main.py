import sys
import cv2
from src.calibration import calibration
from moviepy.editor import VideoFileClip
from src.lanes import lanes
from src.visualize import visualize
from src.image_handler import ImageHandler

# Default input/output
output = "./output/output.mp4"
input = "./videos/project_video.mp4"
image_shape = (720, 1280, 3)

# Helper objects.
vis = None # Visualization helper.
imgh = None # Image Handler.
calib = None # Calibration helper.
lane = None # Main Lane object.
display_image = False
condition = {
    'visualize': False,
    'smoothen' : False,
    'sharpen'  : True
}

def pipeline(img):
    global vis
    global imgh
    global calib
    global condition
    global lane
    img_copy = img
    # vis.draw_img(img)
    if condition['visualize']:
        image = vis.gradient_and_combine(img)
        vis.visualize_colorspace(img)
        vis.draw_img(image)
    if condition['smoothen']:
        img_copy = imgh.blur_image(img_copy)
    dst = calib.undistort_image(img_copy)
    if condition['sharpen']:
        dst = imgh.sharpen_image(dst)
    otp = imgh.mark_area_of_interest(dst)
    warped = imgh.create_perspective_transform(otp)
    lane.pipeline(warped)
    if condition['visualize']:
        vis.draw_img(warped, True)

    curvature = lane.get_curvatures()
    lane_points = lane.get_current_lane_points()
    vehicle_offset = lane.get_vehicle_offset()
    MInv = imgh.get_minv()

    output_image = vis.draw_lane_and_text(img, warped, curvature, vehicle_offset, lane_points, MInv)  # Draw the lane area and text info.
    if display_image:
        vis.draw_img(output_image)
    return output_image


def main():
    """Main pipeline."""
    global vis
    global imgh
    global calib
    global lane
    # Check and do camera calibration if necessary.
    calib = calibration(9, 6, './camera_cal')
    # Process video and read frame.
    vis = visualize(image_shape)
    imgh = ImageHandler()
    lane = lanes(image_shape)
    clip = VideoFileClip(input)
    output_video = clip.fl_image(pipeline)
    output_video.write_videofile(output, audio=False)

if __name__ == '__main__':
    main()