import sys
import cv2
from src.calibration import calibration
from moviepy.editor import VideoFileClip
from src.lanes import lanes
from src.visualize import visualize
from src.image_handler import ImageHandler
import matplotlib.image as mp
import matplotlib.pyplot as plt
import numpy as np
import glob

# Default input/output
output = "./output/output.mp4"
input = "./videos/project_video.mp4"
input_2 = "./videos/clipped.mp4"
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
    img_copy = np.copy(img)
    print(img.shape)
    if condition['visualize']:
        vis.draw_img(img)
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

def main_test():
    global vis
    global imgh
    global calib
    global lane
    calib = calibration(9, 6, './camera_cal')
    # img = mp.imread('./examples/test6.jpg')
    # undist = calib.undistort_image(img)
    # calib.display_2d_grid(img, undist)
    # Process video and read frame.
    vis = visualize(image_shape)
    imgh = ImageHandler()
    lane = lanes(image_shape)

    file = mp.imread('./examples/solidYellowCurve.jpg')
    file = cv2.resize(file, (1280, 720))
    plt.imshow(file)
    plt.show()
    opt = pipeline(file)
    vis.draw_img(opt)

    files = glob.glob('./special_cases/*.png')
    print(files)
    for file in files:
        file = mp.imread(file)
        print(file.shape)
        plt.imshow(file)
        plt.show()
        file = cv2.resize(file, (1280, 720))
        plt.imshow(file)
        plt.show()
        opt = pipeline(file)
        vis.draw_img(opt)

def main():
    """Main pipeline."""
    global vis
    global imgh
    global calib
    global lane
    # Check and do camera calibration if necessary.
    calib = calibration(9, 6, './camera_cal')
    vis = visualize(image_shape)
    imgh = ImageHandler()
    lane = lanes(image_shape)
    clip = VideoFileClip(input)
    output_video = clip.fl_image(pipeline)
    output_video.write_videofile(output, audio=False)

if __name__ == '__main__':
    main()
    # main_test()