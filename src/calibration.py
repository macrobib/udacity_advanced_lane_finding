import cv2
import time
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os.path as path

def test_calib():
    nx = 9
    ny = 5

    fname = '../camera_cal/calibration1.jpg'
    img = cv2.imread(fname)
    plt.imshow(img)
    plt.show()
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners.
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    print(ret)

    if ret == True:
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)
        plt.show()


class calibration:
    """ Do the caliberation on the chess board images."""

    def __init__(self, nx=9, ny=6, directory=None):
        self.nx = nx
        self.ny = ny
        self.calib_files = glob.glob(directory + '/' + '*.jpg')
        print(self.calib_files)
        self.objpoints = []
        self.imgpoints = []
        self.shape = None
        self.visualize = False
        self.mtx = None
        self.dist = None
        self.calib_status = False
        self.pickle_file = './data/cam.p'
        self.caliberation_pipeline()


    def calibrate(self):
        """Objpoint based calibration."""
        print("start calibration..")
        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

        for file in self.calib_files:
            img = cv2.imread(file)
            if self.visualize == True:
                cv2.imshow('image', img)
                cv2.waitKey(0)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.shape = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            if ret == True:
                # print(corners[0][0])
                time.sleep(4)
                self.imgpoints.append(corners)
                self.objpoints.append(objp)
                if self.visualize == True:
                    cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
                    plt.imshow(img)
                    plt.show()
        self.calib_status, self.mtx, self.dist, rvex, tvex = cv2.calibrateCamera(self.objpoints,
                                                             self.imgpoints, self.shape, None, None)

    def undistort_test(self, output_dir = '../calibrated_images'):
        "Undistort the calibration images."
        print("caliberation status", self.calib_status)
        if self.calib_status:
            for file in self.calib_files:
                img = cv2.imread(file)
                dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
                warped = self.perspectiveTransform(dst)
                if warped:
                    self.display_2d_grid(img, warped)
                #cv2.imshow('image', dst)
                #cv2.waitKey(0)
                #plt.imshow(dst)
                #plt.show()
                #write_path = output_dir + '/' + 'undistort_' + str(file).split('\\')[-1]
                #cv2.imwrite(write_path, dst)
                #self.display_2d_grid(img, dst)

    def undistort_image(self, img):
        """
        Undistort the image, as per calibration parameters.
        :param img: Input image.
        :return: Undistorted image.
        """
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return dst

    def perspectiveTransform(self, undistorted):
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        nx = self.nx
        ny = self.ny
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
        if ret == False:
            ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
        if ret == False:
            ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)
        if ret == False:
            ret, corners = cv2.findChessboardCorners(gray, (9, 5), None)
        print("chess board corners: ", ret)
        warped = None
        if ret:
            cv2.drawChessboardCorners(undistorted, (self.nx, self.ny), corners, ret)
            offset = 100
            img_size = (gray.shape[1], gray.shape[0])
            src = np.float32([corners[0][0], corners[self.nx-1][0], corners[-1][0], corners[-self.nx][0]])
            print("src", src, src.shape)

            dst = np.float32([[offset, offset], [img_size[0] - offset,
                              offset], [img_size[0] - offset, img_size[1] - offset], [offset, img_size[1] - offset]])

            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(undistorted, M, img_size)
        return warped


    def display_2d_grid(self, img, undistorted):
        """Display calibrated images in a 2D grid."""
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        f.tight_layout()
        ax1.imshow(img, aspect='auto')
        ax1.set_title('Original Image', fontsize=15)
        ax2.imshow(undistorted, aspect='auto')
        ax2.set_title('Undistorted Image', fontsize=15)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
    
    def pickle_data(self):
        """Save camera matrix and distribution"""
        print("mtx: ",self.mtx)
        print("dist: ", self.dist)
        store = {
                'mtx':self.mtx,
                'dist':self.dist
                }
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(store, f, pickle.HIGHEST_PROTOCOL)

    def unpickle_data(self):
        """Load the pickled calibration data."""
        data = pickle.load(open(self.pickle_file, 'rb'))
        mtx = data['mtx']
        dist = data['dist']
        return (mtx, dist)


    def caliberation_pipeline(self):
        pickle_data = None
        if path.exists(self.pickle_file):
            pickle_data = self.unpickle_data()
            self.mtx = pickle_data[0]
            self.dist = pickle_data[1]
        else:
            self.calibrate()
            self.pickle_data()
            pickle_data = (self.mtx, self.dist)
        return pickle_data