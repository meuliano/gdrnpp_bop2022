import sys
import os
import cv2
import numpy as np
import argparse
import os
import glob
import math
import json
import matplotlib.pyplot as plt
import datetime
import time
import json

class StereoCalibrationProcessor():
    """Class representing Stereo Calibration processing."""

    def __init__(self, calib_file="", alpha=0.0, resolution_multiplier=1.0, aspect_preserve=2.0):

        # Process input calib_file if provided. Otherwise assume calib_stereo has already been loaded and checked.
        if (calib_file):
            with open(calib_file, 'r') as f:
                calib_stereo = json.load(f)
            assert calib_stereo['type'] == 'stereo_camera_calibration', 'Not a stereo calibration!'

        self.compute_rectification_maps(calib_stereo, alpha=alpha, resolution_multiplier=resolution_multiplier,
                                        aspect_preserve=aspect_preserve)

    def compute_rectification_maps(self, calib_stereo, alpha=0.0, resolution_multiplier=1.0, aspect_preserve=2.0):
        """ Given a stereo calibration, calculate maps that can be used to efficiently
        rectify raw images from that camera!


        resolution_multiplier can be set to 2 if we want to "double" the output resolution so as to not
        lose information by rescaling since we're reducing resolution by splitting L/R images in the endoscope! If we
        don't want this, set it to 1.
        """

        # Extract parameters from the calibration data
        self.image_shape_raw = tuple(calib_stereo['left']['resolution'])  # Assume same for the right image too
        self.image_shape_rectified = (int(np.round(self.image_shape_raw[0] * resolution_multiplier)), int(np.round(
            self.image_shape_raw[
                1] * resolution_multiplier * aspect_preserve)))  # Default to the same size; but change this to change amount of detail after distortion etc. # TODO What is optimal value here?

        K_left = np.array(calib_stereo['left']['K'])
        K_right = np.array(calib_stereo['right']['K'])

        dist_left = np.array(calib_stereo['left']['distortion']['coefficients'])
        dist_right = np.array(calib_stereo['right']['distortion']['coefficients'])

        CamLeft_T_CamRight = np.array(calib_stereo['CamLeft_T_CamRight'])
        CamRight_T_CamLeft = np.linalg.inv(CamLeft_T_CamRight)
        CamRight_R_CamLeft = CamRight_T_CamLeft[0:3, 0:3]
        CamRight_trans_CamLeft = CamRight_T_CamLeft[0:3, 3]

        # Given these parameters, calculate a rectification!
        print("Calculating stereo rectification...")
        self.CamLeftRect_R_CamLeft, self.CamRightRect_R_CamRight, self.P_left, self.P_right, self.Q, _, _ \
            = cv2.stereoRectify(K_left, dist_left, K_right, dist_right,
                                   self.image_shape_raw,
                                   CamRight_R_CamLeft, CamRight_trans_CamLeft,
                                   flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha,
                                   newImageSize=self.image_shape_rectified)

        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(K_left, dist_left, self.CamLeftRect_R_CamLeft,
                                                                       self.P_left, self.image_shape_rectified,
                                                                       cv2.CV_32FC1)
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(K_right, dist_right,
                                                                         self.CamRightRect_R_CamRight, self.P_right,
                                                                         self.image_shape_rectified, cv2.CV_32FC1)

        # Convert the rectified projection matrices P_left and P_right to the rectified intrinsics and extrinsics
        self.K_rect = self.P_left[0:3, 0:3]  # Same for left and right
        self.CamLeft_T_CamRight = np.eye(4)
        self.CamLeft_T_CamRight[0, 3] = -self.P_right[0, 3] / self.P_right[
            0, 0]  # Extrinsics: facing the same way, just a shift along the x-axis

    def rectify_left_image(self, image_left):
        """ Returns a rectified left image """
        return self.rectify_image_from_map(image_left, self.map_left_x, self.map_left_y)

    def rectify_right_image(self, image_right):
        """ Returns a rectified right image """
        return self.rectify_image_from_map(image_right, self.map_right_x, self.map_right_y)

    def rectify_image_from_map(self, image, map_x, map_y):
        """ Perform the actual image rectification """
        image_rectified = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        return image_rectified