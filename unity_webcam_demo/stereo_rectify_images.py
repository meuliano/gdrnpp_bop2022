#!/usr/bin/env python3

import argparse
import cv2
import glob
import json
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
from calib_support import StereoCalibrationProcessor



class StereoImageRectifier():
    def __init__(self, calib_stereo, alpha=1.0, resolution_multiplier=1.0, aspect_preserve=2.0):
        # Compute rectification maps & store results in this class
        # self.folder_out = folder_out
        # self.make_folders()

        self.compute_rectification_maps(calib_stereo, alpha, resolution_multiplier, aspect_preserve)

    def split_and_rectify_stereo_image(self, image):
        """ Perform stereo splitting / de-interlacing, and then rectify the images """
        # First split the image. See note in "split-stereo-image.py" for the splitting order.
        image_left =  image[1::2, :, :]
        image_right = image[0::2, :, :]

        # Rectify
        image_left_rectified  = self.stereo_calib_processor.rectify_left_image(image_left)
        image_right_rectified = self.stereo_calib_processor.rectify_right_image(image_right)

        # Return the pair of rectified images
        return (image_left_rectified, image_right_rectified)


    def calculate_max_compression_ratio(self):
        """ Calculate the "compression ratio" -- if we move one pixel in the destination image (anywhere), what is the largest possible change in the raw image?"
        If this ratio is > 1, we will be skipping over pixels and losing information in the output.
        """

        def calculate_compression_ratio(map_x, map_y, mask):
            # Find the difference in x coordinates
            diff_x = np.diff(map_x, axis=1)
            # Find the max, but only caring about masked valid points
            mask_x = mask[:, 0:-1, 0]
            max_diff_x = np.max(diff_x[mask_x == 255])

            # Find the difference in x coordinates
            diff_y = np.diff(map_y, axis=0)
            # Find the max, but only caring about masked valid points
            mask_y = mask[0:-1, :, 0]
            max_diff_y = np.max(diff_y[mask_y == 255])

            return (max_diff_x, max_diff_y)

        max_left_diff_x, max_left_diff_y = calculate_compression_ratio(self.map_left_x, self.map_left_y, self.mask_left_valid_pixels)
        max_right_diff_x, max_right_diff_y = calculate_compression_ratio(self.map_right_x, self.map_right_y, self.mask_right_valid_pixels)
        max_compression_ratio = max(max_left_diff_x, max_left_diff_y, max_right_diff_x, max_right_diff_y)

        print("Compression ratios:")
        print("  Left:")
        print("     X: {}".format(max_left_diff_x))
        print("     Y: {}".format(max_left_diff_y))
        print("  Right:")
        print("     X: {}".format(max_right_diff_x))
        print("     Y: {}".format(max_right_diff_y))
        print()
        print("  Overall: {}".format(max_compression_ratio))
        return max_compression_ratio



    def compute_rectification_maps(self, calib_stereo, alpha=1.0, resolution_multiplier=1.0, aspect_preserve=2.0):
        """ Given a stereo calibration, calculate maps that can be used to efficiently 
        rectify raw images from that camera! 
        
        
        resolution_multiplier can be set to 2 if we want to "double" the output resolution so as to not
        lose information by rescaling since we're reducing resolution by splitting L/R images in the endoscope! If we
        don't want this, set it to 1.
        """

        self.stereo_calib_processor = StereoCalibrationProcessor(calib_stereo)

        self.image_shape_raw = self.stereo_calib_processor.image_shape_raw
        self.image_shape_rectified = self.stereo_calib_processor.image_shape_rectified
        self.CamLeftRect_R_CamLeft = self.stereo_calib_processor.CamLeftRect_R_CamLeft
        self.CamRightRect_R_CamRight = self.stereo_calib_processor.CamRightRect_R_CamRight
        self.P_left = self.stereo_calib_processor.P_left
        self.P_right = self.stereo_calib_processor.P_right
        self.Q = self.stereo_calib_processor.Q
        self.map_left_x = self.stereo_calib_processor.map_left_x
        self.map_left_y = self.stereo_calib_processor.map_left_y
        self.map_right_x = self.stereo_calib_processor.map_right_x
        self.map_right_y = self.stereo_calib_processor.map_right_y
        self.K_rect = self.stereo_calib_processor.K_rect
        self.CamLeft_T_CamRight = self.stereo_calib_processor.CamLeft_T_CamRight


        print("P_left:")
        print(self.P_left)
        print()

        print("P_right:")
        print(self.P_right)
        print()

        print("Q = ")
        print(self.Q)
        print()
        

        print("K_rect (for both left and right):")
        print(self.K_rect)
        print()

        print("CamLeft_T_CamRight:")
        print(self.CamLeft_T_CamRight)
        print()

        print("Original image shape: {}".format(self.image_shape_raw))
        print("Rectified image shape: {}".format(self.image_shape_rectified))

        print()

        print("Done!")




    def construct_distortion_mask(self,calib_stereo):
        # Construct a "valid pixels" mask that's the size of the output rectified images.
        # Each pixel is white if it corresponds to a valid pixel from the input / raw image, and black otherwise
        # (for instance, if it's part of the "distortion boundaries" etc). This one is particularly useful
        # if the rectified images are "bigger" than the input / raw images.
        white_image = np.ones((self.image_shape_raw[1], self.image_shape_raw[0], 3), dtype=np.uint8) * 255
        self.mask_left_valid_pixels = self.stereo_calib_processor.rectify_left_image(white_image)
        self.mask_right_valid_pixels = self.stereo_calib_processor.rectify_right_image(white_image)

        cv2.imwrite(os.path.join(self.folder_masks, "mask_left_valid_pixels_in_rectified_image.png"), self.mask_left_valid_pixels)
        cv2.imwrite(os.path.join(self.folder_masks, "mask_right_valid_pixels_in_rectified_image.png"), self.mask_right_valid_pixels)

        # cv2.imshow('Mask Left (Valid Pixels)', self.mask_left_valid_pixels)
        # cv2.imshow('Mask Right (Valid Pixels)', self.mask_right_valid_pixels)

        # Construct an "outline used pixels" mask that's the size of the input rectified images, and outlines the extents
        # of the rectified output in that raw input. This one is particularly useful if the rectified images are "smaller"
        # than the input / raw images, to see which pixels aren't being used.
        
        # First, make a bunch of outline points corresponding to the extents of the rectified image
        N_points_per_edge = 100
        points = []
        for xi in np.linspace(0.0, self.image_shape_rectified[0], N_points_per_edge):
            points.append(np.array([xi, 0.0, 1.0]))
        for yi in np.linspace(0.0, self.image_shape_rectified[1], N_points_per_edge):
            points.append(np.array([self.image_shape_rectified[0], yi, 1.0]))
        for xi in np.linspace(self.image_shape_rectified[0], 0.0, N_points_per_edge):
            points.append(np.array([xi, self.image_shape_rectified[1], 1.0]))
        for yi in np.linspace(self.image_shape_rectified[1], 0.0, N_points_per_edge):
            points.append(np.array([0.0, yi, 1.0]))
        points = np.array(points)

        # Retrieve raw projection parameters
        K_left_raw  = np.array(calib_stereo['left']['K'])
        K_right_raw = np.array(calib_stereo['right']['K'])
        
        dist_left  = np.array(calib_stereo['left']['distortion']['coefficients'])
        dist_right = np.array(calib_stereo['right']['distortion']['coefficients'])


        K_rect_inv = np.linalg.inv(self.K_rect)

        def construct_extents_mask(CamRect_R_CamRaw, K_side_raw, dist_side, name_side):
            def coordinate(p):
                return tuple(int(np.round(m)) for m in p)
                
            CamRect_ps = np.dot(K_rect_inv, points.T) # Unproject the rectified points from 2D to 3D in CamRect frame
            CamRaw_ps = np.dot(CamRect_R_CamRaw.T, CamRect_ps) # Convert 3D points from rect to raw frame 
            imageraw_ps = cv2.projectPoints(CamRaw_ps.T, np.zeros(3), np.zeros(3), K_side_raw, dist_side)[0].squeeze() # Project onto original image with distortion

            mask_extents = np.ones((self.image_shape_raw[1], self.image_shape_raw[0], 3), dtype=np.uint8) * 255
            for i in range(len(imageraw_ps) - 1):
                cv2.line(mask_extents, coordinate(imageraw_ps[i]), coordinate(imageraw_ps[i+1]), (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

            # cv2.imshow('Mask {} (Extents plot)'.format(name_side), mask_extents)
            cv2.imwrite(os.path.join(self.folder_masks, 'mask_{}_rectified_extents_in_raw_image.png'.format(name_side)), mask_extents)


        construct_extents_mask(self.CamLeftRect_R_CamLeft, K_left_raw, dist_left, 'left')
        construct_extents_mask(self.CamRightRect_R_CamRight, K_right_raw, dist_right, 'right')


        self.compression_ratio = self.calculate_max_compression_ratio()

        # import IPython; IPython.embed()




    def write_stereo_calibration(self, filename):
        calib = {}
        calib['type'] = 'stereo_camera_calibration'
        calib['source'] = 'stereo_rectification'

        def construct_single_camera_calibration():
            d = {}
            d['type'] = 'single_camera_calibration'
            d['K'] = self.K_rect.tolist()
            d['distortion'] = {'model': 'coefficients_opencv', 'coefficients': [0.0, 0.0, 0.0, 0.0, 0.0]}
            d['resolution'] = self.image_shape_rectified
            return d

        calib['left'] = construct_single_camera_calibration()
        calib['right'] = construct_single_camera_calibration()
        calib['CamLeft_T_CamRight'] = self.CamLeft_T_CamRight.tolist()
        calib['CamLeftRect_R_CamLeft'] = self.CamLeftRect_R_CamLeft.tolist()
        calib['CamRightRect_R_CamRight'] = self.CamRightRect_R_CamRight.tolist()

        # Optional parameters
        calib['max_compression_ratio'] = float(self.compression_ratio)

        # Write to a file
        print("Writing stereo calibration to {}".format(filename))
        with open(filename, 'w') as f:
            json.dump(calib, f, indent=4)


    def make_folders(self):
        """ Helper function to create left and right directories """
        self.folder_left = os.path.join(self.folder_out, "left")
        self.folder_right = os.path.join(self.folder_out, "right")
        self.folder_masks = os.path.join(self.folder_out, "masks")
        self.folder_epipolar = os.path.join(self.folder_out, "epipolar_annotations")
        self.folder_combined = os.path.join(self.folder_out, "combined")
        for folder in [self.folder_left, self.folder_right, self.folder_masks, self.folder_epipolar, self.folder_combined]:
            if not os.path.exists(folder):
                os.makedirs(folder)



def save_combined_images (image_left_rect, image_right_rect, fname, comb_horizontal=True):
    h, w = image_left_rect.shape[0:2]
    if (comb_horizontal):
        image_rect_lr = np.zeros((h, w*2, 3), dtype=np.uint8)
        image_rect_lr[:, 0:w, :] = image_left_rect
        image_rect_lr[:, w:,  :] = image_right_rect
    else:
        image_rect_lr = np.zeros((h*2, w, 3), dtype=np.uint8)
        image_rect_lr[0:h, :, :] = image_left_rect
        image_rect_lr[h:,  :, :] = image_right_rect
    cv2.imwrite(fname, image_rect_lr)

# Save epipolar plots showing the features at each row matching for both left and right images
def save_epipolar_plots (image_left_rect, image_right_rect, fname):
    h, w = image_left_rect.shape[0:2]
    image_rect_lr = np.zeros((h, w*2, 3), dtype=np.uint8)
    image_rect_lr[:, 0:w, :] = image_left_rect
    image_rect_lr[:, w:,  :] = image_right_rect

    for hi in range(0, h, 50):
        image_rect_lr[hi, :, 0] = 0
        image_rect_lr[hi, :, 1] = 255
        image_rect_lr[hi, :, 2] = 0

    cv2.imwrite(fname, image_rect_lr)


def stereo_rectify(folder_in,calib_stereo_file,alpha,resolution_multiplier,aspect_preserve,combine,epipolar_plots):

    stereo_folder_out = folder_in+'/stereo_rectified/'
    if not os.path.exists(stereo_folder_out):
        os.mkdir(stereo_folder_out)

    with open(calib_stereo_file, 'r') as f:
        calib_stereo = json.load(f)
    assert calib_stereo['type'] == 'stereo_camera_calibration', 'Not a stereo calibration!'


    sir = StereoImageRectifier(calib_stereo, stereo_folder_out, alpha=alpha, resolution_multiplier=resolution_multiplier, aspect_preserve=aspect_preserve)

  
    sir.construct_distortion_mask(calib_stereo)

    # Write the equivalent stereo calibration
    sir.write_stereo_calibration(os.path.join(stereo_folder_out, "calib_stereo_rectified.json"))

    files = glob.glob(os.path.join(folder_in, '*.png'))
    files.sort()
    for file in files:
        print("Processing {}".format(file))
        image_stereo = cv2.imread(file)

        # print(image_stereo.shape)

        (image_left_rect, image_right_rect) = sir.split_and_rectify_stereo_image(image_stereo)
        h, w = image_left_rect.shape[0:2]

        folder_in, file_basename = os.path.split(file)
        cv2.imwrite(os.path.join(sir.folder_left, file_basename), image_left_rect)
        cv2.imwrite(os.path.join(sir.folder_right, file_basename), image_right_rect)

        if combine:
            fname = os.path.join(sir.folder_combined, file_basename)
            save_combined_images(image_left_rect, image_right_rect, fname, combine == 1)

        if epipolar_plots:
            fname = os.path.join(sir.folder_epipolar, file_basename)
            save_epipolar_plots (image_left_rect, image_right_rect, fname)



if __name__ == '__main__':

    np.set_printoptions(precision=6, suppress=True)

    # Set up an argument parser
    parser = argparse.ArgumentParser(description='Rectify stereo images!')
    parser.add_argument('calib_stereo', type=str, help='stereo calibration data in .json file')
    parser.add_argument('stereo_folder_in', type=str, help='An input directory of stereo interlaced .png images to rectify')
    parser.add_argument('--alpha', help='alpha scaling value for OpenCV. \n\t 0 => only valid pixels (zoomed in without edges) \n\t 1 => dont lose any pixels (zoomed out with edges)', type=float, default=0.0)
    parser.add_argument('--resolution_multiplier', help='resolution multiplier to make the images larger or smaller so as to preserve detail', type=float, default=1.0)
    parser.add_argument('--aspect_preserve', help='aspect preservation multiplier', type=float, default=2.0)
    parser.add_argument('--epipolar-plots', help='generate epipolar plots for comparison of L and R images', action='store_const', const=True, default=False)
    parser.add_argument('--combine', help='combine left-right images. 0 => disable. 1 => horizontal. 2 => vertical.',
                        type=int, default=0)
    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit()

    assert os.path.exists(args.stereo_folder_in), 'Input directory does not exist'

    if (os.path.isdir(args.stereo_folder_in)):
        stereo_rectify(args.stereo_folder_in, args.calib_stereo, args.alpha, args.resolution_multiplier, args.aspect_preserve, args.combine, args.epipolar_plots)
    else:
        count = 0
        with open (args.stereo_folder_in, 'r') as f:
            dirs = [line.rstrip() for line in f]
            for dir in dirs:
                if (not os.path.isdir(dir)):
                    continue
                count += 1
                print("Dir{}: {}".format(count, dir.strip()))
                stereo_rectify(dir, args.calib_stereo, args.alpha, args.resolution_multiplier,
                               args.aspect_preserve, args.combine, args.epipolar_plots)



