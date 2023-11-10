import os
import sys
import time
import numpy as np
import json
import argparse
import torchvision.transforms as transforms
import torch.nn.functional as F
import open3d as o3d
import plyfile
import cv2
import subprocess
import glob
import datetime

script_dir = os.path.dirname(__file__)

from stereo_rectify_images import StereoImageRectifier
# from stereo.deep_learning.PSMNet import predict_img as predict_img_psmnet
# from stereo.deep_learning.HITNet import predict_img as predict_img_hitnet
from depth_mapping import predict_img as predict_img_hsmnet
from depth_mapping.preprocess import get_transform

DISPLAY_TEXT_THICKNESS = 2.0

#TODO Separate preprocess, add normalization to all networks ? Fix the cloud effect.

pcd_no_disk_io = True

def normalize_stereo_images(img_left, img_right):
    image_left = cv2.fastNlMeansDenoisingColored(img_left,None,10,10,5,15)
    image_right = cv2.fastNlMeansDenoisingColored(img_right,None,10,10,5,15)

    return image_left, image_right

def preprocess_image_psmnet(imgL, imgR):
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])

    imgL = infer_transform(imgL)
    imgR = infer_transform(imgR)

    # pad to width and hight to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0)).unsqueeze(0)
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0)).unsqueeze(0)

    return imgL, imgR

def get_model (args):
    """Get the model based on args in order to run any network for stereo reconstruction.
    """
    model = None

    if args.model == 'hsmnet':
        print("::INFO:: Loading HSMNet model..")
        model = predict_img_hsmnet.load_model(args.hsmnet_weights, args.maxdisp, args.clean, args.level)
    else:
        print("::INFO:: No model specified.")

    return model


def stereo_rectify(args):

    with open(args.calib_file, 'r') as f:
        calib_json = json.load(f)

    sir = StereoImageRectifier(calib_json, '', alpha=args.alpha, resolution_multiplier=args.resolution_multiplier, aspect_preserve=args.aspect_preserve)
    sir.construct_distortion_mask(calib_json)
    sir.write_stereo_calibration("calib_stereo_rectified.json")

    return sir

def run_dispartiy (image_left, image_right, model, args):

    disp_L = None
    
    if args.model == 'hsmnet':

        processed = get_transform()
        model.eval()
        imgsize = image_left.shape[:2]
        disp_L = predict_img_hsmnet.test(model, image_left,image_right,args.maxdisp, args.scale_percent/100, processed, verbose=0)
        disp_L = cv2.resize(disp_L/(args.scale_percent/100),(imgsize[1],imgsize[0]),interpolation=cv2.INTER_LINEAR)
        # clip while keep inf
        invalid = np.logical_or(disp_L == np.inf,disp_L!=disp_L)
        disp_L[invalid] = np.inf

    return disp_L

def display_depthmap(depth_map, image_left, title):

    title_left = title + 'Left'
    title_right = title + 'DepthMap'
    start_point = (int(depth_map.shape[1]/2) - 10 , int(depth_map.shape[0]/2) - 10)
    end_point = (int(depth_map.shape[1]/2) + 10 , int(depth_map.shape[0]/2) + 10)
    depth_map_centre = depth_map[int(depth_map.shape[1]/2) - 10:int(depth_map.shape[1]/2) + 10, int(depth_map.shape[0]/2) - 10 : int(depth_map.shape[0]/2)+10]
    avg_depth = np.mean(depth_map_centre)
    depth_map_rgb = np.array(depth_map*255, dtype=np.uint8)
    depth_map_rgb = cv2.cvtColor(depth_map_rgb, cv2.COLOR_GRAY2BGR)
    depth_map_rgb = cv2.rectangle(depth_map_rgb, start_point, end_point,(0,0,255), 2)
    avg_depth = (avg_depth*1000)
    cv2.putText(depth_map_rgb,
                "%.1f"%avg_depth + " mm",
                (int(depth_map.shape[1]/2) - 10 ,
                 int(depth_map.shape[0]/2) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                DISPLAY_TEXT_THICKNESS,
                (0,255,0),
                2)
    # Display left image
    cv2.namedWindow(title_left)
    cv2.moveWindow(title_left, 0, 600)
    cv2.imshow(title_left,image_left)
    # Display right image
    cv2.namedWindow(title_right)
    cv2.moveWindow(title_right,1100,600)
    #depth_map_rgb = cv2.applyColorMap(depth_map_rgb, cv2.COLORMAP_HOT)
    depth_map_rgb = cv2.applyColorMap(depth_map_rgb, cv2.COLORMAP_HSV)
    cv2.imshow(title_right, depth_map_rgb)


def o3d_camera_intrinsics (calib_file):
    # camera settings
    with open(calib_file, 'r') as f:
        calib_json = json.load(f)

    res = calib_json['left']['resolution']
    K_left  = np.array(calib_json['left']['K'])
    width = res[0]
    height = res[1]
    fx = K_left[0][0]
    fy = K_left[1][1]
    cx = K_left[2][0]
    cy = K_left[2][1]
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    camera_intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    return camera_intrinsic


def run_live_depthmapping(dev_name, model, args):

    if ((not args.imagedir) and (not args.dev)):
        print("No video input selected. Exiting")
        return

    if (not args.imagedir):
        cam = cv2.VideoCapture(dev_name)
        # get vcap property
        width = round(cam.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
        height = round(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
        if (width <= 0 or height <= 0):
            return
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Use open3d point cloud generation method without intermediate pcd to disk
    pcd_no_disk_io = not args.pcd_disk_io

    input = init_connector()

    sir = stereo_rectify (args)
    pcr = reproject_pointcloud.PointcloudReprojector("calib_stereo_rectified.json")
    k_values, _ , camera_parameters = reproject_pointcloud.load_camera_calibration("calib_stereo_rectified.json")
    baseline_distance = camera_parameters[0,3]

    o3d_intrinsics = o3d_camera_intrinsics(args.calib_file)

    include_reconstruct_pipeline = False
    flag = 0
    frame_itr = 0

    if (args.imagedir):
        files = sorted(glob.glob(args.imagedir + str("/*.png")))
    next_image_index = 0

    start_timer_fps = time.time()

    while next_image_index > -1:
        # Grab a frame
        if (args.imagedir):
            # Next image for offline mode
            image_stereo = cv2.imread(files[next_image_index])
        else:
            # Next frame for online mode
            ret_val, image_stereo = cam.read()

        if image_stereo is None:
            break
        # De-interlace to grab the left and right
        image_left, image_right = sir.split_and_rectify_stereo_image(image_stereo)
        frame_itr  += 1

        # image_left, image_right = normalize_stereo_images(image_left, image_right)                #This fixes the Cloud effect till a certain extent, but drags down the FPS

        start_time = time.time()
        disp_L = run_dispartiy (image_left, image_right, model, args)
        depth_map = (k_values[0, 0] * baseline_distance) / disp_L
        end_time = time.time() - start_time

        # Display it
        title = dev_name
        display_depthmap(depth_map, image_left, title)

        if args.debug:
            cv2.imwrite("current_frame.png", image_left)
            np.save("current_disp.npy",disp_L)
            cv2.imwrite("current_disp.png", disp_L)
            cv2.imwrite("current_stereo.png", image_stereo)
            cv2.imwrite("current_depth.png", depth_map_rgb)
            np.save("current_depth.npy", depth_map)

        key = cv2.waitKey(1)
        if key == 27:           #Press ESC in the CV2 window to quit.
            break

        if args.pointclouds:
            # Press ENTR to start pointcloud reprojector in an Open3D window.
            if key == 13:
                include_reconstruct_pipeline = True
                current_frame_visualizer = o3d.visualization.Visualizer()
                current_frame_visualizer.create_window(height=540, width=960)

            # Press lower-case r to perform registration
            elif key == 114:
                # TODO: Use relative path here or get the script path and generate this
                plyfpath = "current_pcd.ply"
                # TODO: Get this as arg
                stlfpath = "/data/data/model_store/heart_phantom/heart_cropped.stl"
                if (os.path.exists(plyfpath) and os.path.isfile(plyfpath) ):
                    include_reconstruct_pipeline = False
                    # manual registration
                    stl_flag = subprocess.Popen(["python", "manual_reg.py",
                                                 "--stl", stlfpath,
                                                 "--pcd", plyfpath,
                                                 "--voxel_size", str(0.0007)])

                    while True:
                        stdoutdata, stderrdata = stl_flag.communicate()
                        if not stl_flag.returncode:
                            print("Points recieved.")
                            break

                    # Start a process to dump keyframes.json and run STL model transformation based on it
                    #while True:
                    #    read_dds_connector(input, json=True)

                    if (1):
                        # Start process to transform model
                        subprocess.Popen(["python", "stl.py", "--stl", stlfpath])
                        #subprocess.Popen(["python", "stl.py", "--stl", stlfpath, "--rotate"])
                        #stl_rotate(stlfpath)
                else:
                    print ("::ERROR:: ply file not found: ", plyfpath)
            elif key == 27:          #Press ESC in the CV2 window to quit.
                break

            if include_reconstruct_pipeline:
                if flag == 0:
                    # TODO: FIXME. Don't pass and save images and ply file. Get the vertices directly and pass to add_geometry

                    # <editor-fold desc="TODO pcd using open3d">
                    # TODO: FIXME. Use open3d to generate rgbd image from rgb and depth map.
                    #  Then use open3d to create point cloud from rgbd images and camera calibration parameters
                    # https://towardsdatascience.com/generate-a-3d-mesh-from-an-image-with-python-12210c73e5cc
                    # # create rgbd image
                    # depth_o3d = o3d.geometry.Image(depth_image)
                    # image_o3d = o3d.geometry.Image(image)
                    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d,
                    #                                                                 convert_rgb_to_intensity=False)
                    # # camera settings
                    # camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
                    # camera_intrinsic.set_intrinsics(width, height, 500, 500, width / 2, height / 2)
                    # # create point cloud
                    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
                    # </editor-fold>

                    if (pcd_no_disk_io):
                        # Create O3D format RGB-D image, then create point cloud
                        depth_o3d = o3d.geometry.Image(depth_map)
                        image_o3d = o3d.geometry.Image(image_left)
                        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d,
                                                                                        convert_rgb_to_intensity=False)
                        current_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_intrinsics)

                    else:
                        # Use point cloud projector to create textured point cloud from input image and disparity. Output
                        # in "current_pcd.ply" file
                        current_pcd = pcr.reproject_mat(image_left, disp_L, "current_pcd.ply")
                        # read back the point cloud in o3d format
                        current_pcd = o3d.io.read_point_cloud("current_pcd.ply")

                    current_pcd = pcd_remove_outliers(current_pcd, pcd_filter_neighbors=args.pcd_filter_neighbors, pcd_filter_std=args.pcd_filter_std)
                    current_frame_visualizer.add_geometry(current_pcd)
                    flag = 1
                else:
                    if (pcd_no_disk_io):
                        # Create O3D format RGB-D image, then create point cloud
                        depth_o3d = o3d.geometry.Image(depth_map)
                        image_o3d = o3d.geometry.Image(image_left)
                        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d,
                                                                                        convert_rgb_to_intensity=False)
                        current_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_intrinsics)
                    else:

                        # Use point cloud projector to create textured point cloud from input image and disparity. Output
                        # in "current_pcd.ply" file
                        current_pcd = pcr.reproject_mat(image_left,disp_L, "current_pcd.ply")
                        # read back the point cloud in o3d format
                        current_pcd = o3d.io.read_point_cloud("current_pcd.ply")

                    current_pcd = pcd_remove_outliers(current_pcd, pcd_filter_neighbors=20, pcd_filter_std=20.0)
                    if np.max(depth_map) > 1:   #to avoid noise, skipping clouds with depth reading of more than 1 meter for now.
                        continue
                    cam_control = current_frame_visualizer.get_view_control().convert_to_pinhole_camera_parameters()
                    current_frame_visualizer.clear_geometries()
                    current_frame_visualizer.update_geometry(current_pcd)
                    current_frame_visualizer.add_geometry(current_pcd)
                    cam_control = current_frame_visualizer.get_view_control().convert_from_pinhole_camera_parameters(cam_control)
                current_frame_visualizer.poll_events()
                current_frame_visualizer.update_renderer()

        if time.time() - start_timer_fps >=  5:
            print(datetime.datetime.now(), ' device: ', args.device, '::INFO:: device: FPS acheived ->  %.1f' % (frame_itr/10))
            frame_itr = 0
            start_timer_fps = time.time()

        next_image_index += 1
        if (args.imagedir and next_image_index >= len(files)):
            next_image_index = -1
            break

    current_frame_visualizer.destroy_window()
    cv2.destroyAllWindows()
