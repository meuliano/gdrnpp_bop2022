import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, ".."))
sys.path.insert(0, PROJ_ROOT)
from predictor_yolo import YoloPredictor
from predictor_gdrn import GdrnPredictor
import cv2
from det.yolox.utils import vis
import numpy as np
from utils.dds_connector import DDSReader, DDSWriter
from utils.webcam_utils import VideoCapture
from stereo_rectify_images import StereoImageRectifier

# Stereo Calibration JSON File
calib_file = cur_dir + '/endoscope_files' + '/SN12398_calib_stereo.json'
# Get all class names from classes.txt
classes = [line.rstrip() for line in open(cur_dir + "/utils/classes.txt")]
# Object Poses we are sending over DDS - order matters
xml_str = ['six_dof_pose_marker', 'six_dof_pose_pudding', 'six_dof_pose_banana']
ycbv_str = ['040_large_marker', '008_pudding_box', '011_banana']

def vis_yolo(output, rgb_image, class_names, cls_conf=0.35):
    if output is None:
        return rgb_image
    output = output.cpu()

    bboxes = output[:, 0:4]

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    vis_res = vis(rgb_image, bboxes, scores, cls, cls_conf, class_names)
    return vis_res


def publish_dds(prepare_for_pose=False):
    # If the object is not detected, send an identity matrix in its place
    for k in ycbv_str:
        if k not in poses.keys():
            poses[k] = np.eye(4)

    pose_writer.write_dict({
        'prepare_for_pose':False,
        xml_str[0]:poses[ycbv_str[0]].reshape(-1).tolist(),
        xml_str[1]:poses[ycbv_str[1]].reshape(-1).tolist(),
        xml_str[2]:poses[ycbv_str[2]].reshape(-1).tolist()
    })

if __name__ == "__main__":

    # Load Pretrained YOLOX PredictionModel
    yolo_predictor = YoloPredictor(
                    exp_name="yolox-x",
                    config_file_path=osp.join(PROJ_ROOT,"configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_pbr_ycbv_bop_test.py"),
                    ckpt_file_path=osp.join(PROJ_ROOT,"output/yolox/model_final.pth"),
                    fuse=True,
                    fp16=False
                    )
    # Load Pretrained GDRN Prediction Model
    gdrn_predictor = GdrnPredictor(
                    config_file_path=osp.join(PROJ_ROOT,"configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py"),
                    ckpt_file_path=osp.join(PROJ_ROOT,"output/gdrn/ycbv/model_final_wo_optim.pth"),
                    # ckpt_file_path=osp.join(PROJ_ROOT,"output/gdrn/ycbv/model_tomato_soup_can.pth"),
                    camera_json_path=cur_dir + '/endoscope_files' + '/SN12398_calib_stereo_rectified.json',#osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/camera_cmu.json"),
                    path_to_obj_models=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/models"),
                    is_endoscope=True
                    )

    # Create DDS Publisher
    pose_writer = DDSWriter("SixDofPoseParticipant", "PoseWriter")

    # Setup Image De-Interlacing and Rectification (alpha = 0 crops image)
    img_rectifier = StereoImageRectifier(calib_file, alpha=0.0, resolution_multiplier=1.0, aspect_preserve=2.0)

    # Setup Webcam Capture
    cam = VideoCapture(0)

    while True:

        # Read from Webcam
        img = cam.read()

        # Split and Rectify the images
        (img_right, img_left) = img_rectifier.split_and_rectify_stereo_image(img)
        
        # Resize Left Image to match Training Data
        # Use cropped portion of image (convert 1920x1080 to 1440x1080 that is same aspect ratio as 640x480)
        img_resize = cv2.resize(img_left[:, 240:1680], dsize=(640, 480), interpolation=cv2.INTER_CUBIC)

        # YOLO Inference
        output = yolo_predictor.inference(img_resize)

        # Show output feed with YOLO bounding boxes
        out = vis_yolo(output[0], img_resize, classes, cls_conf=0.5)
        cv2.imshow('video', out)

        # TODO: Implement Disparity Map for Depth Pose Refinement

        # GDRN Inference and Pose Detection
        data_dict = gdrn_predictor.preprocessing(outputs=output, image=img_resize)
        out_dict = gdrn_predictor.inference(data_dict)
        poses = gdrn_predictor.postprocessing(data_dict, out_dict)

        # If the object is not detected, send an identity matrix in its place
        for k in ycbv_str:
            if k not in poses.keys():
                poses[k] = np.eye(4)

        # Publish Poses for Detected Objects
        print("Writing Poses...")
        pose_writer.write_dict({
            xml_str[0]:poses[ycbv_str[0]].reshape(-1).tolist(),
            xml_str[1]:poses[ycbv_str[1]].reshape(-1).tolist(),
            xml_str[2]:poses[ycbv_str[2]].reshape(-1).tolist()
        })

        key = cv2.waitKey(1)
        if key == 27:
            break
    
    print("Shutting Down")
    pose_writer.shutdown()
    cam.release()
    cv2.destroyAllWindows()