import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, ".."))
sys.path.insert(0, PROJ_ROOT)
from core.gdrn_modeling.demo.predictor_yolo import YoloPredictor
from core.gdrn_modeling.demo.predictor_gdrn import GdrnPredictor
import cv2
from det.yolox.utils import vis
import numpy as np
from dds_connector import DDSReader, DDSWriter
from webcam_utils import VideoCapture

# Get all class names from classes.txt
classes = [line.rstrip() for line in open(cur_dir + "/classes.txt")]
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
                    camera_json_path=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/camera_cmu.json"),
                    path_to_obj_models=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/models")
                    )

    # Create DDS Publisher
    pose_writer = DDSWriter("SixDofPoseParticipant", "PoseWriter")

    # Setup Webcam Capture
    cam = VideoCapture(0)

    while True:

        img = cam.read()
        
        pose_writer.write_dict({
            'prepare_for_pose':True,
            xml_str[0]:np.eye(4).reshape(-1).tolist(),
            xml_str[1]:np.eye(4).reshape(-1).tolist(),
            xml_str[2]:np.eye(4).reshape(-1).tolist()
        })
        
        print("YOLO Inference...")
        output = yolo_predictor.inference(img)

        # Show output feed with YOLO bounding boxes
        out = vis_yolo(output[0], img, classes, cls_conf=0.5)
        cv2.imshow('video', out)

        print("GDRN Pose Prediction...")
        data_dict = gdrn_predictor.preprocessing(outputs=output, image=img)
        out_dict = gdrn_predictor.inference(data_dict)
        poses = gdrn_predictor.postprocessing(data_dict, out_dict)

        # If the object is not detected, send an identity matrix in its place
        for k in ycbv_str:
            if k not in poses.keys():
                poses[k] = np.eye(4)

        print("Writing Poses...")
        pose_writer.write_dict({
            'prepare_for_pose':False,
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