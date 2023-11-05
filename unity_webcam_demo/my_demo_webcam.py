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

classes = {
        0: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
        1: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
        2: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
        3: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
        4: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
        5: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
        6: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
        7: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
        8: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
        9: "011_banana",  # [-18.6730, 12.1915, -1.4635]
        10: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
        11: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
        12: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
        13: "025_mug",  # [-8.4675, -0.6995, -1.6145]
        14: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
        15: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
        16: "037_scissors",  # [7.0535, -28.1320, 0.0420]
        17: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
        18: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
        19: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
        20: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
}


def vis_yolo(output, rgb_image, class_names, cls_conf=0.35):
    if output is None:
        return rgb_image
    output = output.cpu()

    bboxes = output[:, 0:4]

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    vis_res = vis(rgb_image, bboxes, scores, cls, cls_conf, class_names)
    return vis_res

if __name__ == "__main__":
    
    # Load Pretrained YOLOX PredictionModel
    yolo_predictor = YoloPredictor(
                    exp_name="yolox-x",
                    config_file_path=osp.join(PROJ_ROOT,"configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_pbr_ycbv_bop_test.py"),
                    ckpt_file_path=osp.join(PROJ_ROOT,"output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_pbr_ycbv_bop_test/model_final.pth"),
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

    # Object Poses we are sending over DDS - order matters
    xlm_str = ['six_dof_pose_marker', 'six_dof_pose_pudding', 'six_dof_pose_banana']
    ycbv_str = ['040_large_marker', '008_pudding_box', '011_banana']

    # Setup Webcam Capture
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)

    while True:
        check, img = cam.read()
        if not check:
            continue
        # cv2.imshow('video', img)

        print("YOLO Inference...")
        output = yolo_predictor.inference(img)

        # Show output feed with YOLO bounding boxes
        out = vis_yolo(output[0], img, classes, cls_conf=0.35)
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
            xlm_str[0]:poses[ycbv_str[0]].reshape(-1).tolist(),
            xlm_str[1]:poses[ycbv_str[1]].reshape(-1).tolist(),
            xlm_str[2]:poses[ycbv_str[2]].reshape(-1).tolist()
        })

        key = cv2.waitKey(1)
        if key == 27:
            break
    
    print("Shutting Down")
    pose_writer.shutdown()
    cam.release()
    cv2.destroyAllWindows()