import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)
from core.gdrn_modeling.demo.predictor_yolo import YoloPredictor
from core.gdrn_modeling.demo.predictor_gdrn import GdrnPredictor
import os
import cv2

from time import sleep
import numpy as np
import rticonnextdds_connector as rti

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

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
def get_image_list(rgb_images_path, depth_images_path=None):
    image_names = []

    rgb_file_names = os.listdir(rgb_images_path)
    rgb_file_names.sort()
    for filename in rgb_file_names:
        apath = os.path.join(rgb_images_path, filename)
        ext = os.path.splitext(apath)[1]
        if ext in IMAGE_EXT:
            image_names.append(apath)

    if depth_images_path is not None:
        depth_file_names = os.listdir(depth_images_path)
        depth_file_names.sort()
        for i, filename in enumerate(depth_file_names):
            apath = os.path.join(depth_images_path, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names[i] = (image_names[i], apath)
                # depth_names.append(apath)

    else:
        for i, filename in enumerate(rgb_file_names):
            image_names[i] = (image_names[i], None)

    return image_names

# TURN INTO A CLASS
def init_connector (participant_str, pubsub_str, is_publisher=False):
    connector = rti.Connector(
            config_name=participant_str,
            url=cur_dir + "/NddsQosProfiles.xml")
    
    if is_publisher:
        output = connector.get_output(pubsub_str)
        print("Waiting for subscriptions...")
        output.wait_for_subscriptions()
        return output
    else:
        input = connector.get_input(pubsub_str)
        print("Waiting for publications...")
        input.wait_for_publications()  # wait for at least one matching publication
        return input

def write_pose(writer: rti.Output, mat: np.array):
    print("Writing...")
    writer.instance.set_dictionary({
        "six_dof_pose":mat.reshape(-1).tolist()})
    writer.write()
    writer.wait()

if __name__ == "__main__":
    
    pose_writer = init_connector(
        participant_str="HugoParticipantLibrary::SixDofPoseParticipant", 
        pubsub_str="Pub::PoseWriter", 
        is_publisher=True)

    
    yolo_predictor = YoloPredictor(
                    exp_name="yolox-x",
                    config_file_path=osp.join(PROJ_ROOT,"configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_pbr_ycbv_bop_test.py"),
                    ckpt_file_path=osp.join(PROJ_ROOT,"output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_pbr_ycbv_bop_test/model_final.pth"),
                    fuse=True,
                    fp16=False
    )
    gdrn_predictor = GdrnPredictor(
                    config_file_path=osp.join(PROJ_ROOT,"configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py"),
                    ckpt_file_path=osp.join(PROJ_ROOT,"output/gdrn/ycbv/model_final_wo_optim.pth"),
                    # ckpt_file_path=osp.join(PROJ_ROOT,"output/gdrn/ycbv/model_tomato_soup_can.pth"),
                    camera_json_path=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/camera_cmu.json"),
                    path_to_obj_models=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/models")
    )

    img_set = "000050"
    img_num = "000923"

    image_paths = get_image_list(osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/test/" + img_set + "/rgb"))#, osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/test/000057/depth"))
    img_dir = osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/test/" + img_set + "/rgb/" + img_num + ".png")
    
    # img_dir = osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/bananaPic.jpg")

    img_dir = osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/fullPic.jpg")
    img = cv2.imread(img_dir)
    print("YOLO Inference...")
    output = yolo_predictor.inference(img)
    print("Preprocessing...")
    data_dict = gdrn_predictor.preprocessing(outputs=output, image=img)
    print("GDRN Inference...")
    out_dict = gdrn_predictor.inference(data_dict)
    print("Postprocessing...")
    poses = gdrn_predictor.postprocessing(data_dict, out_dict)
    print("Done")

    write_pose(pose_writer, poses['008_pudding_box'])
    sleep(0.5  )
    write_pose(pose_writer, poses['040_large_marker'])
    sleep(0.5)
    # write_pose(pose_writer, poses['003_cracker_box'])
    # sleep(0.5)
    # write_pose(pose_writer, poses['005_tomato_soup_can'])
    # sleep(0.5)
    # write_pose(pose_writer, poses['006_mustard_bottle'])
    sleep(0.5)
    write_pose(pose_writer, poses['011_banana'])
    sleep(0.5)
    # write_pose(pose_writer, poses['035_power_drill'])

    print("Exiting...")
     # Wait for all subscriptions to receive the data before exiting
    pose_writer.connector.close()