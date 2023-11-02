from time import sleep
import numpy as np

# pip-install rticonnextdds-connector
from os import path as os_path
file_path = os_path.dirname(os_path.realpath(__file__))

import rticonnextdds_connector as rti

def init_connector (participant_str, pubsub_str, is_publisher=False):
    connector = rti.Connector(
            config_name=participant_str,
            url=file_path + "/NddsQosProfiles.xml")
    
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

def publish_keypoints(writer: rti.Output):
    aca_keypoints_available = [True, True, False, False]

    shaft_start_1 = np.array([0.1, 0.2, 0.3])
    shaft_end_1 = np.array([0.1, 0.2, 0.3])
    joint_s_1 = np.array([0.1, 0.2, 0.3])
    joint_t_1 = np.array([0.1, 0.2, 0.3])
    tip1_1 = np.array([0.1, 0.2, 0.3])
    tip2_1 = np.array([0.1, 0.2, 0.3])

    shaft_start_2 = np.array([0.1, 0.2, 0.3])
    shaft_end_2 = np.array([0.1, 0.2, 0.3])
    joint_s_2 = np.array([0.1, 0.2, 0.3])
    joint_t_2 = np.array([0.1, 0.2, 0.3])
    tip1_2 = np.array([0.1, 0.2, 0.3])
    tip2_2 = np.array([0.1, 0.2, 0.3])

    print("Writing...")
    for i in range(1, 20):
        writer.instance.set_dictionary({
            "aca_keypoints_available": aca_keypoints_available,
            "aca1_keypoints":
            {
                "shaft_start":shaft_start_1.tolist(), 
                "shaft_end":shaft_end_1.tolist(),
                "joint_s":joint_s_1.tolist(),
                "joint_t":joint_t_1.tolist(),
                "tip1":tip1_1.tolist(),
                "tip2":tip2_1.tolist()
            },
            "aca2_keypoints":
            {
                "shaft_start":shaft_start_1.tolist(), 
                "shaft_end":shaft_end_1.tolist(),
                "joint_s":joint_s_1.tolist(),
                "joint_t":joint_t_1.tolist(),
                "tip1":tip1_1.tolist(),
                "tip2":tip2_1.tolist()
            },
            "aca3_keypoints":
            {
                "shaft_start":shaft_start_1.tolist(), 
                "shaft_end":shaft_end_1.tolist(),
                "joint_s":joint_s_1.tolist(),
                "joint_t":joint_t_1.tolist(),
                "tip1":tip1_1.tolist(),
                "tip2":tip2_1.tolist()
            },
            "aca4_keypoints":
            {
                "shaft_start":shaft_start_1.tolist(), 
                "shaft_end":shaft_end_1.tolist(),
                "joint_s":joint_s_1.tolist(),
                "joint_t":joint_t_1.tolist(),
                "tip1":tip1_1.tolist(),
                "tip2":tip2_1.tolist()
            }})
        # output.instance.set_dictionary({"six_dof_pose":np.array([1, 2.0, 3.345345, 63.3412,
        #                                                     23.4, 12, 23, 5,
        #                                                     1, 2, 3, 4,
        #                                                     4, 5, 6, 7]).reshape(-1).tolist()})
        writer.write()

        sleep(0.5) # Write at a rate of one sample every 0.5 seconds, for ex.

    print("Exiting...")
    writer.wait() # Wait for all subscriptions to receive the data before exiting

# Close Connectors

def read_sra_joints(input: rti.Input):
    print("Waiting for data...")
    input.wait()  # wait for data on this input
    input.take()  # read a sample of data

    for sample in input.samples.valid_data_iter:
        data = sample.get_dictionary()  # get all fields
        q = data['q']  # isolate joint angles
        print("Received q: " + repr(q))


if __name__ == "__main__":
    keypoint_writer = init_connector("HugoParticipantLibrary::KeypointParticipant", "Pub::KeypointWriter", is_publisher=True)
    # pose_connector = rti.Connector(
    #     config_name="HugoParticipantLibrary::SixDofPoseParticipant",
    #     url=file_path + "/../NddsQosProfiles.xml")

    # sra1_reader = init_connector("HugoParticipantLibrary::SraJointsReader_Arm1", "Sub::SraReader")
    # sra2_reader = init_connector("HugoParticipantLibrary::SraJointsReader_Arm2", "Sub::SraReader")
    # sra3_reader = init_connector("HugoParticipantLibrary::SraJointsReader_Arm3", "Sub::SraReader")
    # sra4_reader = init_connector("HugoParticipantLibrary::SraJointsReader_Arm4", "Sub::SraReader")
    # read_sra_joints(sra1_reader)
    # read_sra_joints(sra2_reader)
    # Get RCM Estimate from DDS Messages

    # Get RCM Estimate from Keypoints

    # Correlate Keypoint Structure to ACA

    # Publish over DDS to Unity
    # Put this on While Loop
    publish_keypoints(keypoint_writer)

    keypoint_writer.connector.close()
    # sra1_reader.connector.close()
    # sra2_reader.connector.close()
    # sra3_reader.connector.close()
    # sra4_reader.connector.close()