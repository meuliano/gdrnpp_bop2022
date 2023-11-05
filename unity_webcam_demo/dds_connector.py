# pip-install rticonnextdds-connector
from os import path as os_path
file_path = os_path.dirname(os_path.realpath(__file__))
import rticonnextdds_connector as rti

class DDSConnector:
    def __init__(self, participant_str):
        self.connector = rti.Connector(
            config_name="HugoParticipantLibrary::" + participant_str,
            url=file_path + "/NddsQosProfiles.xml")
        
    def shutdown(self):
        self.connector.close()

class DDSWriter(DDSConnector):
    def __init__(self, participant_str, pub_str):
        super().__init__(participant_str)
        self.output = self.connector.get_output("Pub::" + pub_str)
        print("Waiting for subscriptions...")
        self.output.wait_for_subscriptions()
        print("Subscriber Established")
    
    def write_dict(self, pub_dict: dict):
        print("Writing data...")
        self.output.instance.set_dictionary(pub_dict)
        self.output.write()
        self.output.wait() # Wait for all subscriptions to receive the data before exiting

class DDSReader(DDSConnector):
    def __init__(self, participant_str, sub_str):
        super().__init__(participant_str)
        self.input = self.connector.get_input("Sub::" + sub_str)
        print("Waiting for publications...")
        self.input.wait_for_publications()  # wait for at least one matching publication
        print("Publisher Established")

    def read_dict(self):
        print("Waiting for data...")
        self.input.wait()  # wait for data on this input
        self.input.take()  # read a sample of data
        for sample in self.input.samples.valid_data_iter:
            data = sample.get_dictionary()  # get all fields
        return data