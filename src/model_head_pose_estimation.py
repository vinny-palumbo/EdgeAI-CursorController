from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np

class HeadPoseEstimationModel:
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device):
    
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.core = IECore()
        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.net = None


    def load_model(self):

        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)


    def preprocess_input(self, image):

        input_width, input_height = self.input_shape[3], self.input_shape[2]
        image = cv2.resize(image, (input_width,input_height), interpolation = cv2.INTER_AREA)
        image = np.moveaxis(image, -1, 0)
        
        return image
    

    def get_coords(self, outputs):
        
        angle_p_fc = np.squeeze(outputs['angle_p_fc']).tolist()
        angle_r_fc = np.squeeze(outputs['angle_r_fc']).tolist()
        angle_y_fc = np.squeeze(outputs['angle_y_fc']).tolist()
        
        coords = (angle_p_fc, angle_r_fc, angle_y_fc)
        
        return coords
        
        
    def predict(self, image):

        image_p = self.preprocess_input(image.copy())
        input_dict={self.input_name: image_p}
        outputs = self.net.infer(input_dict)
        coords = self.get_coords(outputs)
        
        return coords
        