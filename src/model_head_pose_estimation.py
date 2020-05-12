from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
from model import Model

class HeadPoseEstimationModel(Model):
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device):
    
        super().__init__(model_name, device)
        
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
    

    def get_coords(self, outputs):
        
        angle_p_fc = np.squeeze(outputs['angle_p_fc']).tolist()
        angle_r_fc = np.squeeze(outputs['angle_r_fc']).tolist()
        angle_y_fc = np.squeeze(outputs['angle_y_fc']).tolist()
        
        coords = (angle_p_fc, angle_r_fc, angle_y_fc)
        
        return coords
        
        
    def predict(self, image):

        image_p = self.preprocess_input(image.copy(), self.input_shape)
        input_dict={self.input_name: image_p}
        outputs = self.net.infer(input_dict)
        coords = self.get_coords(outputs)
        
        return coords
        