from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
from model import Model

class GazeEstimationModel(Model):
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device):
    
        super().__init__(model_name, device)
        
        self.input_shape_eye_image = self.model.inputs['left_eye_image'].shape
        self.input_shape_head_pose_coords = self.model.inputs['head_pose_angles'].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
    

    def get_coords(self, outputs):
        
        coords = np.squeeze(outputs[self.output_name]).tolist()
        
        return tuple(coords)
        
        
    def predict(self, image_eye_left, image_eye_right, head_pose_coords):

        image_eye_left_p = self.preprocess_input(image_eye_left.copy(), self.input_shape_eye_image)
        image_eye_right_p = self.preprocess_input(image_eye_right.copy(), self.input_shape_eye_image)
        
        input_dict={'left_eye_image': image_eye_left_p, 
                    'right_eye_image': image_eye_right_p,
                    'head_pose_angles': head_pose_coords}
                    
        outputs = self.net.infer(input_dict)
        coords = self.get_coords(outputs)
        
        return coords
        
        
    def draw_gaze_direction(self, gaze_coords, face_coords, image):
        
        SCALE_FACTOR = 256
        
        gaze_x, gaze_y, _ = gaze_coords
        face_xmin, face_ymin, face_xmax, face_ymax = face_coords
        
        line_start_x = (face_xmin + face_xmax) // 2
        line_start_y = (face_ymin + face_ymax) // 2
        line_end_x = int(line_start_x + gaze_x * SCALE_FACTOR)
        line_end_y = int(line_start_y + gaze_y * SCALE_FACTOR * -1) # the y coord is reversed
        
        image = cv2.arrowedLine(image, 
                                (line_start_x, line_start_y),
                                (line_end_x, line_end_y), 
                                color=[0, 255, 0], 
                                thickness=2) 
                                        
        return image    
        