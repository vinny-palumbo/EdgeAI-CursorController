from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np

class GazeEstimationModel:
    '''
    Class for the Gaze Estimation Model.
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
        self.input_shape_eye_image = self.model.inputs['left_eye_image'].shape
        self.input_shape_head_pose_coords = self.model.inputs['head_pose_angles'].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        self.net = None


    def load_model(self):

        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)


    def preprocess_image_eye(self, image_eye):

        input_width_eye_image, input_height_eye_image = self.input_shape_eye_image[3], self.input_shape_eye_image[2]
        image_eye = cv2.resize(image_eye, (input_width_eye_image, input_height_eye_image), interpolation = cv2.INTER_AREA)
        image_eye = np.moveaxis(image_eye, -1, 0)
        
        return image_eye
    

    def get_coords(self, outputs):
        
        coords = np.squeeze(outputs[self.output_name]).tolist()
        
        return coords
        
        
    def predict(self, image_eye_left, image_eye_right, head_pose_coords):

        image_eye_left_p = self.preprocess_image_eye(image_eye_left.copy())
        image_eye_right_p = self.preprocess_image_eye(image_eye_right.copy())
        
        input_dict={'left_eye_image': image_eye_left_p, 
                    'right_eye_image': image_eye_right_p,
                    'head_pose_angles': head_pose_coords}
                    
        outputs = self.net.infer(input_dict)
        coords = self.get_coords(outputs)
        
        return coords
        