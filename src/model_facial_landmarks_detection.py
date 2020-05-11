from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np

class FacialLandmarksDetectionModel:
    '''
    Class for the Facial Landmarks Detection Model.
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
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        self.net = None


    def load_model(self):

        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)


    def preprocess_input(self, image):

        input_width, input_height = self.input_shape[3], self.input_shape[2]
        image = cv2.resize(image, (input_width,input_height), interpolation = cv2.INTER_AREA)
        image = np.moveaxis(image, -1, 0)
        
        return image
    

    def preprocess_outputs(self, outputs):
        
        outputs = np.squeeze(outputs[self.output_name])
        
        return outputs
        
        
    def get_coords(self, outputs, image):
    
        IMG_HEIGHT, IMG_WIDTH, _ = image.shape
    
        x_coords = outputs[0::2] * IMG_WIDTH
        x_coords = [int(i) for i in x_coords]
        
        y_coords = outputs[1::2] * IMG_HEIGHT
        y_coords = [int(i) for i in y_coords]
        
        coords = []
        for x, y in zip(x_coords, y_coords):
            coords.append([x, y])

        return coords
        
        
    def predict(self, image):

        image_p = self.preprocess_input(image.copy())
        input_dict={self.input_name: image_p}
        outputs = self.net.infer(input_dict)
        outputs = self.preprocess_outputs(outputs)
        coords = self.get_coords(outputs, image)
        
        return coords
        
        
    def draw_landmarks(self, landmarks_crop, face_coords, image):

        PAD = 5 # Padding pixel size
        LANDMARKS_COLOR = (0,255,0) # green
        face_xmin, face_ymin, _, _ = face_coords
        
        for landmark_crop in landmarks_crop: 
            
            x_crop = landmark_crop[0]
            y_crop = landmark_crop[1]
            
            x = face_xmin + x_crop
            y = face_ymin + y_crop
            
            # draw landmark point on original frame
            image[y-PAD: y+PAD, x-PAD: x+PAD] = LANDMARKS_COLOR
            
        return image
        
        
    def crop_eyes(self, landmarks_crop, face_coords, image):
        
        PAD = 30 # eye box pixel size
        landmarks_eyes_crop = landmarks_crop[:2]
        face_xmin, face_ymin, _, _ = face_coords
        
        images_eyes = []
        for landmark_eye_crop in landmarks_eyes_crop:
        
            x_crop = landmark_eye_crop[0]
            y_crop = landmark_eye_crop[1]
            
            x = face_xmin + x_crop
            y = face_ymin + y_crop
            
            # crop eye
            image_eye = image.copy()[y-PAD: y+PAD, x-PAD: x+PAD]
            
            images_eyes.append(image_eye)
            
        return tuple(images_eyes)
        