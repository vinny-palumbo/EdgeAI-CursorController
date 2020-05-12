from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
from model import Model

class FacialLandmarksDetectionModel(Model):
    '''
    Class for the Facial Landmarks Detection Model.
    '''
    def __init__(self, model_name, device):
    
        super().__init__(model_name, device)    
        
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        self.PAD_EYE = 40 # eye box pixel size
        
        
    def get_coords(self, outputs, image):
    
        IMG_HEIGHT, IMG_WIDTH, _ = image.shape
        
        outputs = np.squeeze(outputs[self.output_name])
    
        x_coords = outputs[0::2] * IMG_WIDTH
        x_coords = [int(i) for i in x_coords]
        
        y_coords = outputs[1::2] * IMG_HEIGHT
        y_coords = [int(i) for i in y_coords]
        
        coords = []
        for x, y in zip(x_coords, y_coords):
            coords.append([x, y])

        return coords
        
        
    def predict(self, image):

        image_p = self.preprocess_input(image.copy(), self.input_shape)
        input_dict={self.input_name: image_p}
        outputs = self.net.infer(input_dict)
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
        
        landmarks_eyes_crop = landmarks_crop[:2]
        face_xmin, face_ymin, _, _ = face_coords
        
        images_eyes = []
        for landmark_eye_crop in landmarks_eyes_crop:
        
            x_crop = landmark_eye_crop[0]
            y_crop = landmark_eye_crop[1]
            
            x = face_xmin + x_crop
            y = face_ymin + y_crop
            
            # crop eye
            image_eye = image.copy()[y-self.PAD_EYE: y+self.PAD_EYE, x-self.PAD_EYE: x+self.PAD_EYE]
            
            images_eyes.append(image_eye)
            
        return tuple(images_eyes)
        
        
    def draw_eyes(self, landmarks_crop, face_coords, image):
    
        landmarks_eyes_crop = landmarks_crop[:2]
        face_xmin, face_ymin, _, _ = face_coords
        
        for landmark_eye_crop in landmarks_eyes_crop:
        
            x_crop = landmark_eye_crop[0]
            y_crop = landmark_eye_crop[1]
            
            x = face_xmin + x_crop
            y = face_ymin + y_crop
            
            # draw bbox
            cv2.rectangle(image, (x-self.PAD_EYE, y-self.PAD_EYE), (x+self.PAD_EYE, y+self.PAD_EYE), (0, 255, 0), 1)
            
        return image
    
        