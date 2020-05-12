from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np

from abc import ABCMeta, abstractmethod

class Model(metaclass=ABCMeta):
    '''
    General Model class to create individual model sub-classes
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
        self.net = None


    def load_model(self):

        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)


    def preprocess_input(self, image, input_shape):

        input_width, input_height = input_shape[3], input_shape[2]
        image = cv2.resize(image, (input_width,input_height), interpolation = cv2.INTER_AREA)
        image = np.moveaxis(image, -1, 0)
        
        return image
    
    
    @abstractmethod
    def get_coords(self):
        pass
        
    
    @abstractmethod
    def predict(self):
        pass