from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np

class FaceDetectionModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device, threshold=0.60):
    
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
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

        detections = outputs[self.output_name][-1,-1,:,:] # outputs shape is 1x1x200x7

        coords= []
        for detection in detections: 
            conf = detection[2]
            if conf >= self.threshold:
                xmin = detection[3]
                ymin = detection[4]
                xmax = detection[5]
                ymax = detection[6]
                
                coords.append([xmin,ymin,xmax,ymax])
        
        return coords
    
    
    def get_coords(self, detections, image):
    
        IMG_HEIGHT, IMG_WIDTH, _ = image.shape
        
        coords = []
        for detection in detections: 
            # scale bbox coords
            xmin = int(detection[0] * IMG_WIDTH)
            ymin = int(detection[1] * IMG_HEIGHT)
            xmax = int(detection[2] * IMG_WIDTH)
            ymax = int(detection[3] * IMG_HEIGHT)
            
            coords.append([xmin,ymin,xmax,ymax])
            
        return coords
        
        
    def predict(self, image):

        image_p = self.preprocess_input(image.copy())
        input_dict={self.input_name: image_p}
        outputs = self.net.infer(input_dict)
        outputs = self.preprocess_outputs(outputs)
        coords = self.get_coords(outputs, image)
        
        return coords
        