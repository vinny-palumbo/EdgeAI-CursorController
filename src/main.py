import os
import sys
import time
import cv2

from argparse import ArgumentParser
from sys import platform

from input_feeder import InputFeeder
from model_face_detection import FaceDetectionModel
from model_facial_landmarks_detection import FacialLandmarksDetectionModel


PATH_MODELS_FOLDER = os.path.abspath(r'C:\Users\vin_p\Github\EdgeAI-CursorController\models\intel')
PATH_MODEL_FACE_DETECTION = os.path.join(PATH_MODELS_FOLDER, r'face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001')
PATH_MODEL_FACIAL_LANDMARKS_DETECTION = os.path.join(PATH_MODELS_FOLDER, r'landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009')

# Get correct params according to the OS
if platform == "darwin": # for MACs
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
else: # Windows and Linux
    CODEC = 0x00000021


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")

    return parser
    

def draw_landmarks(facial_landmarks_coords_crop, face_coords, image):

    PAD = 5 # Padding pixel size
    face_xmin, face_ymin, _, _ = face_coords
    
    for facial_landmark_coords_crop in facial_landmarks_coords_crop: 
        
        facial_landmark_x_crop = facial_landmark_coords_crop[0]
        facial_landmark_y_crop = facial_landmark_coords_crop[1]
        
        facial_landmark_x = face_xmin + facial_landmark_x_crop
        facial_landmark_y = face_ymin + facial_landmark_y_crop
        
        # draw landmark point on original frame
        landmarks_color = (0,255,0) # green
        image[facial_landmark_y-PAD: facial_landmark_y+PAD, facial_landmark_x-PAD: facial_landmark_x+PAD] = landmarks_color
        
    return image
    
    
def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    
    # Initialise the models
    model_face_detection = FaceDetectionModel(PATH_MODEL_FACE_DETECTION, args.device, args.prob_threshold)
    model_facial_landmarks_detection = FacialLandmarksDetectionModel(PATH_MODEL_FACIAL_LANDMARKS_DETECTION, args.device)

    # Load the models
    model_face_detection.load_model()
    model_facial_landmarks_detection.load_model()
    
    # Check if the input is a webcam
    if args.input == 'CAM':
        args.input = 0
    
    # Handle the input stream 
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    CAP_WIDTH = int(cap.get(3))
    CAP_HEIGHT = int(cap.get(4))
    CAP_FPS = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create a video writer for the output video
    out = cv2.VideoWriter('results/output_video.mp4', CODEC, CAP_FPS, (CAP_WIDTH,CAP_HEIGHT))
    
    # Loop until stream is over 
    while cap.isOpened():
        
        # Read from the video capture 
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        # Get face detection crops
        face_coords = model_face_detection.predict(frame)
        if len(face_coords) > 1:
            print('Multiple faces detected')
            quit()
        
        # draw face detection bbox on original frame
        face_coords = face_coords[0]
        face_xmin, face_ymin, face_xmax, face_ymax = face_coords
        frame_out = cv2.rectangle(frame, (face_xmin, face_ymin), (face_xmax, face_ymax), (0, 255, 0), 1)
        
        # Get the landmarks from the face crop
        image_face = frame.copy()[face_ymin:face_ymax, face_xmin:face_xmax]
        facial_landmarks_coords_crop = model_facial_landmarks_detection.predict(image_face)
        
        # draw facial landmarks
        frame_out = draw_landmarks(facial_landmarks_coords_crop, face_coords, frame_out)
        
        # Write out the output frame 
        out.write(frame_out)
        
        # Break if escape key pressed
        if key_pressed == 27:
            break
    
    # Release the capture and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)


if __name__ == '__main__':
    main()
