import os
import sys
import time
import cv2

from argparse import ArgumentParser
from sys import platform

from input_feeder import InputFeeder
from model_face_detection import FaceDetectionModel
from model_facial_landmarks_detection import FacialLandmarksDetectionModel
from model_head_pose_estimation import HeadPoseEstimationModel


PATH_MODELS_FOLDER = os.path.abspath(r'C:\Users\vin_p\Github\EdgeAI-CursorController\models\intel')
PATH_MODEL_FACE_DETECTION = os.path.join(PATH_MODELS_FOLDER, r'face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001')
PATH_MODEL_FACIAL_LANDMARKS_DETECTION = os.path.join(PATH_MODELS_FOLDER, r'landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009')
PATH_MODEL_HEAD_POSE_ESTIMATION = os.path.join(PATH_MODELS_FOLDER, r'head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001')

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
    model_head_pose_estimation = HeadPoseEstimationModel(PATH_MODEL_HEAD_POSE_ESTIMATION, args.device)
    
    # Load the models
    model_face_detection.load_model()
    model_facial_landmarks_detection.load_model()
    model_head_pose_estimation.load_model()
    
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
        image_face = frame.copy()[face_ymin:face_ymax, face_xmin:face_xmax]
        
        # Get the landmarks from the face crop
        facial_landmarks_coords_crop = model_facial_landmarks_detection.predict(image_face)
        
        # get eyes crops for the gaze estimation model
        images_eyes = model_facial_landmarks_detection.crop_eyes(facial_landmarks_coords_crop, face_coords, frame_out)
        image_eye_left, image_eye_right = images_eyes
        
        # draw facial landmarks
        frame_out = model_facial_landmarks_detection.draw_landmarks(facial_landmarks_coords_crop, face_coords, frame_out)
        
        # Get the head pose estimation from the face crop
        head_pose_coords = model_head_pose_estimation.predict(image_face)
        
        # TODO: draw head pose on face
        

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
