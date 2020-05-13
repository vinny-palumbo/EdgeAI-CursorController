import os
import sys
import time
import cv2
import logging

from argparse import ArgumentParser
from sys import platform

from mouse_controller import MouseController
from model_face_detection import FaceDetectionModel
from model_facial_landmarks_detection import FacialLandmarksDetectionModel
from model_head_pose_estimation import HeadPoseEstimationModel
from model_gaze_estimation import GazeEstimationModel


logging.basicConfig(filename='logs/logs.txt',level=logging.DEBUG)

# Get correct params according to the OS
if platform in ["darwin", "linux", "linux2"] # for MACs and Linux
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
else: # Windows
    CODEC = 0x00000021


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or CAM for webcam")
    parser.add_argument("-m", "--models_path", type=str, default="models/intel",
                        help="Path to models folder")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD is acceptable (CPU by default)")
    parser.add_argument("-p", "--precision", type=str, default="FP32",
                        help="Precision of the Intermediate Representation models (FP32 by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for face detections filtering (0.5 by default)")
    parser.add_argument("-o", "--output", type=bool, default=False,
                        help="Choose whether to generate an annotated output video or not (False by default)")

    return parser
    
    
def infer_on_stream(args, mouse_controller):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    
    PATH_MODEL_FACE_DETECTION = os.path.join(args.models_path, r'face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001')
    PATH_MODEL_FACIAL_LANDMARKS_DETECTION = os.path.join(args.models_path, r'landmarks-regression-retail-0009\{}\landmarks-regression-retail-0009'.format(args.precision))
    PATH_MODEL_HEAD_POSE_ESTIMATION = os.path.join(args.models_path, r'head-pose-estimation-adas-0001\{}\head-pose-estimation-adas-0001'.format(args.precision))
    PATH_MODEL_GAZE_ESTIMATION = os.path.join(args.models_path, r'gaze-estimation-adas-0002\{}\gaze-estimation-adas-0002'.format(args.precision))

    # Initialise the models
    model_face_detection = FaceDetectionModel(PATH_MODEL_FACE_DETECTION, args.device, args.prob_threshold)
    model_facial_landmarks_detection = FacialLandmarksDetectionModel(PATH_MODEL_FACIAL_LANDMARKS_DETECTION, args.device)
    model_head_pose_estimation = HeadPoseEstimationModel(PATH_MODEL_HEAD_POSE_ESTIMATION, args.device)
    model_gaze_estimation = GazeEstimationModel(PATH_MODEL_GAZE_ESTIMATION, args.device)
    
    # Load the models
    start_models_load_time=time.time()
    model_face_detection.load_model()
    model_facial_landmarks_detection.load_model()
    model_head_pose_estimation.load_model()
    model_gaze_estimation.load_model()
    total_models_load_time = round(time.time() - start_models_load_time, 2)
    
    # Check if the input is a webcam
    if args.input == 'CAM':
        args.input = 0
    
    # Handle the input stream 
    try:
        cap=cv2.VideoCapture(args.input)
    except FileNotFoundError:
        print("Cannot locate video file: "+ args.input)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    cap.open(args.input)
    CAP_WIDTH = int(cap.get(3))
    CAP_HEIGHT = int(cap.get(4))
    CAP_FPS = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create a video writer for the output video
    if args.output:
        out = cv2.VideoWriter('results/output_video.mp4', CODEC, CAP_FPS, (CAP_WIDTH,CAP_HEIGHT))
    
    # Loop until stream is over 
    counter=0
    start_inference_time=time.time()
    try:
        while cap.isOpened():
            
            # Read from the video capture 
            flag, frame = cap.read()
            if not flag:
                break
            counter+=1
            key_pressed = cv2.waitKey(60)
            
            # Get face detection crops
            face_coords = model_face_detection.predict(frame)
            if len(face_coords) > 1:
                logging.info('Multiple faces detected')
                if args.output:
                    out.write(frame)
                continue
            elif len(face_coords) == 0:
                logging.info('No face detected')
                if args.output:
                    out.write(frame)
                continue
            
            # crop face
            face_coords = face_coords[0]
            face_xmin, face_ymin, face_xmax, face_ymax = face_coords
            image_face = frame.copy()[face_ymin:face_ymax, face_xmin:face_xmax]
            
            # Get the landmarks from the face crop
            facial_landmarks_coords_crop = model_facial_landmarks_detection.predict(image_face)
            
            # get eyes crops for the gaze estimation model
            images_eyes = model_facial_landmarks_detection.crop_eyes(facial_landmarks_coords_crop, face_coords, frame)
            image_eye_left, image_eye_right = images_eyes
            
            # Get the head pose estimation from the face crop
            head_pose_coords = model_head_pose_estimation.predict(image_face)
            
            # Get gaze estimation coords
            gaze_coords = model_gaze_estimation.predict(image_eye_left, image_eye_right, head_pose_coords)
            gaze_x, gaze_y, gaze_z = gaze_coords
            
            # Move mouse
            mouse_controller.move(-gaze_y, gaze_x) # the y coord is reversed
            
            if args.output:
                
                # draw face detection bbox on original frame
                frame_out = model_face_detection.draw_face(face_coords, frame)
                
                # draw eyes
                frame_out = model_facial_landmarks_detection.draw_eyes(facial_landmarks_coords_crop, face_coords, frame_out)
                    
                # draw gaze direction   
                frame_out = model_gaze_estimation.draw_gaze_direction(gaze_coords, face_coords, frame_out)
                 
                # Write out the output frame 
                out.write(frame_out)
            
            # Break if escape key pressed
            if key_pressed == 27:
                break
        
        # log stats
        total_inference_time = round(time.time()-start_inference_time, 2)
        fps = round(counter/total_inference_time, 2)
        
        log_filename = 'logs/{}_{}.txt'.format(args.device, args.precision)
        with open(log_filename, 'w') as f:
            f.write('total_inference_time: ' + str(total_inference_time)+'\n')
            f.write('fps: ' + str(fps)+'\n')
            f.write('total_models_load_time: ' + str(total_models_load_time)+'\n')

        # Release the capture and destroy any OpenCV windows
        if args.output:
            out.release()
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print("Could not run Inference: ", e)


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # get mouse controller
    mc = MouseController()
    # Perform inference on the input stream
    infer_on_stream(args, mc)


if __name__ == '__main__':
    main()
