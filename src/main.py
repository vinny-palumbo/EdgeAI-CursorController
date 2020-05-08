import os
import sys
import time
import cv2

from argparse import ArgumentParser
from sys import platform

from input_feeder import InputFeeder
from model_face_detection import FaceDetectionModel

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
    
    # Initialise the class
    path_model_face_detection = os.path.abspath(r'C:\Users\vin_p\Github\EdgeAI-CursorController\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001')
    model_face_detection = FaceDetectionModel(path_model_face_detection, args.device, args.prob_threshold)

    ### TODO: Load the model through `infer_network` ###
    model_face_detection.load_model()
    
    # Check if the input is a webcam
    if args.input == 'CAM':
        args.input = 0
    
    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    CAP_WIDTH = int(cap.get(3))
    CAP_HEIGHT = int(cap.get(4))
    CAP_FPS = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create a video writer for the output video
    out = cv2.VideoWriter('results/output_video.mp4', CODEC, CAP_FPS, (CAP_WIDTH,CAP_HEIGHT))

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        # get output
        _, out_frame = model_face_detection.predict(frame)

        ### TODO: Write out the frame, depending on image or video ###
        out.write(out_frame)
        
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
