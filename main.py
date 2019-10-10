from detect_process import process
import cv2 as cv
import argparse

def get_arguments():

    parser = argparse.ArgumentParser(description='Object Detection and Tracking on Video Streams')
    
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    parser.add_argument('--output', help='Path to save output as video file. If nothing is given, the output will not be saved.')

    parser.add_argument('--model', required=True,
                        help='Path to a binary file of model contains trained weights. '
                             'It could be a file with extensions .caffemodel (Caffe), '
                             '.weights (Darknet)')
    
    parser.add_argument('--config',
                        help='Path to a text file of model contains network configuration. '
                             'It could be a file with extensions .prototxt (Caffe), .cfg (Darknet)')
    
    parser.add_argument('--classes', help='Optional path to a text file with names of classes to label detected objects.')
    
    parser.add_argument('--thr', type=float, default=0.35, help='Confidence threshold for detection')
    
    return parser.parse_args()

def main():
    
    args = get_arguments()

    process(args)

if __name__ == '__main__':
    main()