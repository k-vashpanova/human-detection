"""
Human Detection in Videos using YOLO

This module provides functionality to detect humans in video files using YOLO
object detection models and output the processed video with bounding boxes,
labels and confidence scores.

Features:
    - Human detection using YOLO models from Ultralytics
    - Support for various YOLO model sizes

Example:
    >>> from detect_humans import detect_humans_on_video
    >>> detect_humans_on_video('input.mp4', 'output.avi', 'yolo11n.pt')

Command Line Usage:
    detect_humans <filename> [--options]

Options:
    -m, --model PATH   Path to detection model [default: yolo11n.pt]
    -o, --output FILE  Output file name [default: output.avi]
"""

import os
import argparse
from tqdm import tqdm
import cv2
from ultralytics import YOLO


def get_video_params(file_path):
    """
    Extract video parameters and metadata from a video file.

    This function reads a video file and extracts essential parameters
    needed for video processing and output, including codec, frame rate,
    and frame dimensions.

    Args:
        file_path (str): Path to the input video file.

    Returns:
        tuple: A tuple containing:
            - video_params (dict): Dictionary with video parameters:
                - 'fourcc': Video codec (FourCC code)
                - 'fps': Frames per second
                - 'frameSize': Tuple of (width, height) in pixels
            - num_frames (int): Total number of frames in the video

    Raises:
        ValueError: If the video file cannot be opened.

    Example:
        >>> params, num_frames = get_video_params('input.mp4')
        >>> print(f"FPS: {params['fps']}, Frames: {num_frames}")
    """
    vid = cv2.VideoCapture(file_path)
    if not vid.isOpened():
        raise ValueError(f"Cannot open video file: {file_path}")

    return {
        'fourcc': cv2.VideoWriter_fourcc(*'XVID'),  # Codec for .avi files
        'fps': vid.get(cv2.CAP_PROP_FPS),
        'frameSize': (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    }, int(vid.get(cv2.CAP_PROP_FRAME_COUNT))


def detect_humans_on_video(file_path,
                           output_filename='output.avi',
                           model_path='yolo11n.pt'):
    """
    Process a video file to detect humans and draw bounding boxes.

    The main function that performs human detection on video frames using
    a YOLO model. It processes each frame, detects humans, draws bounding
    boxes with labels and confidence scores, and writes the results to an
    output video file.

    Args:
        file_path (str): Path to the input video file.
        output_filename (str, optional): Name of the output video file.
            Must have .avi extension. Defaults to 'output.avi'.
        model_path (str, optional): Path to the YOLO model weights file.
            Defaults to 'yolo11n.pt'.

    Returns:
        None: The processed video is saved to the specified output file.

    Raises:
        ValueError: If output filename doesn't have .avi extension.
        FileNotFoundError: If input video file cannot be found.

    Note:
        - Processing progress is displayed with a progress bar.

    Example:
        >>> detect_humans_on_video('input.mp4', 'output.avi', 'yolo11s.pt')
        Video 'output.avi' created successfully.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input video not found: {file_path}")

    if not output_filename.lower().endswith('.avi'):
        raise ValueError("Output file must have .avi extension")

    # Detection
    model = YOLO(model_path, task='detect')
    results = model(file_path, stream=True, classes=[0], verbose=False)

    # Writing the output video
    video_params, num_frames = get_video_params(file_path)
    out = cv2.VideoWriter(output_filename, **video_params)
    for r in tqdm(results,
                  total=num_frames,
                  desc='Processing video: '):
        frame = r.plot(labels=True,
                       probs=True,
                       show=False,
                       font='Calibri.tiff')
        out.write(frame)
    out.release()

    print(f"Video '{output_filename}' created successfully.")


def main():
    """
    Command-line entry point for human detection in videos.
    """
    parser = argparse.ArgumentParser(description='Detect humans in video')
    parser.add_argument('filename', help='Path to input video file')
    parser.add_argument('-m', '--model', default='yolo11n.pt',
                        help='Path to detection model')
    parser.add_argument('-o', '--output', default='output.avi',
                        help='Output file name')

    args = parser.parse_args()

    detect_humans_on_video(
        file_path=args.filename,
        output_filename=args.output,
        model_path=args.model
    )

if __name__ == '__main__':
    main()


