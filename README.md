# ObjectTrackingSurveillance

This is a real-time Object Tracking and Surveillance Project using C++ and OpenCV. It uses the YOLOv3-tiny model for object detection and tracks multiple moving objects like people and vehicles using the Kalman filter. The system is capable of processing live webcam input or video files, displaying object identity with motion tracking, and saving the output.  

## Features

- Real-time object detection using YOLOv3-tiny (via OpenCV DNN module)
- Tracks multiple object types: person, car, bus, truck, bicycle, and motorbike
- Saves the output video to file
- Supports both video file input and live webcam feed
- FPS monitoring and pause/resume support

## Requirements

- C++ 17 compatible compiler
- OpenCV
- CMake

## File structure

ObjectTrackingSurveillance/
├── include/
│   ├── Detector.hpp
│   └── Tracker.hpp
├── src/
│   ├── main.cpp
│   ├── detector.cpp
│   └── Tracker.cpp
├── weights/
│   ├── yolov3-tiny.cfg
│   ├── yolov3-tiny.weights
│   └── coco.names
├── recordings/
│   └── [output video will be saved here]
├── CMakeLists.txt
└── README.md

## How to build

mkdir build && cd build
cmake ..
cmake --build .

## How to Run

After building, run the executable:
./ObjectTrackingSurveillance

You will be prompted to enter the path to a video file.
Leave it empty to use the default webcam.

## Important Notes

I have hardcoded YOLO config, weights, and class names files along with the output video path are currently in main.cpp to keep it simple while experimenting locally. If running on a differnt system, you must:
Ensure the weights/ and recordings/ folders exist.
Update the paths inside main.cpp
