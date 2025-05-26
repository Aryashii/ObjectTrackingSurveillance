cmake_minimum_required(VERSION 3.10)
project(ObjectTrackingSurveillance)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "C:/Users/KIIT/Desktop/opencv/build")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include include/doctest)

add_executable(ObjectTrackingSurveillance
                src/main.cpp
                src/Detector.cpp
                src/Tracker.cpp
              )

target_link_libraries(ObjectTrackingSurveillance ${OpenCV_LIBS})

add_executable( test_tracker test/test_tracker.cpp  src/Tracker.cpp)
target_link_libraries(test_tracker ${OpenCV_LIBS})
