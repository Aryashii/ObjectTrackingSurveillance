#include <opencv2/opencv.hpp>
#include "Detector.hpp"
#include "Tracker.hpp"

int main(){
    Detector detector(
        "C:/Users/KIIT/Desktop/ObjectTrackingSurveillance/weights/yolov3-tiny.cfg", 
        "C:/Users/KIIT/Desktop/ObjectTrackingSurveillance/weights/yolov3-tiny.weights", 
        "C:/Users/KIIT/Desktop/ObjectTrackingSurveillance/weights/coco.names"
    );
    Tracker tracker;

    bool paused = false;
    std::string inputPath;
    cv::VideoCapture cap;

    std::cout << "Enter path to video file (leave empty to use webcam): "; 
    std::getline(std::cin, inputPath);
    
    if(!inputPath.empty()){
        cap.open(inputPath);
        std::cout << "Openeing Video File : " << inputPath << std::endl;
    }
    else{
        cap.open(0);
        std::cout << "Opening Default webcam...\n";
    }

    if(!cap.isOpened()){
        std::cerr << "Could not open input source!\n";
        return -1;
    }

    cv::VideoWriter writer;
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    int fps = 30;
    cv::Size frameSize(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    writer.open("C:/Users/KIIT/Desktop/ObjectTrackingSurveillance/recordings/output.avi", codec, fps, frameSize, true);

    if(!writer.isOpened()){
        std::cerr << "Error : Could not open file video for writing\n";
        return -1;
    }

    cv::namedWindow("Detection", cv::WINDOW_NORMAL);

    while(true){
        cv::Mat frame;
        cap >> frame;
        if(frame.empty()) break;

        int64 start = cv::getTickCount();

        std::vector<std::string> classNames;
        std::vector<cv::Rect> boxes = detector.detect(frame, classNames);


        std::vector<cv::Rect> filtered;
        std::vector<std::string> flabels;
        for(size_t i = 0; i < boxes.size(); ++i){
            const std::string& label = classNames[i];

            if(label == "person" || label == "car" || label == "bus" || label == "truck" || label == "bicycle" || label == "motorbike"){
                // cv::rectangle(frame, boxes[i], cv::Scalar(0,255,0), 2);
                // cv::putText(frame, label, boxes[i].tl(), cv::FONT_HERSHEY_SIMPLEX , 0.9, cv::Scalar(0,0,255), 1);

                filtered.push_back(boxes[i]);
                flabels.push_back(classNames[i]);
            }
        }
        int64 end = cv::getTickCount();
        double timeMs = (end - start) * 1000.0 / cv::getTickFrequency();
        double realFps = 1000.0 / timeMs; 

        tracker.update(filtered, flabels, realFps);
        tracker.draw(frame);
    

        writer.write(frame);

        cv::imshow("Detection", frame);

        int key = cv::waitKey(paused ? 0 : 1);
        if (key == 'p' || key == 'P') {
            paused = !paused;
        }
        if (key == 'q' || key == 'Q') {
            break;
        }

        if (paused) {
            continue;
        }
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}
