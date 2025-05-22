#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <opencv2/opencv.hpp>

class Detector{
    public:
    Detector(const std::string& cfg, const std::string& weights, const std::string& names);
    std::vector<cv::Rect> detect(const cv::Mat& frame, std::vector<std::string>& classNames);

    private:
    cv::dnn::Net net;
    std::vector<std::string> outputLayerNames;
    std::vector<std::string> classList;

    float confThreshold = 0.5;
    float nmsThreshold = 0.4;
    
};

#endif