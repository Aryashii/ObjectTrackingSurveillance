#include "Detector.hpp"
#include <fstream>
#include <iostream>

Detector::Detector(const std::string& cfg, const std::string& weights, const std::string& names){
    net = cv::dnn::readNetFromDarknet(cfg, weights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    
    std::ifstream ifs(names);
    if (!ifs.is_open()) {
        std::cerr << "Error: Could not open class names file: " << names << std::endl;
        return;
    }
    std::string line;
    while(std::getline(ifs, line))
        if (!line.empty())
            classList.push_back(line);

    auto outNames = net.getUnconnectedOutLayersNames();
    outputLayerNames = std::vector<std::string>(outNames.begin(), outNames.end());
        
}

std::vector<cv::Rect> Detector::detect(const cv::Mat& frame, std::vector<std::string>& classNames){
    std::vector<cv::Rect> boxes;
    classNames.clear();

    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(416,416), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, outputLayerNames);

    std::vector<int> classIds;
    std::vector<float> confidences;

    for(const auto& output : outputs){
        for(int i = 0; i < output.rows; ++i){
            auto data = output.ptr<float>(i);
            float score = data[4];
            if (score > confThreshold){
                cv::Mat scores = output.row(i).colRange(5, output.cols);
                cv::Point classIdPoint;
                double confidence;
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if(confidence > confThreshold){
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);

                    //Get the top-left corner of the bounding box
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back((float)confidence);
                    classIds.push_back(classIdPoint.x);

                }
            }
        }
    }

    //Apply NMS to remove overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    
    std::vector<cv::Rect> result;
    for(int idx : indices){
        result.push_back(boxes[idx]);
        classNames.push_back(classList[classIds[idx]]);
    }

    return result;
}