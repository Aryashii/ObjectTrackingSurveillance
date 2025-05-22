#include "Tracker.hpp"
#include <cmath>
#include <iostream>

Tracker::Tracker() {}

cv::KalmanFilter Tracker::createKalmanFilter(const cv::Point2f& pt){
    cv::KalmanFilter kf(4,2,0);
    //Setting up the transition matrix to model constant velocity motion
    kf.transitionMatrix = (cv::Mat_<float>(4,4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1
        );

    kf.measurementMatrix = cv::Mat::eye(2,4, CV_32F);
    kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-2;
    kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 1e-1;
    kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F);

    kf.statePost.at<float>(0) = pt.x;
    kf.statePost.at<float>(1) = pt.y;
    kf.statePost.at<float>(2) = 0;
    kf.statePost.at<float>(3) = 0;
    return kf;
}

int Tracker::matchObject(const cv::Point2f& detectionPt) {
    for(auto& [id, obj] : objects) {
        float dist = cv::norm(obj.last_position - detectionPt);
        if(dist < distanceThreshold) {
            return id;
        }
    }
    return -1;
}

void Tracker::update(const std::vector<cv::Rect>& detections, const std::vector<std::string>& labels, double fps){
    std::set<int> updated;

    //Match new detections
    for(size_t i = 0; i < detections.size(); ++i){
        cv::Point2f center(
            detections[i].x + detections[i].width / 2.0f,
            detections[i].y + detections[i].height / 2.0f
        );

        int match_id = matchObject(center);
        if(match_id == -1){
            //New object detected
            TrackedObject obj;
            obj.id = next_id++;
            obj.label = labels[i];
            obj.kf = createKalmanFilter(center);
            obj.last_position = center;
            obj.bounding_box = detections[i]; 
            obj.trajectory.push_back(center);
            obj.velocity = 0.0f;
            objects[obj.id] = obj;
        }
        else{
            //Existing Object
            auto& obj = objects[match_id];
            cv::Mat meas = (cv::Mat_<float>(2,1) << center.x, center.y);
            obj.kf.correct(meas);
            obj.last_position = center;
            obj.bounding_box = detections[i]; 
            obj.trajectory.push_back(center);

            //Computing Velocity 
            if (obj.trajectory.size() >= 2) {
                cv::Point2f p1 = obj.trajectory[obj.trajectory.size() - 2];
                cv::Point2f p2 = obj.trajectory.back();
                float dist = cv::norm(p2 - p1);
                obj.velocity = dist * fps;       
            }

            obj.missing_frames = 0;
            updated.insert(match_id);
        }
    }

    //Predict position for missing detections and mark them for removal if lost
    std::vector<int> to_remove;
    for(auto& [id, obj] : objects) {
        if(updated.find(id) == updated.end()){
            //Object not detetcted, predict new position
            obj.kf.predict();
            obj.last_position = cv::Point2f(
                obj.kf.statePost.at<float>(0),
                obj.kf.statePost.at<float>(1)
            );

            obj.trajectory.push_back(obj.last_position);

            if(obj.trajectory.size() >= 2){
                cv::Point2f p1 = obj.trajectory[obj.trajectory.size() - 2];
                cv::Point2f p2 = obj.trajectory.back();
                float dist = cv::norm(p2 -p1);
                obj.velocity = dist * fps;
            }

            obj.missing_frames++;

            //Removing object if missing for more than 20 frames
            if(obj.missing_frames > 20){
                to_remove.push_back(id);
            }
        }
    }   

    for(int id : to_remove){
        objects.erase(id);
    }
    
}

void Tracker::draw(cv::Mat& frame){
    //Define restricted zone 
    cv::Rect restrictedZone(200,200,150,200);
    cv::Scalar zoneColor(0,0,255);
    cv::rectangle(frame, restrictedZone, zoneColor, 2);
    
    //Iterate through all detected objects to draw their info
    for(const auto& [id, obj] : objects){
        //Draw current position as filled circle
        cv::circle(frame, obj.last_position, 5, cv::Scalar(0,255,255), -1);
        
        //Drawing trajectory lines
        for(size_t i = 1; i < obj.trajectory.size(); ++i){
            cv::line(frame, obj.trajectory[i-1], obj.trajectory[i], cv::Scalar(255, 0, 0), 1);
        }

        std::string labelText = obj.label + " ID: " + std::to_string(id) + " v: " + std::to_string(static_cast<int>(obj.velocity)) + " px/s";

        cv::putText(frame, labelText, obj.last_position + cv::Point2f(5,-5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255,255,255), 1.8);

        // Colored bounding box
        cv::Scalar color;
        if (obj.label == "person") {
            color = cv::Scalar(255, 0, 255);  //Pink
        } else {
            color = cv::Scalar(255, 0, 0);    //Blue
        }
        cv::rectangle(frame, obj.bounding_box, color, 2);


        // Speed Alert
        if(obj.velocity > 600.0f){
            std::cout << "Alert! Object #" << id << " exceeding speed limit!\n";
            cv::putText(frame, "Over Speed!", obj.last_position + cv::Point2f(0, -15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.9, cv::Scalar(0,0,255), 2);
        }

        //Alert if inside zone
        if(restrictedZone.contains(obj.last_position)){
            std::cout << "ALERT! Object #" << id << " entered restricted Zone!\n";
            cv::putText(frame, "Restricted Area!", obj.last_position + cv::Point2f(5,15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 1.9);
        }
    }
}