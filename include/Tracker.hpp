#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <opencv2/opencv.hpp>
#include <map>

struct TrackedObject {
    int id;
    std::string label;
    cv::KalmanFilter kf;
    cv::Point2f last_position;
    std::vector<cv::Point2f> trajectory;
    int missing_frames = 0;
    float velocity = 0.0f;

    cv::Rect bounding_box;  // NEW: stores the actual bounding box
};

class Tracker{
    public:
    Tracker();
    void update(const std::vector<cv::Rect>& detections, const std::vector<std::string>& labels, double fps);
    void draw(cv::Mat& frame);

    private:
    std::map<int, TrackedObject> objects;
    int next_id = 0;
    float distanceThreshold = 60.0f;

    int matchObject(const cv::Point2f& detectionPt);
    cv::KalmanFilter createKalmanFilter(const cv::Point2f& pt);
};

#endif