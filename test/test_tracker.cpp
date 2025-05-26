#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "Tracker.hpp"
#include "doctest.h"
#include <opencv2/opencv.hpp>

TEST_CASE("Test tracker with dummy object"){
    Tracker tracker;
    std::vector<cv::Rect> boxes = { cv::Rect(10, 10,100,100)};
    std::vector<std::string> labels = {"person"};
    double fps = 30.0;
    tracker.update(boxes, labels, fps);
    CHECK(true);
}

TEST_CASE("Test tracker works with zero objects"){
    Tracker tracker;
    std::vector<cv::Rect> boxes;
    std::vector<std::string> labels;
    double fps = 30.0;
    CHECK_NOTHROW(tracker.update(boxes, labels, fps));
}

TEST_CASE("Test tracker handles high fps input"){
    Tracker tracker;
    std::vector<cv::Rect> boxes = { cv::Rect(15, 15, 50, 50)};
    std::vector<std::string> labels = {"person"};
    double fps = 1000.0;
    CHECK_NOTHROW(tracker.update(boxes, labels, fps));
}