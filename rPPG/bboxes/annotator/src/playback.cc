#include "playback.h"
#include "labeller.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <thread>

namespace frontend {
    struct playback_impl {
        std::vector<cv::Mat> images;
        int fps;
        double reportedDuration;
        std::size_t frameNum = 0;
    };

    playback::playback(const std::filesystem::path& video, std::size_t frameCap) {
        data = std::shared_ptr<playback_impl>(new playback_impl);
        cv::VideoCapture cap(video);
        if(!cap.isOpened()) {
            throw std::runtime_error("Error loading video " + video.string());
        }
        data->fps = cap.get(cv::CAP_PROP_FPS);
        while(frameCap == 0 or data->images.size() < frameCap) {
            cv::Mat frame;
            cap >> frame;
            if(frame.empty()) break;
            data->images.push_back(frame);
        }
        data->reportedDuration = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        std::cout << "Reported duration: " << data->reportedDuration << " seconds" << std::endl;
        cap.release();
    }

    void playback::display(const std::string& windowName, const backend::rect &r) const {
        cv::Mat curr = data->images[data->frameNum].clone();
        if(r.nonzero()) {
            // Update with the rectangle
            cv::rectangle(curr, cv::Point(r.x1, r.y1), cv::Point(r.x2, r.y2), cv::Scalar(255, 0, 0), 2, cv::LINE_8);
        }
        cv::imshow(windowName, curr);
    }

    bool playback::seekFrame(std::size_t frameNum) {
        if(frameNum >= data->images.size()) {
            return false;
        }
        data->frameNum = frameNum;
        return true;
    }

    std::size_t playback::getFrame() const {
        return data->frameNum;
    }

    bool playback::seekTime(double time) {
        return seekFrame(std::size_t(time * data->fps));
    }

    double playback::getTime() const {
        return getFrame() / (double) data->fps;
    }
    
    void playback::interFrameSleep() const {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000/data->fps));
    }

    std::size_t playback::getMaxFrame() const {
        return data->images.size() - 1;
    }

    double playback::getMaxTime() const {
        return getMaxFrame() / (double) data->fps;
    }
}
