#pragma once
#include <filesystem>
#include <memory>
#include <string>

namespace backend {
    struct rect;
}

namespace frontend {
    struct playback_impl;
    class playback {
        public:
            playback(const std::filesystem::path& video, std::size_t frameCap=0);
            void display(const std::string& windowName, const backend::rect &r) const;
            bool seekFrame(std::size_t frameNum);
            std::size_t getFrame() const;
            bool seekTime(double time);
            double getTime() const;
            void interFrameSleep() const;
            std::size_t getMaxFrame() const;
            double getMaxTime() const;

        private:
            std::shared_ptr<playback_impl> data;
    };
}
