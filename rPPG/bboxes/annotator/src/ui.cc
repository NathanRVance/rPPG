#include "ui.h"
#include "playback.h"
#include "labeller.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sys/ioctl.h>
#include <unistd.h>

namespace frontend {
    struct ui_impl {
        std::shared_ptr<playback> pb;
        std::shared_ptr<backend::labeller> llr;
        bool play = false;
        bool stalePrintout = true;
    };

    ui::ui(std::shared_ptr<playback> playback, std::shared_ptr<backend::labeller> labeller) {
        data = std::shared_ptr<ui_impl>(new ui_impl);
        data->pb = playback;
        data->llr = labeller;
    }

    bool handleKey(int keycode, const std::shared_ptr<ui_impl>& data) {
        if(keycode != -1) {
            data->stalePrintout = true;
            if(keycode >= 48 && keycode <= 57) { // Number -> label
                std::size_t num = keycode - 48;
                if(num <= data->llr->getLabels().size()) {
                    data->llr->applyLabel(data->llr->getLabels()[num-1], data->pb->getTime());
                }
            } else {
                switch(keycode) {
                    case 27: // Escape quits
                        std::cout << std::endl;
                        return true;
                    case 32: // Space pauses
                        data->play = ! data->play;
                        break;
                    case 104:
                    case 65361: // Left seeks backward 1 frame (or h)
                        data->pb->seekFrame(data->pb->getFrame() - 1);
                        break;
                    case 100:
                    case 65362: // Up seeks backward 1 second (or d)
                        if(data->pb->getTime() < 1) {
                            data->pb->seekTime(data->pb->getTime() * -1);
                        } else {
                            data->pb->seekTime(data->pb->getTime() - 1);
                        }
                        break;
                    case 116:
                    case 65363: // Right seeks forward 1 frame (or t)
                        data->pb->seekFrame(data->pb->getFrame() + 1);
                        break;
                    case 110:
                    case 65364: // Down seeks forward 1 second (or n)
                        data->pb->seekTime(data->pb->getTime() + 1);
                        break;
                    case 117: // u undoes
                        data->llr->undo();
                        break;
                    case 114: // r redoes
                        data->llr->redo();
                        break;
                    case 65535: // DEL deletes
                        data->llr->deleteLabel(data->pb->getTime());
                        break;
                    case 115: // s saves
                        data->llr->save();
                        break;
                    default:
                        std::cout << "Pressed the " << keycode << " key" << std::endl;
                        break;
                }
            }
        }
        return false;
    }

    backend::label* getCurrentRectLabel(ui_impl *data) {
        auto &labs = data->llr->getEditableLabels();
        for(auto rit = labs.rbegin(); rit != labs.rend(); rit++) {
            if(rit->location.nonzero() && rit->time <= data->pb->getTime()) {
                return &(*rit);
            }
        }
        return nullptr;
    }

    void mouseCallback(int event, int x, int y, int flags, void* userdata) {
        // userdata is actually a &shared_ptr<ui_impl>
        auto data = static_cast<ui_impl*>(userdata);
        static bool mouseDown = false;
        // We're interested in left button down/up and movement when down
        if(event == cv::EVENT_LBUTTONDOWN) {
            // If the current frame doesn't have a rectangle label, then make one
            mouseDown = true;
            double time = data->pb->getTime();
            auto lab = getCurrentRectLabel(data);
            if(lab && lab->time == time) {
                // Use this one
            } else { // Make one
                // Try to use the first rectangle annotation, fall back on normal, then "Rectangle"
                std::string name = "Rectangle";
                if(!data->llr->getRectangleLabels().empty()) {
                    name = data->llr->getRectangleLabels()[0];
                } else if(!data->llr->getLabels().empty()) {
                    name = data->llr->getLabels()[0];
                }
                data->llr->applyLabel(name, time);
                lab = &data->llr->getEditableLabels().back();
            }
            lab->location.x1 = lab->location.x2 = x;
            lab->location.y1 = lab->location.y2 = y;
        } else if(event == cv::EVENT_LBUTTONUP) {
            // Mouse is no longer down
            mouseDown = false;
        } else if(event == cv::EVENT_MOUSEMOVE) {
            // If the mouse is down, update x2 and y2 of current window
            if(mouseDown) {
                auto lab = getCurrentRectLabel(data); // It will be valid! TODO: Will it?
                //std::cout << "Updating location from " << lab->location.x2 << ", " << lab->location.y2 << " to " << x << ", " << y << "\n";
                lab->location.x2 = x;
                lab->location.y2 = y;
            }
        }
    }

    void ui::begin() {
        std::cout << "Playing a video that's " << data->pb->getMaxFrame() << " frames (" << data->pb->getMaxTime() << " seconds) long." << std::endl;
        std::cout << "Annotations:" << std::endl;
        int num = 0;
        for(auto ann : data->llr->getLabels()) {
            std::cout << ++num << ": " << ann << std::endl;
        }
        // Get window size
        struct winsize size;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);
        int cols = size.ws_col;
        // Register mouse callback
        data->pb->display("Video", (getCurrentRectLabel(data.get()))? getCurrentRectLabel(data.get())->location : backend::rect());
        cv::setMouseCallback("Video", mouseCallback, data.get());
        while(true) {
            data->pb->display("Video", (getCurrentRectLabel(data.get()))? getCurrentRectLabel(data.get())->location : backend::rect());
            data->pb->interFrameSleep();
            if(handleKey(cv::pollKey(), data)) {
                break;
            }
            if(data->play) {
                data->pb->seekFrame(data->pb->getFrame() + 1);
            }
            if(data->play || data->stalePrintout) {
                std::stringstream toPrint;
                toPrint << "Frame: " << data->pb->getFrame() << " (" << std::setprecision(4) << std::setw(6) << std::left << data->pb->getTime() << " s)";
                std::string timeString = toPrint.str();
                toPrint.str(""); // clear contents
                // Get surrounding labels
                auto labs = data->llr->getSurrounding(data->pb->getTime());
                if(! labs.first.name.empty()) toPrint << labs.first.name << " (t=" << std::setprecision(4) << labs.first.time << ")";
                else toPrint << "START (t=0)";
                toPrint << " CURRENT ";
                if(! labs.second.name.empty()) toPrint << labs.second.name << " (t=" << std::setprecision(4) << labs.second.time << ")";
                else toPrint << "END (t=" << data->pb->getMaxTime() << ")";
                toPrint << std::string(cols - toPrint.str().size() - timeString.size() - 1, ' ') << timeString;
                std::cout << "\r\b\r" << toPrint.str();
                std::cout.flush();
                data->stalePrintout = false;
            }
        }
    }
}
