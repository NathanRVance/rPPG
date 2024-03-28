#include "labeller.h"
#include "settings.h"
#include <fstream>
#include <iostream>

namespace backend {
    struct labeller_impl {
        std::vector<std::string> labels;
        std::vector<std::string> rectangleLabels;
        std::vector<label> annotations;
        // Bool in undo/redo buffers is whether was applied or deleted
        std::vector<std::pair<bool, label>> undoBuffer;
        std::vector<std::pair<bool, label>> redoBuffer;
        std::filesystem::path savepath;
    };

    void load(const std::filesystem::path& savepath, labeller_impl& data) {
        std::ifstream in(savepath);
        std::string line;
        const char delim = ',';
        while(std::getline(in, line)) {
            // Fields are label,time,x1,y1,x2,y2
            std::vector<std::string> parts;
            std::size_t split;
            while((split = line.find(delim)) != std::string::npos) {
                parts.push_back(line.substr(0, split));
                line = line.substr(split+1);
            }
            // And pick up that last part
            parts.push_back(line);
            if(parts.size() < 2) {
                std::cerr << "Error reading " << savepath << std::endl;
                throw std::runtime_error(savepath.string());
            }
            label lab(parts[0], std::stod(parts[1]));
            if(parts.size() == 6) {
                lab.location.x1 = std::stoi(parts[2]);
                lab.location.y1 = std::stoi(parts[3]);
                lab.location.x2 = std::stoi(parts[4]);
                lab.location.y2 = std::stoi(parts[5]);
            }
            data.annotations.push_back(lab);
        }
    }

    labeller::labeller(const std::filesystem::path& savepath) {
        data = std::shared_ptr<labeller_impl>(new labeller_impl);
        data->labels = settings::getLabels();
        data->rectangleLabels = settings::getRectangleLabels();
        data->savepath = savepath;
        if(std::filesystem::directory_entry(savepath).exists()) {
            load(savepath, *data);
        }
    }

    std::pair<label, label> labeller::getSurrounding(double time) const {
        std::pair<label, label> labs;
        if(!data->annotations.empty()) {
            auto above = data->annotations.rbegin();
            auto below = above;
            auto rit = above;
            while(++rit != data->annotations.rend()) {
                if(rit->time > time && (above->time <= time || rit->time < above->time)) above = rit;
                if(rit->time <= time && (below->time > time || rit->time > below->time)) below = rit;
            }
            if(below->time <= time) labs.first = *below;
            if(above->time > time) labs.second = *above;
        }
        return labs;
    }

    std::vector<label>& labeller::getEditableLabels() {
        return data->annotations;
    }


    std::vector<std::string> labeller::getLabels() const {
        return data->labels;
    }

    std::vector<std::string> labeller::getRectangleLabels() const {
        return data->rectangleLabels;
    }

    void appLab(std::vector<label>& annotations, label lab) {
        annotations.push_back(lab);
    }

    void labeller::applyLabel(std::string name, double time) {
        label lab(name, time);
        appLab(data->annotations, lab);
        data->undoBuffer.push_back({true, lab});
        data->redoBuffer.clear();
    }

    double abs(double x) {
        if(x < 0) x *= -1;
        return x;
    }

    label delLab(std::vector<label>& annotations, double time) {
        if(annotations.empty()) {
            return label();
        }
        auto closest = annotations.rbegin();
        auto rit = closest;
        while(++rit != annotations.rend()) {
            if(abs(rit->time - time) < abs(closest->time - time)) {
                closest = rit;
            }
        }
        label c(*closest);
        annotations.erase(closest.base());
        return c;
    }

    void labeller::deleteLabel(double time) {
        label deleted = delLab(data->annotations, time);
        data->undoBuffer.push_back({false, deleted});
        data->redoBuffer.clear();
    }

    // Pops action from 'from', applies, and appends to 'to'.
    void handleUndoRedo(std::vector<std::pair<bool, label>>& from, std::vector<std::pair<bool, label>>& to, std::vector<label>& annotations) {
        if(! from.empty()) {
            auto elem = from.back();
            from.pop_back();
            if(elem.first) { // It was applied, so we must unapply
                delLab(annotations, elem.second.time);
            } else {
                appLab(annotations, elem.second);
            }
            to.push_back({! elem.first, elem.second});
        }
    }

    void labeller::undo() {
        handleUndoRedo(data->undoBuffer, data->redoBuffer, data->annotations);
    }

    void labeller::redo() {
        handleUndoRedo(data->redoBuffer, data->undoBuffer, data->annotations);
    }

    bool compareLabels(label l1, label l2) {
        return (l1.time < l2.time);
    }

    void labeller::save() const {
        std::ofstream out;
        std::filesystem::create_directories(data->savepath.parent_path());
        out.open(data->savepath);
        std::vector<label> a(data->annotations);
        std::sort(a.begin(), a.end(), compareLabels);
        for(label l : a) {
            out << l.name << "," << l.time << "," << l.location.string() << std::endl;
        }
        out.close();
    }

}
