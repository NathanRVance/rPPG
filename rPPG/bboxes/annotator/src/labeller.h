#pragma once
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <filesystem>

namespace backend {
    struct labeller_impl;

    struct rect {
        int x1=0, x2=0, y1=0, y2=0;
        bool nonzero() const {
            return (x1 != 0 || x1 != 0 || y1 != 0 || y2 != 0);
        }
        std::string string() {
            return std::to_string(x1) + "," + std::to_string(y1) + "," + std::to_string(x2) + "," + std::to_string(y2);
        }
    };

    struct label {
        label() {}
        label(const std::string& name, double time) : name(name), time(time) {}
        std::string name;
        double time;
        rect location;
    };

    class labeller {
        public:
            labeller(const std::filesystem::path& savepath);
            std::pair<label, label> getSurrounding(double time) const;
            std::vector<label>& getEditableLabels();
            std::vector<std::string> getLabels() const;
            std::vector<std::string> getRectangleLabels() const;
            void applyLabel(std::string name, double time);
            void deleteLabel(double time); // Deletes closest to time, last added if ties
            void undo();
            void redo();
            void save() const;

        private:
            std::shared_ptr<labeller_impl> data;
    };
}
