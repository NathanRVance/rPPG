#include "playback.h"
#include "ui.h"
#include "labeller.h"
#include <vector>
#include <iostream>
#include <memory>

// Removes flags from args (in-place) and returns vector of flags
std::vector<std::string> extractFlags(std::vector<std::string>& args) {
    std::vector<std::string> ret;
    auto it = args.begin();
    while(it != args.end()) {
        if((*it)[0] == '-') {
            while((*it)[0] == '-') {
                (*it).erase((*it).begin());
            }
            ret.push_back(*it);
            args.erase(it);
        } else {
            it++;
        }
    }
    return ret;
}

int main(int argc, char *argv[]) {
    std::string exename = argv[0];
    std::vector<std::string> args(&argv[1], &argv[argc]);
    std::vector<std::string> flags = extractFlags(args);
    if(args.empty()) {
        std::cout << "Must provide a path to a video to process!" << std::endl;
        return 1;
    }
    int frameCap = -1;
    if(args.size() >= 2) {
        frameCap = std::stoi(args[1]);
        std::cout << "Only using first " << frameCap << " frames." << std::endl;
    }
    std::cout << "Loading video " << args[0] << std::endl;
    std::filesystem::path p(args[0]);
    std::shared_ptr<frontend::playback> playback(new frontend::playback(p, frameCap));
    // Format save path
    auto savepath = std::filesystem::path("saves") / p.parent_path().filename() / (p.stem().string() + ".csv");
    std::cout << "Saves are written to: " << savepath << std::endl;
    std::shared_ptr<backend::labeller> labeller(new backend::labeller(savepath));
    frontend::ui ui(playback, labeller);
    ui.begin();
    return 0;
}
