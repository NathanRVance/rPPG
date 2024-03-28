#include "settings.h"
#include <confuse.h>
#include <errno.h>
#include <cstring>
#include <stdexcept>

namespace settings {
    std::vector<std::string> readVector(std::string heading) {
        cfg_opt_t opts[] = {
            CFG_STR_LIST(heading.c_str(), NULL, CFGF_NONE),
            CFG_END()
        };
        cfg_t *cfg = cfg_init(opts, CFGF_IGNORE_UNKNOWN);
        if(cfg_parse(cfg, "annotator.conf") == CFG_PARSE_ERROR) {
            throw std::runtime_error("Configuration file annotator.conf could not be read: " + std::string(strerror(errno)));
        }
        try {
            std::vector<std::string> ret;
            std::size_t i;
            for(i = 0; i < cfg_size(cfg, heading.c_str()); i++) {
                ret.push_back(cfg_getnstr(cfg, heading.c_str(), i));
            }
            return ret;
        } catch(std::exception& e) {
            throw std::runtime_error("Cannot find '" + heading + "' in configuration file");
        }
    }

    std::vector<std::string> getLabels() {
        return readVector("labels");
    }

    std::vector<std::string> getRectangleLabels() {
        return readVector("rectangles");
    }
}

