#pragma once
#include <memory>

namespace backend {
    class labeller;
}

namespace frontend {
    struct ui_impl;
    class playback;
    class ui {
        public:
            ui(std::shared_ptr<playback> playback, std::shared_ptr<backend::labeller> labeller);
            void begin();

        private:
            std::shared_ptr<ui_impl> data;
    };
}
