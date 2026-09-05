#include "image_container.hpp"
#include <cstdint>
#include <vector>
#include <memory>

struct ImageData {
    size_t size;
    int widith;
    int height;
    int channels;
};

class ImageContainer {
    private:
        ImageData data;
        std::vector<uint8_t> buffer;

    public:
        void release() {
            buffer.clear();
            buffer.shrink_to_fit();

            data.size = 0;
            data.widith = 0;
            data.height = 0;
            data.channels = 0;
        }
};