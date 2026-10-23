// Host-test the native input boundary with SDK headers; no hardware calls are made.
#include "codec.cpp"
#include <cassert>
#include <vector>
int main() {
    std::vector<uint8_t> jpeg{0xff,0xd8,0xff,0xc0,0,11,8,8,112,15,0,1,1,0x11,0,0xff,0xda};
    int w=0, h=0;
    jpeg_size(jpeg.data(), jpeg.size(), w, h);
    assert(w==3840 && h==2160);
    for (size_t n=0; n<jpeg.size()-1; ++n) {
        bool rejected=false;
        try { jpeg_size(jpeg.data(), n, w, h); } catch (const std::exception &) { rejected=true; }
        assert(rejected);
    }
    for (int marker : {0xc1,0xc2,0xc3}) {
        auto invalid=jpeg; invalid[3]=marker;
        bool rejected=false;
        try { jpeg_size(invalid.data(), invalid.size(), w, h); } catch (const std::exception &) { rejected=true; }
        assert(rejected);
    }
    assert(color_mode(MPP_FRAME_SPC_UNSPECIFIED, MPP_FRAME_RANGE_UNSPECIFIED)==IM_YUV_BT601_FULL_RANGE);
    assert(color_mode(MPP_FRAME_SPC_BT709, MPP_FRAME_RANGE_JPEG)==IM_YUV_BT709_FULL_RANGE);
    assert(rga_format(MPP_FMT_YUV422SP)==RK_FORMAT_YCbCr_422_SP);
    bool rejected=false;
    try { rga_format(MPP_FMT_YUV420SP_10BIT); } catch (const std::exception &) { rejected=true; }
    assert(rejected);
    assert(align_up(93,2)==94 && align_up(94,16)==96);
}
