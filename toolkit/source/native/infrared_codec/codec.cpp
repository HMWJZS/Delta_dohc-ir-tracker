#include "codec.h"
#include <rk_mpi.h>
#include <mpp_buffer.h>
#include <mpp_frame.h>
#include <mpp_packet.h>
#include <mpp_meta.h>
#include <im2d.h>
#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#ifndef IR_CODEC_BUILD_INFO
#define IR_CODEC_BUILD_INFO "unrecorded-local-build"
#endif
namespace {
using Clock = std::chrono::steady_clock;
thread_local std::string error;
void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}
void mpp_check(MPP_RET value, const char *operation) {
    require(value == MPP_OK, std::string(operation)+": "+std::to_string(value));
}
void rga_check(IM_STATUS value, const char *operation) {
    require(value == IM_STATUS_SUCCESS || value == IM_STATUS_NOERROR,
            std::string(operation)+": "+imStrError(value));
}
int align_up(int v, int n) { return (v+n-1)/n*n; }
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now()-t).count();
}
void sync_buffer(MppBuffer buffer, uint64_t flags) {
    dma_buf_sync sync{};
    sync.flags = flags;
    require(ioctl(mpp_buffer_get_fd(buffer), DMA_BUF_IOCTL_SYNC, &sync) == 0,
            "DMA buffer cache synchronization failed");
}
struct Decoder {
    MppCtx ctx = nullptr;
    MppApi *api = nullptr;
    MppBufferGroup group = nullptr;
    MppBuffer packet_buffer = nullptr, yuv_buffer = nullptr, bgr_buffer = nullptr;
    MppPacket packet = nullptr;
    MppFrame frame = nullptr;
    IrFrameInfo info{};
    int rga_format = 0, csc = 0;
    bool active = false, failed = false;
    ~Decoder() {
        // Destroy the context first: an errored/timed-out task may still own buffers.
        if (ctx) mpp_destroy(ctx);
        if (packet) mpp_packet_deinit(&packet);
        if (frame) mpp_frame_deinit(&frame);
        if (packet_buffer) mpp_buffer_put(packet_buffer);
        if (yuv_buffer) mpp_buffer_put(yuv_buffer);
        if (bgr_buffer) mpp_buffer_put(bgr_buffer);
        if (group) mpp_buffer_group_put(group);
    }
    void reserve(MppBuffer &buffer, size_t bytes) {
        if (buffer && mpp_buffer_get_size(buffer) >= bytes) return;
        if (buffer) { mpp_buffer_put(buffer); buffer = nullptr; }
        mpp_check(mpp_buffer_get(group, &buffer, bytes), "allocate DMA buffer");
        require(mpp_buffer_get_fd(buffer) >= 0, "MPP did not allocate a DMA fd");
    }
};
// JPEG SOF is checked before allocating DMA memory; only native 8-bit baseline is accepted.
void jpeg_size(const uint8_t *p, size_t n, int &w, int &h) {
    require(p && n >= 4 && n <= 32*1024*1024 && p[0] == 0xff && p[1] == 0xd8,
            "Invalid/oversized JPEG");
    bool found = false;
    for (size_t i = 2; i+1 < n;) {
        require(p[i++] == 0xff, "Invalid JPEG marker");
        while (i < n && p[i] == 0xff) ++i;
        require(i < n, "Truncated JPEG marker");
        int marker = p[i++];
        if (marker == 0xda) { require(found, "JPEG has no SOF"); return; }
        require(marker != 0xd9 && marker != 0x00, "Invalid JPEG before SOS");
        if (marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
        require(i+2 <= n, "Truncated JPEG length");
        size_t length = (size_t(p[i]) << 8) | p[i+1];
        require(length >= 2 && length <= n-i, "Truncated JPEG segment");
        if (marker >= 0xc0 && marker <= 0xcf && marker != 0xc4 && marker != 0xc8 && marker != 0xcc) {
            require(marker == 0xc0 && length >= 8 && p[i+2] == 8 && !found,
                    "MPP input must be 8-bit baseline JPEG");
            h = (int(p[i+3]) << 8) | p[i+4];
            w = (int(p[i+5]) << 8) | p[i+6];
            int components = p[i+7];
            require((components == 1 || components == 3) && length == size_t(8+3*components),
                    "Only grayscale/YCbCr JPEG is supported");
            require(w >= 3 && h >= 3 && w <= 8192 && h <= 8192 && uint64_t(w)*h <= 16777216,
                    "JPEG dimensions exceed decoder limits");
            found = true;
        }
        if (marker == 0xee && length >= 14 && std::memcmp(p+i+2, "Adobe", 5) == 0)
            require(p[i+13] == 1, "Adobe RGB/CMYK JPEG is unsupported");
        i += length;
    }
    throw std::runtime_error("JPEG has no scan");
}
int rga_format(MppFrameFormat format) {
    switch (format) {
    case MPP_FMT_YUV420SP: return RK_FORMAT_YCbCr_420_SP;
    case MPP_FMT_YUV420SP_VU: return RK_FORMAT_YCrCb_420_SP;
    case MPP_FMT_YUV422SP: return RK_FORMAT_YCbCr_422_SP;
    case MPP_FMT_YUV422SP_VU: return RK_FORMAT_YCrCb_422_SP;
    case MPP_FMT_YUV420P: return RK_FORMAT_YCbCr_420_P;
    case MPP_FMT_YUV422P: return RK_FORMAT_YCbCr_422_P;
    case MPP_FMT_YUV400: return RK_FORMAT_YCbCr_400;
    case MPP_FMT_YUV444SP: return RK_FORMAT_YCbCr_444_SP;
    default: throw std::runtime_error("Unsupported MPP format/tiling/bit-depth: "+std::to_string(format));
    }
}
int color_mode(MppFrameColorSpace space, MppFrameColorRange range) {
    bool full = range == MPP_FRAME_RANGE_JPEG || range == MPP_FRAME_RANGE_UNSPECIFIED;
    require(full || range == MPP_FRAME_RANGE_MPEG, "Unsupported MPP color range");
    if (space == MPP_FRAME_SPC_BT709)
        return full ? IM_YUV_BT709_FULL_RANGE : IM_YUV_BT709_LIMIT_RANGE;
    require(space == MPP_FRAME_SPC_UNSPECIFIED || space == MPP_FRAME_SPC_BT470BG ||
            space == MPP_FRAME_SPC_SMPTE170M, "Unsupported MPP color matrix");
    return full ? IM_YUV_BT601_FULL_RANGE : IM_YUV_BT601_LIMIT_RANGE;
}
template <typename F> int guarded(Decoder *decoder, F function) {
    try { function(); return 0; }
    catch (const std::exception &e) { error = e.what(); }
    catch (...) { error = "Unknown MPP/RGA error"; }
    if (decoder) decoder->failed = true;
    return -1;
}
}
extern "C" {
uint32_t ir_decoder_abi() { return 1; }
const char *ir_decoder_error() { return error.c_str(); }
const char *ir_decoder_build_info() { return IR_CODEC_BUILD_INFO; }
int ir_decoder_create(void **out) {
    return guarded(nullptr, [&] {
        require(out, "Missing decoder pointer");
        *out = nullptr;
        auto d = std::make_unique<Decoder>();
        rga_check(imcheckHeader(), "RGA header/runtime compatibility");
        mpp_check(mpp_create(&d->ctx, &d->api), "mpp_create");
        mpp_check(mpp_init(d->ctx, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG), "mpp_init JPEG");
        // Small crops can select RGA2, whose DMA addresses must fit below 4 GiB.
        mpp_check(mpp_buffer_group_get_internal(&d->group,
            MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_DMA32), "DMA32 group");
        *out = d.release();
    });
}
int ir_decoder_decode(void *handle, const void *jpeg, size_t bytes, IrFrameInfo *info) {
    return ir_decoder_decode_profiled(handle, jpeg, bytes, info, nullptr);
}
int ir_decoder_decode_profiled(void *handle, const void *jpeg, size_t bytes,
                              IrFrameInfo *info, IrDecodeProfile *profile) {
    auto d = static_cast<Decoder *>(handle);
    auto mark = [&](int index) {
        if (profile) {
            timespec t{};
            require(clock_gettime(CLOCK_MONOTONIC, &t) == 0, "monotonic clock failed");
            profile->boundary_ns[index] = uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec;
        }
    };
    return guarded(d, [&] {
        if (profile) *profile = {};
        mark(0);
        require(d && info && !d->active && !d->failed, "Decoder unavailable or frame still active");
        int width = 0, height = 0;
        jpeg_size(static_cast<const uint8_t *>(jpeg), bytes, width, height);
        mark(1);
        if (d->packet) mpp_packet_deinit(&d->packet);
        if (d->frame) mpp_frame_deinit(&d->frame);
        d->reserve(d->packet_buffer, align_up(int(bytes)+256, 4096));
        d->reserve(d->yuv_buffer, size_t(align_up(width, 64))*align_up(height, 64)*4);
        mark(2);
        sync_buffer(d->packet_buffer, DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE);
        mark(3);
        auto ptr = static_cast<uint8_t *>(mpp_buffer_get_ptr(d->packet_buffer));
        require(ptr, "MPP packet mapping failed");
        std::memcpy(ptr, jpeg, bytes);
        std::memset(ptr+bytes, 0, 256);
        mark(4);
        sync_buffer(d->packet_buffer, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
        mark(5);
        mpp_check(mpp_packet_init_with_buffer(&d->packet, d->packet_buffer), "MPP packet");
        mpp_packet_set_pos(d->packet, ptr);
        mpp_packet_set_length(d->packet, bytes);
        mpp_check(mpp_frame_init(&d->frame), "MPP frame");
        mpp_frame_set_buffer(d->frame, d->yuv_buffer);
        MppTask task = nullptr;
        mark(6);
        mpp_check(d->api->poll(d->ctx, MPP_PORT_INPUT, static_cast<MppPollType>(250)), "MPP input poll timeout");
        mark(7);
        mpp_check(d->api->dequeue(d->ctx, MPP_PORT_INPUT, &task), "MPP input dequeue");
        require(task, "MPP input task is null");
        mpp_check(mpp_task_meta_set_packet(task, KEY_INPUT_PACKET, d->packet), "MPP task packet");
        mpp_check(mpp_task_meta_set_frame(task, KEY_OUTPUT_FRAME, d->frame), "MPP task frame");
        mpp_check(d->api->enqueue(d->ctx, MPP_PORT_INPUT, task), "MPP submit");
        mark(8);
        mpp_check(d->api->poll(d->ctx, MPP_PORT_OUTPUT, static_cast<MppPollType>(250)), "MPP output poll timeout");
        mark(9);
        task = nullptr;
        mpp_check(d->api->dequeue(d->ctx, MPP_PORT_OUTPUT, &task), "MPP output dequeue");
        require(task, "MPP output task is null");
        MppFrame result = nullptr;
        MPP_RET meta_ret = mpp_task_meta_get_frame(task, KEY_OUTPUT_FRAME, &result);
        mpp_check(d->api->enqueue(d->ctx, MPP_PORT_OUTPUT, task), "MPP output return");
        mpp_check(meta_ret, "MPP output metadata");
        mark(10);
        require(result == d->frame && !mpp_frame_get_errinfo(result) && !mpp_frame_get_discard(result),
                "MPP returned a different/corrupt/discarded frame");
        d->info = {mpp_frame_get_width(result), mpp_frame_get_height(result),
            mpp_frame_get_hor_stride(result), mpp_frame_get_ver_stride(result),
            uint32_t(mpp_frame_get_fmt(result)), uint32_t(mpp_frame_get_colorspace(result)),
            uint32_t(mpp_frame_get_color_range(result))};
        require(d->info.width == uint32_t(width) && d->info.height == uint32_t(height) &&
                d->info.hstride >= d->info.width && d->info.vstride >= d->info.height &&
                mpp_frame_get_buffer(result) == d->yuv_buffer, "Invalid MPP dimensions/strides/buffer");
        require(uint64_t(d->info.hstride)*d->info.vstride*3 <= mpp_buffer_get_size(d->yuv_buffer),
                "MPP strides exceed allocated buffer");
        d->rga_format = rga_format(mpp_frame_get_fmt(result));
        d->csc = color_mode(mpp_frame_get_colorspace(result), mpp_frame_get_color_range(result));
        mark(11);
        *info = d->info;
        d->active = true;
        mark(12);
    });
}
int ir_decoder_crop_bgr(void *handle, int x, int y, int width, int height,
                        void *bgr, size_t row_bytes, IrCropStats *stats) {
    auto d = static_cast<Decoder *>(handle);
    return guarded(d, [&] {
        require(d && d->active && !d->failed && bgr && stats, "No active MPP frame");
        require(x >= 0 && y >= 0 && width > 0 && height > 0 &&
                int64_t(x)+width <= d->info.width && int64_t(y)+height <= d->info.height &&
                row_bytes >= size_t(width)*3, "Invalid BGR crop");
        auto rga_started = Clock::now();
        // Even source edges preserve chroma phase; trim exactly after conversion, never resize.
        int left = x/2*2, top = y/2*2;
        int rw = align_up(x+width, 2)-left, rh = align_up(y+height, 2)-top;
        int stride = align_up(rw, 16);
        d->reserve(d->bgr_buffer, size_t(stride)*rh*3);
        auto src = wrapbuffer_fd(mpp_buffer_get_fd(d->yuv_buffer),
            align_up(int(d->info.width), 2), align_up(int(d->info.height), 2),
            d->rga_format, int(d->info.hstride), int(d->info.vstride));
        auto dst = wrapbuffer_fd(mpp_buffer_get_fd(d->bgr_buffer), rw, rh,
            RK_FORMAT_BGR_888, stride, rh);
        src.color_space_mode = d->csc;
        dst.color_space_mode = IM_RGB_FULL;
        im_rect sr{left, top, rw, rh}, dr{0, 0, rw, rh};
        rga_check(imcheck(src, dst, sr, dr), "RGA crop validation");
        rga_check(improcess(src, dst, {}, sr, dr, {}, -1, nullptr, nullptr, IM_SYNC), "RGA crop/CSC");
        stats->rga_ms = elapsed(rga_started);
        auto copy_started = Clock::now();
        sync_buffer(d->bgr_buffer, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
        auto ptr = static_cast<uint8_t *>(mpp_buffer_get_ptr(d->bgr_buffer));
        require(ptr, "RGA output mapping failed");
        for (int row = 0; row < height; ++row)
            std::memcpy(static_cast<uint8_t *>(bgr)+row*row_bytes,
                        ptr+(row+y-top)*size_t(stride)*3+(x-left)*3, size_t(width)*3);
        sync_buffer(d->bgr_buffer, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
        stats->map_copy_ms = elapsed(copy_started);
        stats->converted_pixels = uint64_t(rw)*rh;
    });
}
void ir_decoder_release_frame(void *handle) {
    if (handle) static_cast<Decoder *>(handle)->active = false;
}
void ir_decoder_destroy(void *handle) { delete static_cast<Decoder *>(handle); }
}
