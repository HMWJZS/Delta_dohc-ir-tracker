#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
uint32_t ir_decoder_abi(void);
const char *ir_decoder_error(void);
const char *ir_decoder_build_info(void);
typedef struct {
    uint32_t width, height, hstride, vstride, format, colorspace, range;
} IrFrameInfo;
typedef struct {
    double rga_ms, map_copy_ms;
    uint64_t converted_pixels;
} IrCropStats;
typedef struct {
    /* CLOCK_MONOTONIC nanoseconds, consecutive boundaries within one decode. */
    uint64_t boundary_ns[13];
} IrDecodeProfile;
int ir_decoder_create(void **decoder);
int ir_decoder_decode(void *decoder, const void *jpeg, size_t bytes, IrFrameInfo *info);
int ir_decoder_decode_profiled(void *decoder, const void *jpeg, size_t bytes,
                              IrFrameInfo *info, IrDecodeProfile *profile);
int ir_decoder_crop_bgr(void *decoder, int x, int y, int width, int height,
                        void *bgr, size_t row_bytes, IrCropStats *stats);
void ir_decoder_release_frame(void *decoder);
void ir_decoder_destroy(void *decoder);
#ifdef __cplusplus
}
#endif
