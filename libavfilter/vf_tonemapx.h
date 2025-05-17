/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef AVFILTER_TONEMAPX_H
#define AVFILTER_TONEMAPX_H

#include "config.h"
#include "colorspace.h"

#define X86_64_V2 __attribute__((target("sse4.2")))
#define X86_64_V3 __attribute__((target("avx2,fma")))

#if defined(__GNUC__) || defined(__clang__)
#    if (__GNUC__ >= 9) || (__clang_major__ >= 11)
#        define CC_SUPPORTS_TONEMAPX_INTRINSICS
#    endif // (__GNUC__ >= 10) || (__clang_major__ >= 11)
#endif // defined(__GNUC__) || defined(__clang__)

#ifdef CC_SUPPORTS_TONEMAPX_INTRINSICS
#    if ARCH_AARCH64
#        if HAVE_INTRINSICS_NEON
#            define ENABLE_TONEMAPX_NEON_INTRINSICS
#        endif
#    endif // ARCH_AARCH64
#    if ARCH_X86
#        if HAVE_INTRINSICS_SSE42
#           define ENABLE_TONEMAPX_SSE_INTRINSICS
#        endif
#        if HAVE_INTRINSICS_AVX2 && HAVE_INTRINSICS_FMA3
#            define ENABLE_TONEMAPX_AVX_INTRINSICS
#        endif
#    endif // ARCH_X86
#endif // CC_SUPPORTS_TONEMAPX_INTRINSICS

#define params_cnt 8
#define pivots_cnt (7+1)
#define coeffs_cnt 8*4
#define mmr_cnt 8*6*4
#define params_sz params_cnt*sizeof(float)
#define pivots_sz pivots_cnt*sizeof(float)
#define coeffs_sz coeffs_cnt*sizeof(float)
#define mmr_sz mmr_cnt*sizeof(float)

#define CHROMA_AVG_ROUNDING 2
#define JPEG_SCALE 32767.0f
#define EIGHT_BIT_ROUNDING 1048576
#define EIGHT_BIT_UV_OFFSET 128
#define EIGHT_BIT_SCALE_SHIFT 21
#define TEN_BIT_SCALE 1023.0f
#define TEN_BIT_UV_OFFSET 512
#define TEN_BIT_ROUNDING 512
#define TEN_BIT_BIPLANAR_SHIFT 6
#define TEN_BIT_SCALE_SHIFT 19

typedef struct TonemapIntParams {
    double lut_peak;
    float *lin_lut;
    float *tonemap_lut;
    uint16_t *delin_lut;
    int in_yuv_off, out_yuv_off;
    int (*yuv2rgb_coeffs)[3][3][8];
    int (*rgb2yuv_coeffs)[3][3][8];
    double  (*rgb2rgb_coeffs)[3][3];
    int rgb2rgb_passthrough;
    const AVLumaCoefficients *coeffs, *ocoeffs;
    double desat;
    struct DoviMetadata *dovi;
    float *dovi_pbuf;
    double (*lms2rgb_matrix)[3][3];
    float (*ycc_offset)[3];
} TonemapIntParams;

enum SIMDVariant {
    SIMD_NONE = -1,
    SIMD_NEON,
    SIMD_SSE,
    SIMD_AVX
};

void tonemap_frame_dovi_2_420p(uint8_t *dsty, uint8_t *dstu, uint8_t *dstv,
                               const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                               const int *dstlinesize, const int *srclinesize,
                               int dstdepth, int srcdepth,
                               int width, int height,
                               const struct TonemapIntParams *params);

void tonemap_frame_420p10_2_420p(uint8_t *dsty, uint8_t *dstu, uint8_t *dstv,
                                 const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                 const int *dstlinesize, const int *srclinesize,
                                 int dstdepth, int srcdepth,
                                 int width, int height,
                                 const struct TonemapIntParams *params);

void tonemap_frame_p010_2_nv12(uint8_t *dsty, uint8_t *dstuv,
                               const uint16_t *srcy, const uint16_t *srcuv,
                               const int *dstlinesize, const int *srclinesize,
                               int dstdepth, int srcdepth,
                               int width, int height,
                               const struct TonemapIntParams *params);

void tonemap_frame_dovi_2_420p10(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                 const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                 const int *dstlinesize, const int *srclinesize,
                                 int dstdepth, int srcdepth,
                                 int width, int height,
                                 const struct TonemapIntParams *params);

void tonemap_frame_dovi_2_420hdr(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                 const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                 const int *dstlinesize, const int *srclinesize,
                                 int dstdepth, int srcdepth,
                                 int width, int height,
                                 const struct TonemapIntParams *params);

void tonemap_frame_420p10_2_420p10(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                   const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                   const int *dstlinesize, const int *srclinesize,
                                   int dstdepth, int srcdepth,
                                   int width, int height,
                                   const struct TonemapIntParams *params);

void tonemap_frame_p010_2_p010(uint16_t *dsty, uint16_t *dstuv,
                               const uint16_t *srcy, const uint16_t *srcuv,
                               const int *dstlinesize, const int *srclinesize,
                               int dstdepth, int srcdepth,
                               int width, int height,
                               const struct TonemapIntParams *params);

#endif // AVFILTER_TONEMAPX_H
