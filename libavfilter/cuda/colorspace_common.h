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

#ifndef AVFILTER_CUDA_COLORSPACE_COMMON_H
#define AVFILTER_CUDA_COLORSPACE_COMMON_H

#include "util.h"
#include "libavutil/pixfmt.h"

#define ST2084_MAX_LUMINANCE 10000.0f

#define ST2084_M1 0.1593017578125f
#define ST2084_M2 78.84375f
#define ST2084_C1 0.8359375f
#define ST2084_C2 18.8515625f
#define ST2084_C3 18.6875f

#define ARIB_B67_A 0.17883277f
#define ARIB_B67_B 0.28466892f
#define ARIB_B67_C 0.55991073f

#define FLOAT_EPS 1.175494351e-38f

extern __constant__ const float ref_white;
extern __constant__ const float3 luma_src, luma_dst;
extern __constant__ const enum AVColorTransferCharacteristic trc_src, trc_dst;
extern __constant__ const enum AVColorRange range_src, range_dst;
extern __constant__ const enum AVChromaLocation chroma_loc_src, chroma_loc_dst;
extern __constant__ const bool rgb2rgb_passthrough;
extern __constant__ const float rgb2rgb_matrix[9];
extern __constant__ const float yuv_matrix[9], rgb_matrix[9];

static __inline__ __device__ float get_luma_dst(float3 c, const float3& luma_dst) {
    return luma_dst.x * c.x + luma_dst.y * c.y + luma_dst.z * c.z;
}

static __inline__ __device__ float get_luma_src(float3 c, const float3& luma_src) {
    return luma_src.x * c.x + luma_src.y * c.y + luma_src.z * c.z;
}

static __inline__ __device__ float3 get_chroma_sample(float3 a, float3 b, float3 c, float3 d) {
    switch (chroma_loc_dst) {
    case AVCHROMA_LOC_LEFT:
        return ((a) + (c)) * 0.5f;
    case AVCHROMA_LOC_CENTER:
    case AVCHROMA_LOC_UNSPECIFIED:
    default:
        return ((a) + (b) + (c) + (d)) * 0.25f;
    case AVCHROMA_LOC_TOPLEFT:
        return a;
    case AVCHROMA_LOC_TOP:
        return ((a) + (b)) * 0.5f;
    case AVCHROMA_LOC_BOTTOMLEFT:
        return c;
    case AVCHROMA_LOC_BOTTOM:
        return ((c) + (d)) * 0.5f;
    }
}

// linearizer for PQ/ST2084
static __inline__ __device__ float eotf_st2084(float x) {
    x = max(x, 0.0f);
    float xpow = __powf(x, 1.0f / ST2084_M2);
    float num = max(xpow - ST2084_C1, 0.0f);
    float den = max(ST2084_C2 - ST2084_C3 * xpow, FLOAT_EPS);
    x = __powf(num / den, 1.0f / ST2084_M1);
    return x * ST2084_MAX_LUMINANCE / ref_white;
}

// delinearizer for PQ/ST2084
static __inline__ __device__ float inverse_eotf_st2084(float x) {
    x = max(x, 0.0f);
    x *= ref_white / ST2084_MAX_LUMINANCE;
    float xpow = __powf(x, ST2084_M1);
#if 0
    // Original formulation from SMPTE ST 2084:2014 publication.
    float num = ST2084_C1 + ST2084_C2 * xpow;
    float den = 1.0f + ST2084_C3 * xpow;
    return __powf(num / den, ST2084_M2);
#else
    // More stable arrangement that avoids some cancellation error.
    float num = (ST2084_C1 - 1.0f) + (ST2084_C2 - ST2084_C3) * xpow;
    float den = 1.0f + ST2084_C3 * xpow;
    return __powf(1.0f + num / den, ST2084_M2);
#endif
}

static __inline__ __device__ float ootf_1_2(float x) {
    return x > 0.0f ? __powf(x, 1.2f) : x;
}

static __inline__ __device__ float inverse_ootf_1_2(float x) {
    return x > 0.0f ? __powf(x, 1.0f / 1.2f) : x;
}

static __inline__ __device__ float oetf_arib_b67(float x) {
    x = max(x, 0.0f);
    return x <= (1.0f / 12.0f)
           ? __sqrtf(3.0f * x)
           : (ARIB_B67_A * __logf(12.0f * x - ARIB_B67_B) + ARIB_B67_C);
}

static __inline__ __device__ float inverse_oetf_arib_b67(float x) {
    x = max(x, 0.0f);
    return x <= 0.5f
           ? (x * x) * (1.0f / 3.0f)
           : (__expf((x - ARIB_B67_C) / ARIB_B67_A) + ARIB_B67_B) * (1.0f / 12.0f);
}

// linearizer for HLG/ARIB-B67
static __inline__ __device__ float eotf_arib_b67(float x) {
    return ootf_1_2(inverse_oetf_arib_b67(x));
}

// delinearizer for HLG/ARIB-B67
static __inline__ __device__ float inverse_eotf_arib_b67(float x) {
    return oetf_arib_b67(inverse_ootf_1_2(x));
}

// delinearizer for BT709, BT2020-10
static __inline__ __device__ float inverse_eotf_bt1886(float x) {
    return x > 0.0f ? __powf(x, 1.0f / 2.4f) : 0.0f;
}

static __inline__ __device__ float linearize(float x)
{
    if (trc_src == AVCOL_TRC_SMPTE2084)
        return eotf_st2084(x);
    else if (trc_src == AVCOL_TRC_ARIB_STD_B67)
        return eotf_arib_b67(x);
    else
        return x;
}

static __inline__ __device__ float delinearize(float x)
{
    if (trc_dst == AVCOL_TRC_BT709 || trc_dst == AVCOL_TRC_BT2020_10)
        return inverse_eotf_bt1886(x);
    else
        return x;
}

static __inline__ __device__ float3 yuv2rgb(float y, float u, float v) {
    if (range_src == AVCOL_RANGE_JPEG) {
        u -= 0.5f; v -= 0.5f;
    } else {
        y = (y * 255.0f -  16.0f) / 219.0f;
        u = (u * 255.0f - 128.0f) / 224.0f;
        v = (v * 255.0f - 128.0f) / 224.0f;
    }
    float r = y * rgb_matrix[0] + u * rgb_matrix[1] + v * rgb_matrix[2];
    float g = y * rgb_matrix[3] + u * rgb_matrix[4] + v * rgb_matrix[5];
    float b = y * rgb_matrix[6] + u * rgb_matrix[7] + v * rgb_matrix[8];
    return make_float3(r, g, b);
}

static __inline__ __device__ float3 yuv2lrgb(float3 yuv) {
    float3 rgb = yuv2rgb(yuv.x, yuv.y, yuv.z);
    return make_float3(linearize(rgb.x),
                       linearize(rgb.y),
                       linearize(rgb.z));
}

static __inline__ __device__ float3 rgb2yuv(float r, float g, float b) {
    float y = r*yuv_matrix[0] + g*yuv_matrix[1] + b*yuv_matrix[2];
    float u = r*yuv_matrix[3] + g*yuv_matrix[4] + b*yuv_matrix[5];
    float v = r*yuv_matrix[6] + g*yuv_matrix[7] + b*yuv_matrix[8];
    if (range_dst == AVCOL_RANGE_JPEG) {
        u += 0.5f; v += 0.5f;
    } else {
        y = (219.0f * y + 16.0f) / 255.0f;
        u = (224.0f * u + 128.0f) / 255.0f;
        v = (224.0f * v + 128.0f) / 255.0f;
    }
    return make_float3(y, u, v);
}

static __inline__ __device__ float rgb2y(float r, float g, float b) {
    float y = r*yuv_matrix[0] + g*yuv_matrix[1] + b*yuv_matrix[2];
    if (range_dst != AVCOL_RANGE_JPEG)
        y = (219.0f * y + 16.0f) / 255.0f;
    return y;
}

static __inline__ __device__ float3 lrgb2yuv(float3 c) {
    float r = delinearize(c.x);
    float g = delinearize(c.y);
    float b = delinearize(c.z);
    return rgb2yuv(r, g, b);
}

static __inline__ __device__ float3 lrgb2lrgb(float3 c) {
    if (rgb2rgb_passthrough) {
        return c;
    } else {
        float r = c.x, g = c.y, b = c.z;
        float rr = rgb2rgb_matrix[0] * r + rgb2rgb_matrix[1] * g + rgb2rgb_matrix[2] * b;
        float gg = rgb2rgb_matrix[3] * r + rgb2rgb_matrix[4] * g + rgb2rgb_matrix[5] * b;
        float bb = rgb2rgb_matrix[6] * r + rgb2rgb_matrix[7] * g + rgb2rgb_matrix[8] * b;
        return make_float3(rr, gg, bb);
    }
}

#endif /* AVFILTER_CUDA_COLORSPACE_COMMON_H */
