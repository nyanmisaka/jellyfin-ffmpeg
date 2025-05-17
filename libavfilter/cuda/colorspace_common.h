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

#define FLOAT_EPS 1e-6f

extern __constant__ const float ref_white;
extern __constant__ const float3 luma_dst;
extern __constant__ const float3 ycc2rgb_offset;
extern __constant__ const enum AVColorTransferCharacteristic trc_src, trc_dst;
extern __constant__ const enum AVColorRange range_src, range_dst;
extern __constant__ const enum AVChromaLocation chroma_loc_src, chroma_loc_dst;
extern __constant__ const bool rgb2rgb_passthrough;
extern __constant__ const float rgb2rgb_matrix[9];
extern __constant__ const float lms2rgb_matrix[9];
extern __constant__ const float yuv_matrix[9], rgb_matrix[9];
extern __constant__ const float pq_max_lum_div_ref_white;
extern __constant__ const float ref_white_div_pq_max_lum;
extern __constant__ const float input_quantization_offset;
extern __constant__ const float output_quantization_offset;
extern __constant__ const float input_y_scale;
extern __constant__ const float input_uv_scale;
extern __constant__ const float output_quantization_factor;
extern __constant__ const float output_quantization_scale;


static __inline__ __device__ float get_luma_dst(float3 c, const float3& luma_dst) {
    return luma_dst.x * c.x + luma_dst.y * c.y + luma_dst.z * c.z;
}

/*
static __inline__ __device__ float get_luma_src(float3 c, const float3& luma_src) {
    return luma_src.x * c.x + luma_src.y * c.y + luma_src.z * c.z;
}
*/

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
static __inline__ __device__ float eotf_st2084_common(float x) {
    x = max(x, 0.0f);
    float xpow = __powf(x, 1.0f / ST2084_M2);
    float num = max(xpow - ST2084_C1, 0.0f);
    float den = max(ST2084_C2 - ST2084_C3 * xpow, FLOAT_EPS);
    x = __powf(num / den, 1.0f / ST2084_M1);
    return x;
}

static __inline__ __device__ float eotf_st2084(float x) {
    return eotf_st2084_common(x) * pq_max_lum_div_ref_white;
}

// delinearizer for PQ/ST2084
static __inline__ __device__ float inverse_eotf_st2084_common(float x) {
    x = max(x, 0.0f);
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

static __inline__ __device__ float inverse_eotf_st2084(float x) {
    x *= ref_white_div_pq_max_lum;
    return inverse_eotf_st2084_common(x);
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
           ? sqrtf(3.0f * x)
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
    return ootf_1_2(inverse_oetf_arib_b67(x)) * 5.0f;
}

// delinearizer for HLG/ARIB-B67
static __inline__ __device__ float inverse_eotf_arib_b67(float x) {
    return oetf_arib_b67(inverse_ootf_1_2(x / 5.0f));
}

// delinearizer for BT709, BT2020-10
static __inline__ __device__ float inverse_eotf_bt1886(float x) {
    return x > 0.0f ? __powf(x, 1.0f / 2.4f) : 0.0f;
}

static __inline__ __device__ float linearize(float x)
{
    if (trc_src == AVCOL_TRC_SMPTE2084 && trc_dst != AVCOL_TRC_SMPTE2084)
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
    y += y > 0.0f ? input_quantization_offset : 0.0f;
    u += u > 0.0f ? input_quantization_offset : 0.0f;
    v += v > 0.0f ? input_quantization_offset : 0.0f;
    if (range_src == AVCOL_RANGE_MPEG) {
        y = input_y_scale * y - 0.07305936073f;
        u = input_uv_scale * u - 0.5714285714f;
        v = input_uv_scale * v - 0.5714285714f;
    } else {
        u -= 0.5f;
        v -= 0.5f;
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
    if (range_dst == AVCOL_RANGE_MPEG) {
        y = floorf(((219.0f * y + 16.0f) * output_quantization_factor) + 0.5f) / output_quantization_scale;
        u = floorf(((224.0f * u + 128.0f) * output_quantization_factor) + 0.5f) / output_quantization_scale;
        v = floorf(((224.0f * v + 128.0f) * output_quantization_factor) + 0.5f) / output_quantization_scale;
    } else {
        u += 0.5f;
        v += 0.5f;
    }
    y -= y > 0.0f ? output_quantization_offset : 0.0f;
    u -= u > 0.0f ? output_quantization_offset : 0.0f;
    v -= v > 0.0f ? output_quantization_offset : 0.0f;
    return make_float3(y, u, v);
}

static __inline__ __device__ float rgb2y(float r, float g, float b) {
    float y = r*yuv_matrix[0] + g*yuv_matrix[1] + b*yuv_matrix[2];
    if (range_dst == AVCOL_RANGE_MPEG) {
        y = floorf(((219.0f * y + 16.0f) * output_quantization_factor) + 0.5f) / output_quantization_scale;
    }
    y -= y > 0.0f ? output_quantization_offset : 0.0f;
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

static __inline__ __device__ float3 rgb2lrgb(float3 c) {
    float r = linearize(c.x);
    float g = linearize(c.y);
    float b = linearize(c.z);
    return make_float3(r, g, b);
}

static __inline__ __device__ float3 ycc2rgb(float y, float cb, float cr) {
    float r = y * rgb_matrix[0] + cb * rgb_matrix[1] + cr * rgb_matrix[2];
    float g = y * rgb_matrix[3] + cb * rgb_matrix[4] + cr * rgb_matrix[5];
    float b = y * rgb_matrix[6] + cb * rgb_matrix[7] + cr * rgb_matrix[8];
    return make_float3(r, g, b) + ycc2rgb_offset;
}

static __inline__ __device__ float3 lms2rgb(float r, float g, float b) {
    r = eotf_st2084_common(r);
    g = eotf_st2084_common(g);
    b = eotf_st2084_common(b);
    float rr = r * lms2rgb_matrix[0] + g * lms2rgb_matrix[1] + b * lms2rgb_matrix[2];
    float gg = r * lms2rgb_matrix[3] + g * lms2rgb_matrix[4] + b * lms2rgb_matrix[5];
    float bb = r * lms2rgb_matrix[6] + g * lms2rgb_matrix[7] + b * lms2rgb_matrix[8];
    rr = inverse_eotf_st2084_common(rr);
    gg = inverse_eotf_st2084_common(gg);
    bb = inverse_eotf_st2084_common(bb);
    return rgb2lrgb(make_float3(rr, gg, bb));
}

static __inline__ __device__ float3 lms2rgb_fast(float r, float g, float b) {
    float rr = r * lms2rgb_matrix[0] + g * lms2rgb_matrix[1] + b * lms2rgb_matrix[2];
    float gg = r * lms2rgb_matrix[3] + g * lms2rgb_matrix[4] + b * lms2rgb_matrix[5];
    float bb = r * lms2rgb_matrix[6] + g * lms2rgb_matrix[7] + b * lms2rgb_matrix[8];
    return rgb2lrgb(make_float3(rr, gg, bb));
}

static __inline__ __device__ float3 lrgb2ictcp(float r, float g, float b) {
    float l = 0.412109375000000f * r + 0.523925781250000f * g + 0.063964843750000f * b;
    float m = 0.166748046875000f * r + 0.720458984375000f * g + 0.112792968750000f * b;
    float s = 0.024169921875000f * r + 0.075439453125000f * g + 0.900390625000000f * b;
    l = inverse_eotf_st2084(l);
    m = inverse_eotf_st2084(m);
    s = inverse_eotf_st2084(s);
    float i = 0.5f * l + 0.5f * m;
    float ct = 1.613769531250000f * l - 3.323486328125000f * m + 1.709716796875000f * s;
    float cp = 4.378173828125000f * l - 4.245605468750000f * m - 0.132568359375000f * s;
    return make_float3(i, ct, cp);
}

static __inline__ __device__ float3 ictcp2lrgb(float i, float ct, float cp) {
    float ll = i + 0.008609037037933f * ct + 0.111029625003026f * cp;
    float mm = i - 0.008609037037933f * ct - 0.111029625003026f * cp;
    float ss = i + 0.560031335710679f * ct - 0.320627174987319f * cp;
    ll = eotf_st2084(ll);
    mm = eotf_st2084(mm);
    ss = eotf_st2084(ss);
    float r = 3.436606694333079f * ll - 2.506452118656270f * mm + 0.069845424323191f * ss;
    float g = -0.791329555598929f * ll + 1.983600451792291f * mm - 0.192270896193362f * ss;
    float b = -0.025949899690593f * ll - 0.098913714711726f * mm + 1.124863614402319f * ss;
    return make_float3(r, g, b);
}

static __inline__ __device__ float parabolic(float x, float t0, float x0, float y0) {
    float s = (y0 - t0) / sqrtf(x0 - y0);
    float ox = t0 - s * s * 0.25f;
    float oy = t0 - s * sqrtf(s * s * 0.25f);
    return (x < t0 ? x : s * sqrtf(x - ox) + oy);
}

static __inline __device__ float3 gamut_compress(float3 rgb) {
    #define cyan_limit 1.5187050250638159f
    #define magenta_limit 1.0750082769546088f
    #define yellow_limit 1.0887800403483898f
    #define cyan_threshold 1.050508660266247f
    #define magenta_threshold 0.940509816042432f
    #define yellow_threshold 0.9771607996420639f

    // Achromatic axis
    float ac = max(max(rgb.x, rgb.y), rgb.z);
    float ac_abs = fabsf(ac);
    float3 ac3 = make_float3(ac, ac, ac);
    float3 ac_abs3 = make_float3(ac_abs, ac_abs, ac_abs);

    // Inverse RGB Ratios: distance from achromatic axis
    float3 d = ac == 0.0f ? make_float3(0.0f, 0.0f, 0.0f) : (ac3 - rgb) / ac_abs3;

    // Compressed distance
    float3 cd = make_float3(
        parabolic(d.x, cyan_threshold, cyan_limit, 1.0f),
        parabolic(d.y, magenta_threshold, magenta_limit, 1.0f),
        parabolic(d.z, yellow_threshold, yellow_limit, 1.0f)
    );

    // Inverse RGB Ratios to RGB
    float3 crgb = ac3 - cd * ac_abs3;

    return crgb;
}

#endif /* AVFILTER_CUDA_COLORSPACE_COMMON_H */
