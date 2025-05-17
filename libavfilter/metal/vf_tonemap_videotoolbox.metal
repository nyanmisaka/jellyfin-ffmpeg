/*
 * Copyright (c) 2024 Gnattu OC <gnattuoc@me.com>
 *
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

#include <metal_stdlib>
#include <metal_texture>
#include <metal_integer>

using namespace metal;

//------------
// Metal Tonemapping

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

constant float ref_white [[function_constant(0)]];
constant float tone_param [[function_constant(1)]];
constant float desat_param [[function_constant(2)]];
constant float target_peak [[function_constant(3)]];
constant float scene_threshold [[function_constant(4)]];
constant float pq_max_lum_div_ref_white [[function_constant(5)]];
constant float ref_white_div_pq_max_lum [[function_constant(6)]];
constant short tonemap_func_type [[function_constant(7)]];
constant bool is_tone_func_bt2390 [[function_constant(8)]];
constant bool is_tone_mode_rgb [[function_constant(9)]];
constant bool is_tone_mode_max [[function_constant(10)]];
constant bool is_non_semi_planar_in [[function_constant(11)]];
constant bool is_non_semi_planar_out [[function_constant(12)]];
constant bool enable_dither [[function_constant(13)]];
constant float dither_size2 [[function_constant(14)]];
constant float dither_quantization [[function_constant(15)]];
constant bool is_full_range_in [[function_constant(16)]];
constant bool is_full_range_out [[function_constant(17)]];
constant int chroma_loc [[function_constant(18)]];
constant bool is_rgb2rgb_passthrough [[function_constant(19)]];
constant float3 rgb2rgb_matrix_1 [[function_constant(20)]];
constant float3 rgb2rgb_matrix_2 [[function_constant(21)]];
constant float3 rgb2rgb_matrix_3 [[function_constant(22)]];
constant bool skip_tonemap [[function_constant(23)]];
constant bool dovi_reshape [[function_constant(24)]];
constant float3 ycc2rgb_offset [[function_constant(25)]];
constant float3 rgb_matrix_1 [[function_constant(26)]];
constant float3 rgb_matrix_2 [[function_constant(27)]];
constant float3 rgb_matrix_3 [[function_constant(28)]];
constant float3 lms2rgb_matrix_1 [[function_constant(29)]];
constant float3 lms2rgb_matrix_2 [[function_constant(30)]];
constant float3 lms2rgb_matrix_3 [[function_constant(31)]];
constant float3 yuv_matrix_1 [[function_constant(32)]];
constant float3 yuv_matrix_2 [[function_constant(33)]];
constant float3 yuv_matrix_3 [[function_constant(34)]];
constant float3 luma_dst [[function_constant(35)]];
constant short linearize_type [[function_constant(36)]];
constant short delinearize_type [[function_constant(37)]];
constant bool map_in_src_space [[function_constant(38)]];
constant bool is_tone_mode_itp [[function_constant(39)]];

enum AVChromaLocation {
    AVCHROMA_LOC_UNSPECIFIED,
    AVCHROMA_LOC_LEFT,
    AVCHROMA_LOC_CENTER,
    AVCHROMA_LOC_TOPLEFT,
    AVCHROMA_LOC_TOP,
    AVCHROMA_LOC_BOTTOMLEFT,
    AVCHROMA_LOC_BOTTOM,
    AVCHROMA_LOC_NB
};

float3 get_chroma_sample(float3 a, float3 b, float3 c,float3 d) {
    if (chroma_loc == AVCHROMA_LOC_LEFT) return (((a) + (c)) * 0.5f);
    if (chroma_loc == AVCHROMA_LOC_TOPLEFT) return a;
    if (chroma_loc == AVCHROMA_LOC_TOP) return (((a) + (b)) * 0.5f);
    if (chroma_loc == AVCHROMA_LOC_BOTTOMLEFT) return c;
    if (chroma_loc == AVCHROMA_LOC_BOTTOM) return (((c) + (d)) * 0.5f);
    return (((a) + (b) + (c) + (d)) * 0.25f);
}

float get_luma_dst(float3 c) {
    return luma_dst.x * c.x + luma_dst.y * c.y + luma_dst.z * c.z;
}

float4 get_luma_dst4(float4 r4, float4 g4, float4 b4) {
    return luma_dst.x * r4 + luma_dst.y * g4 + luma_dst.z * b4;
}

//------------
// linearizers / delinearizers

// linearizer for PQ/ST2084
float eotf_st2084_common(float x) {
    x = fmax(x, 0.0f);
    float xpow = powr(x, 1.0f / ST2084_M2);
    float num = fmax(xpow - ST2084_C1, 0.0f);
    float den = fmax(ST2084_C2 - ST2084_C3 * xpow, FLOAT_EPS);
    x = powr(num / den, 1.0f / ST2084_M1);
    return x;
}

float eotf_st2084(float x) {
    return eotf_st2084_common(x) * pq_max_lum_div_ref_white;
}

float4 eotf_st2084x4(float4 x) {
    x.x = eotf_st2084_common(x.x);
    x.y = eotf_st2084_common(x.y);
    x.z = eotf_st2084_common(x.z);
    x.w = eotf_st2084_common(x.w);
    return x * pq_max_lum_div_ref_white;
}

// delinearizer for PQ/ST2084
float inverse_eotf_st2084_common(float x) {
    x = fmax(x, 0.0f);
    float xpow = powr(x, ST2084_M1);
    float num = (ST2084_C1 - 1.0f) + (ST2084_C2 - ST2084_C3) * xpow;
    float den = 1.0f + ST2084_C3 * xpow;
    return powr(1.0f + num / den, ST2084_M2);
}

float inverse_eotf_st2084(float x) {
    x *= ref_white_div_pq_max_lum;
    return inverse_eotf_st2084_common(x);
}

float4 inverse_eotf_st2084x4(float4 x) {
    x *= ref_white_div_pq_max_lum;
    x.x = inverse_eotf_st2084_common(x.x);
    x.y = inverse_eotf_st2084_common(x.y);
    x.z = inverse_eotf_st2084_common(x.z);
    x.w = inverse_eotf_st2084_common(x.w);
    return x;
}

float ootf_1_2(float x) {
    return x > 0.0f ? powr(x, 1.2f) : x;
}

float inverse_ootf_1_2(float x) {
    return x > 0.0f ? powr(x, 1.0f / 1.2f) : x;
}

float oetf_arib_b67(float x) {
    x = fmax(x, 0.0f);
    return x <= (1.0f / 12.0f)
           ? sqrt(3.0f * x)
           : (ARIB_B67_A * log(12.0f * x - ARIB_B67_B) + ARIB_B67_C);
}

float inverse_oetf_arib_b67(float x) {
    x = fmax(x, 0.0f);
    return x <= 0.5f
           ? (x * x) * (1.0f / 3.0f)
           : (exp((x - ARIB_B67_C) / ARIB_B67_A) + ARIB_B67_B) * (1.0f / 12.0f);
}

// linearizer for HLG/ARIB-B67
float eotf_arib_b67(float x) {
    return ootf_1_2(inverse_oetf_arib_b67(x)) * 5.0f;
}

// delinearizer for HLG/ARIB-B67
float inverse_eotf_arib_b67(float x) {
    return oetf_arib_b67(inverse_ootf_1_2(x / 5.0f));
}

float4 oetf_arib_b67x4(float4 x) {
    x.x = oetf_arib_b67(x.x);
    x.y = oetf_arib_b67(x.y);
    x.z = oetf_arib_b67(x.z);
    x.w = oetf_arib_b67(x.w);
    return x;
}

float4 inverse_oetf_arib_b67x4(float4 x) {
    x.x = inverse_oetf_arib_b67(x.x);
    x.y = inverse_oetf_arib_b67(x.y);
    x.z = inverse_oetf_arib_b67(x.z);
    x.w = inverse_oetf_arib_b67(x.w);
    return x;
}

// delinearizer for BT709, BT2020-10
float inverse_eotf_bt1886(float x) {
    return x > 0.0f ? powr(x, 1.0f / 2.4f) : 0.0f;
}

float linearize(float x) {
    if (linearize_type == 1) {
        return eotf_st2084(x);
    }
    if (linearize_type == 2) {
        return eotf_arib_b67(x);
    }
    return eotf_st2084(x);
}

float delinearize(float x) {
    return inverse_eotf_bt1886(x);
}

// ------------
// Color conversion
// See libavfilter/colorspace.h for derivation of these constants
float3 yuv2rgb(float y, float u, float v) {
    y += mix(0.0f, 0.0009765774014f, y > 0.0f);
    u += mix(0.0f, 0.0009765774014f, u > 0.0f);
    v += mix(0.0f, 0.0009765774014f, v > 0.0f);
    if (is_full_range_in) {
        u -= 0.5f;
        v -= 0.5f;
    } else {
        y = 1.1678082192f * y - 0.07305936073f;
        u = 1.1417410714f * u - 0.5714285714f;
        v = 1.1417410714f * v - 0.5714285714f;
    }
    float r = (y * rgb_matrix_1[0]) + (u * rgb_matrix_1[1]) + (v * rgb_matrix_1[2]);
    float g = (y * rgb_matrix_2[0]) + (u * rgb_matrix_2[1]) + (v * rgb_matrix_2[2]);
    float b = (y * rgb_matrix_3[0]) + (u * rgb_matrix_3[1]) + (v * rgb_matrix_3[2]);
    float3 c = float3(r, g, b);
    return c;
}

float3 yuv2lrgb(float3 yuv) {
    float3 rgb = yuv2rgb(yuv.x, yuv.y, yuv.z);
    if (skip_tonemap) {
        return rgb;
    }
    float r = linearize(rgb.x);
    float g = linearize(rgb.y);
    float b = linearize(rgb.z);
    return float3(r, g, b);
}

float3 rgb2yuv(float r, float g, float b) {
    float y = (r * yuv_matrix_1[0]) + (g * yuv_matrix_1[1]) + (b * yuv_matrix_1[2]);
    float u = (r * yuv_matrix_2[0]) + (g * yuv_matrix_2[1]) + (b * yuv_matrix_2[2]);
    float v = (r * yuv_matrix_3[0]) + (g * yuv_matrix_3[1]) + (b * yuv_matrix_3[2]);
    if (is_full_range_out) {
        u += 0.5f;
        v += 0.5f;
    } else {
        if (enable_dither) {
            y = floor((219.0f * y + 16.0f) + 0.5f) / 255.0f;
            u = floor((224.0f * u + 128.0f) + 0.5f) / 255.0f;
            v = floor((224.0f * v + 128.0f) + 0.5f) / 255.0f;
        } else {
            y = floor(((219.0f * y + 16.0f) * 256.0f) + 0.5f) / 65535.0f;
            u = floor(((224.0f * u + 128.0f) * 256.0f) + 0.5f) / 65535.0f;
            v = floor(((224.0f * v + 128.0f) * 256.0f) + 0.5f) / 65535.0f;
        }
    }
    // in rgb2yuv conversion, enable_dither means output is 8bit in metal pipeline
    // use this to check if we need the 10bit offset
    if (!enable_dither) {
        y -= mix(0.0f, 0.0009765774014f, y > 0.0f);
        u -= mix(0.0f, 0.0009765774014f, u > 0.0f);
        v -= mix(0.0f, 0.0009765774014f, v > 0.0f);
    }
    return float3(y, u, v);
}

float rgb2y(float r, float g, float b) {
    float y = (r*yuv_matrix_1[0]) + (g*yuv_matrix_1[1]) + (b*yuv_matrix_1[2]);
    if (!is_full_range_out) {
        if (enable_dither) {
            y = floor((219.0f * y + 16.0f) + 0.5f) / 255.0f;
        } else {
            y = floor(((219.0f * y + 16.0f) * 256.0f) + 0.5f) / 65535.0f;
        }
    }
    if (!enable_dither) {
        y -= mix(0.0f, 0.0009765774014f, y > 0.0f);
    }
    return y;
}

float3 lrgb2yuv(float3 c) {
    if (skip_tonemap) {
        return rgb2yuv(c.x, c.y, c.z);
    }
    float r = delinearize(c.x);
    float g = delinearize(c.y);
    float b = delinearize(c.z);
    return rgb2yuv(r, g, b);
}

float lrgb2y(float3 c) {
    if (skip_tonemap) {
        return rgb2y(c.x, c.y, c.z);
    }
    float r = delinearize(c.x);
    float g = delinearize(c.y);
    float b = delinearize(c.z);
    return rgb2y(r, g, b);
}

float3 lrgb2lrgb(float3 c) {
    if (is_rgb2rgb_passthrough) {
        return c;
    }
    float r = c.x, g = c.y, b = c.z;
    float rr = (rgb2rgb_matrix_1[0] * r) + (rgb2rgb_matrix_1[1] * g) + (rgb2rgb_matrix_1[2] * b);
    float gg = (rgb2rgb_matrix_2[0] * r) + (rgb2rgb_matrix_2[1] * g) + (rgb2rgb_matrix_2[2] * b);
    float bb = (rgb2rgb_matrix_3[0] * r) + (rgb2rgb_matrix_3[1] * g) + (rgb2rgb_matrix_3[2] * b);
    return float3(rr, gg, bb);
}

float3 rgb2lrgb(float3 c) {
    if (skip_tonemap) {
        return lrgb2lrgb(float3(c.x, c.y, c.z));
    }
    float r = linearize(c.x);
    float g = linearize(c.y);
    float b = linearize(c.z);
    return float3(r, g, b);
}

float3 ycc2rgb(float y, float cb, float cr) {
    float r = y * rgb_matrix_1[0] + cb * rgb_matrix_1[1] + cr * rgb_matrix_1[2];
    float g = y * rgb_matrix_2[0] + cb * rgb_matrix_2[1] + cr * rgb_matrix_2[2];
    float b = y * rgb_matrix_3[0] + cb * rgb_matrix_3[1] + cr * rgb_matrix_3[2];
    return float3(r, g, b) + ycc2rgb_offset;
}

float3 lms2rgb(float r, float g, float b) {
    r = eotf_st2084_common(r);
    g = eotf_st2084_common(g);
    b = eotf_st2084_common(b);
    float rr = r * lms2rgb_matrix_1[0] + g * lms2rgb_matrix_1[1] + b * lms2rgb_matrix_1[2];
    float gg = r * lms2rgb_matrix_2[0] + g * lms2rgb_matrix_2[1] + b * lms2rgb_matrix_2[2];
    float bb = r * lms2rgb_matrix_3[0] + g * lms2rgb_matrix_3[1] + b * lms2rgb_matrix_3[2];
    rr = inverse_eotf_st2084_common(rr);
    gg = inverse_eotf_st2084_common(gg);
    bb = inverse_eotf_st2084_common(bb);
    return float3(rr, gg, bb);
}

// The following assumes bt2020
void lrgb2ictcp(float4 r4, float4 g4, float4 b4, thread float4* i4, thread float4* ct4, thread float4* cp4) {
    float4 l4 = 0.412109375000000f * r4 + 0.523925781250000f * g4 + 0.063964843750000f * b4;
    float4 m4 = 0.166748046875000f * r4 + 0.720458984375000f * g4 + 0.112792968750000f * b4;
    float4 s4 = 0.024169921875000f * r4 + 0.075439453125000f * g4 + 0.900390625000000f * b4;
    l4 = inverse_eotf_st2084x4(l4);
    m4 = inverse_eotf_st2084x4(m4);
    s4 = inverse_eotf_st2084x4(s4);
    *i4 = 0.5f * l4 + 0.5f * m4;
    *ct4 = 1.613769531250000f * l4 - 3.323486328125000f * m4 + 1.709716796875000f * s4;
    *cp4 = 4.378173828125000f * l4 - 4.245605468750000f * m4 - 0.132568359375000f * s4;
}

void ictcp2lrgb(float4 i4, float4 ct4, float4 cp4, thread float4* r4, thread float4* g4, thread float4* b4) {
    float4 ll4 = i4 + 0.008609037037933f * ct4 + 0.111029625003026f * cp4;
    float4 mm4 = i4 - 0.008609037037933f * ct4 - 0.111029625003026f * cp4;
    float4 ss4 = i4 + 0.560031335710679f * ct4 - 0.320627174987319f * cp4;
    ll4 = eotf_st2084x4(ll4);
    mm4 = eotf_st2084x4(mm4);
    ss4 = eotf_st2084x4(ss4);
    *r4 = 3.436606694333079f * ll4 - 2.506452118656270f * mm4 + 0.069845424323191f * ss4;
    *g4 = -0.791329555598929f * ll4 + 1.983600451792291f * mm4 - 0.192270896193362f * ss4;
    *b4 = -0.025949899690593f * ll4 - 0.098913714711726f * mm4 + 1.124863614402319f * ss4;
}

float parabolic(float x, float t0, float x0, float y0) {
    float s = (y0 - t0) / sqrt(x0 - y0);
    float ox = t0 - s * s * 0.25f;
    float oy = t0 - s * sqrt(s * s * 0.25f);
    return (x < t0 ? x : s * sqrt(x - ox) + oy);
}

float3 gamut_compress(float3 rgb) {
    #define cyan_limit 1.5187050250638159f
    #define magenta_limit 1.0750082769546088f
    #define yellow_limit 1.0887800403483898f
    #define cyan_threshold 1.050508660266247f
    #define magenta_threshold 0.940509816042432f
    #define yellow_threshold 0.9771607996420639f

    // Achromatic axis
    float ac = max3(rgb.r, rgb.g, rgb.b);

    // Inverse RGB Ratios: distance from achromatic axis
    float3 d = ac == 0.0f ? float3(0.0f) : (ac - rgb) / abs(ac);

    // Compressed distance
    float3 cd = float3(
        parabolic(d.x, cyan_threshold, cyan_limit, 1.0f),
        parabolic(d.y, magenta_threshold, magenta_limit, 1.0f),
        parabolic(d.z, yellow_threshold, yellow_limit, 1.0f)
    );

    // Inverse RGB Ratios to RGB
    float3 crgb = ac - cd * abs(ac);

    return crgb;
}


//------------
// Tonemapping methods
enum TonemapAlgorithm {
    TONEMAP_NONE,
    TONEMAP_LINEAR,
    TONEMAP_GAMMA,
    TONEMAP_CLIP,
    TONEMAP_REINHARD,
    TONEMAP_HABLE,
    TONEMAP_MOBIUS,
    TONEMAP_BT2390,
    TONEMAP_COUNT,
};

float hable_f(float in) {
    float a = 0.15f, b = 0.50f, c = 0.10f, d = 0.20f, e = 0.02f, f = 0.30f;
    return (in * (in * a + b * c) + d * e) / (in * (in * a + b) + d * f) - e / f;
}

float direct(float s, float peak, float target_peak) {
    return s;
}

float linear(float s, float peak, float target_peak) {
    return s * tone_param / peak;
}

float gamma(float s, float peak, float target_peak) {
    float p = s > 0.05f ? s / peak : 0.05f / peak;
    float v = powr(p, 1.0f / tone_param);
    return s > 0.05f ? v : (s * v / 0.05f);
}

float clip(float s, float peak, float target_peak) {
    return clamp(s * tone_param, 0.0f, 1.0f);
}

float reinhard(float s, float peak, float target_peak) {
    return s / (s + tone_param) * (peak + tone_param) / peak;
}

float hable(float s, float peak, float target_peak) {
    return hable_f(s) / hable_f(peak);
}

float mobius(float s, float peak, float target_peak) {
    float j = tone_param;
    float a, b;

    if (s <= j)
        return s;

    a = -j * j * (peak - 1.0f) / (j * j - 2.0f * j + peak);
    b = (j * j - 2.0f * j * peak + peak) / fmax(peak - 1.0f, FLOAT_EPS);

    return (b * b + 2.0f * b * j + j * j) / (b - a) * (s + a) / (s + b);
}

float bt2390(float s, float peak_inv_pq, float target_peak_inv_pq) {
    float peak_pq = peak_inv_pq;
    float scale = peak_pq > 0.0f ? (1.0f / peak_pq) : 1.0f;

    float s_pq = s * scale;
    float max_lum = target_peak_inv_pq * scale;

    float ks = (1.0f + tone_param) * max_lum - tone_param;
    float tb = (s_pq - ks) / (1.0f - ks);
    float tb2 = tb * tb;
    float tb3 = tb2 * tb;
    float pb = (2.0f * tb3 - 3.0f * tb2 + 1.0f) * ks +
               (tb3 - 2.0f * tb2 + tb) * (1.0f - ks) +
               (-2.0f * tb3 + 3.0f * tb2) * max_lum;
    float sig = mix(pb, s_pq, s_pq < ks);

    return sig * peak_pq;
}

float tonemap(float s, float peak, float target_peak) {
    if (tonemap_func_type == TONEMAP_NONE) {
        return direct(s, peak, target_peak);
    }
    if (tonemap_func_type == TONEMAP_LINEAR) {
        return linear(s, peak, target_peak);
    }
    if (tonemap_func_type == TONEMAP_GAMMA) {
        return gamma(s, peak, target_peak);
    }
    if (tonemap_func_type == TONEMAP_CLIP) {
        return clip(s, peak, target_peak);
    }
    if (tonemap_func_type == TONEMAP_REINHARD) {
        return reinhard(s, peak, target_peak);
    }
    if (tonemap_func_type == TONEMAP_HABLE) {
        return hable(s, peak, target_peak);
    }
    if (tonemap_func_type == TONEMAP_MOBIUS) {
        return mobius(s, peak, target_peak);
    }
    if (tonemap_func_type == TONEMAP_BT2390) {
        return bt2390(s, peak, target_peak);
    }
    return direct(s, peak, target_peak);
}

float get_dithered_y(float y, float d) {
    return floor(y * dither_quantization + d + 0.5f / dither_size2) * 1.0f / dither_quantization;
}

void map_four_pixels(thread float4 *r4, thread float4 *g4, thread float4 *b4, float peak) {
#define MAP_FOUR_PIXELS(sig, peak, target_peak) \
{ \
    sig.x = tonemap(sig.x, peak, target_peak); \
    sig.y = tonemap(sig.y, peak, target_peak); \
    sig.z = tonemap(sig.z, peak, target_peak); \
    sig.w = tonemap(sig.w, peak, target_peak); \
}
    if (is_tone_mode_rgb) {
        float4 sig_r = fmax(*r4, FLOAT_EPS);
        float4 sig_g = fmax(*g4, FLOAT_EPS);
        float4 sig_b = fmax(*b4, FLOAT_EPS);
        float4 sig_ro = sig_r;
        float4 sig_go = sig_g;
        float4 sig_bo = sig_b;
        if (is_tone_func_bt2390) {
            sig_r = inverse_eotf_st2084x4(fmin(sig_r, peak));
            sig_g = inverse_eotf_st2084x4(fmin(sig_g, peak));
            sig_b = inverse_eotf_st2084x4(fmin(sig_b, peak));
        }
        // Desaturate the color using a coefficient dependent on the signal level
        if (desat_param > 0.0f) {
            float4 sig = fmax(fmax(*r4, fmax(*g4, *b4)), FLOAT_EPS);
            float4 luma = get_luma_dst4(*r4, *g4, *b4);
            float4 coeff = fmax(sig - 0.18f, FLOAT_EPS) / fmax(sig, FLOAT_EPS);
            coeff = powr(coeff, 10.0f / desat_param);
            *r4 = mix(*r4, luma, coeff);
            *g4 = mix(*g4, luma, coeff);
            *b4 = mix(*b4, luma, coeff);
        }
        if (is_tone_func_bt2390) {
            float src_peak_delin_pq = inverse_eotf_st2084(peak);
            float dst_peak_delin_pq = inverse_eotf_st2084(1.0f);
            MAP_FOUR_PIXELS(sig_r, src_peak_delin_pq, dst_peak_delin_pq)
            MAP_FOUR_PIXELS(sig_g, src_peak_delin_pq, dst_peak_delin_pq)
            MAP_FOUR_PIXELS(sig_b, src_peak_delin_pq, dst_peak_delin_pq)
            sig_r = fmin(eotf_st2084x4(sig_r), peak);
            sig_g = fmin(eotf_st2084x4(sig_g), peak);
            sig_b = fmin(eotf_st2084x4(sig_b), peak);
        } else {
            MAP_FOUR_PIXELS(sig_r, peak, 1.0f)
            MAP_FOUR_PIXELS(sig_g, peak, 1.0f)
            MAP_FOUR_PIXELS(sig_b, peak, 1.0f)
            sig_r = fmin(sig_r, 1.0f);
            sig_g = fmin(sig_g, 1.0f);
            sig_b = fmin(sig_b, 1.0f);
        }
        float4 factor_r = sig_r / sig_ro;
        float4 factor_g = sig_g / sig_go;
        float4 factor_b = sig_b / sig_bo;
        *r4 *= factor_r;
        *g4 *= factor_g;
        *b4 *= factor_b;
    } else if (is_tone_mode_itp) {
        float4 i4_o, i4, ct4 , cp4;
        if (is_tone_func_bt2390) {
            *r4 = fmin(*r4, peak);
            *g4 = fmin(*g4, peak);
            *b4 = fmin(*b4, peak);
        }
        lrgb2ictcp(*r4, *g4, *b4, &i4, &ct4, &cp4);
        i4 = fmax(i4, FLOAT_EPS);
        i4_o = i4;
        if (desat_param > 0.0f) {
            float4 coeff = exp(-pow(eotf_st2084x4(i4) - (target_peak - desat_param) * 0.5f, 2) / (2.0f * peak));
            ct4 *= coeff;
            cp4 *= coeff;
        }
        if (is_tone_func_bt2390) {
            float src_peak_delin_pq = inverse_eotf_st2084(peak);
            float dst_peak_delin_pq = inverse_eotf_st2084(1.0f);
            MAP_FOUR_PIXELS(i4, src_peak_delin_pq, dst_peak_delin_pq)
        } else {
            i4 = eotf_st2084x4(i4);
            MAP_FOUR_PIXELS(i4, peak, 1.0f)
            i4 = inverse_eotf_st2084x4(i4);
        }
        i4 = fmin(i4, 1.0f);
        float4 factor = min(i4/i4_o, i4_o/i4);
        ct4 *= factor;
        cp4 *= factor;
        ictcp2lrgb(i4, ct4, cp4, r4, g4, b4);
    } else {
        float4 sig;
        if (is_tone_mode_max) {
            sig = fmax(fmax3(*r4, *g4, *b4), FLOAT_EPS);
        } else {
            sig = fmax((*r4 * 0.2627f + *g4 * 0.678f + *b4 * 0.0593f), FLOAT_EPS);
        }
        if (is_tone_func_bt2390) {
            sig = fmin(sig, peak);
        }
        float4 sig_o = sig;
        if (desat_param > 0.0f) {
            float4 luma;
            if (is_tone_mode_max) {
                luma = get_luma_dst4(*r4, *g4, *b4);
            } else {
                luma = sig;
            }
            float4 coeff = fmax(sig - 0.18f, FLOAT_EPS) / fmax(sig, FLOAT_EPS);
            coeff = powr(coeff, 10.0f / desat_param);
            *r4 = mix(*r4, luma, coeff);
            *g4 = mix(*g4, luma, coeff);
            *b4 = mix(*b4, luma, coeff);
        }
        if (is_tone_func_bt2390) {
            float src_peak_delin_pq = inverse_eotf_st2084(peak);
            float dst_peak_delin_pq = inverse_eotf_st2084(1.0f);
            sig = inverse_eotf_st2084x4(sig);
            MAP_FOUR_PIXELS(sig, src_peak_delin_pq, dst_peak_delin_pq)
            sig = fmin(eotf_st2084x4(sig), peak);
        } else {
            MAP_FOUR_PIXELS(sig, peak, 1.0f)
            sig = fmin(sig, 1.0f);
        }
        float4 factor = sig / sig_o;
        *r4 *= factor;
        *g4 *= factor;
        *b4 *= factor;
    }
}

// Map from source space YUV to source space RGB
float3 map_to_src_space_from_yuv(float3 yuv) {
    if (dovi_reshape) {
        float3 c = ycc2rgb(yuv.x, yuv.y, yuv.z);
        c = lms2rgb(c.x, c.y, c.z);
        c = rgb2lrgb(c);
        return c;
    } else {
        float3 c = yuv2lrgb(yuv);
        return c;
    }
}

// Map from source space YUV to destination space RGB
float3 map_to_dst_space_from_yuv(float3 yuv) {
    if (dovi_reshape) {
        float3 c = ycc2rgb(yuv.x, yuv.y, yuv.z);
        c = lms2rgb(c.x, c.y, c.z);
        c = rgb2lrgb(c);
        return lrgb2lrgb(c);
    } else {
        float3 c = yuv2lrgb(yuv);
        c = lrgb2lrgb(c);
        return c;
    }
}

//------------
// DOVI helpers

float reshape_poly(float s, float4 coeffs) {
    return (coeffs.z * s + coeffs.y) * s + coeffs.x;
}

float reshape_mmr(float3 sig,
                  float4 coeffs,
                  constant float4 *dovi_mmr,
                  int dovi_mmr_single,
                  int dovi_min_order,
                  int dovi_max_order)
{
    int mmr_idx = dovi_mmr_single ? 0 : (int)coeffs.y;
    int order = (int)coeffs.w;
    float4 sigX;

    float s = coeffs.x;
    sigX.xyz = sig.xxy * sig.yzz;
    sigX.w = sigX.x * sig.z;
    s += dot(dovi_mmr[mmr_idx + 0].xyz, sig);
    s += dot(dovi_mmr[mmr_idx + 1], sigX);

    int t = dovi_max_order >= 2 && (dovi_min_order >= 2 || order >= 2);
    if (t) {
        float3 sig2 = sig * sig;
        float4 sigX2 = sigX * sigX;
        s += dot(dovi_mmr[mmr_idx + 2].xyz, sig2);
        s += dot(dovi_mmr[mmr_idx + 3], sigX2);
        t = dovi_max_order == 3 && (dovi_min_order == 3 || order >= 3);
        if (t) {
            s += dot(dovi_mmr[mmr_idx + 4].xyz, sig2 * sig);
            s += dot(dovi_mmr[mmr_idx + 5], sigX2 * sigX);
        }
    }

    return s;
}

float3 reshape_dovi_yuv(float3 yuv,
                        constant float *src_dovi_params,
                        constant float *src_dovi_pivots,
                        constant float4 *src_dovi_coeffs,
                        constant float4 *src_dovi_mmr)
{
    int i;
    float s;
    float3 sig = clamp(yuv.xyz, 0.0f, 1.0f);
    float sig_arr[3] = {sig.x, sig.y, sig.z};
    float4 coeffs;
    int dovi_num_pivots, dovi_has_mmr, dovi_has_poly;
    int dovi_mmr_single, dovi_min_order, dovi_max_order;
    float dovi_lo, dovi_hi;
    constant float *dovi_params;
    constant float *dovi_pivots;
    constant float4 *dovi_coeffs, *dovi_mmr;

    #pragma clang loop unroll(full)
    for (i = 0; i < 3; i++) {
        dovi_params = src_dovi_params + i*8;
        dovi_pivots = src_dovi_pivots + i*8;
        dovi_coeffs = src_dovi_coeffs + i*8;
        dovi_mmr = src_dovi_mmr + i*48;
        dovi_num_pivots = dovi_params[0];
        dovi_has_mmr = dovi_params[1];
        dovi_has_poly = dovi_params[2];
        dovi_mmr_single = dovi_params[3];
        dovi_min_order = dovi_params[4];
        dovi_max_order = dovi_params[5];
        dovi_lo = dovi_params[6];
        dovi_hi = dovi_params[7];

        s = sig_arr[i];
        coeffs = dovi_coeffs[0];

        if (i == 0 && dovi_num_pivots > 2) {
            coeffs = mix(mix(mix(dovi_coeffs[0], dovi_coeffs[1], (float4)(s >= dovi_pivots[0])),
                             mix(dovi_coeffs[2], dovi_coeffs[3], (float4)(s >= dovi_pivots[2])),
                             (float4)(s >= dovi_pivots[1])),
                         mix(mix(dovi_coeffs[4], dovi_coeffs[5], (float4)(s >= dovi_pivots[4])),
                             mix(dovi_coeffs[6], dovi_coeffs[7], (float4)(s >= dovi_pivots[6])),
                             (float4)(s >= dovi_pivots[5])),
                         (float4)(s >= dovi_pivots[3]));
        }

        int has_mmr_poly = dovi_has_mmr && dovi_has_poly;

        if ((has_mmr_poly && coeffs.w == 0.0f) || (!has_mmr_poly && dovi_has_poly))
            s = reshape_poly(s, coeffs);
        else
            s = reshape_mmr(sig, coeffs, dovi_mmr,
                            dovi_mmr_single, dovi_min_order, dovi_max_order);

        sig_arr[i] = clamp(s, dovi_lo, dovi_hi);
    }

    return float3(sig_arr[0], sig_arr[1], sig_arr[2]);
}


//------------
// Samplers
constexpr sampler n_sampler(coord::pixel, address::clamp_to_edge, filter::nearest);
constexpr sampler l_sampler(coord::normalized, address::clamp_to_edge, filter::linear);
constexpr sampler d_sampler(coord::normalized, address::repeat, filter::nearest);

//------------
// kernel
kernel void tonemap(texture2d<float, access::write> dst1 [[texture(0)]],
                    texture2d<float, access::sample> src1 [[texture(1)]],
                    texture2d<float, access::write> dst2  [[texture(2)]],
                    texture2d<float, access::sample> src2 [[texture(3)]],
                    texture2d<float, access::write> dst3 [[texture(4), function_constant(is_non_semi_planar_out)]],
                    texture2d<float, access::sample> src3 [[texture(5), function_constant(is_non_semi_planar_in)]],
                    texture2d<float, access::sample> dither [[texture(6), function_constant(enable_dither)]],
                    constant float* dovi_buf [[buffer(7), function_constant(dovi_reshape)]],
                    constant float* peak [[buffer(8)]],
                    uint2 index [[thread_position_in_grid]])
{
    int xi = index.x;
    int yi = index.y;
    // each thread process four pixels
    int x = 2 * xi;
    int y = 2 * yi;

    int2 src1_sz = int2(src1.get_width(),
                        src1.get_height());
    int2 dst2_sz = int2(dst2.get_width(),
                        dst2.get_height());

    if (xi >= dst2_sz.x || yi >= dst2_sz.y)
        return;

    float2 ncoords_yuv0 = float2(int2(x, y)) / float2(src1_sz);
    float2 ncoords_yuv1 = float2(int2(x + 1, y)) / float2(src1_sz);
    float2 ncoords_yuv2 = float2(int2(x, y + 1)) / float2(src1_sz);
    float2 ncoords_yuv3 = float2(int2(x + 1, y + 1)) / float2(src1_sz);

    float3 yuv0, yuv1, yuv2, yuv3;

    yuv0.x = src1.sample(n_sampler, float2(x, y)).x;
    yuv1.x = src1.sample(n_sampler, float2(x + 1, y)).x;
    yuv2.x = src1.sample(n_sampler, float2(x, y + 1)).x;
    yuv3.x = src1.sample(n_sampler, float2(x + 1,y + 1)).x;

    if (is_non_semi_planar_in) {
        yuv0.yz = float2(src2.sample(l_sampler, ncoords_yuv0).x, src3.sample(l_sampler, ncoords_yuv0).x);
        yuv1.yz = float2(src2.sample(l_sampler, ncoords_yuv1).x, src3.sample(l_sampler, ncoords_yuv1).x);
        yuv2.yz = float2(src2.sample(l_sampler, ncoords_yuv2).x, src3.sample(l_sampler, ncoords_yuv2).x);
        yuv3.yz = float2(src2.sample(l_sampler, ncoords_yuv3).x, src3.sample(l_sampler, ncoords_yuv3).x);
    } else {
        yuv0.yz = float2(src2.sample(l_sampler, ncoords_yuv0).xy);
        yuv1.yz = float2(src2.sample(l_sampler, ncoords_yuv1).xy);
        yuv2.yz = float2(src2.sample(l_sampler, ncoords_yuv2).xy);
        yuv3.yz = float2(src2.sample(l_sampler, ncoords_yuv3).xy);
    }

    if (dovi_reshape) {
        constant float *dovi_params = dovi_buf;
        constant float *dovi_pivots = dovi_buf + 24;
        constant float4 *dovi_coeffs = (constant float4 *)(dovi_buf + 48);
        constant float4 *dovi_mmr = (constant float4 *)(dovi_buf + 144);
        yuv0 = reshape_dovi_yuv(yuv0, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
        yuv1 = reshape_dovi_yuv(yuv1, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
        yuv2 = reshape_dovi_yuv(yuv2, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
        yuv3 = reshape_dovi_yuv(yuv3, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
    }

    float3 c0, c1, c2, c3;

    if (map_in_src_space) {
        c0 = map_to_src_space_from_yuv(yuv0);
        c1 = map_to_src_space_from_yuv(yuv1);
        c2 = map_to_src_space_from_yuv(yuv2);
        c3 = map_to_src_space_from_yuv(yuv3);
    } else {
        c0 = map_to_dst_space_from_yuv(yuv0);
        c1 = map_to_dst_space_from_yuv(yuv1);
        c2 = map_to_dst_space_from_yuv(yuv2);
        c3 = map_to_dst_space_from_yuv(yuv3);
    }

    if(!skip_tonemap) {
        float4 r4 = float4(c0.x, c1.x, c2.x, c3.x);
        float4 g4 = float4(c0.y, c1.y, c2.y, c3.y);
        float4 b4 = float4(c0.z, c1.z, c2.z, c3.z);
        map_four_pixels(&r4, &g4, &b4, *peak);
        c0 = float3(r4.x, g4.x, b4.x);
        c1 = float3(r4.y, g4.y, b4.y);
        c2 = float3(r4.z, g4.z, b4.z);
        c3 = float3(r4.w, g4.w, b4.w);
    }

    if (map_in_src_space) {
        c0 = lrgb2lrgb(c0);
        c1 = lrgb2lrgb(c1);
        c2 = lrgb2lrgb(c2);
        c3 = lrgb2lrgb(c3);
        if (!is_rgb2rgb_passthrough) {
            c0 = gamut_compress(c0);
            c1 = gamut_compress(c1);
            c2 = gamut_compress(c2);
            c3 = gamut_compress(c3);
        }
        c0 = clamp(c0, 0.0f, 1.0f);
        c1 = clamp(c1, 0.0f, 1.0f);
        c2 = clamp(c2, 0.0f, 1.0f);
        c3 = clamp(c3, 0.0f, 1.0f);
    }

    float y0 = lrgb2y(c0);
    float y1 = lrgb2y(c1);
    float y2 = lrgb2y(c2);
    float y3 = lrgb2y(c3);

    if (enable_dither && !skip_tonemap) {
        int2 dither_sz = int2(dither.get_width(),
                              dither.get_height());;
        float2 ncoords_d = float2(int2(xi, yi)) / float2(dither_sz);
        float d = dither.sample(d_sampler, ncoords_d).x;
        y0 = get_dithered_y(y0, d), y1 = get_dithered_y(y1, d);
        y2 = get_dithered_y(y2, d), y3 = get_dithered_y(y3, d);
    }

    float3 chroma_c = get_chroma_sample(c0, c1, c2, c3);
    float3 chroma = lrgb2yuv(chroma_c);

    dst1.write(float4(y0, 0.0f, 0.0f, 1.0f), uint2(x, y));
    dst1.write(float4(y1, 0.0f, 0.0f, 1.0f), uint2(x + 1, y));
    dst1.write(float4(y2, 0.0f, 0.0f, 1.0f), uint2(x, y + 1));
    dst1.write(float4(y3, 0.0f, 0.0f, 1.0f), uint2(x + 1, y + 1));
    if (is_non_semi_planar_out) {
        dst2.write(float4(chroma.y, 0.0f, 0.0f, 1.0f), uint2(xi, yi));
        dst3.write(float4(chroma.z, 0.0f, 0.0f, 1.0f), uint2(xi, yi));
    } else {
        dst2.write(float4(chroma.y, chroma.z, 0.0f, 1.0f), uint2(xi, yi));
    }
}
