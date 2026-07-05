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

#ifdef DOVI_RESHAPE
  #undef typedef_vecs
  #undef M_ZERO_VEC
  #define typedef_vecs(base, new) \
    typedef base     new;    \
    typedef base##2  new##2; \
    typedef base##3  new##3; \
    typedef base##4  new##4; \
    typedef base##8  new##8; \
    typedef base##16 new##16;
  #ifdef DOVI_PERF_FP16
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    #define M_ZERO_VEC 0.0h
    typedef_vecs(half,  vec)
  #else
    #define M_ZERO_VEC 0.0f
    typedef_vecs(float, vec)
  #endif
  #undef typedef_vecs
#endif

#define FLOAT_EPS 1e-6f

extern float3 lrgb2yuv(float3);
extern float  lrgb2y(float3);
extern float3 yuv2lrgb(float3);
extern float3 lrgb2lrgb(float3);
extern float  eotf_st2084(float);
extern float  inverse_eotf_st2084(float);
extern float4 get_luma_dstx4(float4, float4, float4);
extern float3 get_chroma_sample(float3, float3, float3, float3);
#ifdef DOVI_RESHAPE
extern float3 rgb2lrgb(float3);
extern float3 ycc2rgb(float, float, float);
extern float3 lms2rgb(float, float, float);
#endif
extern float4 eotf_st2084x4(float4 x);
extern float4 inverse_eotf_st2084x4(float4 x);
#ifdef TONE_MODE_ITP
extern void lrgb2ictcp(float4 r4, float4 g4, float4 b4, float4* i4, float4* ct4, float4* cp4);
extern void ictcp2lrgb(float4 i4, float4 ct4, float4 cp4, float4* r4, float4* g4, float4* b4);
#endif
extern float3 gamut_compress(float3 rgb);

#ifdef ENABLE_DITHER
float get_dithered_y(float y, float d) {
    return floor(y * dither_quantization + d + 0.5f / dither_size2) * 1.0f / dither_quantization;
}
#endif

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
    float v = native_powr(p, 1.0f / tone_param);
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

#define MAP_FOUR_PIXELS(sig, peak, target_peak) \
{ \
    sig.x = TONE_FUNC(sig.x, peak, target_peak); \
    sig.y = TONE_FUNC(sig.y, peak, target_peak); \
    sig.z = TONE_FUNC(sig.z, peak, target_peak); \
    sig.w = TONE_FUNC(sig.w, peak, target_peak); \
}

#ifndef TONE_MODE_ITP
void map_four_pixels_rgb(float4 *r4, float4 *g4, float4 *b4, float peak) {
#ifdef TONE_MODE_RGB
    float4 sig_r = fmax(*r4, FLOAT_EPS), sig_ro = sig_r;
    float4 sig_g = fmax(*g4, FLOAT_EPS), sig_go = sig_g;
    float4 sig_b = fmax(*b4, FLOAT_EPS), sig_bo = sig_b;
#else
  #ifdef TONE_MODE_MAX
    float4 sig = fmax(fmax(*r4, fmax(*g4, *b4)), FLOAT_EPS);
  #else
    float4 sig = fmax((*r4 * 0.2627f + *g4 * 0.678f + *b4 * 0.0593f), FLOAT_EPS);
  #endif
    float4 sig_o = sig;
#endif

    // Desaturate the color using a coefficient dependent on the signal level
    if (desat_param > 0.0f) {
#ifdef TONE_MODE_RGB
        float4 sig = fmax(fmax(*r4, fmax(*g4, *b4)), FLOAT_EPS);
#endif
#ifdef MAP_IN_DST_SPACE
        float4 luma = get_luma_dstx4(*r4, *g4, *b4);
#else // only LUM mode currently
        float4 luma = sig;
#endif
        float4 coeff = fmax(sig - 0.18f, FLOAT_EPS) / fmax(sig, FLOAT_EPS);
        coeff = native_powr(coeff, 10.0f / desat_param);
        *r4 = mix(*r4, luma, coeff);
        *g4 = mix(*g4, luma, coeff);
        *b4 = mix(*b4, luma, coeff);
    }

#ifdef TONE_FUNC_BT2390
    float src_peak_delin_pq = inverse_eotf_st2084(peak);
    float dst_peak_delin_pq = inverse_eotf_st2084(target_peak);
  #ifdef TONE_MODE_RGB
    sig_r = inverse_eotf_st2084x4(fmin(sig_r, peak));
    sig_g = inverse_eotf_st2084x4(fmin(sig_g, peak));
    sig_b = inverse_eotf_st2084x4(fmin(sig_b, peak));
    MAP_FOUR_PIXELS(sig_r, src_peak_delin_pq, dst_peak_delin_pq)
    MAP_FOUR_PIXELS(sig_g, src_peak_delin_pq, dst_peak_delin_pq)
    MAP_FOUR_PIXELS(sig_b, src_peak_delin_pq, dst_peak_delin_pq)
    sig_r = eotf_st2084x4(sig_r);
    sig_g = eotf_st2084x4(sig_g);
    sig_b = eotf_st2084x4(sig_b);
  #else
    sig = inverse_eotf_st2084x4(fmin(sig, peak));
    MAP_FOUR_PIXELS(sig, src_peak_delin_pq, dst_peak_delin_pq)
    sig = eotf_st2084x4(sig);
  #endif
#else
  #ifdef TONE_MODE_RGB
    MAP_FOUR_PIXELS(sig_r, peak, target_peak)
    MAP_FOUR_PIXELS(sig_g, peak, target_peak)
    MAP_FOUR_PIXELS(sig_b, peak, target_peak)
  #else
    MAP_FOUR_PIXELS(sig, peak, target_peak)
  #endif
#endif

#ifdef TONE_MODE_RGB
    sig_r = fmin(sig_r, 1.0f);
    sig_g = fmin(sig_g, 1.0f);
    sig_b = fmin(sig_b, 1.0f);
    float4 factor_r = sig_r / sig_ro;
    float4 factor_g = sig_g / sig_go;
    float4 factor_b = sig_b / sig_bo;
    *r4 *= factor_r;
    *g4 *= factor_g;
    *b4 *= factor_b;
#else
    sig = fmin(sig, 1.0f);
    float4 factor = sig / sig_o;
    *r4 *= factor;
    *g4 *= factor;
    *b4 *= factor;
#endif
}
#endif

#ifdef TONE_MODE_ITP
void map_four_pixels_itp(float4 *r4, float4 *g4, float4 *b4, float peak) {
    float4 i4_o, i4, ct4 , cp4;
#ifdef TONE_FUNC_BT2390
    *r4 = fmin(*r4, peak);
    *g4 = fmin(*g4, peak);
    *b4 = fmin(*b4, peak);
#endif
    lrgb2ictcp(*r4, *g4, *b4, &i4, &ct4, &cp4);
    i4 = fmax(i4, FLOAT_EPS);
    i4_o = i4;
    if (desat_param > 0.0f) {
        float4 coeff = native_exp(-pow(eotf_st2084x4(i4) - (target_peak - desat_param) * 0.5f, 2) / (2.0f * peak));
        ct4 *= coeff;
        cp4 *= coeff;
    }
#ifdef TONE_FUNC_BT2390
    float src_peak_delin_pq = inverse_eotf_st2084(peak);
    float dst_peak_delin_pq = inverse_eotf_st2084(target_peak);
    MAP_FOUR_PIXELS(i4, src_peak_delin_pq, dst_peak_delin_pq)
#else
    i4 = eotf_st2084x4(i4);
    MAP_FOUR_PIXELS(i4, peak, target_peak)
    i4 = inverse_eotf_st2084x4(i4);
#endif
    i4 = fmin(i4, 1.0f);
    float4 factor = min(i4/i4_o, i4_o/i4);
    ct4 *= factor;
    cp4 *= factor;
    ictcp2lrgb(i4, ct4, cp4, r4, g4, b4);
}
#endif

// Map from source space YUV to source space RGB
float3 map_to_src_space_from_yuv(float3 yuv) {
#ifdef DOVI_RESHAPE
    float3 c = ycc2rgb(yuv.x, yuv.y, yuv.z);
    c = lms2rgb(c.x, c.y, c.z);
    c = rgb2lrgb(c);
#else
    float3 c = yuv2lrgb(yuv);
#endif
    return c;
}

// Map from source space YUV to destination space RGB
float3 map_to_dst_space_from_yuv(float3 yuv) {
#ifdef DOVI_RESHAPE
    float3 c = ycc2rgb(yuv.x, yuv.y, yuv.z);
    c = lms2rgb(c.x, c.y, c.z);
    c = rgb2lrgb(c);
    c = lrgb2lrgb(c);
#else
    float3 c = yuv2lrgb(yuv);
    c = lrgb2lrgb(c);
#endif
    return c;
}

#ifdef DOVI_RESHAPE
vec reshape_mmr(vec3 sig,
                vec4 coeffs,
                __global const vec4 *dovi_mmr,
                uchar dovi_mmr_single,
                uchar dovi_min_order,
                uchar dovi_max_order)
{
    uchar mmr_idx = dovi_mmr_single ? 0 : (uchar)coeffs.y;
    uchar order = (uchar)coeffs.w;
    vec4 sigX;
    bool t;

    vec s = coeffs.x;
    sigX.xyz = sig.xxy * sig.yzz;
    sigX.w = sigX.x * sig.z;
    s += dot(dovi_mmr[mmr_idx + 0].xyz, sig);
    s += dot(dovi_mmr[mmr_idx + 1], sigX);

    // Branching here is faster from testing, divergence rate for I channel seems to be low
    t = dovi_max_order >= 2 && (dovi_min_order >= 2 || order >= 2);
    if (t) {
        vec3 sig2 = sig * sig;
        vec4 sigX2 = sigX * sigX;
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

#ifndef IS_QCOM_GPU
vec4 reshape_polyx4(vec4 s, vec4 coeffsx, vec4 coeffsy, vec4 coeffsz) {
    return mad(mad(coeffsz, s, coeffsy), s, coeffsx);
}

vec4 reshape_mmrx4(vec4 sig_i4,
                   vec4 sig_p4,
                   vec4 sig_t4,
                   vec4 coeffsx,
                   vec4 coeffsy,
                   vec4 coeffsz,
                   vec4 coeffsw,
                   __global const vec4 *dovi_mmr,
                   uchar dovi_mmr_single,
                   uchar dovi_min_order,
                   uchar dovi_max_order)
{
    vec4 coeffs = (vec4)(coeffsx.x, coeffsy.x, coeffsz.x, coeffsw.x);
    vec4 result;
    result.x = reshape_mmr((vec3)(sig_i4.x, sig_p4.x, sig_t4.x), coeffs, dovi_mmr,
                           dovi_mmr_single, dovi_min_order, dovi_max_order);
    coeffs = (vec4)(coeffsx.y, coeffsy.y, coeffsz.y, coeffsw.y);
    result.y = reshape_mmr((vec3)(sig_i4.y, sig_p4.y, sig_t4.y), coeffs, dovi_mmr,
                           dovi_mmr_single, dovi_min_order, dovi_max_order);
    coeffs = (vec4)(coeffsx.z, coeffsy.z, coeffsz.z, coeffsw.z);
    result.z = reshape_mmr((vec3)(sig_i4.z, sig_p4.z, sig_t4.z), coeffs, dovi_mmr,
                           dovi_mmr_single, dovi_min_order, dovi_max_order);
    coeffs = (vec4)(coeffsx.w, coeffsy.w, coeffsz.w, coeffsw.w);
    result.w = reshape_mmr((vec3)(sig_i4.w, sig_p4.w, sig_t4.w), coeffs, dovi_mmr,
                           dovi_mmr_single, dovi_min_order, dovi_max_order);
    return result;
}

vec4 reshape_mmr_ptx4(vec4 sig_i4,
                      vec4 sig_p4,
                      vec4 sig_t4,
                      vec4 coeffs,
                      __global const vec4 *dovi_mmr,
                      uchar dovi_mmr_single,
                      uchar dovi_min_order,
                      uchar dovi_max_order)
{
    uchar mmr_idx = dovi_mmr_single ? 0 : (uchar)coeffs.y;
  #ifdef DOVI_PERF_FP16
    const vec8 mmr_order1 = vload8(0, (__global const vec *)&dovi_mmr[mmr_idx]);
  #else
    const vec8 mmr_order1 = (vec8)(dovi_mmr[mmr_idx + 0], dovi_mmr[mmr_idx + 1]);
  #endif
    uchar order = (uchar)coeffs.w;
    bool t;
    vec4 sigXx, sigXy, sigXz, sigXw;
    vec4 sx4 = (vec4)coeffs.x;
    sigXx = sig_i4 * sig_p4;
    sigXy = sig_i4 * sig_t4;
    sigXz = sig_p4 * sig_t4;
    sigXw = sigXx * sig_t4;

  #define DOT_MMR(A, B, C, D, E, F, G, ORD)  \
    sx4 = mad((A), (vec4)(ORD.lo.x), sx4);   \
    sx4 = mad((B), (vec4)(ORD.lo.y), sx4);   \
    sx4 = mad((C), (vec4)(ORD.lo.z), sx4);   \
    sx4 = mad((D), (vec4)(ORD.hi.x), sx4);   \
    sx4 = mad((E), (vec4)(ORD.hi.y), sx4);   \
    sx4 = mad((F), (vec4)(ORD.hi.z), sx4);   \
    sx4 = mad((G), (vec4)(ORD.hi.w), sx4);   \

    DOT_MMR(sig_i4, sig_p4, sig_t4, sigXx, sigXy, sigXz, sigXw, mmr_order1)

    // Branching is free for PT channels as the condition t is same for all threads.
    t = dovi_max_order >= 2 && (dovi_min_order >= 2 || order >= 2);
    if (t) {
  #ifdef DOVI_PERF_FP16
        const vec8 mmr_order2 = vload8(1, (__global const vec *)&dovi_mmr[mmr_idx]);
  #else
        const vec8 mmr_order2 = (vec8)(dovi_mmr[mmr_idx + 2], dovi_mmr[mmr_idx + 3]);
  #endif
        vec4 sig2_i4 = sig_i4 * sig_i4;
        vec4 sig2_p4 = sig_p4 * sig_p4;
        vec4 sig2_t4 = sig_t4 * sig_t4;
        vec4 sigX2x = sigXx * sigXx;
        vec4 sigX2y = sigXy * sigXy;
        vec4 sigX2z = sigXz * sigXz;
        vec4 sigX2w = sigXw * sigXw;

        DOT_MMR(sig2_i4, sig2_p4, sig2_t4, sigX2x, sigX2y, sigX2z, sigX2w, mmr_order2);

        t = dovi_max_order == 3 && (dovi_min_order == 3 || order >= 3);
        if (t) {
  #ifdef DOVI_PERF_FP16
            const vec8 mmr_order3 = vload8(2, (__global const vec *)&dovi_mmr[mmr_idx]);
  #else
            const vec8 mmr_order3 = (vec8)(dovi_mmr[mmr_idx + 4], dovi_mmr[mmr_idx + 5]);
  #endif
            DOT_MMR(sig2_i4 * sig_i4, sig2_p4 * sig_p4, sig2_t4 * sig_t4,
                    sigX2x * sigXx, sigX2y * sigXy, sigX2z * sigXz, sigX2w * sigXw,
                    mmr_order3);
        }
    }

    return sx4;
  #undef DOT_MMR
}

void reshape_dovi_iptx4(float3 *ipt0,
                        float3 *ipt1,
                        float3 *ipt2,
                        float3 *ipt3,
                        __global const vec *src_dovi_params,
                        __global const vec *src_dovi_pivots,
                        __global const vec4 *src_dovi_coeffs,
                        __global const vec4 *src_dovi_mmr)
{
    bool has_mmr_poly, t;
    vec4 do_poly, coeffw_is_zero;
    vec4 coeffs, coeffsx, coeffsy, coeffsz, coeffsw, sx4;
    float4 result;
    uchar dovi_num_pivots, dovi_has_mmr, dovi_has_poly;
    uchar dovi_mmr_single, dovi_min_order, dovi_max_order;
    vec dovi_lo, dovi_hi;
    __global const vec *dovi_params;
    __global const vec *dovi_pivots;
    __global const vec4 *dovi_coeffs, *dovi_mmr;

  #ifdef DOVI_PERF_FP16
    vec4 sig_i4 = convert_half4(clamp((vec4)((*ipt0).x,(*ipt1).x,(*ipt2).x,(*ipt3).x), 0.0f, 1.0f));
    vec4 sig_p4 = convert_half4(clamp((vec4)((*ipt0).y,(*ipt1).y,(*ipt2).y,(*ipt3).y), 0.0f, 1.0f));
    vec4 sig_t4 = convert_half4(clamp((vec4)((*ipt0).z,(*ipt1).z,(*ipt2).z,(*ipt3).z), 0.0f, 1.0f));
  #else
    vec4 sig_i4 = clamp((vec4)((*ipt0).x,(*ipt1).x,(*ipt2).x,(*ipt3).x), 0.0f, 1.0f);
    vec4 sig_p4 = clamp((vec4)((*ipt0).y,(*ipt1).y,(*ipt2).y,(*ipt3).y), 0.0f, 1.0f);
    vec4 sig_t4 = clamp((vec4)((*ipt0).z,(*ipt1).z,(*ipt2).z,(*ipt3).z), 0.0f, 1.0f);
  #endif

  #define SETUP_DOVI_PARAMS(channel_offset) \
    dovi_params = src_dovi_params + (channel_offset)*8; \
    dovi_pivots = src_dovi_pivots + (channel_offset)*8; \
    dovi_coeffs = src_dovi_coeffs + (channel_offset)*8; \
    dovi_mmr = src_dovi_mmr + (channel_offset)*48;      \
    dovi_num_pivots = dovi_params[0];                   \
    dovi_has_mmr = dovi_params[1];                      \
    dovi_has_poly = dovi_params[2];                     \
    dovi_mmr_single = dovi_params[3];                   \
    dovi_min_order = dovi_params[4];                    \
    dovi_max_order = dovi_params[5];                    \
    dovi_lo = dovi_params[6];                           \
    dovi_hi = dovi_params[7];

  #define EXTRACT_COEFFS() \
    coeffs = dovi_coeffs[0];  \
    coeffsx = (vec4)coeffs.x; \
    coeffsy = (vec4)coeffs.y; \
    coeffsz = (vec4)coeffs.z; \
    coeffsw = (vec4)coeffs.w;

    // Reshape I
    SETUP_DOVI_PARAMS(0)
    EXTRACT_COEFFS()

    if (dovi_num_pivots > 2) {
  #ifdef DOVI_PERF_FP16
        const vec8 pivots0 = vload8(0, (__global const vec *)dovi_pivots);
        const vec8 coeffs0 = vload8(0, (__global const vec *)dovi_coeffs);
        const vec8 coeffs1 = vload8(1, (__global const vec *)dovi_coeffs);
        const vec8 coeffs2 = vload8(2, (__global const vec *)dovi_coeffs);
        const vec8 coeffs3 = vload8(3, (__global const vec *)dovi_coeffs);

        const vec *pivots = (const vec *)&pivots0;
  #else
        __global const vec *pivots = dovi_pivots;
        const vec8 coeffs0 = (vec8)(dovi_coeffs[0], dovi_coeffs[1]);
        const vec8 coeffs1 = (vec8)(dovi_coeffs[2], dovi_coeffs[3]);
        const vec8 coeffs2 = (vec8)(dovi_coeffs[4], dovi_coeffs[5]);
        const vec8 coeffs3 = (vec8)(dovi_coeffs[6], dovi_coeffs[7]);
  #endif

  #define PICK_COEFF_FOR(LANE) \
        mix(                                                          \
          mix(                                                        \
            mix(coeffs0.lo, coeffs0.hi, (vec4)((LANE) >= pivots[0])), \
            mix(coeffs1.lo, coeffs1.hi, (vec4)((LANE) >= pivots[2])), \
            (vec4)((LANE) >= pivots[1])                               \
          ),                                                          \
          mix(                                                        \
            mix(coeffs2.lo, coeffs2.hi, (vec4)((LANE) >= pivots[4])), \
            mix(coeffs3.lo, coeffs3.hi, (vec4)((LANE) >= pivots[6])), \
            (vec4)((LANE) >= pivots[5])                               \
          ),                                                          \
          (vec4)((LANE) >= pivots[3])                                 \
        )

  #define PACK_COEFFS(LANE) \
        coeffsx.LANE = coeffs_temp.x; \
        coeffsy.LANE = coeffs_temp.y; \
        coeffsz.LANE = coeffs_temp.z; \
        coeffsw.LANE = coeffs_temp.w;

        vec4 coeffs_temp = PICK_COEFF_FOR(sig_i4.x);
        PACK_COEFFS(x)

        coeffs_temp = PICK_COEFF_FOR(sig_i4.y);
        PACK_COEFFS(y)

        coeffs_temp = PICK_COEFF_FOR(sig_i4.z);
        PACK_COEFFS(z)

        coeffs_temp = PICK_COEFF_FOR(sig_i4.w);
        PACK_COEFFS(w)

  #undef PICK_COEFF_FOR
  #undef PACK_COEFFS
    }

    has_mmr_poly = dovi_has_mmr && dovi_has_poly;
  #ifdef DOVI_PERF_FP16
    coeffw_is_zero = convert_half4(coeffsw == (vec4)M_ZERO_VEC);
  #else
    coeffw_is_zero = convert_float4(coeffsw == (vec4)M_ZERO_VEC);
  #endif
    do_poly = has_mmr_poly
              ? coeffw_is_zero
              : (vec4)(dovi_has_poly != M_ZERO_VEC);

    sx4 = mix(reshape_mmrx4(sig_i4, sig_p4, sig_t4,
                            coeffsx, coeffsy, coeffsz, coeffsw, dovi_mmr,
                            dovi_mmr_single, dovi_min_order, dovi_max_order),
              reshape_polyx4(sig_i4, coeffsx, coeffsy, coeffsz),
              do_poly);
  #ifdef DOVI_PERF_FP16
    result = convert_float4(clamp(sx4, dovi_lo, dovi_hi));
  #else
    result = clamp(sx4, dovi_lo, dovi_hi);
  #endif

  #define STORE_RESULTS(LANE) \
    (*ipt0).LANE = result.x; \
    (*ipt1).LANE = result.y; \
    (*ipt2).LANE = result.z; \
    (*ipt3).LANE = result.w;

    STORE_RESULTS(x)

    // Reshape P
    SETUP_DOVI_PARAMS(1)
    EXTRACT_COEFFS()

  #define RESHAPE_P_T(sig_x4) \
    has_mmr_poly = dovi_has_mmr && dovi_has_poly;                               \
    t = has_mmr_poly ? coeffs.w == M_ZERO_VEC : dovi_has_poly != M_ZERO_VEC;    \
    sx4 = t ? reshape_polyx4(sig_x4, coeffsx, coeffsy, coeffsz)                 \
            : reshape_mmr_ptx4(sig_i4, sig_p4, sig_t4,                          \
                               coeffs, dovi_mmr,                                \
                               dovi_mmr_single, dovi_min_order, dovi_max_order);

    RESHAPE_P_T(sig_p4)
  #ifdef DOVI_PERF_FP16
    result = convert_float4(clamp(sx4, dovi_lo, dovi_hi));
  #else
    result = clamp(sx4, dovi_lo, dovi_hi);
  #endif
    STORE_RESULTS(y)

    // Reshape T
    SETUP_DOVI_PARAMS(2)
    EXTRACT_COEFFS()

    RESHAPE_P_T(sig_t4)
  #ifdef DOVI_PERF_FP16
    result = convert_float4(clamp(sx4, dovi_lo, dovi_hi));
  #else
    result = clamp(sx4, dovi_lo, dovi_hi);
  #endif
    STORE_RESULTS(z)

  #undef RESHAPE_P_T
  #undef STORE_RESULTS
  #undef SETUP_DOVI_PARAMS
  #undef EXTRACT_COEFFS
}
#endif //#ifndef IS_QCOM_GPU

// Qualcomm has a really bad OpenCL compiler that is having performance regression with vectorized reshaping kernel
// Make a scalar version just for them
#ifdef IS_QCOM_GPU
static inline vec reshape_poly(vec s, vec4 coeffs) {
    return (coeffs.z * s + coeffs.y) * s + coeffs.x;
}

static inline void reshape_dovi_sig(float *dst,
                                    const vec src,
                                    const vec3 *sig,
                                    const uchar idx,
                                    __global const vec *src_dovi_params,
                                    __global const vec *src_dovi_pivots,
                                    __global const vec4 *src_dovi_coeffs,
                                    __global const vec4 *src_dovi_mmr)
{
    bool t, has_mmr_poly;
    vec s = src;
    vec4 coeffs;
    uchar dovi_num_pivots, dovi_has_mmr, dovi_has_poly;
    uchar dovi_mmr_single, dovi_min_order, dovi_max_order;
    vec dovi_lo, dovi_hi;
    __global const vec *dovi_params;
    __global const vec *dovi_pivots;
    __global const vec4 *dovi_coeffs, *dovi_mmr;

    dovi_params = src_dovi_params + (idx<<3);
    dovi_pivots = src_dovi_pivots + (idx<<3);
    dovi_coeffs = src_dovi_coeffs + (idx<<3);
    dovi_mmr = src_dovi_mmr + idx*48;
    dovi_num_pivots = dovi_params[0];
    dovi_has_mmr = dovi_params[1];
    dovi_has_poly = dovi_params[2];
    dovi_mmr_single = dovi_params[3];
    dovi_min_order = dovi_params[4];
    dovi_max_order = dovi_params[5];
    dovi_lo = dovi_params[6];
    dovi_hi = dovi_params[7];

    coeffs = dovi_coeffs[0];

    if (idx == 0 && dovi_num_pivots > 2) {
  #ifdef DOVI_PERF_FP16
        const vec8 pivots0 = vload8(0, (__global const vec *)dovi_pivots);
        const vec8 coeffs0 = vload8(0, (__global const vec *)dovi_coeffs);
        const vec8 coeffs1 = vload8(1, (__global const vec *)dovi_coeffs);
        const vec8 coeffs2 = vload8(2, (__global const vec *)dovi_coeffs);
        const vec8 coeffs3 = vload8(3, (__global const vec *)dovi_coeffs);

        const vec *pivots = (const vec *)&pivots0;
  #else
        __global const vec *pivots = dovi_pivots;
        const vec8 coeffs0 = (vec8)(dovi_coeffs[0], dovi_coeffs[1]);
        const vec8 coeffs1 = (vec8)(dovi_coeffs[2], dovi_coeffs[3]);
        const vec8 coeffs2 = (vec8)(dovi_coeffs[4], dovi_coeffs[5]);
        const vec8 coeffs3 = (vec8)(dovi_coeffs[6], dovi_coeffs[7]);
  #endif
        coeffs = mix(mix(mix(coeffs0.lo, coeffs0.hi, (vec4)(s >= pivots[0])),
                         mix(coeffs1.lo, coeffs1.hi, (vec4)(s >= pivots[2])),
                         (vec4)(s >= pivots[1])),
                     mix(mix(coeffs2.lo, coeffs2.hi, (vec4)(s >= pivots[4])),
                         mix(coeffs3.lo, coeffs3.hi, (vec4)(s >= pivots[6])),
                         (vec4)(s >= pivots[5])),
                     (vec4)(s >= pivots[3]));
    }

    has_mmr_poly = dovi_has_mmr && dovi_has_poly;
    t = (has_mmr_poly && coeffs.w == M_ZERO_VEC) || (!has_mmr_poly && dovi_has_poly);

    s = t ? reshape_poly(s, coeffs)
          : reshape_mmr(*sig, coeffs, dovi_mmr,
                        dovi_mmr_single, dovi_min_order, dovi_max_order);
  #ifdef DOVI_PERF_FP16
    *dst = convert_float(clamp(s, dovi_lo, dovi_hi));
  #else
    *dst = clamp(s, dovi_lo, dovi_hi);
  #endif
}

void reshape_dovi_ipt(float3 *ipt,
                      __global const vec *src_dovi_params,
                      __global const vec *src_dovi_pivots,
                      __global const vec4 *src_dovi_coeffs,
                      __global const vec4 *src_dovi_mmr)
{
  #ifdef DOVI_PERF_FP16
    const vec3 sig = convert_half3(clamp(*ipt, 0.0f, 1.0f));
  #else
    const vec3 sig = clamp(*ipt, 0.0f, 1.0f);
  #endif
    float dsti, dstp, dstt;
    reshape_dovi_sig(&dsti, sig.x, &sig, 0, src_dovi_params, src_dovi_pivots, src_dovi_coeffs, src_dovi_mmr);
    reshape_dovi_sig(&dstp, sig.y, &sig, 1, src_dovi_params, src_dovi_pivots, src_dovi_coeffs, src_dovi_mmr);
    reshape_dovi_sig(&dstt, sig.z, &sig, 2, src_dovi_params, src_dovi_pivots, src_dovi_coeffs, src_dovi_mmr);
    *ipt = (float3)(dsti, dstp, dstt);
}
#endif //#ifdef IS_QCOM_GPU
#endif //#ifdef DOVI_RESHAPE

__constant sampler_t n_sampler = (CLK_NORMALIZED_COORDS_FALSE |
                                  CLK_ADDRESS_CLAMP_TO_EDGE   |
                                  CLK_FILTER_NEAREST);

__constant sampler_t l_sampler = (CLK_NORMALIZED_COORDS_TRUE  |
                                  CLK_ADDRESS_CLAMP_TO_EDGE   |
                                  CLK_FILTER_LINEAR);

__constant sampler_t d_sampler = (CLK_NORMALIZED_COORDS_TRUE  |
                                  CLK_ADDRESS_REPEAT          |
                                  CLK_FILTER_NEAREST);

__kernel void tonemap(__write_only image2d_t dst1,
                      __read_only  image2d_t src1,
                      __write_only image2d_t dst2,
                      __read_only  image2d_t src2,
#ifdef NON_SEMI_PLANAR_OUT
                      __write_only image2d_t dst3,
#endif
#ifdef NON_SEMI_PLANAR_IN
                      __read_only  image2d_t src3,
#endif
#ifdef ENABLE_DITHER
                      __read_only  image2d_t dither,
#endif
#ifdef DOVI_RESHAPE
                      __global const vec *dovi_buf,
#endif
                      float peak)
{
    int xi = get_global_id(0);
    int yi = get_global_id(1);
    // each work item process four pixels
    int x = xi << 1;
    int y = yi << 1;

    int2 src1_sz = get_image_dim(src1);
    int2 dst2_sz = get_image_dim(dst2);

    if (xi >= dst2_sz.x || yi >= dst2_sz.y)
        return;

    float2 src1_sz_recip = native_recip(convert_float2(src1_sz));
    float2 ncoords_yuv0 = convert_float2((int2)(x,     y)) * src1_sz_recip;
    float2 ncoords_yuv1 = convert_float2((int2)(x + 1, y)) * src1_sz_recip;
    float2 ncoords_yuv2 = convert_float2((int2)(x,     y + 1)) * src1_sz_recip;
    float2 ncoords_yuv3 = convert_float2((int2)(x + 1, y + 1)) * src1_sz_recip;

    float3 yuv0, yuv1, yuv2, yuv3;

#ifndef P010LE_COMPACT_IN
    yuv0.x = read_imagef(src1, n_sampler, (int2)(x,     y)).x;
    yuv1.x = read_imagef(src1, n_sampler, (int2)(x + 1, y)).x;
    yuv2.x = read_imagef(src1, n_sampler, (int2)(x,     y + 1)).x;
    yuv3.x = read_imagef(src1, n_sampler, (int2)(x + 1, y + 1)).x;
#else
    uint off0 = ((x + 0) << 1) & 7;
    uint off1 = ((x + 1) << 1) & 7;
    int2 pos0 = (int2)((x + 0) * 1.25f,     y + 0);
    int2 pos1 = (int2)((x + 0) * 1.25f + 1, y + 0);
    int2 pos2 = (int2)((x + 1) * 1.25f,     y + 0);
    int2 pos3 = (int2)((x + 1) * 1.25f + 1, y + 0);
    int2 pos4 = (int2)((x + 0) * 1.25f,     y + 1);
    int2 pos5 = (int2)((x + 0) * 1.25f + 1, y + 1);
    int2 pos6 = (int2)((x + 1) * 1.25f,     y + 1);
    int2 pos7 = (int2)((x + 1) * 1.25f + 1, y + 1);
    uint4 px4ui;
    float4 px4f;
    px4ui.x = read_imageui(src1, n_sampler, pos0).x >> off0 |
              read_imageui(src1, n_sampler, pos1).x << (8 - off0);
    px4ui.y = read_imageui(src1, n_sampler, pos2).x >> off1 |
              read_imageui(src1, n_sampler, pos3).x << (8 - off1);
    px4ui.z = read_imageui(src1, n_sampler, pos4).x >> off0 |
              read_imageui(src1, n_sampler, pos5).x << (8 - off0);
    px4ui.w = read_imageui(src1, n_sampler, pos6).x >> off1 |
              read_imageui(src1, n_sampler, pos7).x << (8 - off1);
    px4f = convert_float4((px4ui & 0x3FF) << 6) / USHRT_MAX;
    yuv0.x = px4f.x, yuv1.x = px4f.y, yuv2.x = px4f.z, yuv3.x = px4f.w;
#endif

#ifdef NON_SEMI_PLANAR_IN
    yuv0.yz = (float2)(read_imagef(src2, l_sampler, ncoords_yuv0).x,
                       read_imagef(src3, l_sampler, ncoords_yuv0).x);
    yuv1.yz = (float2)(read_imagef(src2, l_sampler, ncoords_yuv1).x,
                       read_imagef(src3, l_sampler, ncoords_yuv1).x);
    yuv2.yz = (float2)(read_imagef(src2, l_sampler, ncoords_yuv2).x,
                       read_imagef(src3, l_sampler, ncoords_yuv2).x);
    yuv3.yz = (float2)(read_imagef(src2, l_sampler, ncoords_yuv3).x,
                       read_imagef(src3, l_sampler, ncoords_yuv3).x);
#else
  #ifndef P010LE_COMPACT_IN
    yuv0.yz = read_imagef(src2, l_sampler, ncoords_yuv0).xy;
    yuv1.yz = read_imagef(src2, l_sampler, ncoords_yuv1).xy;
    yuv2.yz = read_imagef(src2, l_sampler, ncoords_yuv2).xy;
    yuv3.yz = read_imagef(src2, l_sampler, ncoords_yuv3).xy;
  #else
    off0 = ((xi * 2 + 0) << 1) & 7;
    off1 = ((xi * 2 + 1) << 1) & 7;
    pos0 = (int2)((xi * 2 + 0) * 1.25f,     yi);
    pos1 = (int2)((xi * 2 + 0) * 1.25f + 1, yi);
    pos2 = (int2)((xi * 2 + 1) * 1.25f,     yi);
    pos3 = (int2)((xi * 2 + 1) * 1.25f + 1, yi);
    px4ui.x = read_imageui(src2, n_sampler, pos0).x >> off0 |
              read_imageui(src2, n_sampler, pos1).x << (8 - off0);
    px4ui.y = read_imageui(src2, n_sampler, pos2).x >> off1 |
              read_imageui(src2, n_sampler, pos3).x << (8 - off1);
    yuv0.yz = convert_float2((px4ui.xy & 0x3FF) << 6) / USHRT_MAX;
    yuv1.yz = yuv2.yz = yuv3.yz = yuv0.yz;
  #endif
#endif

#ifdef DOVI_RESHAPE
    __global const vec *dovi_params = dovi_buf;
    __global const vec *dovi_pivots = dovi_buf + 24;
    __global const vec4 *dovi_coeffs = (__global const vec4 *)(dovi_buf + 48);
    __global const vec4 *dovi_mmr = (__global const vec4 *)(dovi_buf + 144);
  #ifdef IS_QCOM_GPU
    reshape_dovi_ipt(&yuv0, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
    reshape_dovi_ipt(&yuv1, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
    reshape_dovi_ipt(&yuv2, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
    reshape_dovi_ipt(&yuv3, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
  #else
    reshape_dovi_iptx4(&yuv0, &yuv1, &yuv2, &yuv3, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
  #endif
#endif

    float3 c0, c1, c2, c3;
#ifndef MAP_IN_DST_SPACE
    c0 = map_to_src_space_from_yuv(yuv0);
    c1 = map_to_src_space_from_yuv(yuv1);
    c2 = map_to_src_space_from_yuv(yuv2);
    c3 = map_to_src_space_from_yuv(yuv3);
#else
    c0 = map_to_dst_space_from_yuv(yuv0);
    c1 = map_to_dst_space_from_yuv(yuv1);
    c2 = map_to_dst_space_from_yuv(yuv2);
    c3 = map_to_dst_space_from_yuv(yuv3);
#endif

#ifndef SKIP_TONEMAP
    float4 r4 = (float4)(c0.x, c1.x, c2.x, c3.x);
    float4 g4 = (float4)(c0.y, c1.y, c2.y, c3.y);
    float4 b4 = (float4)(c0.z, c1.z, c2.z, c3.z);
  #ifdef TONE_MODE_ITP
    map_four_pixels_itp(&r4, &g4, &b4, peak);
  #else
    map_four_pixels_rgb(&r4, &g4, &b4, peak);
  #endif
    c0 = (float3)(r4.x, g4.x, b4.x);
    c1 = (float3)(r4.y, g4.y, b4.y);
    c2 = (float3)(r4.z, g4.z, b4.z);
    c3 = (float3)(r4.w, g4.w, b4.w);
#endif

#ifndef MAP_IN_DST_SPACE
    c0 = lrgb2lrgb(c0);
    c1 = lrgb2lrgb(c1);
    c2 = lrgb2lrgb(c2);
    c3 = lrgb2lrgb(c3);
  #if !defined(RGB2RGB_PASSTHROUGH)
    c0 = gamut_compress(c0);
    c1 = gamut_compress(c1);
    c2 = gamut_compress(c2);
    c3 = gamut_compress(c3);
  #endif
    c0 = clamp(c0, 0.0f, 1.0f);
    c1 = clamp(c1, 0.0f, 1.0f);
    c2 = clamp(c2, 0.0f, 1.0f);
    c3 = clamp(c3, 0.0f, 1.0f);
#endif

    float y0 = lrgb2y(c0);
    float y1 = lrgb2y(c1);
    float y2 = lrgb2y(c2);
    float y3 = lrgb2y(c3);

#if defined(ENABLE_DITHER) && !defined(SKIP_TONEMAP)
    int2 dither_sz = get_image_dim(dither);
    float2 dither_sz_recip = native_recip(convert_float2(dither_sz));
    float2 ncoords_d = convert_float2((int2)(xi, yi)) * dither_sz_recip;
    float d = read_imagef(dither, d_sampler, ncoords_d).x;
    y0 = get_dithered_y(y0, d), y1 = get_dithered_y(y1, d);
    y2 = get_dithered_y(y2, d), y3 = get_dithered_y(y3, d);
#endif

    float3 chroma_c = get_chroma_sample(c0, c1, c2, c3);
    float3 chroma = lrgb2yuv(chroma_c);

    write_imagef(dst1, (int2)(x,     y), (float4)(y0, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, (int2)(x + 1, y), (float4)(y1, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, (int2)(x,     y + 1), (float4)(y2, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, (int2)(x + 1, y + 1), (float4)(y3, 0.0f, 0.0f, 1.0f));
#ifdef NON_SEMI_PLANAR_OUT
    write_imagef(dst2, (int2)(xi, yi), (float4)(chroma.y, 0.0f, 0.0f, 1.0f));
    write_imagef(dst3, (int2)(xi, yi), (float4)(chroma.z, 0.0f, 0.0f, 1.0f));
#else
    write_imagef(dst2, (int2)(xi, yi), (float4)(chroma.y, chroma.z, 0.0f, 1.0f));
#endif
}

#undef lut3d_read_t
#ifdef LUT_PERF_IMAGE3D
  #define lut3d_read_t __read_only image3d_t
#else
  #define lut3d_read_t __global const float4 *restrict
#endif

float3 apply_lut3d(lut3d_read_t lut, float3 color)
{
    color = clamp(color, 0.0f, 1.0f);

    // Scale the color to the LUT grid
    float3 pos = color * (float)(LUT_SIZE - 1);

    // Get the integer base indices in the LUT
    int3 base = clamp(convert_int3(floor(pos)), 0, LUT_SIZE - 2);

    // Compute the fractional part within the cell
    float3 f = pos - convert_float3(base);

    // Sort the fraction offsets, so that we always have f_max>=f_mid>=f_min
    float f_max = fmax(f.x, fmax(f.y, f.z));
    float f_min = fmin(f.x, fmin(f.y, f.z));
    float f_mid = f.x + f.y + f.z - f_max - f_min;

    // The initial and the last corner values of current cube will always be fetched
#ifdef LUT_PERF_IMAGE3D
    float3 c000 = read_imagef(lut, n_sampler, (int4)(base + 0, 0)).xyz;
    float3 c111 = read_imagef(lut, n_sampler, (int4)(base + 1, 0)).xyz;
#else
    // Compute the base linear index
    uint base_idx = base.x + base.y * LUT_SIZE + base.z * LUT_SIZE * LUT_SIZE;
    uint last_idx = base_idx + 1 + LUT_SIZE + LUT_SIZE * LUT_SIZE;
    float3 c000 = lut[base_idx].xyz;
    float3 c111 = lut[last_idx].xyz;
#endif

    // Select the index for vertices of the tetrahedron:
    // Although we have f_max and f_min values, we cannot use them in the
    // following selection as float equality comparison is not accurate on GPU
#ifdef LUT_PERF_IMAGE3D
    int3 y_max = (int3)(-(f.y >= f.z && f.y >= f.x));
    int3 x_max = (int3)(-(f.x >= f.y && f.x >= f.z));
    int3 d0 = select(select((int3)(0, 0, 1), (int3)(0, 1, 0), y_max), (int3)(1, 0, 0), x_max);
    int3 y_min = (int3)(-(f.y <= f.z && f.y <= f.x));
    int3 x_min = (int3)(-(f.x <= f.y && f.x <= f.z));
    int3 d1 = select(select((int3)(1, 1, 0), (int3)(1, 0, 1), y_min), (int3)(0, 1, 1), x_min);
#else
    uint idx100 = base_idx + 1;
    uint idx010 = base_idx + LUT_SIZE;
    uint idx110 = base_idx + 1 + LUT_SIZE;
    uint idx001 = base_idx + LUT_SIZE * LUT_SIZE;
    uint idx101 = base_idx + 1 + LUT_SIZE * LUT_SIZE;
    uint idx011 = base_idx + LUT_SIZE + LUT_SIZE * LUT_SIZE;
    uint y_max = f.y >= f.z && f.y >= f.x;
    uint x_max = f.x >= f.y && f.x >= f.z;
    uint idx0 = select(select(idx001, idx010, y_max), idx100, x_max);
    uint y_min = f.y <= f.z && f.y <= f.x;
    uint x_min = f.x <= f.y && f.x <= f.z;
    uint idx1 = select(select(idx110, idx101, y_min), idx011, x_min);
#endif

    // Fetch LUT value with determined tetrahedron
#ifdef LUT_PERF_IMAGE3D
    float3 c0 = read_imagef(lut, n_sampler, (int4)(base + d0, 0)).xyz;
    float3 c1 = read_imagef(lut, n_sampler, (int4)(base + d1, 0)).xyz;
#else
    float3 c0 = lut[idx0].xyz;
    float3 c1 = lut[idx1].xyz;
#endif

    return clamp(c000 + f_max * (c0   - c000)
                      + f_mid * (c1   - c0)
                      + f_min * (c111 - c1), 0.0f, 1.0f);
}

__kernel void tonemap_lut(          lut3d_read_t  lut,
                          __write_only image2d_t dst1,
                          __read_only  image2d_t src1,
                          __write_only image2d_t dst2,
                          __read_only  image2d_t src2,
#ifdef NON_SEMI_PLANAR_OUT
                          __write_only image2d_t dst3,
#endif
#ifdef NON_SEMI_PLANAR_IN
                          __read_only  image2d_t src3,
#endif
#ifdef ENABLE_DITHER
                          __read_only  image2d_t dither,
#endif
#ifdef DOVI_RESHAPE
                          __global const vec *restrict dovi_buf,
#endif
                          float peak)
{
    int xi = get_global_id(0);
    int yi = get_global_id(1);
    // each work item process four pixels
    int x = xi << 1;
    int y = yi << 1;

    int2 src1_sz = get_image_dim(src1);
    int2 dst2_sz = get_image_dim(dst2);

    if (xi >= dst2_sz.x || yi >= dst2_sz.y)
        return;

    float2 src1_sz_recip = native_recip(convert_float2(src1_sz));
    float2 ncoords_yuv0 = convert_float2((int2)(x,     y)) * src1_sz_recip;
    float2 ncoords_yuv1 = convert_float2((int2)(x + 1, y)) * src1_sz_recip;
    float2 ncoords_yuv2 = convert_float2((int2)(x,     y + 1)) * src1_sz_recip;
    float2 ncoords_yuv3 = convert_float2((int2)(x + 1, y + 1)) * src1_sz_recip;

    float3 yuv0, yuv1, yuv2, yuv3;

#ifndef P010LE_COMPACT_IN
    yuv0.x = read_imagef(src1, n_sampler, (int2)(x,     y)).x;
    yuv1.x = read_imagef(src1, n_sampler, (int2)(x + 1, y)).x;
    yuv2.x = read_imagef(src1, n_sampler, (int2)(x,     y + 1)).x;
    yuv3.x = read_imagef(src1, n_sampler, (int2)(x + 1, y + 1)).x;
#else
    uint off0 = ((x + 0) << 1) & 7;
    uint off1 = ((x + 1) << 1) & 7;
    int2 pos0 = (int2)((x + 0) * 1.25f,     y + 0);
    int2 pos1 = (int2)((x + 0) * 1.25f + 1, y + 0);
    int2 pos2 = (int2)((x + 1) * 1.25f,     y + 0);
    int2 pos3 = (int2)((x + 1) * 1.25f + 1, y + 0);
    int2 pos4 = (int2)((x + 0) * 1.25f,     y + 1);
    int2 pos5 = (int2)((x + 0) * 1.25f + 1, y + 1);
    int2 pos6 = (int2)((x + 1) * 1.25f,     y + 1);
    int2 pos7 = (int2)((x + 1) * 1.25f + 1, y + 1);
    uint4 px4ui;
    float4 px4f;
    px4ui.x = read_imageui(src1, n_sampler, pos0).x >> off0 |
              read_imageui(src1, n_sampler, pos1).x << (8 - off0);
    px4ui.y = read_imageui(src1, n_sampler, pos2).x >> off1 |
              read_imageui(src1, n_sampler, pos3).x << (8 - off1);
    px4ui.z = read_imageui(src1, n_sampler, pos4).x >> off0 |
              read_imageui(src1, n_sampler, pos5).x << (8 - off0);
    px4ui.w = read_imageui(src1, n_sampler, pos6).x >> off1 |
              read_imageui(src1, n_sampler, pos7).x << (8 - off1);
    px4f = convert_float4((px4ui & 0x3FF) << 6) / USHRT_MAX;
    yuv0.x = px4f.x, yuv1.x = px4f.y, yuv2.x = px4f.z, yuv3.x = px4f.w;
#endif

#ifdef NON_SEMI_PLANAR_IN
    yuv0.yz = (float2)(read_imagef(src2, l_sampler, ncoords_yuv0).x,
                       read_imagef(src3, l_sampler, ncoords_yuv0).x);
    yuv1.yz = (float2)(read_imagef(src2, l_sampler, ncoords_yuv1).x,
                       read_imagef(src3, l_sampler, ncoords_yuv1).x);
    yuv2.yz = (float2)(read_imagef(src2, l_sampler, ncoords_yuv2).x,
                       read_imagef(src3, l_sampler, ncoords_yuv2).x);
    yuv3.yz = (float2)(read_imagef(src2, l_sampler, ncoords_yuv3).x,
                       read_imagef(src3, l_sampler, ncoords_yuv3).x);
#else
  #ifndef P010LE_COMPACT_IN
    yuv0.yz = read_imagef(src2, l_sampler, ncoords_yuv0).xy;
    yuv1.yz = read_imagef(src2, l_sampler, ncoords_yuv1).xy;
    yuv2.yz = read_imagef(src2, l_sampler, ncoords_yuv2).xy;
    yuv3.yz = read_imagef(src2, l_sampler, ncoords_yuv3).xy;
  #else
    off0 = ((xi * 2 + 0) << 1) & 7;
    off1 = ((xi * 2 + 1) << 1) & 7;
    pos0 = (int2)((xi * 2 + 0) * 1.25f,     yi);
    pos1 = (int2)((xi * 2 + 0) * 1.25f + 1, yi);
    pos2 = (int2)((xi * 2 + 1) * 1.25f,     yi);
    pos3 = (int2)((xi * 2 + 1) * 1.25f + 1, yi);
    px4ui.x = read_imageui(src2, n_sampler, pos0).x >> off0 |
              read_imageui(src2, n_sampler, pos1).x << (8 - off0);
    px4ui.y = read_imageui(src2, n_sampler, pos2).x >> off1 |
              read_imageui(src2, n_sampler, pos3).x << (8 - off1);
    yuv0.yz = convert_float2((px4ui.xy & 0x3FF) << 6) / USHRT_MAX;
    yuv1.yz = yuv2.yz = yuv3.yz = yuv0.yz;
  #endif
#endif

#ifdef DOVI_RESHAPE
    __global const vec *dovi_params = dovi_buf;
    __global const vec *dovi_pivots = dovi_buf + 24;
    __global const vec4 *dovi_coeffs = (__global const vec4 *)(dovi_buf + 48);
    __global const vec4 *dovi_mmr = (__global const vec4 *)(dovi_buf + 144);
  #ifdef IS_QCOM_GPU
    reshape_dovi_ipt(&yuv0, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
    reshape_dovi_ipt(&yuv1, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
    reshape_dovi_ipt(&yuv2, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
    reshape_dovi_ipt(&yuv3, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
  #else
    reshape_dovi_iptx4(&yuv0, &yuv1, &yuv2, &yuv3, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
  #endif
#endif

    float3 c0, c1, c2, c3;

    c0 = apply_lut3d(lut, yuv0);
    c1 = apply_lut3d(lut, yuv1);
    c2 = apply_lut3d(lut, yuv2);
    c3 = apply_lut3d(lut, yuv3);

    float3 chroma = get_chroma_sample(c0, c1, c2, c3);

    write_imagef(dst1, (int2)(x,     y), (float4)(c0.x, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, (int2)(x + 1, y), (float4)(c1.x, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, (int2)(x,     y + 1), (float4)(c2.x, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, (int2)(x + 1, y + 1), (float4)(c3.x, 0.0f, 0.0f, 1.0f));
#ifdef NON_SEMI_PLANAR_OUT
    write_imagef(dst2, (int2)(xi, yi), (float4)(chroma.y, 0.0f, 0.0f, 1.0f));
    write_imagef(dst3, (int2)(xi, yi), (float4)(chroma.z, 0.0f, 0.0f, 1.0f));
#else
    write_imagef(dst2, (int2)(xi, yi), (float4)(chroma.y, chroma.z, 0.0f, 1.0f));
#endif
}

__kernel void build_lut(__global float4 *lut, float peak)
{
    const int total_entries = LUT_SIZE * LUT_SIZE * LUT_SIZE;
    int idx = get_global_id(0);
    if (idx >= total_entries) return;
    int z = idx / (LUT_SIZE * LUT_SIZE);
    int rem = idx - (z * LUT_SIZE * LUT_SIZE);
    int y = rem / LUT_SIZE;
    int x = rem % LUT_SIZE;
    float fx = (float)x / (LUT_SIZE - 1);
    float fy = (float)y / (LUT_SIZE - 1);
    float fz = (float)z / (LUT_SIZE - 1);
    float3 c = (float3)(fx, fy, fz);
#ifndef MAP_IN_DST_SPACE
    c = map_to_src_space_from_yuv(c);
#else
    c = map_to_dst_space_from_yuv(c);
#endif
    float4 r4 = (float4)(c.x, c.x, c.x, c.x);
    float4 g4 = (float4)(c.y, c.y, c.y, c.y);
    float4 b4 = (float4)(c.z, c.z, c.z, c.z);
#ifndef SKIP_TONEMAP
  #ifdef TONE_MODE_ITP
    map_four_pixels_itp(&r4, &g4, &b4, peak);
  #else
    map_four_pixels_rgb(&r4, &g4, &b4, peak);
  #endif
#endif
    c = (float3)(r4.x, g4.x, b4.x);
#ifndef MAP_IN_DST_SPACE
    c = lrgb2lrgb(c);
  #ifndef RGB2RGB_PASSTHROUGH
    c = gamut_compress(c);
  #endif
    c = clamp(c, 0.0f, 1.0f);
#endif
    c = lrgb2yuv(c);
    lut[idx] = clamp((float4)(c, 0.0f), 0.0f, 1.0f);
}
