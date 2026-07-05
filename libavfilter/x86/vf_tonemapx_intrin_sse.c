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

#include "vf_tonemapx_intrin_sse.h"

#ifdef ENABLE_TONEMAPX_SSE_INTRINSICS
#    include <immintrin.h>
#endif // ENABLE_TONEMAPX_SSE_INTRINSICS

#ifdef ENABLE_TONEMAPX_SSE_INTRINSICS
// GCC 10 and below does not implement _mm_storeu_si32 with movd instruction
// cast the register into float register and store with movss as a workaround
#if (defined(__GNUC__) && !defined(__clang__)) && (__GNUC__ <= 10)
__attribute__((always_inline))
X86_64_V2 static inline void _mm_storeu_si32(void* mem_addr, __m128i a) {
    _mm_store_ss((float*)mem_addr, _mm_castsi128_ps(a));
    return;
}
#endif

X86_64_V2 static inline __m128i av_clip_uint16_sse(__m128i a)
{
    __m128i mask = _mm_set1_epi32(0x7FFF);
    __m128i condition = _mm_and_si128(a, _mm_set1_epi32(~0x7FFF));

    __m128i zero = _mm_setzero_si128();
    __m128i cmp = _mm_cmpeq_epi32(condition, zero);

    __m128i neg_a = _mm_and_si128(_mm_srai_epi32(_mm_xor_si128(a, _mm_set1_epi32(-1)), 31), mask);
    __m128i result = _mm_or_si128(_mm_and_si128(cmp, a), _mm_andnot_si128(cmp, neg_a));

    return result;
}

X86_64_V2 static inline __m128i av_clip_int16_sse(__m128i a)
{
    __m128i add_result = _mm_add_epi32(a, _mm_set1_epi32(0x8000U));
    __m128i mask = _mm_set1_epi32(~0xFFFF);
    __m128i condition = _mm_and_si128(add_result, mask);
    __m128i cmp = _mm_cmpeq_epi32(condition, _mm_setzero_si128());

    __m128i shifted = _mm_srai_epi32(a, 31);
    __m128i xor_result = _mm_xor_si128(shifted, _mm_set1_epi32(0x7FFF));

    return _mm_or_si128(_mm_and_si128(cmp, a), _mm_andnot_si128(cmp, xor_result));
}

/*
X86_64_V2 inline static __m128 mix_float32x4(__m128 x, __m128 y, __m128 a)
{
    __m128 n = _mm_sub_ps(y, x);
    n = _mm_mul_ps(n, a);
    n = _mm_add_ps(n, x);
    return n;
}
*/

X86_64_V2 inline static float reduce_floatx4(__m128 x) {
    x = _mm_hadd_ps(x, x);
    x = _mm_hadd_ps(x, x);
    return _mm_cvtss_f32(x);
}

X86_64_V2 static inline float reshape_poly(float s, __m128 coeffs)
{
    __m128 ps = _mm_set_ps(0.0f, s * s, s, 1.0f);
    ps = _mm_mul_ps(ps, coeffs);
    return reduce_floatx4(ps);
}

X86_64_V2 inline static float reshape_mmr(__m128 sig, __m128 coeffs, const float* mmr,
                                          int mmr_single, int min_order, int max_order)
{
    float s = _mm_cvtss_f32(coeffs);
    int mmr_idx = 0;
    int order = 0;

    __m128 mmr_coeffs, ps;
    __m128 sigX01 = _mm_mul_ps(sig, _mm_shuffle_ps(sig, sig, _MM_SHUFFLE(1, 1, 1, 1))); // {sig[0]*sig[1], sig[1]*sig[1], sig[2]*sig[1], sig[3]*sig[1]}
    __m128 sigX02 = _mm_mul_ps(sig, _mm_shuffle_ps(sig, sig, _MM_SHUFFLE(2, 2, 2, 2))); // {sig[0]*sig[2], sig[1]*sig[2], sig[2]*sig[2], sig[3]*sig[2]}
    __m128 sigX12 = _mm_mul_ps(sigX01, _mm_shuffle_ps(sig, sig, _MM_SHUFFLE(2, 2, 2, 2))); // {sig[0]*sig[1]*sig[2], sig[1]*sig[1]*sig[2], sig[2]*sig[1]*sig[2], sig[3]*sig[1]*sig[2]}
    __m128 sigX = sigX01; // sig[0]*sig[1] now positioned at 0

    sigX = _mm_insert_ps(sigX, sigX02, _MM_MK_INSERTPS_NDX(0, 1, 0)); // sig[0]*sig[2] at 1
    sigX = _mm_insert_ps(sigX, sigX02, _MM_MK_INSERTPS_NDX(1, 2, 0)); // sig[1]*sig[2] at 2
    sigX = _mm_insert_ps(sigX, sigX12, _MM_MK_INSERTPS_NDX(0, 3, 0)); // sig[0]*sig[1]*sig[2] at 3

    mmr_idx = mmr_single ? 0 : (int)_mm_cvtss_f32(_mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3, 2, 0, 1)));
    order = (int)_mm_cvtss_f32(_mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(1, 2, 0, 3)));

    // dot first order
    mmr_coeffs = _mm_loadu_ps(&mmr[mmr_idx + 0*4]);
    ps = _mm_mul_ps(sig, mmr_coeffs);
    s += reduce_floatx4(ps);
    mmr_coeffs = _mm_loadu_ps(&mmr[mmr_idx + 1*4]);
    ps = _mm_mul_ps(sigX, mmr_coeffs);
    s += reduce_floatx4(ps);

    if (max_order >= 2 && (min_order >= 2 || order >= 2)) {
        __m128 sig2 = _mm_mul_ps(sig, sig);
        __m128 sigX2 = _mm_mul_ps(sigX, sigX);

        mmr_coeffs = _mm_loadu_ps(&mmr[mmr_idx + 2*4]);
        ps = _mm_mul_ps(sig2, mmr_coeffs);
        s += reduce_floatx4(ps);
        mmr_coeffs = _mm_loadu_ps(&mmr[mmr_idx + 3*4]);
        ps = _mm_mul_ps(sigX2, mmr_coeffs);
        s += reduce_floatx4(ps);

        if (max_order == 3 && (min_order == 3 || order >= 3)) {
            __m128 sig3 = _mm_mul_ps(sig2, sig);
            __m128 sigX3 = _mm_mul_ps(sigX2, sigX);

            mmr_coeffs = _mm_loadu_ps(&mmr[mmr_idx + 4*4]);
            ps = _mm_mul_ps(sig3, mmr_coeffs);
            s += reduce_floatx4(ps);
            mmr_coeffs = _mm_loadu_ps(&mmr[mmr_idx + 5*4]);
            ps = _mm_mul_ps(sigX3, mmr_coeffs);
            s += reduce_floatx4(ps);
        }
    }

    return s;
}

#define CLAMP(a, b, c) (FFMIN(FFMAX((a), (b)), (c)))
X86_64_V2 inline static __m128 reshape_dovi_iptpqc2(__m128 sig, const TonemapIntParams *ctx)
{
    int has_mmr_poly;
    float s;

    float *src_dovi_params = ctx->dovi_pbuf;
    float *src_dovi_pivots = ctx->dovi_pbuf + 24;
    float *src_dovi_coeffs = ctx->dovi_pbuf + 48; //float4*
    float *src_dovi_mmr = ctx->dovi_pbuf + 144; //float4*

    float* dovi_params_i = src_dovi_params + 0*8;
    float* dovi_pivots_i = src_dovi_pivots + 0*8;
    float* dovi_coeffs_i = src_dovi_coeffs + 0 * 8 * 4; //float4*
    float* dovi_mmr_i = src_dovi_mmr + 0 * 48 * 4; //float4*
    int dovi_num_pivots_i = dovi_params_i[0];
    int dovi_has_mmr_i = dovi_params_i[1];
    int dovi_has_poly_i = dovi_params_i[2];
    int dovi_mmr_single_i = dovi_params_i[3];
    int dovi_min_order_i = dovi_params_i[4];
    int dovi_max_order_i = dovi_params_i[5];
    float dovi_lo_i = dovi_params_i[6];
    float dovi_hi_i = dovi_params_i[7];

    float* dovi_params_p = src_dovi_params + 1*8;
    float* dovi_coeffs_p = src_dovi_coeffs + 1*8 * 4; //float4*
    float* dovi_mmr_p = src_dovi_mmr + 1*48 * 4; //float4*
    int dovi_has_mmr_p = dovi_params_p[1];
    int dovi_has_poly_p = dovi_params_p[2];
    int dovi_mmr_single_p = dovi_params_p[3];
    int dovi_min_order_p = dovi_params_p[4];
    int dovi_max_order_p = dovi_params_p[5];
    float dovi_lo_p = dovi_params_p[6];
    float dovi_hi_p = dovi_params_p[7];

    float* dovi_params_t = src_dovi_params + 2*8;
    float* dovi_coeffs_t = src_dovi_coeffs + 2*8 * 4; //float4*
    float* dovi_mmr_t = src_dovi_mmr + 2*48 * 4; //float4*
    int dovi_has_mmr_t = dovi_params_t[1];
    int dovi_has_poly_t = dovi_params_t[2];
    int dovi_mmr_single_t = dovi_params_t[3];
    int dovi_min_order_t = dovi_params_t[4];
    int dovi_max_order_t = dovi_params_t[5];
    float dovi_lo_t = dovi_params_t[6];
    float dovi_hi_t = dovi_params_t[7];

    __m128 coeffs, result;

    // reshape I
    s = _mm_cvtss_f32(sig);
    result = sig;
    if (dovi_num_pivots_i > 2) {
        __m128 m01 = s >= dovi_pivots_i[0] ? _mm_loadu_ps(dovi_coeffs_i + 4) : _mm_loadu_ps(dovi_coeffs_i);
        __m128 m23 = s >= dovi_pivots_i[2] ? _mm_loadu_ps(dovi_coeffs_i + 3*4) : _mm_loadu_ps(dovi_coeffs_i + 2*4);
        __m128 m0123 = s >= dovi_pivots_i[1] ? m23 : m01;
        __m128 m45 = s >= dovi_pivots_i[4] ? _mm_loadu_ps(dovi_coeffs_i + 5*4) : _mm_loadu_ps(dovi_coeffs_i + 4*4);
        __m128 m67 = s >= dovi_pivots_i[6] ? _mm_loadu_ps(dovi_coeffs_i + 7*4) : _mm_loadu_ps(dovi_coeffs_i + 6*4);
        __m128 m4567 = s >= dovi_pivots_i[5] ? m67 : m45;
        coeffs = s >= dovi_pivots_i[3] ? m4567 : m0123;
    } else {
        coeffs = _mm_loadu_ps(dovi_coeffs_i);
    }

    has_mmr_poly = dovi_has_mmr_i && dovi_has_poly_i;

    if ((has_mmr_poly && _mm_cvtss_f32(_mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3, 3, 3, 3))) == 0.0f) || (!has_mmr_poly && dovi_has_poly_i))
        s = reshape_poly(s, coeffs);
    else
        s = reshape_mmr(result, coeffs, dovi_mmr_i,
                        dovi_mmr_single_i, dovi_min_order_i, dovi_max_order_i);

    result = _mm_insert_ps(result, _mm_set1_ps(CLAMP(s, dovi_lo_i, dovi_hi_i)), _MM_MK_INSERTPS_NDX(0, 0, 0));

    // reshape P
    s = _mm_cvtss_f32(_mm_shuffle_ps(sig, sig, _MM_SHUFFLE(1, 1, 1, 1)));
    coeffs = _mm_loadu_ps(dovi_coeffs_p);
    has_mmr_poly = dovi_has_mmr_p && dovi_has_poly_p;

    if ((has_mmr_poly && _mm_cvtss_f32(_mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3, 3, 3, 3))) == 0.0f) || (!has_mmr_poly && dovi_has_poly_p))
        s = reshape_poly(s, coeffs);
    else
        s = reshape_mmr(result, coeffs, dovi_mmr_p,
                        dovi_mmr_single_p, dovi_min_order_p, dovi_max_order_p);

    result = _mm_insert_ps(result, _mm_set1_ps(CLAMP(s, dovi_lo_p, dovi_hi_p)), _MM_MK_INSERTPS_NDX(0, 1, 0));

    // reshape T
    s = _mm_cvtss_f32(_mm_shuffle_ps(sig, sig, _MM_SHUFFLE(2, 2, 2, 2)));
    coeffs = _mm_loadu_ps(dovi_coeffs_t);
    has_mmr_poly = dovi_has_mmr_t && dovi_has_poly_t;

    if ((has_mmr_poly && _mm_cvtss_f32(_mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3, 3, 3, 3))) == 0.0f) || (!has_mmr_poly && dovi_has_poly_t))
        s = reshape_poly(s, coeffs);
    else
        s = reshape_mmr(result, coeffs, dovi_mmr_t,
                        dovi_mmr_single_t, dovi_min_order_t, dovi_max_order_t);

    result = _mm_insert_ps(result, _mm_set1_ps(CLAMP(s, dovi_lo_t, dovi_hi_t)), _MM_MK_INSERTPS_NDX(0, 2, 0));

    return result;
}

X86_64_V2 inline static void ycc2rgbx4(__m128* dy, __m128* dcb, __m128* dcr,
                                       __m128 y, __m128 cb, __m128 cr,
                                       const double nonlinear[3][3], const float ycc_offset[3])
{
    *dy = _mm_mul_ps(y, _mm_set1_ps((float)nonlinear[0][0]));
    *dy = _mm_add_ps(*dy, _mm_mul_ps(cb, _mm_set1_ps((float)nonlinear[0][1])));
    *dy = _mm_add_ps(*dy, _mm_mul_ps(cr, _mm_set1_ps((float)nonlinear[0][2])));
    *dy = _mm_sub_ps(*dy, _mm_set1_ps(ycc_offset[0]));

    *dcb = _mm_mul_ps(y, _mm_set1_ps((float)nonlinear[1][0]));
    *dcb = _mm_add_ps(*dcb, _mm_mul_ps(cb, _mm_set1_ps((float)nonlinear[1][1])));
    *dcb = _mm_add_ps(*dcb, _mm_mul_ps(cr, _mm_set1_ps((float)nonlinear[1][2])));
    *dcb = _mm_sub_ps(*dcb, _mm_set1_ps(ycc_offset[1]));

    *dcr = _mm_mul_ps(y, _mm_set1_ps((float)nonlinear[2][0]));
    *dcr = _mm_add_ps(*dcr, _mm_mul_ps(cb, _mm_set1_ps((float)nonlinear[2][1])));
    *dcr = _mm_add_ps(*dcr, _mm_mul_ps(cr, _mm_set1_ps((float)nonlinear[2][2])));
    *dcr = _mm_sub_ps(*dcr, _mm_set1_ps(ycc_offset[2]));
}

X86_64_V2 inline static void lms2rgbx4(__m128* dl, __m128* dm, __m128* ds,
                                       __m128 l, __m128 m, __m128 s,
                                       const double lms2rgb_matrix[3][3])
{
    *dl = _mm_mul_ps(l, _mm_set1_ps((float)lms2rgb_matrix[0][0]));
    *dl = _mm_add_ps(*dl, _mm_mul_ps(m, _mm_set1_ps((float)lms2rgb_matrix[0][1])));
    *dl = _mm_add_ps(*dl, _mm_mul_ps(s, _mm_set1_ps((float)lms2rgb_matrix[0][2])));

    *dm = _mm_mul_ps(l, _mm_set1_ps((float)lms2rgb_matrix[1][0]));
    *dm = _mm_add_ps(*dm, _mm_mul_ps(m, _mm_set1_ps((float)lms2rgb_matrix[1][1])));
    *dm = _mm_add_ps(*dm, _mm_mul_ps(s, _mm_set1_ps((float)lms2rgb_matrix[1][2])));

    *ds = _mm_mul_ps(l, _mm_set1_ps((float)lms2rgb_matrix[2][0]));
    *ds = _mm_add_ps(*ds, _mm_mul_ps(m, _mm_set1_ps((float)lms2rgb_matrix[2][1])));
    *ds = _mm_add_ps(*ds, _mm_mul_ps(s, _mm_set1_ps((float)lms2rgb_matrix[2][2])));
}

X86_64_V2 static inline void tonemap_int32x4_sse(__m128i r_in, __m128i g_in, __m128i b_in,
                                                 int16_t *r_out, int16_t *g_out, int16_t *b_out,
                                                 float *lin_lut, float *tonemap_lut, uint16_t *delin_lut,
                                                 const AVLumaCoefficients *coeffs,
                                                 const AVLumaCoefficients *ocoeffs, double desat,
                                                 double (*rgb2rgb)[3][3],
                                                 int rgb2rgb_passthrough)
{
    __m128i sig4;
    __m128 mapvalx4, r_linx4, g_linx4, b_linx4;
    __m128 offset = _mm_set1_ps(0.5f);
    __m128 intermediate_upper_bound = _mm_set1_ps(JPEG_SCALE);
    __m128i r, g, b, rx4, gx4, bx4;

    float mapval4[4], r_lin4[4], g_lin4[4], b_lin4[4];

    r = av_clip_uint16_sse(r_in);
    g = av_clip_uint16_sse(g_in);
    b = av_clip_uint16_sse(b_in);

    sig4 = _mm_max_epi32(r, _mm_max_epi32(g, b));

    // Cannot use loop here as the lane has to be compile-time constant
#define LOAD_LUT(i) mapval4[i] = tonemap_lut[_mm_extract_epi32(sig4, i)]; \
r_lin4[i] = lin_lut[_mm_extract_epi32(r, i)];                             \
g_lin4[i] = lin_lut[_mm_extract_epi32(g, i)];                             \
b_lin4[i] = lin_lut[_mm_extract_epi32(b, i)];

    LOAD_LUT(0)
    LOAD_LUT(1)
    LOAD_LUT(2)
    LOAD_LUT(3)

#undef LOAD_LUT

    mapvalx4 = _mm_loadu_ps(mapval4);
    r_linx4 = _mm_loadu_ps(r_lin4);
    g_linx4 = _mm_loadu_ps(g_lin4);
    b_linx4 = _mm_loadu_ps(b_lin4);

    if (!rgb2rgb_passthrough) {
        __m128 r_tmpx4, g_tmpx4, b_tmpx4;

        r_tmpx4 = _mm_mul_ps(r_linx4, _mm_set1_ps((float)(*rgb2rgb)[0][0]));
        r_tmpx4 = _mm_add_ps(r_tmpx4, _mm_mul_ps(g_linx4, _mm_set1_ps((float)(*rgb2rgb)[0][1])));
        r_tmpx4 = _mm_add_ps(r_tmpx4, _mm_mul_ps(b_linx4, _mm_set1_ps((float)(*rgb2rgb)[0][2])));

        g_tmpx4 = _mm_mul_ps(g_linx4, _mm_set1_ps((float)(*rgb2rgb)[1][1]));
        g_tmpx4 = _mm_add_ps(g_tmpx4, _mm_mul_ps(r_linx4, _mm_set1_ps((float)(*rgb2rgb)[1][0])));
        g_tmpx4 = _mm_add_ps(g_tmpx4, _mm_mul_ps(b_linx4, _mm_set1_ps((float)(*rgb2rgb)[1][2])));

        b_tmpx4 = _mm_mul_ps(b_linx4, _mm_set1_ps((float)(*rgb2rgb)[2][2]));
        b_tmpx4 = _mm_add_ps(b_tmpx4, _mm_mul_ps(r_linx4, _mm_set1_ps((float)(*rgb2rgb)[2][0])));
        b_tmpx4 = _mm_add_ps(b_tmpx4, _mm_mul_ps(g_linx4, _mm_set1_ps((float)(*rgb2rgb)[2][1])));

        r_linx4 = r_tmpx4;
        g_linx4 = g_tmpx4;
        b_linx4 = b_tmpx4;
    }

    if (desat > 0) {
        __m128 eps_x4 = _mm_set1_ps(FLOAT_EPS);
        __m128 desat4 = _mm_set1_ps((float)desat);
        __m128 luma4 = _mm_set1_ps(0);
        __m128 overbright4;

        luma4 = _mm_add_ps(luma4, _mm_mul_ps(r_linx4, _mm_set1_ps((float)av_q2d(coeffs->cr))));
        luma4 = _mm_add_ps(luma4, _mm_mul_ps(g_linx4, _mm_set1_ps((float)av_q2d(coeffs->cg))));
        luma4 = _mm_add_ps(luma4, _mm_mul_ps(b_linx4, _mm_set1_ps((float)av_q2d(coeffs->cb))));
        overbright4 = _mm_div_ps(_mm_max_ps(_mm_sub_ps(luma4, desat4), eps_x4), _mm_max_ps(luma4, eps_x4));
        r_linx4 = _mm_sub_ps(r_linx4, _mm_mul_ps(r_linx4, overbright4));
        r_linx4 = _mm_add_ps(r_linx4, _mm_mul_ps(luma4, overbright4));
        g_linx4 = _mm_sub_ps(g_linx4, _mm_mul_ps(g_linx4, overbright4));
        g_linx4 = _mm_add_ps(g_linx4, _mm_mul_ps(luma4, overbright4));
        b_linx4 = _mm_sub_ps(b_linx4, _mm_mul_ps(b_linx4, overbright4));
        b_linx4 = _mm_add_ps(b_linx4, _mm_mul_ps(luma4, overbright4));
    }

    r_linx4 = _mm_mul_ps(r_linx4, mapvalx4);
    g_linx4 = _mm_mul_ps(g_linx4, mapvalx4);
    b_linx4 = _mm_mul_ps(b_linx4, mapvalx4);

    r_linx4 = _mm_mul_ps(r_linx4, intermediate_upper_bound);
    r_linx4 = _mm_add_ps(r_linx4, offset);

    g_linx4 = _mm_mul_ps(g_linx4, intermediate_upper_bound);
    g_linx4 = _mm_add_ps(g_linx4, offset);

    b_linx4 = _mm_mul_ps(b_linx4, intermediate_upper_bound);
    b_linx4 = _mm_add_ps(b_linx4, offset);

    rx4 = _mm_cvttps_epi32(r_linx4);
    rx4 = av_clip_uint16_sse(rx4);
    gx4 = _mm_cvttps_epi32(g_linx4);
    gx4 = av_clip_uint16_sse(gx4);
    bx4 = _mm_cvttps_epi32(b_linx4);
    bx4 = av_clip_uint16_sse(bx4);

#define SAVE_COLOR(i) r_out[i] = delin_lut[_mm_extract_epi32(rx4, i)]; \
g_out[i] = delin_lut[_mm_extract_epi32(gx4, i)];                       \
b_out[i] = delin_lut[_mm_extract_epi32(bx4, i)];

    SAVE_COLOR(0)
    SAVE_COLOR(1)
    SAVE_COLOR(2)
    SAVE_COLOR(3)

#undef SAVE_COLOR
}
#endif // ENABLE_TONEMAPX_SSE_INTRINSICS

X86_64_V2 void tonemap_frame_dovi_2_420p_sse(uint8_t *dsty, uint8_t *dstu, uint8_t *dstv,
                                             const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                             const int *dstlinesize, const int *srclinesize,
                                             int dstdepth, int srcdepth,
                                             int width, int height,
                                             const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_SSE_INTRINSICS
    uint8_t *rdsty = dsty;
    uint8_t *rdstu = dstu;
    uint8_t *rdstv = dstv;

    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcu = srcu;
    const uint16_t *rsrcv = srcv;

    int rheight = height;
    // not zero when not divisible by 8
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 6;

    const int in_depth = srcdepth;
    const float in_rng = (float)((1 << in_depth) - 1);

    const int out_depth = dstdepth;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    int16_t r[8], g[8], b[8];
    int16_t r1[8], g1[8], b1[8];

    __m128i zero128 = _mm_setzero_si128();
    __m128i ux4, vx4;
    __m128i y0x8, y1x8;
    __m128i y0x4a, y0x4b, y1x4a, y1x4b, ux4a, ux4b, vx4a, vx4b;
    __m128i r0x4a, g0x4a, b0x4a, r0x4b, g0x4b, b0x4b;
    __m128i r1x4a, g1x4a, b1x4a, r1x4b, g1x4b, b1x4b;

    __m128i r0ox8, g0ox8, b0ox8;
    __m128i y0ox8;
    __m128i roax4, robx4, goax4, gobx4, boax4, bobx4;
    __m128i yoax4, yobx4;

    __m128i r1ox8, g1ox8, b1ox8;
    __m128i y1ox8;
    __m128i r1oax4, r1obx4, g1oax4, g1obx4, b1oax4, b1obx4;
    __m128i y1oax4, y1obx4;
    __m128i uox4, vox4, ravgx4, gavgx4, bavgx4;

    __m128 ipt0, ipt1, ipt2, ipt3;
    __m128 ia1, ib1, ia2, ib2;
    __m128 ix4, px4, tx4;
    __m128 lx4, mx4, sx4;
    __m128 rx4a, gx4a, bx4a, rx4b, gx4b, bx4b;
    __m128 y0x4af, y0x4bf, y1x4af, y1x4bf, ux4af, ux4bf, vx4af, vx4bf;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0] * 2, dstu += dstlinesize[1], dstv += dstlinesize[2],
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[2] / 2) {
        for (int xx = 0; xx < width >> 3; xx++) {
            int x = xx << 3;

            y0x8 = _mm_lddqu_si128((__m128i*)(srcy + x));
            y1x8 = _mm_lddqu_si128((__m128i*)(srcy + (srclinesize[0] / 2 + x)));
            ux4 = _mm_loadu_si64((__m128i*)(srcu + (x >> 1)));
            vx4 = _mm_loadu_si64((__m128i*)(srcv + (x >> 1)));

            y0x4a = _mm_cvtepu16_epi32(y0x8);
            y0x4b = _mm_unpackhi_epi16(y0x8, zero128);
            y1x4a = _mm_cvtepu16_epi32(y1x8);
            y1x4b = _mm_unpackhi_epi16(y1x8, zero128);
            ux4 = _mm_cvtepu16_epi32(ux4);
            vx4 = _mm_cvtepu16_epi32(vx4);

            ux4a = _mm_unpacklo_epi32(ux4, ux4);
            ux4b = _mm_unpackhi_epi32(ux4, ux4);
            vx4a = _mm_unpacklo_epi32(vx4, vx4);
            vx4b = _mm_unpackhi_epi32(vx4, vx4);

            y0x4af = _mm_cvtepi32_ps(y0x4a);
            y0x4bf = _mm_cvtepi32_ps(y0x4b);
            y1x4af = _mm_cvtepi32_ps(y1x4a);
            y1x4bf = _mm_cvtepi32_ps(y1x4b);
            ux4af = _mm_cvtepi32_ps(ux4a);
            ux4bf = _mm_cvtepi32_ps(ux4b);
            vx4af = _mm_cvtepi32_ps(vx4a);
            vx4bf = _mm_cvtepi32_ps(vx4b);

            y0x4af = _mm_div_ps(y0x4af, _mm_set1_ps(in_rng));
            y0x4bf = _mm_div_ps(y0x4bf, _mm_set1_ps(in_rng));
            y1x4af = _mm_div_ps(y1x4af, _mm_set1_ps(in_rng));
            y1x4bf = _mm_div_ps(y1x4bf, _mm_set1_ps(in_rng));
            ux4af = _mm_div_ps(ux4af, _mm_set1_ps(in_rng));
            ux4bf = _mm_div_ps(ux4bf, _mm_set1_ps(in_rng));
            vx4af = _mm_div_ps(vx4af, _mm_set1_ps(in_rng));
            vx4bf = _mm_div_ps(vx4bf, _mm_set1_ps(in_rng));

            // Reshape y0x4a
            ia1 = _mm_unpacklo_ps(y0x4af, ux4af);
            ia2 = _mm_unpackhi_ps(y0x4af, ux4af);
            ib1 = _mm_unpacklo_ps(vx4af, _mm_setzero_ps());
            ib2 = _mm_unpackhi_ps(vx4af, _mm_setzero_ps());
            ipt0 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(1, 0, 1, 0));
            ipt1 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(3, 2, 3, 2));
            ipt2 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            ipt3 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(3, 2, 3, 2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ipt0 = _mm_shuffle_ps(ipt0, ipt0, _MM_SHUFFLE(3, 1, 2, 0));
            ipt1 = _mm_shuffle_ps(ipt1, ipt1, _MM_SHUFFLE(3, 1, 2, 0));
            ipt2 = _mm_shuffle_ps(ipt2, ipt2, _MM_SHUFFLE(3, 1, 2, 0));
            ipt3 = _mm_shuffle_ps(ipt3, ipt3, _MM_SHUFFLE(3, 1, 2, 0));

            ia1 = _mm_unpacklo_ps(ipt0, ipt1);
            ia2 = _mm_unpacklo_ps(ipt2, ipt3);
            ib1 = _mm_unpackhi_ps(ipt0, ipt1);
            ib2 = _mm_unpackhi_ps(ipt2, ipt3);

            ix4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(1, 0, 1, 0));
            px4 = _mm_shuffle_ps(ib1, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            tx4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(3, 2, 3, 2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4a, &gx4a, &bx4a, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4a = _mm_mul_ps(rx4a, _mm_set1_ps(JPEG_SCALE));
            gx4a = _mm_mul_ps(gx4a, _mm_set1_ps(JPEG_SCALE));
            bx4a = _mm_mul_ps(bx4a, _mm_set1_ps(JPEG_SCALE));

            r0x4a = _mm_cvtps_epi32(rx4a);
            r0x4a = av_clip_int16_sse(r0x4a);
            g0x4a = _mm_cvtps_epi32(gx4a);
            g0x4a = av_clip_int16_sse(g0x4a);
            b0x4a = _mm_cvtps_epi32(bx4a);
            b0x4a = av_clip_int16_sse(b0x4a);

            // Reshape y1x4a
            ia1 = _mm_unpacklo_ps(y1x4af, ux4af);
            ia2 = _mm_unpackhi_ps(y1x4af, ux4af);
            ib1 = _mm_unpacklo_ps(vx4af, _mm_setzero_ps());
            ib2 = _mm_unpackhi_ps(vx4af, _mm_setzero_ps());
            ipt0 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(1, 0, 1, 0));
            ipt1 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(3, 2, 3, 2));
            ipt2 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            ipt3 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(3, 2, 3, 2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ipt0 = _mm_shuffle_ps(ipt0, ipt0, _MM_SHUFFLE(3, 1, 2, 0));
            ipt1 = _mm_shuffle_ps(ipt1, ipt1, _MM_SHUFFLE(3, 1, 2, 0));
            ipt2 = _mm_shuffle_ps(ipt2, ipt2, _MM_SHUFFLE(3, 1, 2, 0));
            ipt3 = _mm_shuffle_ps(ipt3, ipt3, _MM_SHUFFLE(3, 1, 2, 0));

            ia1 = _mm_unpacklo_ps(ipt0, ipt1);
            ia2 = _mm_unpacklo_ps(ipt2, ipt3);
            ib1 = _mm_unpackhi_ps(ipt0, ipt1);
            ib2 = _mm_unpackhi_ps(ipt2, ipt3);

            ix4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(1, 0, 1, 0));
            px4 = _mm_shuffle_ps(ib1, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            tx4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(3, 2, 3, 2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4a, &gx4a, &bx4a, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4a = _mm_mul_ps(rx4a, _mm_set1_ps(JPEG_SCALE));
            gx4a = _mm_mul_ps(gx4a, _mm_set1_ps(JPEG_SCALE));
            bx4a = _mm_mul_ps(bx4a, _mm_set1_ps(JPEG_SCALE));

            r1x4a = _mm_cvtps_epi32(rx4a);
            r1x4a = av_clip_int16_sse(r1x4a);
            g1x4a = _mm_cvtps_epi32(gx4a);
            g1x4a = av_clip_int16_sse(g1x4a);
            b1x4a = _mm_cvtps_epi32(bx4a);
            b1x4a = av_clip_int16_sse(b1x4a);

            // Reshape y0x4b
            ia1 = _mm_unpacklo_ps(y0x4bf, ux4bf);
            ia2 = _mm_unpackhi_ps(y0x4bf, ux4bf);
            ib1 = _mm_unpacklo_ps(vx4bf, _mm_setzero_ps());
            ib2 = _mm_unpackhi_ps(vx4bf, _mm_setzero_ps());
            ipt0 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(1, 0, 1, 0));
            ipt1 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(3, 2, 3, 2));
            ipt2 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            ipt3 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(3, 2, 3, 2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ipt0 = _mm_shuffle_ps(ipt0, ipt0, _MM_SHUFFLE(3, 1, 2, 0));
            ipt1 = _mm_shuffle_ps(ipt1, ipt1, _MM_SHUFFLE(3, 1, 2, 0));
            ipt2 = _mm_shuffle_ps(ipt2, ipt2, _MM_SHUFFLE(3, 1, 2, 0));
            ipt3 = _mm_shuffle_ps(ipt3, ipt3, _MM_SHUFFLE(3, 1, 2, 0));

            ia1 = _mm_unpacklo_ps(ipt0, ipt1);
            ia2 = _mm_unpacklo_ps(ipt2, ipt3);
            ib1 = _mm_unpackhi_ps(ipt0, ipt1);
            ib2 = _mm_unpackhi_ps(ipt2, ipt3);

            ix4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(1, 0, 1, 0));
            px4 = _mm_shuffle_ps(ib1, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            tx4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(3, 2, 3, 2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4b, &gx4b, &bx4b, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4b = _mm_mul_ps(rx4b, _mm_set1_ps(JPEG_SCALE));
            gx4b = _mm_mul_ps(gx4b, _mm_set1_ps(JPEG_SCALE));
            bx4b = _mm_mul_ps(bx4b, _mm_set1_ps(JPEG_SCALE));

            r0x4b = _mm_cvtps_epi32(rx4b);
            r0x4b = av_clip_int16_sse(r0x4b);
            g0x4b = _mm_cvtps_epi32(gx4b);
            g0x4b = av_clip_int16_sse(g0x4b);
            b0x4b = _mm_cvtps_epi32(bx4b);
            b0x4b = av_clip_int16_sse(b0x4b);

            // Reshape y1x4b
            ia1 = _mm_unpacklo_ps(y1x4bf, ux4bf);
            ia2 = _mm_unpackhi_ps(y1x4bf, ux4bf);
            ib1 = _mm_unpacklo_ps(vx4bf, _mm_setzero_ps());
            ib2 = _mm_unpackhi_ps(vx4bf, _mm_setzero_ps());
            ipt0 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(1, 0, 1, 0));
            ipt1 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(3, 2, 3, 2));
            ipt2 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            ipt3 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(3, 2, 3, 2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ipt0 = _mm_shuffle_ps(ipt0, ipt0, _MM_SHUFFLE(3, 1, 2, 0));
            ipt1 = _mm_shuffle_ps(ipt1, ipt1, _MM_SHUFFLE(3, 1, 2, 0));
            ipt2 = _mm_shuffle_ps(ipt2, ipt2, _MM_SHUFFLE(3, 1, 2, 0));
            ipt3 = _mm_shuffle_ps(ipt3, ipt3, _MM_SHUFFLE(3, 1, 2, 0));

            ia1 = _mm_unpacklo_ps(ipt0, ipt1);
            ia2 = _mm_unpacklo_ps(ipt2, ipt3);
            ib1 = _mm_unpackhi_ps(ipt0, ipt1);
            ib2 = _mm_unpackhi_ps(ipt2, ipt3);

            ix4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(1, 0, 1, 0));
            px4 = _mm_shuffle_ps(ib1, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            tx4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(3, 2, 3, 2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4b, &gx4b, &bx4b, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4b = _mm_mul_ps(rx4b, _mm_set1_ps(JPEG_SCALE));
            gx4b = _mm_mul_ps(gx4b, _mm_set1_ps(JPEG_SCALE));
            bx4b = _mm_mul_ps(bx4b, _mm_set1_ps(JPEG_SCALE));

            r1x4b = _mm_cvtps_epi32(rx4b);
            r1x4b = av_clip_int16_sse(r1x4b);
            g1x4b = _mm_cvtps_epi32(gx4b);
            g1x4b = av_clip_int16_sse(g1x4b);
            b1x4b = _mm_cvtps_epi32(bx4b);
            b1x4b = av_clip_int16_sse(b1x4b);

            tonemap_int32x4_sse(r0x4a, g0x4a, b0x4a, r, g, b,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r1x4a, g1x4a, b1x4a, r1, g1, b1,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r0x4b, g0x4b, b0x4b, &r[4], &g[4], &b[4],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r1x4b, g1x4b, b1x4b, &r1[4], &g1[4], &b1[4],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);

            r0ox8 = _mm_lddqu_si128((const __m128i_u *)r);
            g0ox8 = _mm_lddqu_si128((const __m128i_u *)g);
            b0ox8 = _mm_lddqu_si128((const __m128i_u *)b);

            roax4 = _mm_cvtepi16_epi32(r0ox8);
            goax4 = _mm_cvtepi16_epi32(g0ox8);
            boax4 = _mm_cvtepi16_epi32(b0ox8);

            robx4 = _mm_unpackhi_epi16(r0ox8, zero128);
            gobx4 = _mm_unpackhi_epi16(g0ox8, zero128);
            bobx4 = _mm_unpackhi_epi16(b0ox8, zero128);

            yoax4 = _mm_mullo_epi32(roax4, _mm_set1_epi32(cry));
            yoax4 = _mm_add_epi32(yoax4, _mm_mullo_epi32(goax4, _mm_set1_epi32(cgy)));
            yoax4 = _mm_add_epi32(yoax4, _mm_mullo_epi32(boax4, _mm_set1_epi32(cby)));
            yoax4 = _mm_add_epi32(yoax4, _mm_set1_epi32(out_rnd));
            // output shift bits for 8bit outputs is 29 - 8 = 21
            yoax4 = _mm_srai_epi32(yoax4, 21);
            yoax4 = _mm_add_epi32(yoax4, _mm_set1_epi32(params->out_yuv_off));

            yobx4 = _mm_mullo_epi32(robx4, _mm_set1_epi32(cry));
            yobx4 = _mm_add_epi32(yobx4, _mm_mullo_epi32(gobx4, _mm_set1_epi32(cgy)));
            yobx4 = _mm_add_epi32(yobx4, _mm_mullo_epi32(bobx4, _mm_set1_epi32(cby)));
            yobx4 = _mm_add_epi32(yobx4, _mm_set1_epi32(out_rnd));
            yobx4 = _mm_srai_epi32(yobx4, 21);
            yobx4 = _mm_add_epi32(yobx4, _mm_set1_epi32(params->out_yuv_off));

            y0ox8 = _mm_packs_epi32(yoax4, yobx4);
            _mm_storeu_si64(&dsty[x], _mm_packus_epi16(y0ox8, zero128));

            r1ox8 = _mm_lddqu_si128((const __m128i_u *)r1);
            g1ox8 = _mm_lddqu_si128((const __m128i_u *)g1);
            b1ox8 = _mm_lddqu_si128((const __m128i_u *)b1);

            r1oax4 = _mm_cvtepi16_epi32(r1ox8);
            g1oax4 = _mm_cvtepi16_epi32(g1ox8);
            b1oax4 = _mm_cvtepi16_epi32(b1ox8);

            r1obx4 = _mm_unpackhi_epi16(r1ox8, zero128);
            g1obx4 = _mm_unpackhi_epi16(g1ox8, zero128);
            b1obx4 = _mm_unpackhi_epi16(b1ox8, zero128);

            y1oax4 = _mm_mullo_epi32(r1oax4, _mm_set1_epi32(cry));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_mullo_epi32(g1oax4, _mm_set1_epi32(cgy)));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_mullo_epi32(b1oax4, _mm_set1_epi32(cby)));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_set1_epi32(out_rnd));
            y1oax4 = _mm_srai_epi32(y1oax4, 21);
            y1oax4 = _mm_add_epi32(y1oax4, _mm_set1_epi32(params->out_yuv_off));

            y1obx4 = _mm_mullo_epi32(r1obx4, _mm_set1_epi32(cry));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_mullo_epi32(g1obx4, _mm_set1_epi32(cgy)));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_mullo_epi32(b1obx4, _mm_set1_epi32(cby)));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_set1_epi32(out_rnd));
            y1obx4 = _mm_srai_epi32(y1obx4, 21);
            y1obx4 = _mm_add_epi32(y1obx4, _mm_set1_epi32(params->out_yuv_off));

            y1ox8 = _mm_packs_epi32(y1oax4, y1obx4);
            _mm_storeu_si64(&dsty[x + dstlinesize[0]], _mm_packus_epi16(y1ox8, zero128));

            ravgx4 = _mm_hadd_epi32(roax4, robx4);
            ravgx4 = _mm_add_epi32(ravgx4, _mm_hadd_epi32(r1oax4, r1obx4));
            ravgx4 = _mm_add_epi32(ravgx4, _mm_set1_epi32(2));
            ravgx4 = _mm_srai_epi32(ravgx4, 2);

            gavgx4 = _mm_hadd_epi32(goax4, gobx4);
            gavgx4 = _mm_add_epi32(gavgx4, _mm_hadd_epi32(g1oax4, g1obx4));
            gavgx4 = _mm_add_epi32(gavgx4, _mm_set1_epi32(2));
            gavgx4 = _mm_srai_epi32(gavgx4, 2);

            bavgx4 = _mm_hadd_epi32(boax4, bobx4);
            bavgx4 = _mm_add_epi32(bavgx4, _mm_hadd_epi32(b1oax4, b1obx4));
            bavgx4 = _mm_add_epi32(bavgx4, _mm_set1_epi32(2));
            bavgx4 = _mm_srai_epi32(bavgx4, 2);

            uox4 = _mm_add_epi32(_mm_set1_epi32(out_rnd), _mm_mullo_epi32(ravgx4, _mm_set1_epi32(cru)));
            uox4 = _mm_add_epi32(uox4, _mm_mullo_epi32(gavgx4, _mm_set1_epi32(ocgu)));
            uox4 = _mm_add_epi32(uox4, _mm_mullo_epi32(bavgx4, _mm_set1_epi32(cburv)));
            uox4 = _mm_srai_epi32(uox4, 21);
            uox4 = _mm_add_epi32(uox4, _mm_set1_epi32(out_uv_offset));
            _mm_storeu_si32(&dstu[x >> 1], _mm_packus_epi16(_mm_packs_epi32(uox4, zero128), zero128));

            vox4 = _mm_add_epi32(_mm_set1_epi32(out_rnd), _mm_mullo_epi32(ravgx4, _mm_set1_epi32(cburv)));
            vox4 = _mm_add_epi32(vox4, _mm_mullo_epi32(gavgx4, _mm_set1_epi32(ocgv)));
            vox4 = _mm_add_epi32(vox4, _mm_mullo_epi32(bavgx4, _mm_set1_epi32(cbv)));
            vox4 = _mm_srai_epi32(vox4, 21);
            vox4 = _mm_add_epi32(vox4, _mm_set1_epi32(out_uv_offset));
            _mm_storeu_si32(&dstv[x >> 1], _mm_packus_epi16(_mm_packs_epi32(vox4, zero128), zero128));
        }
    }

    // Process remaining pixels cannot fill the full simd register with scalar version
    if (remainw) {
        int offset = width & (int)0xfffffff8;
        rdsty += offset;
        rdstu += offset >> 1;
        rdstv += offset >> 1;
        rsrcy += offset;
        rsrcu += offset >> 1;
        rsrcv += offset >> 1;
        tonemap_frame_dovi_2_420p(rdsty, rdstu, rdstv,
                                    rsrcy, rsrcu, rsrcv,
                                    dstlinesize, srclinesize,
                                    dstdepth, srcdepth,
                                    remainw, rheight, params);
    }
#endif // ENABLE_TONEMAPX_SSE_INTRINSICS
}

X86_64_V2 void tonemap_frame_dovi_2_420p10_sse(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                               const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                               const int *dstlinesize, const int *srclinesize,
                                               int dstdepth, int srcdepth,
                                               int width, int height,
                                               const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_SSE_INTRINSICS
    uint16_t *rdsty = dsty;
    uint16_t *rdstu = dstu;
    uint16_t *rdstv = dstv;
    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcu = srcu;
    const uint16_t *rsrcv = srcv;
    int rheight = height;
    // not zero when not divisible by 8
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 6;

    const int in_depth = srcdepth;
    const float in_rng = (float)((1 << in_depth) - 1);

    const int out_depth = dstdepth;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    int16_t r[8], g[8], b[8];
    int16_t r1[8], g1[8], b1[8];

    __m128i zero128 = _mm_setzero_si128();
    __m128i ux4, vx4;
    __m128i y0x8, y1x8;
    __m128i y0x4a, y0x4b, y1x4a, y1x4b, ux4a, ux4b, vx4a, vx4b;
    __m128i r0x4a, g0x4a, b0x4a, r0x4b, g0x4b, b0x4b;
    __m128i r1x4a, g1x4a, b1x4a, r1x4b, g1x4b, b1x4b;

    __m128i r0ox8, g0ox8, b0ox8;
    __m128i y0ox8;
    __m128i roax4, robx4, goax4, gobx4, boax4, bobx4;
    __m128i yoax4, yobx4;

    __m128i r1ox8, g1ox8, b1ox8;
    __m128i y1ox8;
    __m128i r1oax4, r1obx4, g1oax4, g1obx4, b1oax4, b1obx4;
    __m128i y1oax4, y1obx4;
    __m128i uox4, vox4, ravgx4, gavgx4, bavgx4;

    __m128 ipt0, ipt1, ipt2, ipt3;
    __m128 ia1, ib1, ia2, ib2;
    __m128 ix4, px4, tx4;
    __m128 lx4, mx4, sx4;
    __m128 rx4a, gx4a, bx4a, rx4b, gx4b, bx4b;
    __m128 y0x4af, y0x4bf, y1x4af, y1x4bf, ux4af, ux4bf, vx4af, vx4bf;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstu += dstlinesize[1] / 2, dstv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 3; xx++) {
            int x = xx << 3;

            y0x8 = _mm_lddqu_si128((__m128i*)(srcy + x));
            y1x8 = _mm_lddqu_si128((__m128i*)(srcy + (srclinesize[0] / 2 + x)));
            ux4 = _mm_loadu_si64((__m128i*)(srcu + (x >> 1)));
            vx4 = _mm_loadu_si64((__m128i*)(srcv + (x >> 1)));

            y0x4a = _mm_cvtepu16_epi32(y0x8);
            y0x4b = _mm_unpackhi_epi16(y0x8, zero128);
            y1x4a = _mm_cvtepu16_epi32(y1x8);
            y1x4b = _mm_unpackhi_epi16(y1x8, zero128);
            ux4 = _mm_cvtepu16_epi32(ux4);
            vx4 = _mm_cvtepu16_epi32(vx4);

            ux4a = _mm_unpacklo_epi32(ux4, ux4);
            ux4b = _mm_unpackhi_epi32(ux4, ux4);
            vx4a = _mm_unpacklo_epi32(vx4, vx4);
            vx4b = _mm_unpackhi_epi32(vx4, vx4);

            y0x4af = _mm_cvtepi32_ps(y0x4a);
            y0x4bf = _mm_cvtepi32_ps(y0x4b);
            y1x4af = _mm_cvtepi32_ps(y1x4a);
            y1x4bf = _mm_cvtepi32_ps(y1x4b);
            ux4af = _mm_cvtepi32_ps(ux4a);
            ux4bf = _mm_cvtepi32_ps(ux4b);
            vx4af = _mm_cvtepi32_ps(vx4a);
            vx4bf = _mm_cvtepi32_ps(vx4b);

            y0x4af = _mm_div_ps(y0x4af, _mm_set1_ps(in_rng));
            y0x4bf = _mm_div_ps(y0x4bf, _mm_set1_ps(in_rng));
            y1x4af = _mm_div_ps(y1x4af, _mm_set1_ps(in_rng));
            y1x4bf = _mm_div_ps(y1x4bf, _mm_set1_ps(in_rng));
            ux4af = _mm_div_ps(ux4af, _mm_set1_ps(in_rng));
            ux4bf = _mm_div_ps(ux4bf, _mm_set1_ps(in_rng));
            vx4af = _mm_div_ps(vx4af, _mm_set1_ps(in_rng));
            vx4bf = _mm_div_ps(vx4bf, _mm_set1_ps(in_rng));

            // Reshape y0x4a
            ia1 = _mm_unpacklo_ps(y0x4af, ux4af);
            ia2 = _mm_unpackhi_ps(y0x4af, ux4af);
            ib1 = _mm_unpacklo_ps(vx4af, _mm_setzero_ps());
            ib2 = _mm_unpackhi_ps(vx4af, _mm_setzero_ps());
            ipt0 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(1, 0, 1, 0));
            ipt1 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(3, 2, 3, 2));
            ipt2 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            ipt3 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(3, 2, 3, 2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ipt0 = _mm_shuffle_ps(ipt0, ipt0, _MM_SHUFFLE(3, 1, 2, 0));
            ipt1 = _mm_shuffle_ps(ipt1, ipt1, _MM_SHUFFLE(3, 1, 2, 0));
            ipt2 = _mm_shuffle_ps(ipt2, ipt2, _MM_SHUFFLE(3, 1, 2, 0));
            ipt3 = _mm_shuffle_ps(ipt3, ipt3, _MM_SHUFFLE(3, 1, 2, 0));

            ia1 = _mm_unpacklo_ps(ipt0, ipt1);
            ia2 = _mm_unpacklo_ps(ipt2, ipt3);
            ib1 = _mm_unpackhi_ps(ipt0, ipt1);
            ib2 = _mm_unpackhi_ps(ipt2, ipt3);

            ix4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(1, 0, 1, 0));
            px4 = _mm_shuffle_ps(ib1, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            tx4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(3, 2, 3, 2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4a, &gx4a, &bx4a, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4a = _mm_mul_ps(rx4a, _mm_set1_ps(JPEG_SCALE));
            gx4a = _mm_mul_ps(gx4a, _mm_set1_ps(JPEG_SCALE));
            bx4a = _mm_mul_ps(bx4a, _mm_set1_ps(JPEG_SCALE));

            r0x4a = _mm_cvtps_epi32(rx4a);
            r0x4a = av_clip_int16_sse(r0x4a);
            g0x4a = _mm_cvtps_epi32(gx4a);
            g0x4a = av_clip_int16_sse(g0x4a);
            b0x4a = _mm_cvtps_epi32(bx4a);
            b0x4a = av_clip_int16_sse(b0x4a);

            // Reshape y1x4a
            ia1 = _mm_unpacklo_ps(y1x4af, ux4af);
            ia2 = _mm_unpackhi_ps(y1x4af, ux4af);
            ib1 = _mm_unpacklo_ps(vx4af, _mm_setzero_ps());
            ib2 = _mm_unpackhi_ps(vx4af, _mm_setzero_ps());
            ipt0 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(1, 0, 1, 0));
            ipt1 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(3, 2, 3, 2));
            ipt2 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            ipt3 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(3, 2, 3, 2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ipt0 = _mm_shuffle_ps(ipt0, ipt0, _MM_SHUFFLE(3, 1, 2, 0));
            ipt1 = _mm_shuffle_ps(ipt1, ipt1, _MM_SHUFFLE(3, 1, 2, 0));
            ipt2 = _mm_shuffle_ps(ipt2, ipt2, _MM_SHUFFLE(3, 1, 2, 0));
            ipt3 = _mm_shuffle_ps(ipt3, ipt3, _MM_SHUFFLE(3, 1, 2, 0));

            ia1 = _mm_unpacklo_ps(ipt0, ipt1);
            ia2 = _mm_unpacklo_ps(ipt2, ipt3);
            ib1 = _mm_unpackhi_ps(ipt0, ipt1);
            ib2 = _mm_unpackhi_ps(ipt2, ipt3);

            ix4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(1, 0, 1, 0));
            px4 = _mm_shuffle_ps(ib1, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            tx4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(3, 2, 3, 2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4a, &gx4a, &bx4a, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4a = _mm_mul_ps(rx4a, _mm_set1_ps(JPEG_SCALE));
            gx4a = _mm_mul_ps(gx4a, _mm_set1_ps(JPEG_SCALE));
            bx4a = _mm_mul_ps(bx4a, _mm_set1_ps(JPEG_SCALE));

            r1x4a = _mm_cvtps_epi32(rx4a);
            r1x4a = av_clip_int16_sse(r1x4a);
            g1x4a = _mm_cvtps_epi32(gx4a);
            g1x4a = av_clip_int16_sse(g1x4a);
            b1x4a = _mm_cvtps_epi32(bx4a);
            b1x4a = av_clip_int16_sse(b1x4a);

            // Reshape y0x4b
            ia1 = _mm_unpacklo_ps(y0x4bf, ux4bf);
            ia2 = _mm_unpackhi_ps(y0x4bf, ux4bf);
            ib1 = _mm_unpacklo_ps(vx4bf, _mm_setzero_ps());
            ib2 = _mm_unpackhi_ps(vx4bf, _mm_setzero_ps());
            ipt0 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(1, 0, 1, 0));
            ipt1 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(3, 2, 3, 2));
            ipt2 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            ipt3 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(3, 2, 3, 2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ipt0 = _mm_shuffle_ps(ipt0, ipt0, _MM_SHUFFLE(3, 1, 2, 0));
            ipt1 = _mm_shuffle_ps(ipt1, ipt1, _MM_SHUFFLE(3, 1, 2, 0));
            ipt2 = _mm_shuffle_ps(ipt2, ipt2, _MM_SHUFFLE(3, 1, 2, 0));
            ipt3 = _mm_shuffle_ps(ipt3, ipt3, _MM_SHUFFLE(3, 1, 2, 0));

            ia1 = _mm_unpacklo_ps(ipt0, ipt1);
            ia2 = _mm_unpacklo_ps(ipt2, ipt3);
            ib1 = _mm_unpackhi_ps(ipt0, ipt1);
            ib2 = _mm_unpackhi_ps(ipt2, ipt3);

            ix4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(1, 0, 1, 0));
            px4 = _mm_shuffle_ps(ib1, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            tx4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(3, 2, 3, 2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4b, &gx4b, &bx4b, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4b = _mm_mul_ps(rx4b, _mm_set1_ps(JPEG_SCALE));
            gx4b = _mm_mul_ps(gx4b, _mm_set1_ps(JPEG_SCALE));
            bx4b = _mm_mul_ps(bx4b, _mm_set1_ps(JPEG_SCALE));

            r0x4b = _mm_cvtps_epi32(rx4b);
            r0x4b = av_clip_int16_sse(r0x4b);
            g0x4b = _mm_cvtps_epi32(gx4b);
            g0x4b = av_clip_int16_sse(g0x4b);
            b0x4b = _mm_cvtps_epi32(bx4b);
            b0x4b = av_clip_int16_sse(b0x4b);

            // Reshape y1x4b
            ia1 = _mm_unpacklo_ps(y1x4bf, ux4bf);
            ia2 = _mm_unpackhi_ps(y1x4bf, ux4bf);
            ib1 = _mm_unpacklo_ps(vx4bf, _mm_setzero_ps());
            ib2 = _mm_unpackhi_ps(vx4bf, _mm_setzero_ps());
            ipt0 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(1, 0, 1, 0));
            ipt1 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(3, 2, 3, 2));
            ipt2 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            ipt3 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(3, 2, 3, 2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ipt0 = _mm_shuffle_ps(ipt0, ipt0, _MM_SHUFFLE(3, 1, 2, 0));
            ipt1 = _mm_shuffle_ps(ipt1, ipt1, _MM_SHUFFLE(3, 1, 2, 0));
            ipt2 = _mm_shuffle_ps(ipt2, ipt2, _MM_SHUFFLE(3, 1, 2, 0));
            ipt3 = _mm_shuffle_ps(ipt3, ipt3, _MM_SHUFFLE(3, 1, 2, 0));

            ia1 = _mm_unpacklo_ps(ipt0, ipt1);
            ia2 = _mm_unpacklo_ps(ipt2, ipt3);
            ib1 = _mm_unpackhi_ps(ipt0, ipt1);
            ib2 = _mm_unpackhi_ps(ipt2, ipt3);

            ix4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(1, 0, 1, 0));
            px4 = _mm_shuffle_ps(ib1, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            tx4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(3, 2, 3, 2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4b, &gx4b, &bx4b, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4b = _mm_mul_ps(rx4b, _mm_set1_ps(JPEG_SCALE));
            gx4b = _mm_mul_ps(gx4b, _mm_set1_ps(JPEG_SCALE));
            bx4b = _mm_mul_ps(bx4b, _mm_set1_ps(JPEG_SCALE));

            r1x4b = _mm_cvtps_epi32(rx4b);
            r1x4b = av_clip_int16_sse(r1x4b);
            g1x4b = _mm_cvtps_epi32(gx4b);
            g1x4b = av_clip_int16_sse(g1x4b);
            b1x4b = _mm_cvtps_epi32(bx4b);
            b1x4b = av_clip_int16_sse(b1x4b);

            tonemap_int32x4_sse(r0x4a, g0x4a, b0x4a, r, g, b,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r1x4a, g1x4a, b1x4a, r1, g1, b1,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r0x4b, g0x4b, b0x4b, &r[4], &g[4], &b[4],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r1x4b, g1x4b, b1x4b, &r1[4], &g1[4], &b1[4],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);

            r0ox8 = _mm_lddqu_si128((const __m128i_u *)r);
            g0ox8 = _mm_lddqu_si128((const __m128i_u *)g);
            b0ox8 = _mm_lddqu_si128((const __m128i_u *)b);

            roax4 = _mm_cvtepi16_epi32(r0ox8);
            goax4 = _mm_cvtepi16_epi32(g0ox8);
            boax4 = _mm_cvtepi16_epi32(b0ox8);

            robx4 = _mm_unpackhi_epi16(r0ox8, zero128);
            gobx4 = _mm_unpackhi_epi16(g0ox8, zero128);
            bobx4 = _mm_unpackhi_epi16(b0ox8, zero128);

            yoax4 = _mm_mullo_epi32(roax4, _mm_set1_epi32(cry));
            yoax4 = _mm_add_epi32(yoax4, _mm_mullo_epi32(goax4, _mm_set1_epi32(cgy)));
            yoax4 = _mm_add_epi32(yoax4, _mm_mullo_epi32(boax4, _mm_set1_epi32(cby)));
            yoax4 = _mm_add_epi32(yoax4, _mm_set1_epi32(out_rnd));
            yoax4 = _mm_srai_epi32(yoax4, out_sh);
            yoax4 = _mm_add_epi32(yoax4, _mm_set1_epi32(params->out_yuv_off));

            yobx4 = _mm_mullo_epi32(robx4, _mm_set1_epi32(cry));
            yobx4 = _mm_add_epi32(yobx4, _mm_mullo_epi32(gobx4, _mm_set1_epi32(cgy)));
            yobx4 = _mm_add_epi32(yobx4, _mm_mullo_epi32(bobx4, _mm_set1_epi32(cby)));
            yobx4 = _mm_add_epi32(yobx4, _mm_set1_epi32(out_rnd));
            yobx4 = _mm_srai_epi32(yobx4, out_sh);
            yobx4 = _mm_add_epi32(yobx4, _mm_set1_epi32(params->out_yuv_off));

            y0ox8 = _mm_packus_epi32(yoax4, yobx4);
            _mm_storeu_si128((__m128i_u *) &dsty[x], y0ox8);

            r1ox8 = _mm_lddqu_si128((const __m128i_u *)r1);
            g1ox8 = _mm_lddqu_si128((const __m128i_u *)g1);
            b1ox8 = _mm_lddqu_si128((const __m128i_u *)b1);

            r1oax4 = _mm_cvtepi16_epi32(r1ox8);
            g1oax4 = _mm_cvtepi16_epi32(g1ox8);
            b1oax4 = _mm_cvtepi16_epi32(b1ox8);

            r1obx4 = _mm_unpackhi_epi16(r1ox8, zero128);
            g1obx4 = _mm_unpackhi_epi16(g1ox8, zero128);
            b1obx4 = _mm_unpackhi_epi16(b1ox8, zero128);

            y1oax4 = _mm_mullo_epi32(r1oax4, _mm_set1_epi32(cry));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_mullo_epi32(g1oax4, _mm_set1_epi32(cgy)));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_mullo_epi32(b1oax4, _mm_set1_epi32(cby)));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_set1_epi32(out_rnd));
            y1oax4 = _mm_srai_epi32(y1oax4, out_sh);
            y1oax4 = _mm_add_epi32(y1oax4, _mm_set1_epi32(params->out_yuv_off));

            y1obx4 = _mm_mullo_epi32(r1obx4, _mm_set1_epi32(cry));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_mullo_epi32(g1obx4, _mm_set1_epi32(cgy)));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_mullo_epi32(b1obx4, _mm_set1_epi32(cby)));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_set1_epi32(out_rnd));
            y1obx4 = _mm_srai_epi32(y1obx4, out_sh);
            y1obx4 = _mm_add_epi32(y1obx4, _mm_set1_epi32(params->out_yuv_off));

            y1ox8 = _mm_packus_epi32(y1oax4, y1obx4);
            _mm_storeu_si128((__m128i_u *) &dsty[x + dstlinesize[0] / 2], y1ox8);

            ravgx4 = _mm_hadd_epi32(roax4, robx4);
            ravgx4 = _mm_add_epi32(ravgx4, _mm_hadd_epi32(r1oax4, r1obx4));
            ravgx4 = _mm_add_epi32(ravgx4, _mm_set1_epi32(2));
            ravgx4 = _mm_srai_epi32(ravgx4, 2);

            gavgx4 = _mm_hadd_epi32(goax4, gobx4);
            gavgx4 = _mm_add_epi32(gavgx4, _mm_hadd_epi32(g1oax4, g1obx4));
            gavgx4 = _mm_add_epi32(gavgx4, _mm_set1_epi32(2));
            gavgx4 = _mm_srai_epi32(gavgx4, 2);

            bavgx4 = _mm_hadd_epi32(boax4, bobx4);
            bavgx4 = _mm_add_epi32(bavgx4, _mm_hadd_epi32(b1oax4, b1obx4));
            bavgx4 = _mm_add_epi32(bavgx4, _mm_set1_epi32(2));
            bavgx4 = _mm_srai_epi32(bavgx4, 2);

            uox4 = _mm_add_epi32(_mm_set1_epi32(out_rnd), _mm_mullo_epi32(ravgx4, _mm_set1_epi32(cru)));
            uox4 = _mm_add_epi32(uox4, _mm_mullo_epi32(gavgx4, _mm_set1_epi32(ocgu)));
            uox4 = _mm_add_epi32(uox4, _mm_mullo_epi32(bavgx4, _mm_set1_epi32(cburv)));
            uox4 = _mm_srai_epi32(uox4, out_sh);
            uox4 = _mm_add_epi32(uox4, _mm_set1_epi32(out_uv_offset));
            _mm_storeu_si64((__m128i_u *) &dstu[x >> 1], _mm_packus_epi32(uox4, zero128));

            vox4 = _mm_add_epi32(_mm_set1_epi32(out_rnd), _mm_mullo_epi32(ravgx4, _mm_set1_epi32(cburv)));
            vox4 = _mm_add_epi32(vox4, _mm_mullo_epi32(gavgx4, _mm_set1_epi32(ocgv)));
            vox4 = _mm_add_epi32(vox4, _mm_mullo_epi32(bavgx4, _mm_set1_epi32(cbv)));
            vox4 = _mm_srai_epi32(vox4, out_sh);
            vox4 = _mm_add_epi32(vox4, _mm_set1_epi32(out_uv_offset));
            _mm_storeu_si64((__m128i_u *) &dstv[x >> 1], _mm_packus_epi32(vox4, zero128));
        }
    }

    // Process remaining pixels cannot fill the full simd register with scalar version
    if (remainw) {
        int offset = width & (int)0xfffffff8;
        rdsty += offset;
        rdstu += offset >> 1;
        rdstv += offset >> 1;
        rsrcy += offset;
        rsrcu += offset >> 1;
        rsrcv += offset >> 1;
        tonemap_frame_dovi_2_420p10(rdsty, rdstu, rdstv,
                                      rsrcy, rsrcu, rsrcv,
                                      dstlinesize, srclinesize,
                                      dstdepth, srcdepth,
                                      remainw, rheight, params);
    }
#endif // ENABLE_TONEMAPX_SSE_INTRINSICS
}

X86_64_V2 void tonemap_frame_dovi_2_420hdr_sse(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                               const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                               const int *dstlinesize, const int *srclinesize,
                                               int dstdepth, int srcdepth,
                                               int width, int height,
                                               const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_SSE_INTRINSICS
    uint16_t *rdsty = dsty;
    uint16_t *rdstu = dstu;
    uint16_t *rdstv = dstv;
    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcu = srcu;
    const uint16_t *rsrcv = srcv;
    int rheight = height;
    // not zero when not divisible by 8
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 6;

    const int in_depth = srcdepth;
    const float in_rng = (float)((1 << in_depth) - 1);

    const int out_depth = dstdepth;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    __m128i zero128 = _mm_setzero_si128();
    __m128i ux4, vx4;
    __m128i y0x8, y1x8;
    __m128i y0x4a, y0x4b, y1x4a, y1x4b, ux4a, ux4b, vx4a, vx4b;
    __m128i r0x4a, g0x4a, b0x4a, r0x4b, g0x4b, b0x4b;
    __m128i r1x4a, g1x4a, b1x4a, r1x4b, g1x4b, b1x4b;

    __m128i y0ox8;
    __m128i roax4, robx4, goax4, gobx4, boax4, bobx4;
    __m128i yoax4, yobx4;

    __m128i y1ox8;
    __m128i r1oax4, r1obx4, g1oax4, g1obx4, b1oax4, b1obx4;
    __m128i y1oax4, y1obx4;
    __m128i uox4, vox4, ravgx4, gavgx4, bavgx4;

    __m128 ipt0, ipt1, ipt2, ipt3;
    __m128 ia1, ib1, ia2, ib2;
    __m128 ix4, px4, tx4;
    __m128 lx4, mx4, sx4;
    __m128 rx4a, gx4a, bx4a, rx4b, gx4b, bx4b;
    __m128 y0x4af, y0x4bf, y1x4af, y1x4bf, ux4af, ux4bf, vx4af, vx4bf;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstu += dstlinesize[1] / 2, dstv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 3; xx++) {
            int x = xx << 3;

            y0x8 = _mm_lddqu_si128((__m128i*)(srcy + x));
            y1x8 = _mm_lddqu_si128((__m128i*)(srcy + (srclinesize[0] / 2 + x)));
            ux4 = _mm_loadu_si64((__m128i*)(srcu + (x >> 1)));
            vx4 = _mm_loadu_si64((__m128i*)(srcv + (x >> 1)));

            y0x4a = _mm_cvtepu16_epi32(y0x8);
            y0x4b = _mm_unpackhi_epi16(y0x8, zero128);
            y1x4a = _mm_cvtepu16_epi32(y1x8);
            y1x4b = _mm_unpackhi_epi16(y1x8, zero128);
            ux4 = _mm_cvtepu16_epi32(ux4);
            vx4 = _mm_cvtepu16_epi32(vx4);

            ux4a = _mm_unpacklo_epi32(ux4, ux4);
            ux4b = _mm_unpackhi_epi32(ux4, ux4);
            vx4a = _mm_unpacklo_epi32(vx4, vx4);
            vx4b = _mm_unpackhi_epi32(vx4, vx4);

            y0x4af = _mm_cvtepi32_ps(y0x4a);
            y0x4bf = _mm_cvtepi32_ps(y0x4b);
            y1x4af = _mm_cvtepi32_ps(y1x4a);
            y1x4bf = _mm_cvtepi32_ps(y1x4b);
            ux4af = _mm_cvtepi32_ps(ux4a);
            ux4bf = _mm_cvtepi32_ps(ux4b);
            vx4af = _mm_cvtepi32_ps(vx4a);
            vx4bf = _mm_cvtepi32_ps(vx4b);

            y0x4af = _mm_div_ps(y0x4af, _mm_set1_ps(in_rng));
            y0x4bf = _mm_div_ps(y0x4bf, _mm_set1_ps(in_rng));
            y1x4af = _mm_div_ps(y1x4af, _mm_set1_ps(in_rng));
            y1x4bf = _mm_div_ps(y1x4bf, _mm_set1_ps(in_rng));
            ux4af = _mm_div_ps(ux4af, _mm_set1_ps(in_rng));
            ux4bf = _mm_div_ps(ux4bf, _mm_set1_ps(in_rng));
            vx4af = _mm_div_ps(vx4af, _mm_set1_ps(in_rng));
            vx4bf = _mm_div_ps(vx4bf, _mm_set1_ps(in_rng));

            // Reshape y0x4a
            ia1 = _mm_unpacklo_ps(y0x4af, ux4af);
            ia2 = _mm_unpackhi_ps(y0x4af, ux4af);
            ib1 = _mm_unpacklo_ps(vx4af, _mm_setzero_ps());
            ib2 = _mm_unpackhi_ps(vx4af, _mm_setzero_ps());
            ipt0 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(1, 0, 1, 0));
            ipt1 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(3, 2, 3, 2));
            ipt2 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            ipt3 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(3, 2, 3, 2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ipt0 = _mm_shuffle_ps(ipt0, ipt0, _MM_SHUFFLE(3, 1, 2, 0));
            ipt1 = _mm_shuffle_ps(ipt1, ipt1, _MM_SHUFFLE(3, 1, 2, 0));
            ipt2 = _mm_shuffle_ps(ipt2, ipt2, _MM_SHUFFLE(3, 1, 2, 0));
            ipt3 = _mm_shuffle_ps(ipt3, ipt3, _MM_SHUFFLE(3, 1, 2, 0));

            ia1 = _mm_unpacklo_ps(ipt0, ipt1);
            ia2 = _mm_unpacklo_ps(ipt2, ipt3);
            ib1 = _mm_unpackhi_ps(ipt0, ipt1);
            ib2 = _mm_unpackhi_ps(ipt2, ipt3);

            ix4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(1, 0, 1, 0));
            px4 = _mm_shuffle_ps(ib1, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            tx4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(3, 2, 3, 2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4a, &gx4a, &bx4a, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4a = _mm_mul_ps(rx4a, _mm_set1_ps(JPEG_SCALE));
            gx4a = _mm_mul_ps(gx4a, _mm_set1_ps(JPEG_SCALE));
            bx4a = _mm_mul_ps(bx4a, _mm_set1_ps(JPEG_SCALE));

            r0x4a = _mm_cvtps_epi32(rx4a);
            r0x4a = av_clip_int16_sse(r0x4a);
            g0x4a = _mm_cvtps_epi32(gx4a);
            g0x4a = av_clip_int16_sse(g0x4a);
            b0x4a = _mm_cvtps_epi32(bx4a);
            b0x4a = av_clip_int16_sse(b0x4a);

            // Reshape y1x4a
            ia1 = _mm_unpacklo_ps(y1x4af, ux4af);
            ia2 = _mm_unpackhi_ps(y1x4af, ux4af);
            ib1 = _mm_unpacklo_ps(vx4af, _mm_setzero_ps());
            ib2 = _mm_unpackhi_ps(vx4af, _mm_setzero_ps());
            ipt0 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(1, 0, 1, 0));
            ipt1 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(3, 2, 3, 2));
            ipt2 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            ipt3 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(3, 2, 3, 2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ipt0 = _mm_shuffle_ps(ipt0, ipt0, _MM_SHUFFLE(3, 1, 2, 0));
            ipt1 = _mm_shuffle_ps(ipt1, ipt1, _MM_SHUFFLE(3, 1, 2, 0));
            ipt2 = _mm_shuffle_ps(ipt2, ipt2, _MM_SHUFFLE(3, 1, 2, 0));
            ipt3 = _mm_shuffle_ps(ipt3, ipt3, _MM_SHUFFLE(3, 1, 2, 0));

            ia1 = _mm_unpacklo_ps(ipt0, ipt1);
            ia2 = _mm_unpacklo_ps(ipt2, ipt3);
            ib1 = _mm_unpackhi_ps(ipt0, ipt1);
            ib2 = _mm_unpackhi_ps(ipt2, ipt3);

            ix4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(1, 0, 1, 0));
            px4 = _mm_shuffle_ps(ib1, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            tx4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(3, 2, 3, 2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4a, &gx4a, &bx4a, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4a = _mm_mul_ps(rx4a, _mm_set1_ps(JPEG_SCALE));
            gx4a = _mm_mul_ps(gx4a, _mm_set1_ps(JPEG_SCALE));
            bx4a = _mm_mul_ps(bx4a, _mm_set1_ps(JPEG_SCALE));

            r1x4a = _mm_cvtps_epi32(rx4a);
            r1x4a = av_clip_int16_sse(r1x4a);
            g1x4a = _mm_cvtps_epi32(gx4a);
            g1x4a = av_clip_int16_sse(g1x4a);
            b1x4a = _mm_cvtps_epi32(bx4a);
            b1x4a = av_clip_int16_sse(b1x4a);

            // Reshape y0x4b
            ia1 = _mm_unpacklo_ps(y0x4bf, ux4bf);
            ia2 = _mm_unpackhi_ps(y0x4bf, ux4bf);
            ib1 = _mm_unpacklo_ps(vx4bf, _mm_setzero_ps());
            ib2 = _mm_unpackhi_ps(vx4bf, _mm_setzero_ps());
            ipt0 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(1, 0, 1, 0));
            ipt1 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(3, 2, 3, 2));
            ipt2 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            ipt3 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(3, 2, 3, 2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ipt0 = _mm_shuffle_ps(ipt0, ipt0, _MM_SHUFFLE(3, 1, 2, 0));
            ipt1 = _mm_shuffle_ps(ipt1, ipt1, _MM_SHUFFLE(3, 1, 2, 0));
            ipt2 = _mm_shuffle_ps(ipt2, ipt2, _MM_SHUFFLE(3, 1, 2, 0));
            ipt3 = _mm_shuffle_ps(ipt3, ipt3, _MM_SHUFFLE(3, 1, 2, 0));

            ia1 = _mm_unpacklo_ps(ipt0, ipt1);
            ia2 = _mm_unpacklo_ps(ipt2, ipt3);
            ib1 = _mm_unpackhi_ps(ipt0, ipt1);
            ib2 = _mm_unpackhi_ps(ipt2, ipt3);

            ix4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(1, 0, 1, 0));
            px4 = _mm_shuffle_ps(ib1, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            tx4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(3, 2, 3, 2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4b, &gx4b, &bx4b, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4b = _mm_mul_ps(rx4b, _mm_set1_ps(JPEG_SCALE));
            gx4b = _mm_mul_ps(gx4b, _mm_set1_ps(JPEG_SCALE));
            bx4b = _mm_mul_ps(bx4b, _mm_set1_ps(JPEG_SCALE));

            r0x4b = _mm_cvtps_epi32(rx4b);
            r0x4b = av_clip_int16_sse(r0x4b);
            g0x4b = _mm_cvtps_epi32(gx4b);
            g0x4b = av_clip_int16_sse(g0x4b);
            b0x4b = _mm_cvtps_epi32(bx4b);
            b0x4b = av_clip_int16_sse(b0x4b);

            // Reshape y1x4b
            ia1 = _mm_unpacklo_ps(y1x4bf, ux4bf);
            ia2 = _mm_unpackhi_ps(y1x4bf, ux4bf);
            ib1 = _mm_unpacklo_ps(vx4bf, _mm_setzero_ps());
            ib2 = _mm_unpackhi_ps(vx4bf, _mm_setzero_ps());
            ipt0 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(1, 0, 1, 0));
            ipt1 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(3, 2, 3, 2));
            ipt2 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            ipt3 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(3, 2, 3, 2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ipt0 = _mm_shuffle_ps(ipt0, ipt0, _MM_SHUFFLE(3, 1, 2, 0));
            ipt1 = _mm_shuffle_ps(ipt1, ipt1, _MM_SHUFFLE(3, 1, 2, 0));
            ipt2 = _mm_shuffle_ps(ipt2, ipt2, _MM_SHUFFLE(3, 1, 2, 0));
            ipt3 = _mm_shuffle_ps(ipt3, ipt3, _MM_SHUFFLE(3, 1, 2, 0));

            ia1 = _mm_unpacklo_ps(ipt0, ipt1);
            ia2 = _mm_unpacklo_ps(ipt2, ipt3);
            ib1 = _mm_unpackhi_ps(ipt0, ipt1);
            ib2 = _mm_unpackhi_ps(ipt2, ipt3);

            ix4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(1, 0, 1, 0));
            px4 = _mm_shuffle_ps(ib1, ib2, _MM_SHUFFLE(1, 0, 1, 0));
            tx4 = _mm_shuffle_ps(ia1, ia2, _MM_SHUFFLE(3, 2, 3, 2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4b, &gx4b, &bx4b, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4b = _mm_mul_ps(rx4b, _mm_set1_ps(JPEG_SCALE));
            gx4b = _mm_mul_ps(gx4b, _mm_set1_ps(JPEG_SCALE));
            bx4b = _mm_mul_ps(bx4b, _mm_set1_ps(JPEG_SCALE));

            r1x4b = _mm_cvtps_epi32(rx4b);
            r1x4b = av_clip_int16_sse(r1x4b);
            g1x4b = _mm_cvtps_epi32(gx4b);
            g1x4b = av_clip_int16_sse(g1x4b);
            b1x4b = _mm_cvtps_epi32(bx4b);
            b1x4b = av_clip_int16_sse(b1x4b);

            roax4 = r0x4a;
            goax4 = g0x4a;
            boax4 = b0x4a;

            robx4 = r0x4b;
            gobx4 = g0x4b;
            bobx4 = b0x4b;

            yoax4 = _mm_mullo_epi32(roax4, _mm_set1_epi32(cry));
            yoax4 = _mm_add_epi32(yoax4, _mm_mullo_epi32(goax4, _mm_set1_epi32(cgy)));
            yoax4 = _mm_add_epi32(yoax4, _mm_mullo_epi32(boax4, _mm_set1_epi32(cby)));
            yoax4 = _mm_add_epi32(yoax4, _mm_set1_epi32(out_rnd));
            yoax4 = _mm_srai_epi32(yoax4, out_sh);
            yoax4 = _mm_add_epi32(yoax4, _mm_set1_epi32(params->out_yuv_off));

            yobx4 = _mm_mullo_epi32(robx4, _mm_set1_epi32(cry));
            yobx4 = _mm_add_epi32(yobx4, _mm_mullo_epi32(gobx4, _mm_set1_epi32(cgy)));
            yobx4 = _mm_add_epi32(yobx4, _mm_mullo_epi32(bobx4, _mm_set1_epi32(cby)));
            yobx4 = _mm_add_epi32(yobx4, _mm_set1_epi32(out_rnd));
            yobx4 = _mm_srai_epi32(yobx4, out_sh);
            yobx4 = _mm_add_epi32(yobx4, _mm_set1_epi32(params->out_yuv_off));

            y0ox8 = _mm_packus_epi32(yoax4, yobx4);
            _mm_storeu_si128((__m128i_u *) &dsty[x], y0ox8);

            r1oax4 = r1x4a;
            g1oax4 = g1x4a;
            b1oax4 = b1x4a;

            r1obx4 = r1x4b;
            g1obx4 = g1x4b;
            b1obx4 = b1x4b;

            y1oax4 = _mm_mullo_epi32(r1oax4, _mm_set1_epi32(cry));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_mullo_epi32(g1oax4, _mm_set1_epi32(cgy)));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_mullo_epi32(b1oax4, _mm_set1_epi32(cby)));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_set1_epi32(out_rnd));
            y1oax4 = _mm_srai_epi32(y1oax4, out_sh);
            y1oax4 = _mm_add_epi32(y1oax4, _mm_set1_epi32(params->out_yuv_off));

            y1obx4 = _mm_mullo_epi32(r1obx4, _mm_set1_epi32(cry));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_mullo_epi32(g1obx4, _mm_set1_epi32(cgy)));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_mullo_epi32(b1obx4, _mm_set1_epi32(cby)));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_set1_epi32(out_rnd));
            y1obx4 = _mm_srai_epi32(y1obx4, out_sh);
            y1obx4 = _mm_add_epi32(y1obx4, _mm_set1_epi32(params->out_yuv_off));

            y1ox8 = _mm_packus_epi32(y1oax4, y1obx4);
            _mm_storeu_si128((__m128i_u *) &dsty[x + dstlinesize[0] / 2], y1ox8);

            ravgx4 = _mm_hadd_epi32(roax4, robx4);
            ravgx4 = _mm_add_epi32(ravgx4, _mm_hadd_epi32(r1oax4, r1obx4));
            ravgx4 = _mm_add_epi32(ravgx4, _mm_set1_epi32(2));
            ravgx4 = _mm_srai_epi32(ravgx4, 2);

            gavgx4 = _mm_hadd_epi32(goax4, gobx4);
            gavgx4 = _mm_add_epi32(gavgx4, _mm_hadd_epi32(g1oax4, g1obx4));
            gavgx4 = _mm_add_epi32(gavgx4, _mm_set1_epi32(2));
            gavgx4 = _mm_srai_epi32(gavgx4, 2);

            bavgx4 = _mm_hadd_epi32(boax4, bobx4);
            bavgx4 = _mm_add_epi32(bavgx4, _mm_hadd_epi32(b1oax4, b1obx4));
            bavgx4 = _mm_add_epi32(bavgx4, _mm_set1_epi32(2));
            bavgx4 = _mm_srai_epi32(bavgx4, 2);

            uox4 = _mm_add_epi32(_mm_set1_epi32(out_rnd), _mm_mullo_epi32(ravgx4, _mm_set1_epi32(cru)));
            uox4 = _mm_add_epi32(uox4, _mm_mullo_epi32(gavgx4, _mm_set1_epi32(ocgu)));
            uox4 = _mm_add_epi32(uox4, _mm_mullo_epi32(bavgx4, _mm_set1_epi32(cburv)));
            uox4 = _mm_srai_epi32(uox4, out_sh);
            uox4 = _mm_add_epi32(uox4, _mm_set1_epi32(out_uv_offset));
            _mm_storeu_si64((__m128i_u *) &dstu[x >> 1], _mm_packus_epi32(uox4, zero128));

            vox4 = _mm_add_epi32(_mm_set1_epi32(out_rnd), _mm_mullo_epi32(ravgx4, _mm_set1_epi32(cburv)));
            vox4 = _mm_add_epi32(vox4, _mm_mullo_epi32(gavgx4, _mm_set1_epi32(ocgv)));
            vox4 = _mm_add_epi32(vox4, _mm_mullo_epi32(bavgx4, _mm_set1_epi32(cbv)));
            vox4 = _mm_srai_epi32(vox4, out_sh);
            vox4 = _mm_add_epi32(vox4, _mm_set1_epi32(out_uv_offset));
            _mm_storeu_si64((__m128i_u *) &dstv[x >> 1], _mm_packus_epi32(vox4, zero128));
        }
    }

    // Process remaining pixels cannot fill the full simd register with scalar version
    if (remainw) {
        int offset = width & (int)0xfffffff8;
        rdsty += offset;
        rdstu += offset >> 1;
        rdstv += offset >> 1;
        rsrcy += offset;
        rsrcu += offset >> 1;
        rsrcv += offset >> 1;
        tonemap_frame_dovi_2_420hdr(rdsty, rdstu, rdstv,
                                      rsrcy, rsrcu, rsrcv,
                                      dstlinesize, srclinesize,
                                      dstdepth, srcdepth,
                                      remainw, rheight, params);
    }
#endif // ENABLE_TONEMAPX_SSE_INTRINSICS
}

X86_64_V2 void tonemap_frame_420p10_2_420p_sse(uint8_t *dsty, uint8_t *dstu, uint8_t *dstv,
                                               const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                               const int *dstlinesize, const int *srclinesize,
                                               int dstdepth, int srcdepth,
                                               int width, int height,
                                               const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_SSE_INTRINSICS
    uint8_t *rdsty = dsty;
    uint8_t *rdstu = dstu;
    uint8_t *rdstv = dstv;

    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcu = srcu;
    const uint16_t *rsrcv = srcv;

    int rheight = height;
    // not zero when not divisible by 8
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 6;

    const int in_depth = srcdepth;
    const int in_uv_offset = 128 << (in_depth - 8);
    const int in_sh = in_depth - 1;
    const int in_rnd = 1 << (in_sh - 1);

    const int out_depth = dstdepth;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);

    int cy  = (*params->yuv2rgb_coeffs)[0][0][0];
    int crv = (*params->yuv2rgb_coeffs)[0][2][0];
    int cgu = (*params->yuv2rgb_coeffs)[1][1][0];
    int cgv = (*params->yuv2rgb_coeffs)[1][2][0];
    int cbu = (*params->yuv2rgb_coeffs)[2][1][0];

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    int16_t r[8], g[8], b[8];
    int16_t r1[8], g1[8], b1[8];

    __m128i in_yuv_offx4 = _mm_set1_epi32(params->in_yuv_off);
    __m128i in_uv_offx4= _mm_set1_epi32(in_uv_offset);
    __m128i cyx4 = _mm_set1_epi32(cy);
    __m128i rndx4 = _mm_set1_epi32(in_rnd);
    __m128i zero128 = _mm_setzero_si128();
    __m128i ux4, vx4;
    __m128i y0x8, y1x8;
    __m128i y0x4a, y0x4b, y1x4a, y1x4b, ux4a, ux4b, vx4a, vx4b;
    __m128i r0x4a, g0x4a, b0x4a, r0x4b, g0x4b, b0x4b;
    __m128i r1x4a, g1x4a, b1x4a, r1x4b, g1x4b, b1x4b;

    __m128i r0ox8, g0ox8, b0ox8;
    __m128i y0ox8;
    __m128i roax4, robx4, goax4, gobx4, boax4, bobx4;
    __m128i yoax4, yobx4;

    __m128i r1ox8, g1ox8, b1ox8;
    __m128i y1ox8;
    __m128i r1oax4, r1obx4, g1oax4, g1obx4, b1oax4, b1obx4;
    __m128i y1oax4, y1obx4;
    __m128i uox4, vox4, ravgx4, gavgx4, bavgx4;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0] * 2, dstu += dstlinesize[1], dstv += dstlinesize[2],
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[2] / 2) {
        for (int xx = 0; xx < width >> 3; xx++) {
            int x = xx << 3;

            y0x8 = _mm_lddqu_si128((__m128i*)(srcy + x));
            y1x8 = _mm_lddqu_si128((__m128i*)(srcy + (srclinesize[0] / 2 + x)));
            ux4 = _mm_loadu_si64((__m128i*)(srcu + (x >> 1)));
            vx4 = _mm_loadu_si64((__m128i*)(srcv + (x >> 1)));

            y0x4a = _mm_cvtepu16_epi32(y0x8);
            y0x4b = _mm_unpackhi_epi16(y0x8, zero128);
            y1x4a = _mm_cvtepu16_epi32(y1x8);
            y1x4b = _mm_unpackhi_epi16(y1x8, zero128);
            ux4 = _mm_cvtepu16_epi32(ux4);
            vx4 = _mm_cvtepu16_epi32(vx4);
            y0x4a = _mm_sub_epi32(y0x4a, in_yuv_offx4);
            y1x4a = _mm_sub_epi32(y1x4a, in_yuv_offx4);
            y0x4b = _mm_sub_epi32(y0x4b, in_yuv_offx4);
            y1x4b = _mm_sub_epi32(y1x4b, in_yuv_offx4);
            ux4 = _mm_sub_epi32(ux4, in_uv_offx4);
            vx4 = _mm_sub_epi32(vx4, in_uv_offx4);

            ux4a = _mm_unpacklo_epi32(ux4, ux4);
            ux4b = _mm_unpackhi_epi32(ux4, ux4);
            vx4a = _mm_unpacklo_epi32(vx4, vx4);
            vx4b = _mm_unpackhi_epi32(vx4, vx4);

            // r = av_clip_int16((y * cy + crv * v + in_rnd) >> in_sh);
            r0x4a = g0x4a = b0x4a = _mm_mullo_epi32(y0x4a, cyx4);
            r0x4a = _mm_add_epi32(r0x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(crv)));
            r0x4a = _mm_add_epi32(r0x4a, rndx4);
            r0x4a = _mm_srai_epi32(r0x4a, in_sh);
            r0x4a = av_clip_int16_sse(r0x4a);

            r1x4a = g1x4a = b1x4a = _mm_mullo_epi32(y1x4a, cyx4);
            r1x4a = _mm_add_epi32(r1x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(crv)));
            r1x4a = _mm_add_epi32(r1x4a, rndx4);
            r1x4a = _mm_srai_epi32(r1x4a, in_sh);
            r1x4a = av_clip_int16_sse(r1x4a);

            // g = av_clip_int16((y * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g0x4a = _mm_add_epi32(g0x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cgu)));
            g0x4a = _mm_add_epi32(g0x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(cgv)));
            g0x4a = _mm_add_epi32(g0x4a, rndx4);
            g0x4a = _mm_srai_epi32(g0x4a, in_sh);
            g0x4a = av_clip_int16_sse(g0x4a);

            g1x4a = _mm_add_epi32(g1x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cgu)));
            g1x4a = _mm_add_epi32(g1x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(cgv)));
            g1x4a = _mm_add_epi32(g1x4a, rndx4);
            g1x4a = _mm_srai_epi32(g1x4a, in_sh);
            g1x4a = av_clip_int16_sse(g1x4a);

            // b = av_clip_int16((y * cy + cbu * u + in_rnd) >> in_sh);
            b0x4a = _mm_add_epi32(b0x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cbu)));
            b0x4a = _mm_add_epi32(b0x4a, rndx4);
            b0x4a = _mm_srai_epi32(b0x4a, in_sh);
            b0x4a = av_clip_int16_sse(b0x4a);

            b1x4a = _mm_add_epi32(b1x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cbu)));
            b1x4a = _mm_add_epi32(b1x4a, rndx4);
            b1x4a = _mm_srai_epi32(b1x4a, in_sh);
            b1x4a = av_clip_int16_sse(b1x4a);

            r0x4b = g0x4b = b0x4b = _mm_mullo_epi32(y0x4b, cyx4);
            r0x4b = _mm_add_epi32(r0x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(crv)));
            r0x4b = _mm_add_epi32(r0x4b, rndx4);
            r0x4b = _mm_srai_epi32(r0x4b, in_sh);
            r0x4b = av_clip_int16_sse(r0x4b);

            r1x4b = g1x4b = b1x4b = _mm_mullo_epi32(y1x4b, cyx4);
            r1x4b = _mm_add_epi32(r1x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(crv)));
            r1x4b = _mm_add_epi32(r1x4b, rndx4);
            r1x4b = _mm_srai_epi32(r1x4b, in_sh);
            r1x4b = av_clip_int16_sse(r1x4b);

            g0x4b = _mm_add_epi32(g0x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cgu)));
            g0x4b = _mm_add_epi32(g0x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(cgv)));
            g0x4b = _mm_add_epi32(g0x4b, rndx4);
            g0x4b = _mm_srai_epi32(g0x4b, in_sh);
            g0x4b = av_clip_int16_sse(g0x4b);

            g1x4b = _mm_add_epi32(g1x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cgu)));
            g1x4b = _mm_add_epi32(g1x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(cgv)));
            g1x4b = _mm_add_epi32(g1x4b, rndx4);
            g1x4b = _mm_srai_epi32(g1x4b, in_sh);
            g1x4b = av_clip_int16_sse(g1x4b);

            b0x4b = _mm_add_epi32(b0x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cbu)));
            b0x4b = _mm_add_epi32(b0x4b, rndx4);
            b0x4b = _mm_srai_epi32(b0x4b, in_sh);
            b0x4b = av_clip_int16_sse(b0x4b);

            b1x4b = _mm_add_epi32(b1x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cbu)));
            b1x4b = _mm_add_epi32(b1x4b, rndx4);
            b1x4b = _mm_srai_epi32(b1x4b, in_sh);
            b1x4b = av_clip_int16_sse(b1x4b);

            tonemap_int32x4_sse(r0x4a, g0x4a, b0x4a, r, g, b,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r1x4a, g1x4a, b1x4a, r1, g1, b1,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r0x4b, g0x4b, b0x4b, &r[4], &g[4], &b[4],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r1x4b, g1x4b, b1x4b, &r1[4], &g1[4], &b1[4],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);

            r0ox8 = _mm_lddqu_si128((const __m128i_u *)r);
            g0ox8 = _mm_lddqu_si128((const __m128i_u *)g);
            b0ox8 = _mm_lddqu_si128((const __m128i_u *)b);

            roax4 = _mm_cvtepi16_epi32(r0ox8);
            goax4 = _mm_cvtepi16_epi32(g0ox8);
            boax4 = _mm_cvtepi16_epi32(b0ox8);

            robx4 = _mm_unpackhi_epi16(r0ox8, zero128);
            gobx4 = _mm_unpackhi_epi16(g0ox8, zero128);
            bobx4 = _mm_unpackhi_epi16(b0ox8, zero128);

            yoax4 = _mm_mullo_epi32(roax4, _mm_set1_epi32(cry));
            yoax4 = _mm_add_epi32(yoax4, _mm_mullo_epi32(goax4, _mm_set1_epi32(cgy)));
            yoax4 = _mm_add_epi32(yoax4, _mm_mullo_epi32(boax4, _mm_set1_epi32(cby)));
            yoax4 = _mm_add_epi32(yoax4, _mm_set1_epi32(out_rnd));
            // output shift bits for 8bit outputs is 29 - 8 = 21
            yoax4 = _mm_srai_epi32(yoax4, 21);
            yoax4 = _mm_add_epi32(yoax4, _mm_set1_epi32(params->out_yuv_off));

            yobx4 = _mm_mullo_epi32(robx4, _mm_set1_epi32(cry));
            yobx4 = _mm_add_epi32(yobx4, _mm_mullo_epi32(gobx4, _mm_set1_epi32(cgy)));
            yobx4 = _mm_add_epi32(yobx4, _mm_mullo_epi32(bobx4, _mm_set1_epi32(cby)));
            yobx4 = _mm_add_epi32(yobx4, _mm_set1_epi32(out_rnd));
            yobx4 = _mm_srai_epi32(yobx4, 21);
            yobx4 = _mm_add_epi32(yobx4, _mm_set1_epi32(params->out_yuv_off));

            y0ox8 = _mm_packs_epi32(yoax4, yobx4);
            _mm_storeu_si64(&dsty[x], _mm_packus_epi16(y0ox8, zero128));

            r1ox8 = _mm_lddqu_si128((const __m128i_u *)r1);
            g1ox8 = _mm_lddqu_si128((const __m128i_u *)g1);
            b1ox8 = _mm_lddqu_si128((const __m128i_u *)b1);

            r1oax4 = _mm_cvtepi16_epi32(r1ox8);
            g1oax4 = _mm_cvtepi16_epi32(g1ox8);
            b1oax4 = _mm_cvtepi16_epi32(b1ox8);

            r1obx4 = _mm_unpackhi_epi16(r1ox8, zero128);
            g1obx4 = _mm_unpackhi_epi16(g1ox8, zero128);
            b1obx4 = _mm_unpackhi_epi16(b1ox8, zero128);

            y1oax4 = _mm_mullo_epi32(r1oax4, _mm_set1_epi32(cry));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_mullo_epi32(g1oax4, _mm_set1_epi32(cgy)));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_mullo_epi32(b1oax4, _mm_set1_epi32(cby)));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_set1_epi32(out_rnd));
            y1oax4 = _mm_srai_epi32(y1oax4, 21);
            y1oax4 = _mm_add_epi32(y1oax4, _mm_set1_epi32(params->out_yuv_off));

            y1obx4 = _mm_mullo_epi32(r1obx4, _mm_set1_epi32(cry));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_mullo_epi32(g1obx4, _mm_set1_epi32(cgy)));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_mullo_epi32(b1obx4, _mm_set1_epi32(cby)));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_set1_epi32(out_rnd));
            y1obx4 = _mm_srai_epi32(y1obx4, 21);
            y1obx4 = _mm_add_epi32(y1obx4, _mm_set1_epi32(params->out_yuv_off));

            y1ox8 = _mm_packs_epi32(y1oax4, y1obx4);
            _mm_storeu_si64(&dsty[x + dstlinesize[0]], _mm_packus_epi16(y1ox8, zero128));

            ravgx4 = _mm_hadd_epi32(roax4, robx4);
            ravgx4 = _mm_add_epi32(ravgx4, _mm_hadd_epi32(r1oax4, r1obx4));
            ravgx4 = _mm_add_epi32(ravgx4, _mm_set1_epi32(2));
            ravgx4 = _mm_srai_epi32(ravgx4, 2);

            gavgx4 = _mm_hadd_epi32(goax4, gobx4);
            gavgx4 = _mm_add_epi32(gavgx4, _mm_hadd_epi32(g1oax4, g1obx4));
            gavgx4 = _mm_add_epi32(gavgx4, _mm_set1_epi32(2));
            gavgx4 = _mm_srai_epi32(gavgx4, 2);

            bavgx4 = _mm_hadd_epi32(boax4, bobx4);
            bavgx4 = _mm_add_epi32(bavgx4, _mm_hadd_epi32(b1oax4, b1obx4));
            bavgx4 = _mm_add_epi32(bavgx4, _mm_set1_epi32(2));
            bavgx4 = _mm_srai_epi32(bavgx4, 2);

            uox4 = _mm_add_epi32(_mm_set1_epi32(out_rnd), _mm_mullo_epi32(ravgx4, _mm_set1_epi32(cru)));
            uox4 = _mm_add_epi32(uox4, _mm_mullo_epi32(gavgx4, _mm_set1_epi32(ocgu)));
            uox4 = _mm_add_epi32(uox4, _mm_mullo_epi32(bavgx4, _mm_set1_epi32(cburv)));
            uox4 = _mm_srai_epi32(uox4, 21);
            uox4 = _mm_add_epi32(uox4, _mm_set1_epi32(out_uv_offset));
            _mm_storeu_si32(&dstu[x >> 1], _mm_packus_epi16(_mm_packs_epi32(uox4, zero128), zero128));

            vox4 = _mm_add_epi32(_mm_set1_epi32(out_rnd), _mm_mullo_epi32(ravgx4, _mm_set1_epi32(cburv)));
            vox4 = _mm_add_epi32(vox4, _mm_mullo_epi32(gavgx4, _mm_set1_epi32(ocgv)));
            vox4 = _mm_add_epi32(vox4, _mm_mullo_epi32(bavgx4, _mm_set1_epi32(cbv)));
            vox4 = _mm_srai_epi32(vox4, 21);
            vox4 = _mm_add_epi32(vox4, _mm_set1_epi32(out_uv_offset));
            _mm_storeu_si32(&dstv[x >> 1], _mm_packus_epi16(_mm_packs_epi32(vox4, zero128), zero128));
        }
    }

    // Process remaining pixels cannot fill the full simd register with scalar version
    if (remainw) {
        int offset = width & (int)0xfffffff8;
        rdsty += offset;
        rdstu += offset >> 1;
        rdstv += offset >> 1;
        rsrcy += offset;
        rsrcu += offset >> 1;
        rsrcv += offset >> 1;
        tonemap_frame_420p10_2_420p(rdsty, rdstu, rdstv,
                                    rsrcy, rsrcu, rsrcv,
                                    dstlinesize, srclinesize,
                                    dstdepth, srcdepth,
                                    remainw, rheight, params);
    }
#endif // ENABLE_TONEMAPX_SSE_INTRINSICS
}

X86_64_V2 void tonemap_frame_420p10_2_420p10_sse(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                                 const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                                 const int *dstlinesize, const int *srclinesize,
                                                 int dstdepth, int srcdepth,
                                                 int width, int height,
                                                 const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_SSE_INTRINSICS
    uint16_t *rdsty = dsty;
    uint16_t *rdstu = dstu;
    uint16_t *rdstv = dstv;
    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcu = srcu;
    const uint16_t *rsrcv = srcv;
    int rheight = height;
    // not zero when not divisible by 8
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 6;

    const int in_depth = srcdepth;
    const int in_uv_offset = 128 << (in_depth - 8);
    const int in_sh = in_depth - 1;
    const int in_rnd = 1 << (in_sh - 1);

    const int out_depth = dstdepth;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);

    int cy  = (*params->yuv2rgb_coeffs)[0][0][0];
    int crv = (*params->yuv2rgb_coeffs)[0][2][0];
    int cgu = (*params->yuv2rgb_coeffs)[1][1][0];
    int cgv = (*params->yuv2rgb_coeffs)[1][2][0];
    int cbu = (*params->yuv2rgb_coeffs)[2][1][0];

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    int16_t r[8], g[8], b[8];
    int16_t r1[8], g1[8], b1[8];

    __m128i in_yuv_offx4 = _mm_set1_epi32(params->in_yuv_off);
    __m128i in_uv_offx4= _mm_set1_epi32(in_uv_offset);
    __m128i cyx4 = _mm_set1_epi32(cy);
    __m128i rndx4 = _mm_set1_epi32(in_rnd);
    __m128i zero128 = _mm_setzero_si128();
    __m128i ux4, vx4;
    __m128i y0x8, y1x8;
    __m128i y0x4a, y0x4b, y1x4a, y1x4b, ux4a, ux4b, vx4a, vx4b;
    __m128i r0x4a, g0x4a, b0x4a, r0x4b, g0x4b, b0x4b;
    __m128i r1x4a, g1x4a, b1x4a, r1x4b, g1x4b, b1x4b;

    __m128i r0ox8, g0ox8, b0ox8;
    __m128i y0ox8;
    __m128i roax4, robx4, goax4, gobx4, boax4, bobx4;
    __m128i yoax4, yobx4;

    __m128i r1ox8, g1ox8, b1ox8;
    __m128i y1ox8;
    __m128i r1oax4, r1obx4, g1oax4, g1obx4, b1oax4, b1obx4;
    __m128i y1oax4, y1obx4;
    __m128i uox4, vox4, ravgx4, gavgx4, bavgx4;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstu += dstlinesize[1] / 2, dstv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 3; xx++) {
            int x = xx << 3;

            y0x8 = _mm_lddqu_si128((__m128i*)(srcy + x));
            y1x8 = _mm_lddqu_si128((__m128i*)(srcy + (srclinesize[0] / 2 + x)));
            ux4 = _mm_loadu_si64((__m128i*)(srcu + (x >> 1)));
            vx4 = _mm_loadu_si64((__m128i*)(srcv + (x >> 1)));

            y0x4a = _mm_cvtepu16_epi32(y0x8);
            y0x4b = _mm_unpackhi_epi16(y0x8, zero128);
            y1x4a = _mm_cvtepu16_epi32(y1x8);
            y1x4b = _mm_unpackhi_epi16(y1x8, zero128);
            ux4 = _mm_cvtepu16_epi32(ux4);
            vx4 = _mm_cvtepu16_epi32(vx4);
            y0x4a = _mm_sub_epi32(y0x4a, in_yuv_offx4);
            y1x4a = _mm_sub_epi32(y1x4a, in_yuv_offx4);
            y0x4b = _mm_sub_epi32(y0x4b, in_yuv_offx4);
            y1x4b = _mm_sub_epi32(y1x4b, in_yuv_offx4);
            ux4 = _mm_sub_epi32(ux4, in_uv_offx4);
            vx4 = _mm_sub_epi32(vx4, in_uv_offx4);

            ux4a = _mm_unpacklo_epi32(ux4, ux4);
            ux4b = _mm_unpackhi_epi32(ux4, ux4);
            vx4a = _mm_unpacklo_epi32(vx4, vx4);
            vx4b = _mm_unpackhi_epi32(vx4, vx4);

            // r = av_clip_int16((y * cy + crv * v + in_rnd) >> in_sh);
            r0x4a = g0x4a = b0x4a = _mm_mullo_epi32(y0x4a, cyx4);
            r0x4a = _mm_add_epi32(r0x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(crv)));
            r0x4a = _mm_add_epi32(r0x4a, rndx4);
            r0x4a = _mm_srai_epi32(r0x4a, in_sh);
            r0x4a = av_clip_int16_sse(r0x4a);

            r1x4a = g1x4a = b1x4a = _mm_mullo_epi32(y1x4a, cyx4);
            r1x4a = _mm_add_epi32(r1x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(crv)));
            r1x4a = _mm_add_epi32(r1x4a, rndx4);
            r1x4a = _mm_srai_epi32(r1x4a, in_sh);
            r1x4a = av_clip_int16_sse(r1x4a);

            // g = av_clip_int16((y * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g0x4a = _mm_add_epi32(g0x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cgu)));
            g0x4a = _mm_add_epi32(g0x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(cgv)));
            g0x4a = _mm_add_epi32(g0x4a, rndx4);
            g0x4a = _mm_srai_epi32(g0x4a, in_sh);
            g0x4a = av_clip_int16_sse(g0x4a);

            g1x4a = _mm_add_epi32(g1x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cgu)));
            g1x4a = _mm_add_epi32(g1x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(cgv)));
            g1x4a = _mm_add_epi32(g1x4a, rndx4);
            g1x4a = _mm_srai_epi32(g1x4a, in_sh);
            g1x4a = av_clip_int16_sse(g1x4a);

            // b = av_clip_int16((y * cy + cbu * u + in_rnd) >> in_sh);
            b0x4a = _mm_add_epi32(b0x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cbu)));
            b0x4a = _mm_add_epi32(b0x4a, rndx4);
            b0x4a = _mm_srai_epi32(b0x4a, in_sh);
            b0x4a = av_clip_int16_sse(b0x4a);

            b1x4a = _mm_add_epi32(b1x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cbu)));
            b1x4a = _mm_add_epi32(b1x4a, rndx4);
            b1x4a = _mm_srai_epi32(b1x4a, in_sh);
            b1x4a = av_clip_int16_sse(b1x4a);

            r0x4b = g0x4b = b0x4b = _mm_mullo_epi32(y0x4b, cyx4);
            r0x4b = _mm_add_epi32(r0x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(crv)));
            r0x4b = _mm_add_epi32(r0x4b, rndx4);
            r0x4b = _mm_srai_epi32(r0x4b, in_sh);
            r0x4b = av_clip_int16_sse(r0x4b);

            r1x4b = g1x4b = b1x4b = _mm_mullo_epi32(y1x4b, cyx4);
            r1x4b = _mm_add_epi32(r1x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(crv)));
            r1x4b = _mm_add_epi32(r1x4b, rndx4);
            r1x4b = _mm_srai_epi32(r1x4b, in_sh);
            r1x4b = av_clip_int16_sse(r1x4b);

            g0x4b = _mm_add_epi32(g0x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cgu)));
            g0x4b = _mm_add_epi32(g0x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(cgv)));
            g0x4b = _mm_add_epi32(g0x4b, rndx4);
            g0x4b = _mm_srai_epi32(g0x4b, in_sh);
            g0x4b = av_clip_int16_sse(g0x4b);

            g1x4b = _mm_add_epi32(g1x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cgu)));
            g1x4b = _mm_add_epi32(g1x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(cgv)));
            g1x4b = _mm_add_epi32(g1x4b, rndx4);
            g1x4b = _mm_srai_epi32(g1x4b, in_sh);
            g1x4b = av_clip_int16_sse(g1x4b);

            b0x4b = _mm_add_epi32(b0x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cbu)));
            b0x4b = _mm_add_epi32(b0x4b, rndx4);
            b0x4b = _mm_srai_epi32(b0x4b, in_sh);
            b0x4b = av_clip_int16_sse(b0x4b);

            b1x4b = _mm_add_epi32(b1x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cbu)));
            b1x4b = _mm_add_epi32(b1x4b, rndx4);
            b1x4b = _mm_srai_epi32(b1x4b, in_sh);
            b1x4b = av_clip_int16_sse(b1x4b);

            tonemap_int32x4_sse(r0x4a, g0x4a, b0x4a, r, g, b,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r1x4a, g1x4a, b1x4a, r1, g1, b1,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r0x4b, g0x4b, b0x4b, &r[4], &g[4], &b[4],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r1x4b, g1x4b, b1x4b, &r1[4], &g1[4], &b1[4],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);

            r0ox8 = _mm_lddqu_si128((const __m128i_u *)r);
            g0ox8 = _mm_lddqu_si128((const __m128i_u *)g);
            b0ox8 = _mm_lddqu_si128((const __m128i_u *)b);

            roax4 = _mm_cvtepi16_epi32(r0ox8);
            goax4 = _mm_cvtepi16_epi32(g0ox8);
            boax4 = _mm_cvtepi16_epi32(b0ox8);

            robx4 = _mm_unpackhi_epi16(r0ox8, zero128);
            gobx4 = _mm_unpackhi_epi16(g0ox8, zero128);
            bobx4 = _mm_unpackhi_epi16(b0ox8, zero128);

            yoax4 = _mm_mullo_epi32(roax4, _mm_set1_epi32(cry));
            yoax4 = _mm_add_epi32(yoax4, _mm_mullo_epi32(goax4, _mm_set1_epi32(cgy)));
            yoax4 = _mm_add_epi32(yoax4, _mm_mullo_epi32(boax4, _mm_set1_epi32(cby)));
            yoax4 = _mm_add_epi32(yoax4, _mm_set1_epi32(out_rnd));
            yoax4 = _mm_srai_epi32(yoax4, out_sh);
            yoax4 = _mm_add_epi32(yoax4, _mm_set1_epi32(params->out_yuv_off));

            yobx4 = _mm_mullo_epi32(robx4, _mm_set1_epi32(cry));
            yobx4 = _mm_add_epi32(yobx4, _mm_mullo_epi32(gobx4, _mm_set1_epi32(cgy)));
            yobx4 = _mm_add_epi32(yobx4, _mm_mullo_epi32(bobx4, _mm_set1_epi32(cby)));
            yobx4 = _mm_add_epi32(yobx4, _mm_set1_epi32(out_rnd));
            yobx4 = _mm_srai_epi32(yobx4, out_sh);
            yobx4 = _mm_add_epi32(yobx4, _mm_set1_epi32(params->out_yuv_off));

            y0ox8 = _mm_packus_epi32(yoax4, yobx4);
            _mm_storeu_si128((__m128i_u *) &dsty[x], y0ox8);

            r1ox8 = _mm_lddqu_si128((const __m128i_u *)r1);
            g1ox8 = _mm_lddqu_si128((const __m128i_u *)g1);
            b1ox8 = _mm_lddqu_si128((const __m128i_u *)b1);

            r1oax4 = _mm_cvtepi16_epi32(r1ox8);
            g1oax4 = _mm_cvtepi16_epi32(g1ox8);
            b1oax4 = _mm_cvtepi16_epi32(b1ox8);

            r1obx4 = _mm_unpackhi_epi16(r1ox8, zero128);
            g1obx4 = _mm_unpackhi_epi16(g1ox8, zero128);
            b1obx4 = _mm_unpackhi_epi16(b1ox8, zero128);

            y1oax4 = _mm_mullo_epi32(r1oax4, _mm_set1_epi32(cry));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_mullo_epi32(g1oax4, _mm_set1_epi32(cgy)));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_mullo_epi32(b1oax4, _mm_set1_epi32(cby)));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_set1_epi32(out_rnd));
            y1oax4 = _mm_srai_epi32(y1oax4, out_sh);
            y1oax4 = _mm_add_epi32(y1oax4, _mm_set1_epi32(params->out_yuv_off));

            y1obx4 = _mm_mullo_epi32(r1obx4, _mm_set1_epi32(cry));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_mullo_epi32(g1obx4, _mm_set1_epi32(cgy)));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_mullo_epi32(b1obx4, _mm_set1_epi32(cby)));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_set1_epi32(out_rnd));
            y1obx4 = _mm_srai_epi32(y1obx4, out_sh);
            y1obx4 = _mm_add_epi32(y1obx4, _mm_set1_epi32(params->out_yuv_off));

            y1ox8 = _mm_packus_epi32(y1oax4, y1obx4);
            _mm_storeu_si128((__m128i_u *) &dsty[x + dstlinesize[0] / 2], y1ox8);

            ravgx4 = _mm_hadd_epi32(roax4, robx4);
            ravgx4 = _mm_add_epi32(ravgx4, _mm_hadd_epi32(r1oax4, r1obx4));
            ravgx4 = _mm_add_epi32(ravgx4, _mm_set1_epi32(2));
            ravgx4 = _mm_srai_epi32(ravgx4, 2);

            gavgx4 = _mm_hadd_epi32(goax4, gobx4);
            gavgx4 = _mm_add_epi32(gavgx4, _mm_hadd_epi32(g1oax4, g1obx4));
            gavgx4 = _mm_add_epi32(gavgx4, _mm_set1_epi32(2));
            gavgx4 = _mm_srai_epi32(gavgx4, 2);

            bavgx4 = _mm_hadd_epi32(boax4, bobx4);
            bavgx4 = _mm_add_epi32(bavgx4, _mm_hadd_epi32(b1oax4, b1obx4));
            bavgx4 = _mm_add_epi32(bavgx4, _mm_set1_epi32(2));
            bavgx4 = _mm_srai_epi32(bavgx4, 2);

            uox4 = _mm_add_epi32(_mm_set1_epi32(out_rnd), _mm_mullo_epi32(ravgx4, _mm_set1_epi32(cru)));
            uox4 = _mm_add_epi32(uox4, _mm_mullo_epi32(gavgx4, _mm_set1_epi32(ocgu)));
            uox4 = _mm_add_epi32(uox4, _mm_mullo_epi32(bavgx4, _mm_set1_epi32(cburv)));
            uox4 = _mm_srai_epi32(uox4, out_sh);
            uox4 = _mm_add_epi32(uox4, _mm_set1_epi32(out_uv_offset));
            _mm_storeu_si64((__m128i_u *) &dstu[x >> 1], _mm_packus_epi32(uox4, zero128));

            vox4 = _mm_add_epi32(_mm_set1_epi32(out_rnd), _mm_mullo_epi32(ravgx4, _mm_set1_epi32(cburv)));
            vox4 = _mm_add_epi32(vox4, _mm_mullo_epi32(gavgx4, _mm_set1_epi32(ocgv)));
            vox4 = _mm_add_epi32(vox4, _mm_mullo_epi32(bavgx4, _mm_set1_epi32(cbv)));
            vox4 = _mm_srai_epi32(vox4, out_sh);
            vox4 = _mm_add_epi32(vox4, _mm_set1_epi32(out_uv_offset));
            _mm_storeu_si64((__m128i_u *) &dstv[x >> 1], _mm_packus_epi32(vox4, zero128));
        }
    }

    // Process remaining pixels cannot fill the full simd register with scalar version
    if (remainw) {
        int offset = width & (int)0xfffffff8;
        rdsty += offset;
        rdstu += offset >> 1;
        rdstv += offset >> 1;
        rsrcy += offset;
        rsrcu += offset >> 1;
        rsrcv += offset >> 1;
        tonemap_frame_420p10_2_420p10(rdsty, rdstu, rdstv,
                                      rsrcy, rsrcu, rsrcv,
                                      dstlinesize, srclinesize,
                                      dstdepth, srcdepth,
                                      remainw, rheight, params);
    }
#endif // ENABLE_TONEMAPX_SSE_INTRINSICS
}

X86_64_V2 void tonemap_frame_p010_2_nv12_sse(uint8_t *dsty, uint8_t *dstuv,
                                             const uint16_t *srcy, const uint16_t *srcuv,
                                             const int *dstlinesize, const int *srclinesize,
                                             int dstdepth, int srcdepth,
                                             int width, int height,
                                             const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_SSE_INTRINSICS
    uint8_t *rdsty = dsty;
    uint8_t *rdstuv = dstuv;
    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcuv = srcuv;
    int rheight = height;
    // not zero when not divisible by 8
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 6;

    const int in_depth = srcdepth;
    const int in_uv_offset = 128 << (in_depth - 8);
    const int in_sh = in_depth - 1;
    const int in_rnd = 1 << (in_sh - 1);

    const int out_depth = dstdepth;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);

    int cy  = (*params->yuv2rgb_coeffs)[0][0][0];
    int crv = (*params->yuv2rgb_coeffs)[0][2][0];
    int cgu = (*params->yuv2rgb_coeffs)[1][1][0];
    int cgv = (*params->yuv2rgb_coeffs)[1][2][0];
    int cbu = (*params->yuv2rgb_coeffs)[2][1][0];

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    int16_t r[8], g[8], b[8];
    int16_t r1[8], g1[8], b1[8];

    __m128i in_yuv_offx4 = _mm_set1_epi32(params->in_yuv_off);
    __m128i in_uv_offx4= _mm_set1_epi32(in_uv_offset);
    __m128i cyx4 = _mm_set1_epi32(cy);
    __m128i rndx4 = _mm_set1_epi32(in_rnd);
    __m128i zero128 = _mm_setzero_si128();
    __m128i uvx8, uvx4a, uvx4b;
    __m128i y0x8, y1x8;
    __m128i y0x4a, y0x4b, y1x4a, y1x4b, ux4a, ux4b, vx4a, vx4b;
    __m128i r0x4a, g0x4a, b0x4a, r0x4b, g0x4b, b0x4b;
    __m128i r1x4a, g1x4a, b1x4a, r1x4b, g1x4b, b1x4b;

    __m128i r0ox8, g0ox8, b0ox8;
    __m128i y0ox8;
    __m128i roax4, robx4, goax4, gobx4, boax4, bobx4;
    __m128i yoax4, yobx4;

    __m128i r1ox8, g1ox8, b1ox8;
    __m128i y1ox8;
    __m128i r1oax4, r1obx4, g1oax4, g1obx4, b1oax4, b1obx4;
    __m128i y1oax4, y1obx4, uvoax4, uvobx4;
    __m128i uoax4, voax4, ravgx4, gavgx4, bavgx4;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0] * 2, dstuv += dstlinesize[1],
                       srcy += srclinesize[0], srcuv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 3; xx++) {
            int x = xx << 3;

            y0x8 = _mm_lddqu_si128((__m128i*)(srcy + x));
            y1x8 = _mm_lddqu_si128((__m128i*)(srcy + (srclinesize[0] / 2 + x)));
            uvx8 = _mm_lddqu_si128((__m128i*)(srcuv + x));

            // shift to low10bits for 10bit input
            // shift bit has to be compile-time constant
            y0x8 = _mm_srli_epi16(y0x8, TEN_BIT_BIPLANAR_SHIFT);
            y1x8 = _mm_srli_epi16(y1x8, TEN_BIT_BIPLANAR_SHIFT);
            uvx8 = _mm_srli_epi16(uvx8, TEN_BIT_BIPLANAR_SHIFT);
            y0x4a = _mm_cvtepu16_epi32(y0x8);
            y0x4b = _mm_unpackhi_epi16(y0x8, zero128);
            y1x4a = _mm_cvtepu16_epi32(y1x8);
            y1x4b = _mm_unpackhi_epi16(y1x8, zero128);
            uvx4a = _mm_cvtepu16_epi32(uvx8);
            uvx4b = _mm_unpackhi_epi16(uvx8, zero128);
            y0x4a = _mm_sub_epi32(y0x4a, in_yuv_offx4);
            y1x4a = _mm_sub_epi32(y1x4a, in_yuv_offx4);
            y0x4b = _mm_sub_epi32(y0x4b, in_yuv_offx4);
            y1x4b = _mm_sub_epi32(y1x4b, in_yuv_offx4);
            uvx4a = _mm_sub_epi32(uvx4a, in_uv_offx4);
            uvx4b = _mm_sub_epi32(uvx4b, in_uv_offx4);

            ux4a = _mm_shuffle_epi32(uvx4a, _MM_SHUFFLE(2, 2, 0, 0));
            ux4b = _mm_shuffle_epi32(uvx4b, _MM_SHUFFLE(2, 2, 0, 0));
            vx4a = _mm_shuffle_epi32(uvx4a, _MM_SHUFFLE(3, 3, 1, 1));
            vx4b = _mm_shuffle_epi32(uvx4b, _MM_SHUFFLE(3, 3, 1, 1));

            // r = av_clip_int16((y * cy + crv * v + in_rnd) >> in_sh);
            r0x4a = g0x4a = b0x4a = _mm_mullo_epi32(y0x4a, cyx4);
            r0x4a = _mm_add_epi32(r0x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(crv)));
            r0x4a = _mm_add_epi32(r0x4a, rndx4);
            r0x4a = _mm_srai_epi32(r0x4a, in_sh);
            r0x4a = av_clip_int16_sse(r0x4a);

            r1x4a = g1x4a = b1x4a = _mm_mullo_epi32(y1x4a, cyx4);
            r1x4a = _mm_add_epi32(r1x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(crv)));
            r1x4a = _mm_add_epi32(r1x4a, rndx4);
            r1x4a = _mm_srai_epi32(r1x4a, in_sh);
            r1x4a = av_clip_int16_sse(r1x4a);

            // g = av_clip_int16((y * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g0x4a = _mm_add_epi32(g0x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cgu)));
            g0x4a = _mm_add_epi32(g0x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(cgv)));
            g0x4a = _mm_add_epi32(g0x4a, rndx4);
            g0x4a = _mm_srai_epi32(g0x4a, in_sh);
            g0x4a = av_clip_int16_sse(g0x4a);

            g1x4a = _mm_add_epi32(g1x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cgu)));
            g1x4a = _mm_add_epi32(g1x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(cgv)));
            g1x4a = _mm_add_epi32(g1x4a, rndx4);
            g1x4a = _mm_srai_epi32(g1x4a, in_sh);
            g1x4a = av_clip_int16_sse(g1x4a);

            // b = av_clip_int16((y * cy + cbu * u + in_rnd) >> in_sh);
            b0x4a = _mm_add_epi32(b0x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cbu)));
            b0x4a = _mm_add_epi32(b0x4a, rndx4);
            b0x4a = _mm_srai_epi32(b0x4a, in_sh);
            b0x4a = av_clip_int16_sse(b0x4a);

            b1x4a = _mm_add_epi32(b1x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cbu)));
            b1x4a = _mm_add_epi32(b1x4a, rndx4);
            b1x4a = _mm_srai_epi32(b1x4a, in_sh);
            b1x4a = av_clip_int16_sse(b1x4a);

            r0x4b = g0x4b = b0x4b = _mm_mullo_epi32(y0x4b, cyx4);
            r0x4b = _mm_add_epi32(r0x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(crv)));
            r0x4b = _mm_add_epi32(r0x4b, rndx4);
            r0x4b = _mm_srai_epi32(r0x4b, in_sh);
            r0x4b = av_clip_int16_sse(r0x4b);

            r1x4b = g1x4b = b1x4b = _mm_mullo_epi32(y1x4b, cyx4);
            r1x4b = _mm_add_epi32(r1x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(crv)));
            r1x4b = _mm_add_epi32(r1x4b, rndx4);
            r1x4b = _mm_srai_epi32(r1x4b, in_sh);
            r1x4b = av_clip_int16_sse(r1x4b);

            g0x4b = _mm_add_epi32(g0x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cgu)));
            g0x4b = _mm_add_epi32(g0x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(cgv)));
            g0x4b = _mm_add_epi32(g0x4b, rndx4);
            g0x4b = _mm_srai_epi32(g0x4b, in_sh);
            g0x4b = av_clip_int16_sse(g0x4b);

            g1x4b = _mm_add_epi32(g1x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cgu)));
            g1x4b = _mm_add_epi32(g1x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(cgv)));
            g1x4b = _mm_add_epi32(g1x4b, rndx4);
            g1x4b = _mm_srai_epi32(g1x4b, in_sh);
            g1x4b = av_clip_int16_sse(g1x4b);

            b0x4b = _mm_add_epi32(b0x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cbu)));
            b0x4b = _mm_add_epi32(b0x4b, rndx4);
            b0x4b = _mm_srai_epi32(b0x4b, in_sh);
            b0x4b = av_clip_int16_sse(b0x4b);

            b1x4b = _mm_add_epi32(b1x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cbu)));
            b1x4b = _mm_add_epi32(b1x4b, rndx4);
            b1x4b = _mm_srai_epi32(b1x4b, in_sh);
            b1x4b = av_clip_int16_sse(b1x4b);

            tonemap_int32x4_sse(r0x4a, g0x4a, b0x4a, r, g, b,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r1x4a, g1x4a, b1x4a, r1, g1, b1,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r0x4b, g0x4b, b0x4b, &r[4], &g[4], &b[4],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r1x4b, g1x4b, b1x4b, &r1[4], &g1[4], &b1[4],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);

            r0ox8 = _mm_lddqu_si128((const __m128i_u *)r);
            g0ox8 = _mm_lddqu_si128((const __m128i_u *)g);
            b0ox8 = _mm_lddqu_si128((const __m128i_u *)b);

            roax4 = _mm_cvtepi16_epi32(r0ox8);
            goax4 = _mm_cvtepi16_epi32(g0ox8);
            boax4 = _mm_cvtepi16_epi32(b0ox8);

            robx4 = _mm_unpackhi_epi16(r0ox8, zero128);
            gobx4 = _mm_unpackhi_epi16(g0ox8, zero128);
            bobx4 = _mm_unpackhi_epi16(b0ox8, zero128);

            yoax4 = _mm_mullo_epi32(roax4, _mm_set1_epi32(cry));
            yoax4 = _mm_add_epi32(yoax4, _mm_mullo_epi32(goax4, _mm_set1_epi32(cgy)));
            yoax4 = _mm_add_epi32(yoax4, _mm_mullo_epi32(boax4, _mm_set1_epi32(cby)));
            yoax4 = _mm_add_epi32(yoax4, _mm_set1_epi32(out_rnd));
            // output shift bits for 8bit outputs is 29 - 8 = 21
            yoax4 = _mm_srai_epi32(yoax4, 21);
            yoax4 = _mm_add_epi32(yoax4, _mm_set1_epi32(params->out_yuv_off));

            yobx4 = _mm_mullo_epi32(robx4, _mm_set1_epi32(cry));
            yobx4 = _mm_add_epi32(yobx4, _mm_mullo_epi32(gobx4, _mm_set1_epi32(cgy)));
            yobx4 = _mm_add_epi32(yobx4, _mm_mullo_epi32(bobx4, _mm_set1_epi32(cby)));
            yobx4 = _mm_add_epi32(yobx4, _mm_set1_epi32(out_rnd));
            yobx4 = _mm_srai_epi32(yobx4, 21);
            yobx4 = _mm_add_epi32(yobx4, _mm_set1_epi32(params->out_yuv_off));

            y0ox8 = _mm_packs_epi32(yoax4, yobx4);
            _mm_storeu_si64(&dsty[x], _mm_packus_epi16(y0ox8, zero128));

            r1ox8 = _mm_lddqu_si128((const __m128i_u *)r1);
            g1ox8 = _mm_lddqu_si128((const __m128i_u *)g1);
            b1ox8 = _mm_lddqu_si128((const __m128i_u *)b1);

            r1oax4 = _mm_cvtepi16_epi32(r1ox8);
            g1oax4 = _mm_cvtepi16_epi32(g1ox8);
            b1oax4 = _mm_cvtepi16_epi32(b1ox8);

            r1obx4 = _mm_unpackhi_epi16(r1ox8, zero128);
            g1obx4 = _mm_unpackhi_epi16(g1ox8, zero128);
            b1obx4 = _mm_unpackhi_epi16(b1ox8, zero128);

            y1oax4 = _mm_mullo_epi32(r1oax4, _mm_set1_epi32(cry));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_mullo_epi32(g1oax4, _mm_set1_epi32(cgy)));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_mullo_epi32(b1oax4, _mm_set1_epi32(cby)));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_set1_epi32(out_rnd));
            y1oax4 = _mm_srai_epi32(y1oax4, 21);
            y1oax4 = _mm_add_epi32(y1oax4, _mm_set1_epi32(params->out_yuv_off));

            y1obx4 = _mm_mullo_epi32(r1obx4, _mm_set1_epi32(cry));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_mullo_epi32(g1obx4, _mm_set1_epi32(cgy)));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_mullo_epi32(b1obx4, _mm_set1_epi32(cby)));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_set1_epi32(out_rnd));
            y1obx4 = _mm_srai_epi32(y1obx4, 21);
            y1obx4 = _mm_add_epi32(y1obx4, _mm_set1_epi32(params->out_yuv_off));

            y1ox8 = _mm_packs_epi32(y1oax4, y1obx4);
            _mm_storeu_si64(&dsty[x + dstlinesize[0]], _mm_packus_epi16(y1ox8, zero128));

            ravgx4 = _mm_hadd_epi32(roax4, robx4);
            ravgx4 = _mm_add_epi32(ravgx4, _mm_hadd_epi32(r1oax4, r1obx4));
            ravgx4 = _mm_add_epi32(ravgx4, _mm_set1_epi32(2));
            ravgx4 = _mm_srai_epi32(ravgx4, 2);

            gavgx4 = _mm_hadd_epi32(goax4, gobx4);
            gavgx4 = _mm_add_epi32(gavgx4, _mm_hadd_epi32(g1oax4, g1obx4));
            gavgx4 = _mm_add_epi32(gavgx4, _mm_set1_epi32(2));
            gavgx4 = _mm_srai_epi32(gavgx4, 2);

            bavgx4 = _mm_hadd_epi32(boax4, bobx4);
            bavgx4 = _mm_add_epi32(bavgx4, _mm_hadd_epi32(b1oax4, b1obx4));
            bavgx4 = _mm_add_epi32(bavgx4, _mm_set1_epi32(2));
            bavgx4 = _mm_srai_epi32(bavgx4, 2);

            uoax4 = _mm_add_epi32(_mm_set1_epi32(out_rnd), _mm_mullo_epi32(ravgx4, _mm_set1_epi32(cru)));
            uoax4 = _mm_add_epi32(uoax4, _mm_mullo_epi32(gavgx4, _mm_set1_epi32(ocgu)));
            uoax4 = _mm_add_epi32(uoax4, _mm_mullo_epi32(bavgx4, _mm_set1_epi32(cburv)));
            uoax4 = _mm_srai_epi32(uoax4, 21);
            uoax4 = _mm_add_epi32(uoax4, _mm_set1_epi32(out_uv_offset));

            voax4 = _mm_add_epi32(_mm_set1_epi32(out_rnd), _mm_mullo_epi32(ravgx4, _mm_set1_epi32(cburv)));
            voax4 = _mm_add_epi32(voax4, _mm_mullo_epi32(gavgx4, _mm_set1_epi32(ocgv)));
            voax4 = _mm_add_epi32(voax4, _mm_mullo_epi32(bavgx4, _mm_set1_epi32(cbv)));
            voax4 = _mm_srai_epi32(voax4, 21);
            voax4 = _mm_add_epi32(voax4, _mm_set1_epi32(out_uv_offset));

            uvoax4 = _mm_unpacklo_epi32(uoax4, voax4);
            uvobx4 = _mm_unpackhi_epi32(uoax4, voax4);
            _mm_storeu_si64(&dstuv[x], _mm_packus_epi16(_mm_packs_epi32(uvoax4, uvobx4), zero128));
        }
    }

    // Process remaining pixels cannot fill the full simd register with scalar version
    if (remainw) {
        int offset = width & (int)0xfffffff8;
        rdsty += offset;
        rdstuv += offset;
        rsrcy += offset;
        rsrcuv += offset;
        tonemap_frame_p010_2_nv12(rdsty, rdstuv,
                                  rsrcy, rsrcuv,
                                  dstlinesize, srclinesize,
                                  dstdepth, srcdepth,
                                  remainw, rheight, params);
    }
#endif // ENABLE_TONEMAPX_SSE_INTRINSICS
}

X86_64_V2 void tonemap_frame_p010_2_p010_sse(uint16_t *dsty, uint16_t *dstuv,
                                             const uint16_t *srcy, const uint16_t *srcuv,
                                             const int *dstlinesize, const int *srclinesize,
                                             int dstdepth, int srcdepth,
                                             int width, int height,
                                             const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_SSE_INTRINSICS
    uint16_t *rdsty = dsty;
    uint16_t *rdstuv = dstuv;
    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcuv = srcuv;
    int rheight = height;
    // not zero when not divisible by 8
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 6;

    const int in_depth = srcdepth;
    const int in_uv_offset = 128 << (in_depth - 8);
    const int in_sh = in_depth - 1;
    const int in_rnd = 1 << (in_sh - 1);

    const int out_depth = dstdepth;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);
    const int out_sh2 = 16 - out_depth;

    int cy  = (*params->yuv2rgb_coeffs)[0][0][0];
    int crv = (*params->yuv2rgb_coeffs)[0][2][0];
    int cgu = (*params->yuv2rgb_coeffs)[1][1][0];
    int cgv = (*params->yuv2rgb_coeffs)[1][2][0];
    int cbu = (*params->yuv2rgb_coeffs)[2][1][0];

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    int16_t r[8], g[8], b[8];
    int16_t r1[8], g1[8], b1[8];

    __m128i in_yuv_offx4 = _mm_set1_epi32(params->in_yuv_off);
    __m128i in_uv_offx4= _mm_set1_epi32(in_uv_offset);
    __m128i cyx4 = _mm_set1_epi32(cy);
    __m128i rndx4 = _mm_set1_epi32(in_rnd);
    __m128i zero128 = _mm_setzero_si128();
    __m128i uvx8, uvx4a, uvx4b;
    __m128i y0x8, y1x8;
    __m128i y0x4a, y0x4b, y1x4a, y1x4b, ux4a, ux4b, vx4a, vx4b;
    __m128i r0x4a, g0x4a, b0x4a, r0x4b, g0x4b, b0x4b;
    __m128i r1x4a, g1x4a, b1x4a, r1x4b, g1x4b, b1x4b;

    __m128i r0ox8, g0ox8, b0ox8;
    __m128i y0ox8;
    __m128i roax4, robx4, goax4, gobx4, boax4, bobx4;
    __m128i yoax4, yobx4;

    __m128i r1ox8, g1ox8, b1ox8;
    __m128i y1ox8;
    __m128i r1oax4, r1obx4, g1oax4, g1obx4, b1oax4, b1obx4;
    __m128i y1oax4, y1obx4, uvoax4, uvobx4;
    __m128i uoax4, voax4, ravgx4, gavgx4, bavgx4, uvox8;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstuv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcuv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 3; xx++) {
            int x = xx << 3;

            y0x8 = _mm_lddqu_si128((__m128i*)(srcy + x));
            y1x8 = _mm_lddqu_si128((__m128i*)(srcy + (srclinesize[0] / 2 + x)));
            uvx8 = _mm_lddqu_si128((__m128i*)(srcuv + x));

            // shift to low10bits for 10bit input
            // shift bit has to be compile-time constant
            y0x8 = _mm_srli_epi16(y0x8, TEN_BIT_BIPLANAR_SHIFT);
            y1x8 = _mm_srli_epi16(y1x8, TEN_BIT_BIPLANAR_SHIFT);
            uvx8 = _mm_srli_epi16(uvx8, TEN_BIT_BIPLANAR_SHIFT);
            y0x4a = _mm_cvtepu16_epi32(y0x8);
            y0x4b = _mm_unpackhi_epi16(y0x8, zero128);
            y1x4a = _mm_cvtepu16_epi32(y1x8);
            y1x4b = _mm_unpackhi_epi16(y1x8, zero128);
            uvx4a = _mm_cvtepu16_epi32(uvx8);
            uvx4b = _mm_unpackhi_epi16(uvx8, zero128);
            y0x4a = _mm_sub_epi32(y0x4a, in_yuv_offx4);
            y1x4a = _mm_sub_epi32(y1x4a, in_yuv_offx4);
            y0x4b = _mm_sub_epi32(y0x4b, in_yuv_offx4);
            y1x4b = _mm_sub_epi32(y1x4b, in_yuv_offx4);
            uvx4a = _mm_sub_epi32(uvx4a, in_uv_offx4);
            uvx4b = _mm_sub_epi32(uvx4b, in_uv_offx4);

            ux4a = _mm_shuffle_epi32(uvx4a, _MM_SHUFFLE(2, 2, 0, 0));
            ux4b = _mm_shuffle_epi32(uvx4b, _MM_SHUFFLE(2, 2, 0, 0));
            vx4a = _mm_shuffle_epi32(uvx4a, _MM_SHUFFLE(3, 3, 1, 1));
            vx4b = _mm_shuffle_epi32(uvx4b, _MM_SHUFFLE(3, 3, 1, 1));

            // r = av_clip_int16((y * cy + crv * v + in_rnd) >> in_sh);
            r0x4a = g0x4a = b0x4a = _mm_mullo_epi32(y0x4a, cyx4);
            r0x4a = _mm_add_epi32(r0x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(crv)));
            r0x4a = _mm_add_epi32(r0x4a, rndx4);
            r0x4a = _mm_srai_epi32(r0x4a, in_sh);
            r0x4a = av_clip_int16_sse(r0x4a);

            r1x4a = g1x4a = b1x4a = _mm_mullo_epi32(y1x4a, cyx4);
            r1x4a = _mm_add_epi32(r1x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(crv)));
            r1x4a = _mm_add_epi32(r1x4a, rndx4);
            r1x4a = _mm_srai_epi32(r1x4a, in_sh);
            r1x4a = av_clip_int16_sse(r1x4a);

            // g = av_clip_int16((y * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g0x4a = _mm_add_epi32(g0x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cgu)));
            g0x4a = _mm_add_epi32(g0x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(cgv)));
            g0x4a = _mm_add_epi32(g0x4a, rndx4);
            g0x4a = _mm_srai_epi32(g0x4a, in_sh);
            g0x4a = av_clip_int16_sse(g0x4a);

            g1x4a = _mm_add_epi32(g1x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cgu)));
            g1x4a = _mm_add_epi32(g1x4a, _mm_mullo_epi32(vx4a, _mm_set1_epi32(cgv)));
            g1x4a = _mm_add_epi32(g1x4a, rndx4);
            g1x4a = _mm_srai_epi32(g1x4a, in_sh);
            g1x4a = av_clip_int16_sse(g1x4a);

            // b = av_clip_int16((y * cy + cbu * u + in_rnd) >> in_sh);
            b0x4a = _mm_add_epi32(b0x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cbu)));
            b0x4a = _mm_add_epi32(b0x4a, rndx4);
            b0x4a = _mm_srai_epi32(b0x4a, in_sh);
            b0x4a = av_clip_int16_sse(b0x4a);

            b1x4a = _mm_add_epi32(b1x4a, _mm_mullo_epi32(ux4a, _mm_set1_epi32(cbu)));
            b1x4a = _mm_add_epi32(b1x4a, rndx4);
            b1x4a = _mm_srai_epi32(b1x4a, in_sh);
            b1x4a = av_clip_int16_sse(b1x4a);

            r0x4b = g0x4b = b0x4b = _mm_mullo_epi32(y0x4b, cyx4);
            r0x4b = _mm_add_epi32(r0x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(crv)));
            r0x4b = _mm_add_epi32(r0x4b, rndx4);
            r0x4b = _mm_srai_epi32(r0x4b, in_sh);
            r0x4b = av_clip_int16_sse(r0x4b);

            r1x4b = g1x4b = b1x4b = _mm_mullo_epi32(y1x4b, cyx4);
            r1x4b = _mm_add_epi32(r1x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(crv)));
            r1x4b = _mm_add_epi32(r1x4b, rndx4);
            r1x4b = _mm_srai_epi32(r1x4b, in_sh);
            r1x4b = av_clip_int16_sse(r1x4b);

            g0x4b = _mm_add_epi32(g0x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cgu)));
            g0x4b = _mm_add_epi32(g0x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(cgv)));
            g0x4b = _mm_add_epi32(g0x4b, rndx4);
            g0x4b = _mm_srai_epi32(g0x4b, in_sh);
            g0x4b = av_clip_int16_sse(g0x4b);

            g1x4b = _mm_add_epi32(g1x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cgu)));
            g1x4b = _mm_add_epi32(g1x4b, _mm_mullo_epi32(vx4b, _mm_set1_epi32(cgv)));
            g1x4b = _mm_add_epi32(g1x4b, rndx4);
            g1x4b = _mm_srai_epi32(g1x4b, in_sh);
            g1x4b = av_clip_int16_sse(g1x4b);

            b0x4b = _mm_add_epi32(b0x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cbu)));
            b0x4b = _mm_add_epi32(b0x4b, rndx4);
            b0x4b = _mm_srai_epi32(b0x4b, in_sh);
            b0x4b = av_clip_int16_sse(b0x4b);

            b1x4b = _mm_add_epi32(b1x4b, _mm_mullo_epi32(ux4b, _mm_set1_epi32(cbu)));
            b1x4b = _mm_add_epi32(b1x4b, rndx4);
            b1x4b = _mm_srai_epi32(b1x4b, in_sh);
            b1x4b = av_clip_int16_sse(b1x4b);

            tonemap_int32x4_sse(r0x4a, g0x4a, b0x4a, r, g, b,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r1x4a, g1x4a, b1x4a, r1, g1, b1,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r0x4b, g0x4b, b0x4b, &r[4], &g[4], &b[4],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x4_sse(r1x4b, g1x4b, b1x4b, &r1[4], &g1[4], &b1[4],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);

            r0ox8 = _mm_lddqu_si128((const __m128i_u *)r);
            g0ox8 = _mm_lddqu_si128((const __m128i_u *)g);
            b0ox8 = _mm_lddqu_si128((const __m128i_u *)b);

            roax4 = _mm_cvtepi16_epi32(r0ox8);
            goax4 = _mm_cvtepi16_epi32(g0ox8);
            boax4 = _mm_cvtepi16_epi32(b0ox8);

            robx4 = _mm_unpackhi_epi16(r0ox8, zero128);
            gobx4 = _mm_unpackhi_epi16(g0ox8, zero128);
            bobx4 = _mm_unpackhi_epi16(b0ox8, zero128);

            yoax4 = _mm_mullo_epi32(roax4, _mm_set1_epi32(cry));
            yoax4 = _mm_add_epi32(yoax4, _mm_mullo_epi32(goax4, _mm_set1_epi32(cgy)));
            yoax4 = _mm_add_epi32(yoax4, _mm_mullo_epi32(boax4, _mm_set1_epi32(cby)));
            yoax4 = _mm_add_epi32(yoax4, _mm_set1_epi32(out_rnd));
            yoax4 = _mm_srai_epi32(yoax4, out_sh);
            yoax4 = _mm_add_epi32(yoax4, _mm_set1_epi32(params->out_yuv_off));
            yoax4 = _mm_slli_epi32(yoax4, out_sh2);

            yobx4 = _mm_mullo_epi32(robx4, _mm_set1_epi32(cry));
            yobx4 = _mm_add_epi32(yobx4, _mm_mullo_epi32(gobx4, _mm_set1_epi32(cgy)));
            yobx4 = _mm_add_epi32(yobx4, _mm_mullo_epi32(bobx4, _mm_set1_epi32(cby)));
            yobx4 = _mm_add_epi32(yobx4, _mm_set1_epi32(out_rnd));
            yobx4 = _mm_srai_epi32(yobx4, out_sh);
            yobx4 = _mm_add_epi32(yobx4, _mm_set1_epi32(params->out_yuv_off));
            yobx4 = _mm_slli_epi32(yobx4, out_sh2);

            y0ox8 = _mm_packus_epi32(yoax4, yobx4);
            _mm_storeu_si128((__m128i_u *) &dsty[x], y0ox8);

            r1ox8 = _mm_lddqu_si128((const __m128i_u *)r1);
            g1ox8 = _mm_lddqu_si128((const __m128i_u *)g1);
            b1ox8 = _mm_lddqu_si128((const __m128i_u *)b1);

            r1oax4 = _mm_cvtepi16_epi32(r1ox8);
            g1oax4 = _mm_cvtepi16_epi32(g1ox8);
            b1oax4 = _mm_cvtepi16_epi32(b1ox8);

            r1obx4 = _mm_unpackhi_epi16(r1ox8, zero128);
            g1obx4 = _mm_unpackhi_epi16(g1ox8, zero128);
            b1obx4 = _mm_unpackhi_epi16(b1ox8, zero128);

            y1oax4 = _mm_mullo_epi32(r1oax4, _mm_set1_epi32(cry));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_mullo_epi32(g1oax4, _mm_set1_epi32(cgy)));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_mullo_epi32(b1oax4, _mm_set1_epi32(cby)));
            y1oax4 = _mm_add_epi32(y1oax4, _mm_set1_epi32(out_rnd));
            y1oax4 = _mm_srai_epi32(y1oax4, out_sh);
            y1oax4 = _mm_add_epi32(y1oax4, _mm_set1_epi32(params->out_yuv_off));
            y1oax4 = _mm_slli_epi32(y1oax4, out_sh2);

            y1obx4 = _mm_mullo_epi32(r1obx4, _mm_set1_epi32(cry));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_mullo_epi32(g1obx4, _mm_set1_epi32(cgy)));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_mullo_epi32(b1obx4, _mm_set1_epi32(cby)));
            y1obx4 = _mm_add_epi32(y1obx4, _mm_set1_epi32(out_rnd));
            y1obx4 = _mm_srai_epi32(y1obx4, out_sh);
            y1obx4 = _mm_add_epi32(y1obx4, _mm_set1_epi32(params->out_yuv_off));
            y1obx4 = _mm_slli_epi32(y1obx4, out_sh2);

            y1ox8 = _mm_packus_epi32(y1oax4, y1obx4);
            _mm_storeu_si128((__m128i_u *) &dsty[x + dstlinesize[0] / 2], y1ox8);

            ravgx4 = _mm_hadd_epi32(roax4, robx4);
            ravgx4 = _mm_add_epi32(ravgx4, _mm_hadd_epi32(r1oax4, r1obx4));
            ravgx4 = _mm_add_epi32(ravgx4, _mm_set1_epi32(2));
            ravgx4 = _mm_srai_epi32(ravgx4, 2);

            gavgx4 = _mm_hadd_epi32(goax4, gobx4);
            gavgx4 = _mm_add_epi32(gavgx4, _mm_hadd_epi32(g1oax4, g1obx4));
            gavgx4 = _mm_add_epi32(gavgx4, _mm_set1_epi32(2));
            gavgx4 = _mm_srai_epi32(gavgx4, 2);

            bavgx4 = _mm_hadd_epi32(boax4, bobx4);
            bavgx4 = _mm_add_epi32(bavgx4, _mm_hadd_epi32(b1oax4, b1obx4));
            bavgx4 = _mm_add_epi32(bavgx4, _mm_set1_epi32(2));
            bavgx4 = _mm_srai_epi32(bavgx4, 2);

            uoax4 = _mm_add_epi32(_mm_set1_epi32(out_rnd), _mm_mullo_epi32(ravgx4, _mm_set1_epi32(cru)));
            uoax4 = _mm_add_epi32(uoax4, _mm_mullo_epi32(gavgx4, _mm_set1_epi32(ocgu)));
            uoax4 = _mm_add_epi32(uoax4, _mm_mullo_epi32(bavgx4, _mm_set1_epi32(cburv)));
            uoax4 = _mm_srai_epi32(uoax4, out_sh);
            uoax4 = _mm_add_epi32(uoax4, _mm_set1_epi32(out_uv_offset));

            voax4 = _mm_add_epi32(_mm_set1_epi32(out_rnd), _mm_mullo_epi32(ravgx4, _mm_set1_epi32(cburv)));
            voax4 = _mm_add_epi32(voax4, _mm_mullo_epi32(gavgx4, _mm_set1_epi32(ocgv)));
            voax4 = _mm_add_epi32(voax4, _mm_mullo_epi32(bavgx4, _mm_set1_epi32(cbv)));
            voax4 = _mm_srai_epi32(voax4, out_sh);
            voax4 = _mm_add_epi32(voax4, _mm_set1_epi32(out_uv_offset));

            uvoax4 = _mm_unpacklo_epi32(uoax4, voax4);
            uvobx4 = _mm_unpackhi_epi32(uoax4, voax4);
            uvoax4 = _mm_slli_epi32(uvoax4, out_sh2);
            uvobx4 = _mm_slli_epi32(uvobx4, out_sh2);
            uvox8 = _mm_packus_epi32(uvoax4, uvobx4);
            _mm_storeu_si128((__m128i_u *) &dstuv[x], uvox8);
        }
    }

    // Process remaining pixels cannot fill the full simd register with scalar version
    if (remainw) {
        int offset = width & (int)0xfffffff8;
        rdsty += offset;
        rdstuv += offset;
        rsrcy += offset;
        rsrcuv += offset;
        tonemap_frame_p010_2_p010(rdsty, rdstuv,
                                  rsrcy, rsrcuv,
                                  dstlinesize, srclinesize,
                                  dstdepth, srcdepth,
                                  remainw, rheight, params);
    }
#endif // ENABLE_TONEMAPX_SSE_INTRINSICS
}
