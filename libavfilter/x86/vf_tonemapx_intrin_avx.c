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

#include "vf_tonemapx_intrin_avx.h"

#ifdef ENABLE_TONEMAPX_AVX_INTRINSICS
#    include <immintrin.h>
#endif // ENABLE_TONEMAPX_AVX_INTRINSICS

#ifdef ENABLE_TONEMAPX_AVX_INTRINSICS
X86_64_V3 static inline __m256i av_clip_int16_avx(__m256i a)
{
    __m256i add_result = _mm256_add_epi32(a, _mm256_set1_epi32(0x8000U));
    __m256i mask = _mm256_set1_epi32(~0xFFFF);
    __m256i condition = _mm256_and_si256(add_result, mask);
    __m256i cmp = _mm256_cmpeq_epi32(condition, _mm256_setzero_si256());

    __m256i shifted = _mm256_srai_epi32(a, 31);
    __m256i xor_result = _mm256_xor_si256(shifted, _mm256_set1_epi32(0x7FFF));

    return _mm256_or_si256(_mm256_and_si256(cmp, a), _mm256_andnot_si256(cmp, xor_result));
}

X86_64_V3 inline static float reduce_floatx4(__m128 x) {
    x = _mm_hadd_ps(x, x);
    x = _mm_hadd_ps(x, x);
    return _mm_cvtss_f32(x);
}

X86_64_V3 inline static float reduce_floatx8(__m256 x) {
    __m256 x2 = _mm256_permute2f128_ps(x , x , 1);
    x = _mm256_add_ps(x, x2);
    x = _mm256_hadd_ps(x, x);
    x = _mm256_hadd_ps(x, x);
    return _mm256_cvtss_f32(x);
}

X86_64_V3 static inline float reshape_poly(float s, __m128 coeffs)
{
    __m128 ps = _mm_set_ps(0.0f, s * s, s, 1.0f);
    ps = _mm_mul_ps(ps, coeffs);
    return reduce_floatx4(ps);
}

X86_64_V3 inline static float reshape_mmr(__m128 sig, __m128 coeffs, const float* mmr,
                                          int mmr_single, int min_order, int max_order)
{
    float s = _mm_cvtss_f32(coeffs);
    int mmr_idx = 0;
    int order = 0;

    __m256 sigX, mmr_coeffs, ps;
    __m128 sigX01 = _mm_mul_ps(sig, _mm_shuffle_ps(sig, sig, _MM_SHUFFLE(1, 1, 1, 1))); // {sig[0]*sig[1], sig[1]*sig[1], sig[2]*sig[1], sig[3]*sig[1]}
    __m128 sigX02 = _mm_mul_ps(sig, _mm_shuffle_ps(sig, sig, _MM_SHUFFLE(2, 2, 2, 2))); // {sig[0]*sig[2], sig[1]*sig[2], sig[2]*sig[2], sig[3]*sig[2]}
    __m128 sigX12 = _mm_mul_ps(sigX01, _mm_shuffle_ps(sig, sig, _MM_SHUFFLE(2, 2, 2, 2))); // {sig[0]*sig[1]*sig[2], sig[1]*sig[1]*sig[2], sig[2]*sig[1]*sig[2], sig[3]*sig[1]*sig[2]}
    __m128 sigX0 = sigX01; // sig[0]*sig[1] now positioned at 0

    sigX0 = _mm_insert_ps(sigX0, sigX02, _MM_MK_INSERTPS_NDX(0, 1, 0)); // sig[0]*sig[2] at 1
    sigX0 = _mm_insert_ps(sigX0, sigX02, _MM_MK_INSERTPS_NDX(1, 2, 0)); // sig[1]*sig[2] at 2
    sigX0 = _mm_insert_ps(sigX0, sigX12, _MM_MK_INSERTPS_NDX(0, 3, 0)); // sig[0]*sig[1]*sig[2] at 3

    sigX = _mm256_set_m128(sigX0, sig);

    mmr_idx = mmr_single ? 0 : (int)_mm_cvtss_f32(_mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3, 2, 0, 1)));
    order = (int)_mm_cvtss_f32(_mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(1, 2, 0, 3)));

    // dot first order
    mmr_coeffs = _mm256_loadu_ps(&mmr[mmr_idx + 0*4]);
    ps = _mm256_mul_ps(sigX, mmr_coeffs);
    s += reduce_floatx8(ps);

    if (max_order >= 2 && (min_order >= 2 || order >= 2)) {
        __m256 sigX2 = _mm256_mul_ps(sigX, sigX);
        mmr_coeffs = _mm256_loadu_ps(&mmr[mmr_idx + 2*4]);
        ps = _mm256_mul_ps(sigX2, mmr_coeffs);
        s += reduce_floatx8(ps);

        if (max_order == 3 && (min_order == 3 || order >= 3)) {
            __m256 sigX3 = _mm256_mul_ps(sigX2, sigX);
            mmr_coeffs = _mm256_loadu_ps(&mmr[mmr_idx + 4*4]);
            ps = _mm256_mul_ps(sigX3, mmr_coeffs);
            s += reduce_floatx8(ps);
        }
    }

    return s;
}

#define CLAMP(a, b, c) (FFMIN(FFMAX((a), (b)), (c)))
X86_64_V3 inline static __m128 reshape_dovi_iptpqc2(__m128 sig, const TonemapIntParams *ctx)
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

X86_64_V3 inline static void ycc2rgbx8(__m256* dy, __m256* dcb, __m256* dcr,
                                       __m256 y, __m256 cb, __m256 cr,
                                       const double nonlinear[3][3], const float ycc_offset[3])
{
    *dy = _mm256_mul_ps(y, _mm256_set1_ps((float)nonlinear[0][0]));
    *dy = _mm256_fmadd_ps(cb, _mm256_set1_ps((float)nonlinear[0][1]), *dy);
    *dy = _mm256_fmadd_ps(cr, _mm256_set1_ps((float)nonlinear[0][2]), *dy);
    *dy = _mm256_sub_ps(*dy, _mm256_set1_ps(ycc_offset[0]));

    *dcb = _mm256_mul_ps(y, _mm256_set1_ps((float)nonlinear[1][0]));
    *dcb = _mm256_fmadd_ps(cb, _mm256_set1_ps((float)nonlinear[1][1]), *dcb);
    *dcb = _mm256_fmadd_ps(cr, _mm256_set1_ps((float)nonlinear[1][2]), *dcb);
    *dcb = _mm256_sub_ps(*dcb, _mm256_set1_ps(ycc_offset[1]));

    *dcr = _mm256_mul_ps(y, _mm256_set1_ps((float)nonlinear[2][0]));
    *dcr = _mm256_fmadd_ps(cb, _mm256_set1_ps((float)nonlinear[2][1]), *dcr);
    *dcr = _mm256_fmadd_ps(cr, _mm256_set1_ps((float)nonlinear[2][2]), *dcr);
    *dcr = _mm256_sub_ps(*dcr, _mm256_set1_ps(ycc_offset[2]));
}

X86_64_V3 inline static void lms2rgbx8(__m256* dl, __m256* dm, __m256* ds,
                                       __m256 l, __m256 m, __m256 s,
                                       const double lms2rgb_matrix[3][3])
{
    *dl = _mm256_mul_ps(l, _mm256_set1_ps((float)lms2rgb_matrix[0][0]));
    *dl = _mm256_fmadd_ps(m, _mm256_set1_ps((float)lms2rgb_matrix[0][1]), *dl);
    *dl = _mm256_fmadd_ps(s, _mm256_set1_ps((float)lms2rgb_matrix[0][2]), *dl);

    *dm = _mm256_mul_ps(l, _mm256_set1_ps((float)lms2rgb_matrix[1][0]));
    *dm = _mm256_fmadd_ps(m, _mm256_set1_ps((float)lms2rgb_matrix[1][1]), *dm);
    *dm = _mm256_fmadd_ps(s, _mm256_set1_ps((float)lms2rgb_matrix[1][2]), *dm);

    *ds = _mm256_mul_ps(l, _mm256_set1_ps((float)lms2rgb_matrix[2][0]));
    *ds = _mm256_fmadd_ps(m, _mm256_set1_ps((float)lms2rgb_matrix[2][1]), *ds);
    *ds = _mm256_fmadd_ps(s, _mm256_set1_ps((float)lms2rgb_matrix[2][2]), *ds);
}

X86_64_V3 inline static void reshapeiptx8(__m128* ipt0, __m128* ipt1, __m128* ipt2, __m128* ipt3,
                                          __m128* ipt4, __m128* ipt5, __m128* ipt6, __m128* ipt7,
                                          __m256 yx8, __m256 ux8, __m256 vx8,
                                          const struct TonemapIntParams *params)
{
    __m128 yx4a = _mm256_castps256_ps128(yx8);
    __m128 yx4b = _mm256_extractf128_ps(yx8, 1);
    __m128 ux4a = _mm256_castps256_ps128(ux8);
    __m128 ux4b = _mm256_extractf128_ps(ux8, 1);
    __m128 vx4a = _mm256_castps256_ps128(vx8);
    __m128 vx4b = _mm256_extractf128_ps(vx8, 1);

    __m128 ia1 = _mm_unpacklo_ps(yx4a, ux4a);
    __m128 ia2 = _mm_unpackhi_ps(yx4a, ux4a);
    __m128 ib1 = _mm_unpacklo_ps(vx4a, _mm_setzero_ps());
    __m128 ib2 = _mm_unpackhi_ps(vx4a, _mm_setzero_ps());

    *ipt0 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(1, 0, 1, 0));
    *ipt1 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(3, 2, 3, 2));
    *ipt2 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(1, 0, 1, 0));
    *ipt3 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(3, 2, 3, 2));

    *ipt0 = reshape_dovi_iptpqc2(*ipt0, params);
    *ipt1 = reshape_dovi_iptpqc2(*ipt1, params);
    *ipt2 = reshape_dovi_iptpqc2(*ipt2, params);
    *ipt3 = reshape_dovi_iptpqc2(*ipt3, params);

    ia1 = _mm_unpacklo_ps(yx4b, ux4b);
    ia2 = _mm_unpackhi_ps(yx4b, ux4b);
    ib1 = _mm_unpacklo_ps(vx4b, _mm_setzero_ps());
    ib2 = _mm_unpackhi_ps(vx4b, _mm_setzero_ps());

    *ipt4 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(1, 0, 1, 0));
    *ipt5 = _mm_shuffle_ps(ia1, ib1, _MM_SHUFFLE(3, 2, 3, 2));
    *ipt6 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(1, 0, 1, 0));
    *ipt7 = _mm_shuffle_ps(ia2, ib2, _MM_SHUFFLE(3, 2, 3, 2));

    *ipt4 = reshape_dovi_iptpqc2(*ipt4, params);
    *ipt5 = reshape_dovi_iptpqc2(*ipt5, params);
    *ipt6 = reshape_dovi_iptpqc2(*ipt6, params);
    *ipt7 = reshape_dovi_iptpqc2(*ipt7, params);
}

X86_64_V3 inline static void transpose_ipt8x4(__m128 ipt0, __m128 ipt1, __m128 ipt2, __m128 ipt3,
                                              __m128 ipt4, __m128 ipt5, __m128 ipt6, __m128 ipt7,
                                              __m256* ix8, __m256* px8, __m256* tx8)
{
    __m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    tmp0 = _mm256_castps128_ps256(ipt0);
    tmp0 = _mm256_insertf128_ps(tmp0, ipt4, 1);

    tmp1 = _mm256_castps128_ps256(ipt1);
    tmp1 = _mm256_insertf128_ps(tmp1, ipt5, 1);

    tmp2 = _mm256_castps128_ps256(ipt2);
    tmp2 = _mm256_insertf128_ps(tmp2, ipt6, 1);

    tmp3 = _mm256_castps128_ps256(ipt3);
    tmp3 = _mm256_insertf128_ps(tmp3, ipt7, 1);

    tmp4 = _mm256_unpacklo_ps(tmp0, tmp1);
    tmp5 = _mm256_unpackhi_ps(tmp0, tmp1);
    tmp6 = _mm256_unpacklo_ps(tmp2, tmp3);
    tmp7 = _mm256_unpackhi_ps(tmp2, tmp3);

    *ix8 = _mm256_shuffle_ps(tmp4, tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    *px8 = _mm256_shuffle_ps(tmp4, tmp6, _MM_SHUFFLE(3, 2, 3, 2));
    *tx8 = _mm256_shuffle_ps(tmp5, tmp7, _MM_SHUFFLE(1, 0, 1, 0));
}

X86_64_V3 static inline void tonemap_int32x8_avx(__m256i r_in, __m256i g_in, __m256i b_in,
                                                 int16_t *r_out, int16_t *g_out, int16_t *b_out,
                                                 float *lin_lut, float *tonemap_lut, uint16_t *delin_lut,
                                                 const AVLumaCoefficients *coeffs,
                                                 const AVLumaCoefficients *ocoeffs, double desat,
                                                 double (*rgb2rgb)[3][3],
                                                 int rgb2rgb_passthrough)
{
    __m256i sig8;
    __m256 mapvalx8, r_linx8, g_linx8, b_linx8;
    __m256 offset = _mm256_set1_ps(0.5f);
    __m256i zerox8 = _mm256_setzero_si256();
    __m256i upper_bound = _mm256_set1_epi32(32767);
    __m256 intermediate_upper_bound = _mm256_set1_ps(32767.0f);
    __m256i r, g, b, rx8, gx8, bx8;

    float mapval8[8], r_lin8[8], g_lin8[8], b_lin8[8];

    r = _mm256_max_epi32(r_in, zerox8);
    g = _mm256_max_epi32(g_in, zerox8);
    b = _mm256_max_epi32(b_in, zerox8);

    sig8 = _mm256_max_epi32(r, _mm256_max_epi32(g, b));

#define LOAD_LUT(i) mapval8[i] = tonemap_lut[_mm256_extract_epi32(sig8, i)]; \
r_lin8[i] = lin_lut[_mm256_extract_epi32(r, i)];                             \
g_lin8[i] = lin_lut[_mm256_extract_epi32(g, i)];                             \
b_lin8[i] = lin_lut[_mm256_extract_epi32(b, i)];

    LOAD_LUT(0)
    LOAD_LUT(1)
    LOAD_LUT(2)
    LOAD_LUT(3)
    LOAD_LUT(4)
    LOAD_LUT(5)
    LOAD_LUT(6)
    LOAD_LUT(7)

#undef LOAD_LUT

    mapvalx8 = _mm256_loadu_ps(mapval8);
    r_linx8 = _mm256_loadu_ps(r_lin8);
    g_linx8 = _mm256_loadu_ps(g_lin8);
    b_linx8 = _mm256_loadu_ps(b_lin8);

    if (!rgb2rgb_passthrough) {
        __m256 r_tmpx8, g_tmpx8, b_tmpx8;

        r_tmpx8 = _mm256_mul_ps(r_linx8, _mm256_set1_ps((float)(*rgb2rgb)[0][0]));
        r_tmpx8 = _mm256_fmadd_ps(g_linx8, _mm256_set1_ps((float)(*rgb2rgb)[0][1]), r_tmpx8);
        r_tmpx8 = _mm256_fmadd_ps(b_linx8, _mm256_set1_ps((float)(*rgb2rgb)[0][2]), r_tmpx8);

        g_tmpx8 = _mm256_mul_ps(g_linx8, _mm256_set1_ps((float)(*rgb2rgb)[1][1]));
        g_tmpx8 = _mm256_fmadd_ps(r_linx8, _mm256_set1_ps((float)(*rgb2rgb)[1][0]), g_tmpx8);
        g_tmpx8 = _mm256_fmadd_ps(b_linx8, _mm256_set1_ps((float)(*rgb2rgb)[1][2]), g_tmpx8);

        b_tmpx8 = _mm256_mul_ps(b_linx8, _mm256_set1_ps((float)(*rgb2rgb)[2][2]));
        b_tmpx8 = _mm256_fmadd_ps(r_linx8, _mm256_set1_ps((float)(*rgb2rgb)[2][0]), b_tmpx8);
        b_tmpx8 = _mm256_fmadd_ps(g_linx8, _mm256_set1_ps((float)(*rgb2rgb)[2][1]), b_tmpx8);

        r_linx8 = r_tmpx8;
        g_linx8 = g_tmpx8;
        b_linx8 = b_tmpx8;
    }

    if (desat > 0) {
        __m256 eps_x8 = _mm256_set1_ps(FLOAT_EPS);
        __m256 desat8 = _mm256_set1_ps((float)desat);
        __m256 luma8 = _mm256_set1_ps(0);
        __m256 overbright8;

        luma8 = _mm256_fmadd_ps(r_linx8, _mm256_set1_ps((float)av_q2d(coeffs->cr)), luma8);
        luma8 = _mm256_fmadd_ps(g_linx8, _mm256_set1_ps((float)av_q2d(coeffs->cg)), luma8);
        luma8 = _mm256_fmadd_ps(b_linx8, _mm256_set1_ps((float)av_q2d(coeffs->cb)), luma8);
        overbright8 = _mm256_div_ps(_mm256_max_ps(_mm256_sub_ps(luma8, desat8), eps_x8), _mm256_max_ps(luma8, eps_x8));
        r_linx8 = _mm256_fnmadd_ps(r_linx8, overbright8, r_linx8);
        r_linx8 = _mm256_fmadd_ps(luma8, overbright8, r_linx8);
        g_linx8 = _mm256_fnmadd_ps(g_linx8, overbright8, g_linx8);
        g_linx8 = _mm256_fmadd_ps(luma8, overbright8, g_linx8);
        b_linx8 = _mm256_fnmadd_ps(b_linx8, overbright8, b_linx8);
        b_linx8 = _mm256_fmadd_ps(luma8, overbright8, b_linx8);
    }

    r_linx8 = _mm256_mul_ps(r_linx8, mapvalx8);
    g_linx8 = _mm256_mul_ps(g_linx8, mapvalx8);
    b_linx8 = _mm256_mul_ps(b_linx8, mapvalx8);

    r_linx8 = _mm256_fmadd_ps(r_linx8, intermediate_upper_bound, offset);
    g_linx8 = _mm256_fmadd_ps(g_linx8, intermediate_upper_bound, offset);
    b_linx8 = _mm256_fmadd_ps(b_linx8, intermediate_upper_bound, offset);

    rx8 = _mm256_cvttps_epi32(r_linx8);
    rx8 = _mm256_min_epi32(rx8, upper_bound);
    rx8 = _mm256_max_epi32(rx8, zerox8);

    gx8 = _mm256_cvttps_epi32(g_linx8);
    gx8 = _mm256_min_epi32(gx8, upper_bound);
    gx8 = _mm256_max_epi32(gx8, zerox8);

    bx8 = _mm256_cvttps_epi32(b_linx8);
    bx8 = _mm256_min_epi32(bx8, upper_bound);
    bx8 = _mm256_max_epi32(bx8, zerox8);

#define SAVE_COLOR(i) r_out[i] = delin_lut[_mm256_extract_epi32(rx8, i)]; \
g_out[i] = delin_lut[_mm256_extract_epi32(gx8, i)];                       \
b_out[i] = delin_lut[_mm256_extract_epi32(bx8, i)];

    SAVE_COLOR(0)
    SAVE_COLOR(1)
    SAVE_COLOR(2)
    SAVE_COLOR(3)
    SAVE_COLOR(4)
    SAVE_COLOR(5)
    SAVE_COLOR(6)
    SAVE_COLOR(7)

#undef SAVE_COLOR
}
#endif // ENABLE_TONEMAPX_AVX_INTRINSICS

X86_64_V3 void tonemap_frame_dovi_2_420p_avx(uint8_t *dsty, uint8_t *dstu, uint8_t *dstv,
                                             const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                             const int *dstlinesize, const int *srclinesize,
                                             int dstdepth, int srcdepth,
                                             int width, int height,
                                             const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_AVX_INTRINSICS
    uint8_t *rdsty = dsty;
    uint8_t *rdstu = dstu;
    uint8_t *rdstv = dstv;
    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcu = srcu;
    const uint16_t *rsrcv = srcv;
    int rheight = height;
    // not zero when not divisible by 16
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 14;

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

    int16_t r[16], g[16], b[16];
    int16_t r1[16], g1[16], b1[16];

    __m256i ux8, vx8;
    __m256i y0x16, y1x16;
    __m256i y0x8a, y0x8b, y1x8a, y1x8b, ux8a, ux8b, vx8a, vx8b;
    __m256i r0x8a, g0x8a, b0x8a, r0x8b, g0x8b, b0x8b;
    __m256i r1x8a, g1x8a, b1x8a, r1x8b, g1x8b, b1x8b;

    __m256i r0ox16, g0ox16, b0ox16;
    __m256i y0ox16;
    __m256i roax8, robx8, goax8, gobx8, boax8, bobx8;
    __m256i yoax8, yobx8;

    __m256i r1ox16, g1ox16, b1ox16;
    __m256i y1ox16;
    __m256i r1oax8, r1obx8, g1oax8, g1obx8, b1oax8, b1obx8;
    __m256i y1oax8, y1obx8;
    __m256i uox8, vox8, ravgx8, gavgx8, bavgx8;

    __m128 ipt0, ipt1, ipt2, ipt3, ipt4, ipt5, ipt6, ipt7;
    __m256 ix8, px8, tx8;
    __m256 lx8, mx8, sx8;
    __m256 rx8a, gx8a, bx8a, rx8b, gx8b, bx8b;
    __m256 y0x8af, y0x8bf, y1x8af, y1x8bf, ux8af, ux8bf, vx8af, vx8bf;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0] * 2, dstu += dstlinesize[1], dstv += dstlinesize[2],
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[2] / 2) {
        for (int xx = 0; xx < width >> 4; xx++) {
            int x = xx << 4;

            y0x16 = _mm256_lddqu_si256((__m256i*)(srcy + x));
            y1x16 = _mm256_lddqu_si256((__m256i*)(srcy + (srclinesize[0] / 2 + x)));
            ux8 = _mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i_u *)(srcu + (x >> 1))));
            vx8 = _mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i_u *)(srcv + (x >> 1))));

            y0x8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y0x16));
            y0x8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y0x16, 1));
            y1x8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y1x16));
            y1x8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y1x16, 1));

            ux8a = _mm256_permutevar8x32_epi32(ux8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
            ux8b = _mm256_permutevar8x32_epi32(ux8, _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4));
            vx8a = _mm256_permutevar8x32_epi32(vx8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
            vx8b = _mm256_permutevar8x32_epi32(vx8, _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4));

            y0x8af = _mm256_cvtepi32_ps(y0x8a);
            y0x8bf = _mm256_cvtepi32_ps(y0x8b);
            y1x8af = _mm256_cvtepi32_ps(y1x8a);
            y1x8bf = _mm256_cvtepi32_ps(y1x8b);
            ux8af = _mm256_cvtepi32_ps(ux8a);
            ux8bf = _mm256_cvtepi32_ps(ux8b);
            vx8af = _mm256_cvtepi32_ps(vx8a);
            vx8bf = _mm256_cvtepi32_ps(vx8b);

            y0x8af = _mm256_div_ps(y0x8af, _mm256_set1_ps(in_rng));
            y0x8bf = _mm256_div_ps(y0x8bf, _mm256_set1_ps(in_rng));
            y1x8af = _mm256_div_ps(y1x8af, _mm256_set1_ps(in_rng));
            y1x8bf = _mm256_div_ps(y1x8bf, _mm256_set1_ps(in_rng));
            ux8af = _mm256_div_ps(ux8af, _mm256_set1_ps(in_rng));
            ux8bf = _mm256_div_ps(ux8bf, _mm256_set1_ps(in_rng));
            vx8af = _mm256_div_ps(vx8af, _mm256_set1_ps(in_rng));
            vx8bf = _mm256_div_ps(vx8bf, _mm256_set1_ps(in_rng));

            // Reshape y0x8a
            reshapeiptx8(&ipt0, &ipt1, &ipt2, &ipt3,
                         &ipt4, &ipt5, &ipt6, &ipt7,
                         y0x8af, ux8af, vx8af, params);

            transpose_ipt8x4(ipt0, ipt1, ipt2, ipt3,
                             ipt4, ipt5, ipt6, ipt7,
                             &ix8, &px8, &tx8);

            ycc2rgbx8(&lx8, &mx8, &sx8, ix8, px8, tx8, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx8(&rx8a, &gx8a, &bx8a, lx8, mx8, sx8, *params->lms2rgb_matrix);

            rx8a = _mm256_mul_ps(rx8a, _mm256_set1_ps(JPEG_SCALE));
            gx8a = _mm256_mul_ps(gx8a, _mm256_set1_ps(JPEG_SCALE));
            bx8a = _mm256_mul_ps(bx8a, _mm256_set1_ps(JPEG_SCALE));

            r0x8a = _mm256_cvtps_epi32(rx8a);
            r0x8a = av_clip_int16_avx(r0x8a);
            g0x8a = _mm256_cvtps_epi32(gx8a);
            g0x8a = av_clip_int16_avx(g0x8a);
            b0x8a = _mm256_cvtps_epi32(bx8a);
            b0x8a = av_clip_int16_avx(b0x8a);

            // Reshape y1x8a
            reshapeiptx8(&ipt0, &ipt1, &ipt2, &ipt3,
                         &ipt4, &ipt5, &ipt6, &ipt7,
                         y1x8af, ux8af, vx8af, params);

            transpose_ipt8x4(ipt0, ipt1, ipt2, ipt3,
                             ipt4, ipt5, ipt6, ipt7,
                             &ix8, &px8, &tx8);

            ycc2rgbx8(&lx8, &mx8, &sx8, ix8, px8, tx8, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx8(&rx8a, &gx8a, &bx8a, lx8, mx8, sx8, *params->lms2rgb_matrix);

            rx8a = _mm256_mul_ps(rx8a, _mm256_set1_ps(JPEG_SCALE));
            gx8a = _mm256_mul_ps(gx8a, _mm256_set1_ps(JPEG_SCALE));
            bx8a = _mm256_mul_ps(bx8a, _mm256_set1_ps(JPEG_SCALE));

            r1x8a = _mm256_cvtps_epi32(rx8a);
            r1x8a = av_clip_int16_avx(r1x8a);
            g1x8a = _mm256_cvtps_epi32(gx8a);
            g1x8a = av_clip_int16_avx(g1x8a);
            b1x8a = _mm256_cvtps_epi32(bx8a);
            b1x8a = av_clip_int16_avx(b1x8a);

            // Reshape y0x8b
            reshapeiptx8(&ipt0, &ipt1, &ipt2, &ipt3,
                         &ipt4, &ipt5, &ipt6, &ipt7,
                         y0x8bf, ux8bf, vx8bf, params);

            transpose_ipt8x4(ipt0, ipt1, ipt2, ipt3,
                             ipt4, ipt5, ipt6, ipt7,
                             &ix8, &px8, &tx8);

            ycc2rgbx8(&lx8, &mx8, &sx8, ix8, px8, tx8, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx8(&rx8b, &gx8b, &bx8b, lx8, mx8, sx8, *params->lms2rgb_matrix);

            rx8b = _mm256_mul_ps(rx8b, _mm256_set1_ps(JPEG_SCALE));
            gx8b = _mm256_mul_ps(gx8b, _mm256_set1_ps(JPEG_SCALE));
            bx8b = _mm256_mul_ps(bx8b, _mm256_set1_ps(JPEG_SCALE));

            r0x8b = _mm256_cvtps_epi32(rx8b);
            r0x8b = av_clip_int16_avx(r0x8b);
            g0x8b = _mm256_cvtps_epi32(gx8b);
            g0x8b = av_clip_int16_avx(g0x8b);
            b0x8b = _mm256_cvtps_epi32(bx8b);
            b0x8b = av_clip_int16_avx(b0x8b);

            // Reshape y1x8b
            reshapeiptx8(&ipt0, &ipt1, &ipt2, &ipt3,
                         &ipt4, &ipt5, &ipt6, &ipt7,
                         y1x8bf, ux8bf, vx8bf, params);

            transpose_ipt8x4(ipt0, ipt1, ipt2, ipt3,
                             ipt4, ipt5, ipt6, ipt7,
                             &ix8, &px8, &tx8);

            ycc2rgbx8(&lx8, &mx8, &sx8, ix8, px8, tx8, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx8(&rx8b, &gx8b, &bx8b, lx8, mx8, sx8, *params->lms2rgb_matrix);

            rx8b = _mm256_mul_ps(rx8b, _mm256_set1_ps(JPEG_SCALE));
            gx8b = _mm256_mul_ps(gx8b, _mm256_set1_ps(JPEG_SCALE));
            bx8b = _mm256_mul_ps(bx8b, _mm256_set1_ps(JPEG_SCALE));

            r1x8b = _mm256_cvtps_epi32(rx8b);
            r1x8b = av_clip_int16_avx(r1x8b);
            g1x8b = _mm256_cvtps_epi32(gx8b);
            g1x8b = av_clip_int16_avx(g1x8b);
            b1x8b = _mm256_cvtps_epi32(bx8b);
            b1x8b = av_clip_int16_avx(b1x8b);

            tonemap_int32x8_avx(r0x8a, g0x8a, b0x8a, r, g, b,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r1x8a, g1x8a, b1x8a, r1, g1, b1,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r0x8b, g0x8b, b0x8b, &r[8], &g[8], &b[8],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r1x8b, g1x8b, b1x8b, &r1[8], &g1[8], &b1[8],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);

            r0ox16 = _mm256_lddqu_si256((const __m256i_u *)r);
            g0ox16 = _mm256_lddqu_si256((const __m256i_u *)g);
            b0ox16 = _mm256_lddqu_si256((const __m256i_u *)b);

            roax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r0ox16));
            goax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(g0ox16));
            boax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0ox16));

            robx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r0ox16, 1));
            gobx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(g0ox16, 1));
            bobx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0ox16, 1));

            yoax8 = _mm256_mullo_epi32(roax8, _mm256_set1_epi32(cry));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_mullo_epi32(goax8, _mm256_set1_epi32(cgy)));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_mullo_epi32(boax8, _mm256_set1_epi32(cby)));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_set1_epi32(out_rnd));
            yoax8 = _mm256_srai_epi32(yoax8, out_sh);
            yoax8 = _mm256_add_epi32(yoax8, _mm256_set1_epi32(params->out_yuv_off));

            yobx8 = _mm256_mullo_epi32(robx8, _mm256_set1_epi32(cry));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_mullo_epi32(gobx8, _mm256_set1_epi32(cgy)));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_mullo_epi32(bobx8, _mm256_set1_epi32(cby)));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_set1_epi32(out_rnd));
            yobx8 = _mm256_srai_epi32(yobx8, out_sh);
            yobx8 = _mm256_add_epi32(yobx8, _mm256_set1_epi32(params->out_yuv_off));

            y0ox16 = _mm256_packs_epi32(yoax8, yobx8);
            y0ox16 = _mm256_permute4x64_epi64(y0ox16, _MM_SHUFFLE(3, 1, 2, 0));
            _mm_storeu_si128((__m128i_u *) &dsty[x], _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(y0ox16, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0))));

            r1ox16 = _mm256_lddqu_si256((const __m256i_u *)r1);
            g1ox16 = _mm256_lddqu_si256((const __m256i_u *)g1);
            b1ox16 = _mm256_lddqu_si256((const __m256i_u *)b1);

            r1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r1ox16));
            g1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(g1ox16));
            b1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1ox16));

            r1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r1ox16, 1));
            g1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(g1ox16, 1));
            b1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1ox16, 1));

            y1oax8 = _mm256_mullo_epi32(r1oax8, _mm256_set1_epi32(cry));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_mullo_epi32(g1oax8, _mm256_set1_epi32(cgy)));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_mullo_epi32(b1oax8, _mm256_set1_epi32(cby)));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_set1_epi32(out_rnd));
            y1oax8 = _mm256_srai_epi32(y1oax8, out_sh);
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_set1_epi32(params->out_yuv_off));

            y1obx8 = _mm256_mullo_epi32(r1obx8, _mm256_set1_epi32(cry));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_mullo_epi32(g1obx8, _mm256_set1_epi32(cgy)));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_mullo_epi32(b1obx8, _mm256_set1_epi32(cby)));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_set1_epi32(out_rnd));
            y1obx8 = _mm256_srai_epi32(y1obx8, out_sh);
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_set1_epi32(params->out_yuv_off));

            y1ox16 = _mm256_packs_epi32(y1oax8, y1obx8);
            y1ox16 = _mm256_permute4x64_epi64(y1ox16, _MM_SHUFFLE(3, 1, 2, 0));
            _mm_storeu_si128((__m128i_u *) &dsty[x + dstlinesize[0]], _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(y1ox16, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0))));

            ravgx8 = _mm256_hadd_epi32(roax8, robx8);
            ravgx8 = _mm256_add_epi32(ravgx8, _mm256_hadd_epi32(r1oax8, r1obx8));
            ravgx8 = _mm256_permute4x64_epi64(ravgx8, _MM_SHUFFLE(3, 1, 2, 0));
            ravgx8 = _mm256_add_epi32(ravgx8, _mm256_set1_epi32(2));
            ravgx8 = _mm256_srai_epi32(ravgx8, 2);

            gavgx8 = _mm256_hadd_epi32(goax8, gobx8);
            gavgx8 = _mm256_add_epi32(gavgx8, _mm256_hadd_epi32(g1oax8, g1obx8));
            gavgx8 = _mm256_permute4x64_epi64(gavgx8, _MM_SHUFFLE(3, 1, 2, 0));
            gavgx8 = _mm256_add_epi32(gavgx8, _mm256_set1_epi32(2));
            gavgx8 = _mm256_srai_epi32(gavgx8, 2);

            bavgx8 = _mm256_hadd_epi32(boax8, bobx8);
            bavgx8 = _mm256_add_epi32(bavgx8, _mm256_hadd_epi32(b1oax8, b1obx8));
            bavgx8 = _mm256_permute4x64_epi64(bavgx8, _MM_SHUFFLE(3, 1, 2, 0));
            bavgx8 = _mm256_add_epi32(bavgx8, _mm256_set1_epi32(2));
            bavgx8 = _mm256_srai_epi32(bavgx8, 2);

            uox8 = _mm256_add_epi32(_mm256_set1_epi32(out_rnd), _mm256_mullo_epi32(ravgx8, _mm256_set1_epi32(cru)));
            uox8 = _mm256_add_epi32(uox8, _mm256_mullo_epi32(gavgx8, _mm256_set1_epi32(ocgu)));
            uox8 = _mm256_add_epi32(uox8, _mm256_mullo_epi32(bavgx8, _mm256_set1_epi32(cburv)));
            uox8 = _mm256_srai_epi32(uox8, out_sh);
            uox8 = _mm256_add_epi32(uox8, _mm256_set1_epi32(out_uv_offset));
            uox8 = _mm256_packs_epi32(uox8, _mm256_setzero_si256());
            uox8 = _mm256_permute4x64_epi64(uox8, _MM_SHUFFLE(3, 1, 2, 0));
            uox8 = _mm256_packus_epi16(uox8, _mm256_setzero_si256());
            _mm_storeu_si64(&dstu[x >> 1], _mm256_castsi256_si128(uox8));

            vox8 = _mm256_add_epi32(_mm256_set1_epi32(out_rnd), _mm256_mullo_epi32(ravgx8, _mm256_set1_epi32(cburv)));
            vox8 = _mm256_add_epi32(vox8, _mm256_mullo_epi32(gavgx8, _mm256_set1_epi32(ocgv)));
            vox8 = _mm256_add_epi32(vox8, _mm256_mullo_epi32(bavgx8, _mm256_set1_epi32(cbv)));
            vox8 = _mm256_srai_epi32(vox8, out_sh);
            vox8 = _mm256_add_epi32(vox8, _mm256_set1_epi32(out_uv_offset));
            vox8 = _mm256_packs_epi32(vox8, _mm256_setzero_si256());
            vox8 = _mm256_permute4x64_epi64(vox8, _MM_SHUFFLE(3, 1, 2, 0));
            vox8 = _mm256_packus_epi16(vox8, _mm256_setzero_si256());
            _mm_storeu_si64(&dstv[x >> 1], _mm256_castsi256_si128(vox8));
        }
    }

    // Process remaining pixels cannot fill the full simd register with scalar version
    if (remainw) {
        int offset = width & (int)0xfffffff0;
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
#endif // ENABLE_TONEMAPX_AVX_INTRINSICS
}

X86_64_V3 void tonemap_frame_dovi_2_420p10_avx(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                               const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                               const int *dstlinesize, const int *srclinesize,
                                               int dstdepth, int srcdepth,
                                               int width, int height,
                                               const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_AVX_INTRINSICS
    uint16_t *rdsty = dsty;
    uint16_t *rdstu = dstu;
    uint16_t *rdstv = dstv;
    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcu = srcu;
    const uint16_t *rsrcv = srcv;
    int rheight = height;
    // not zero when not divisible by 8
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 14;

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

    int16_t r[16], g[16], b[16];
    int16_t r1[16], g1[16], b1[16];

    __m256i ux8, vx8;
    __m256i y0x16, y1x16;
    __m256i y0x8a, y0x8b, y1x8a, y1x8b, ux8a, ux8b, vx8a, vx8b;
    __m256i r0x8a, g0x8a, b0x8a, r0x8b, g0x8b, b0x8b;
    __m256i r1x8a, g1x8a, b1x8a, r1x8b, g1x8b, b1x8b;

    __m256i r0ox16, g0ox16, b0ox16;
    __m256i y0ox16;
    __m256i roax8, robx8, goax8, gobx8, boax8, bobx8;
    __m256i yoax8, yobx8;

    __m256i r1ox16, g1ox16, b1ox16;
    __m256i y1ox16;
    __m256i r1oax8, r1obx8, g1oax8, g1obx8, b1oax8, b1obx8;
    __m256i y1oax8, y1obx8;
    __m256i uox8, vox8, ravgx8, gavgx8, bavgx8;

    __m128 ipt0, ipt1, ipt2, ipt3, ipt4, ipt5, ipt6, ipt7;
    __m256 ix8, px8, tx8;
    __m256 lx8, mx8, sx8;
    __m256 rx8a, gx8a, bx8a, rx8b, gx8b, bx8b;
    __m256 y0x8af, y0x8bf, y1x8af, y1x8bf, ux8af, ux8bf, vx8af, vx8bf;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstu += dstlinesize[1] / 2, dstv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 4; xx++) {
            int x = xx << 4;

            y0x16 = _mm256_lddqu_si256((__m256i*)(srcy + x));
            y1x16 = _mm256_lddqu_si256((__m256i*)(srcy + (srclinesize[0] / 2 + x)));
            ux8 = _mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i_u *)(srcu + (x >> 1))));
            vx8 = _mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i_u *)(srcv + (x >> 1))));

            y0x8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y0x16));
            y0x8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y0x16, 1));
            y1x8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y1x16));
            y1x8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y1x16, 1));

            ux8a = _mm256_permutevar8x32_epi32(ux8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
            ux8b = _mm256_permutevar8x32_epi32(ux8, _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4));
            vx8a = _mm256_permutevar8x32_epi32(vx8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
            vx8b = _mm256_permutevar8x32_epi32(vx8, _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4));

            y0x8af = _mm256_cvtepi32_ps(y0x8a);
            y0x8bf = _mm256_cvtepi32_ps(y0x8b);
            y1x8af = _mm256_cvtepi32_ps(y1x8a);
            y1x8bf = _mm256_cvtepi32_ps(y1x8b);
            ux8af = _mm256_cvtepi32_ps(ux8a);
            ux8bf = _mm256_cvtepi32_ps(ux8b);
            vx8af = _mm256_cvtepi32_ps(vx8a);
            vx8bf = _mm256_cvtepi32_ps(vx8b);

            y0x8af = _mm256_div_ps(y0x8af, _mm256_set1_ps(in_rng));
            y0x8bf = _mm256_div_ps(y0x8bf, _mm256_set1_ps(in_rng));
            y1x8af = _mm256_div_ps(y1x8af, _mm256_set1_ps(in_rng));
            y1x8bf = _mm256_div_ps(y1x8bf, _mm256_set1_ps(in_rng));
            ux8af = _mm256_div_ps(ux8af, _mm256_set1_ps(in_rng));
            ux8bf = _mm256_div_ps(ux8bf, _mm256_set1_ps(in_rng));
            vx8af = _mm256_div_ps(vx8af, _mm256_set1_ps(in_rng));
            vx8bf = _mm256_div_ps(vx8bf, _mm256_set1_ps(in_rng));

            // Reshape y0x8a
            reshapeiptx8(&ipt0, &ipt1, &ipt2, &ipt3,
                         &ipt4, &ipt5, &ipt6, &ipt7,
                         y0x8af, ux8af, vx8af, params);

            transpose_ipt8x4(ipt0, ipt1, ipt2, ipt3,
                             ipt4, ipt5, ipt6, ipt7,
                             &ix8, &px8, &tx8);

            ycc2rgbx8(&lx8, &mx8, &sx8, ix8, px8, tx8, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx8(&rx8a, &gx8a, &bx8a, lx8, mx8, sx8, *params->lms2rgb_matrix);

            rx8a = _mm256_mul_ps(rx8a, _mm256_set1_ps(JPEG_SCALE));
            gx8a = _mm256_mul_ps(gx8a, _mm256_set1_ps(JPEG_SCALE));
            bx8a = _mm256_mul_ps(bx8a, _mm256_set1_ps(JPEG_SCALE));

            r0x8a = _mm256_cvtps_epi32(rx8a);
            r0x8a = av_clip_int16_avx(r0x8a);
            g0x8a = _mm256_cvtps_epi32(gx8a);
            g0x8a = av_clip_int16_avx(g0x8a);
            b0x8a = _mm256_cvtps_epi32(bx8a);
            b0x8a = av_clip_int16_avx(b0x8a);

            // Reshape y1x8a
            reshapeiptx8(&ipt0, &ipt1, &ipt2, &ipt3,
                         &ipt4, &ipt5, &ipt6, &ipt7,
                         y1x8af, ux8af, vx8af, params);

            transpose_ipt8x4(ipt0, ipt1, ipt2, ipt3,
                             ipt4, ipt5, ipt6, ipt7,
                             &ix8, &px8, &tx8);

            ycc2rgbx8(&lx8, &mx8, &sx8, ix8, px8, tx8, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx8(&rx8a, &gx8a, &bx8a, lx8, mx8, sx8, *params->lms2rgb_matrix);

            rx8a = _mm256_mul_ps(rx8a, _mm256_set1_ps(JPEG_SCALE));
            gx8a = _mm256_mul_ps(gx8a, _mm256_set1_ps(JPEG_SCALE));
            bx8a = _mm256_mul_ps(bx8a, _mm256_set1_ps(JPEG_SCALE));

            r1x8a = _mm256_cvtps_epi32(rx8a);
            r1x8a = av_clip_int16_avx(r1x8a);
            g1x8a = _mm256_cvtps_epi32(gx8a);
            g1x8a = av_clip_int16_avx(g1x8a);
            b1x8a = _mm256_cvtps_epi32(bx8a);
            b1x8a = av_clip_int16_avx(b1x8a);

            // Reshape y0x8b
            reshapeiptx8(&ipt0, &ipt1, &ipt2, &ipt3,
                         &ipt4, &ipt5, &ipt6, &ipt7,
                         y0x8bf, ux8bf, vx8bf, params);

            transpose_ipt8x4(ipt0, ipt1, ipt2, ipt3,
                             ipt4, ipt5, ipt6, ipt7,
                             &ix8, &px8, &tx8);

            ycc2rgbx8(&lx8, &mx8, &sx8, ix8, px8, tx8, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx8(&rx8b, &gx8b, &bx8b, lx8, mx8, sx8, *params->lms2rgb_matrix);

            rx8b = _mm256_mul_ps(rx8b, _mm256_set1_ps(JPEG_SCALE));
            gx8b = _mm256_mul_ps(gx8b, _mm256_set1_ps(JPEG_SCALE));
            bx8b = _mm256_mul_ps(bx8b, _mm256_set1_ps(JPEG_SCALE));

            r0x8b = _mm256_cvtps_epi32(rx8b);
            r0x8b = av_clip_int16_avx(r0x8b);
            g0x8b = _mm256_cvtps_epi32(gx8b);
            g0x8b = av_clip_int16_avx(g0x8b);
            b0x8b = _mm256_cvtps_epi32(bx8b);
            b0x8b = av_clip_int16_avx(b0x8b);

            // Reshape y1x8b
            reshapeiptx8(&ipt0, &ipt1, &ipt2, &ipt3,
                         &ipt4, &ipt5, &ipt6, &ipt7,
                         y1x8bf, ux8bf, vx8bf, params);

            transpose_ipt8x4(ipt0, ipt1, ipt2, ipt3,
                             ipt4, ipt5, ipt6, ipt7,
                             &ix8, &px8, &tx8);

            ycc2rgbx8(&lx8, &mx8, &sx8, ix8, px8, tx8, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx8(&rx8b, &gx8b, &bx8b, lx8, mx8, sx8, *params->lms2rgb_matrix);

            rx8b = _mm256_mul_ps(rx8b, _mm256_set1_ps(JPEG_SCALE));
            gx8b = _mm256_mul_ps(gx8b, _mm256_set1_ps(JPEG_SCALE));
            bx8b = _mm256_mul_ps(bx8b, _mm256_set1_ps(JPEG_SCALE));

            r1x8b = _mm256_cvtps_epi32(rx8b);
            r1x8b = av_clip_int16_avx(r1x8b);
            g1x8b = _mm256_cvtps_epi32(gx8b);
            g1x8b = av_clip_int16_avx(g1x8b);
            b1x8b = _mm256_cvtps_epi32(bx8b);
            b1x8b = av_clip_int16_avx(b1x8b);

            tonemap_int32x8_avx(r0x8a, g0x8a, b0x8a, r, g, b,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r1x8a, g1x8a, b1x8a, r1, g1, b1,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r0x8b, g0x8b, b0x8b, &r[8], &g[8], &b[8],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r1x8b, g1x8b, b1x8b, &r1[8], &g1[8], &b1[8],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);

            r0ox16 = _mm256_lddqu_si256((const __m256i_u *)r);
            g0ox16 = _mm256_lddqu_si256((const __m256i_u *)g);
            b0ox16 = _mm256_lddqu_si256((const __m256i_u *)b);

            roax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r0ox16));
            goax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(g0ox16));
            boax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0ox16));

            robx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r0ox16, 1));
            gobx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(g0ox16, 1));
            bobx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0ox16, 1));

            yoax8 = _mm256_mullo_epi32(roax8, _mm256_set1_epi32(cry));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_mullo_epi32(goax8, _mm256_set1_epi32(cgy)));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_mullo_epi32(boax8, _mm256_set1_epi32(cby)));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_set1_epi32(out_rnd));
            yoax8 = _mm256_srai_epi32(yoax8, out_sh);
            yoax8 = _mm256_add_epi32(yoax8, _mm256_set1_epi32(params->out_yuv_off));

            yobx8 = _mm256_mullo_epi32(robx8, _mm256_set1_epi32(cry));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_mullo_epi32(gobx8, _mm256_set1_epi32(cgy)));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_mullo_epi32(bobx8, _mm256_set1_epi32(cby)));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_set1_epi32(out_rnd));
            yobx8 = _mm256_srai_epi32(yobx8, out_sh);
            yobx8 = _mm256_add_epi32(yobx8, _mm256_set1_epi32(params->out_yuv_off));

            y0ox16 = _mm256_packus_epi32(yoax8, yobx8);
            y0ox16 = _mm256_permute4x64_epi64(y0ox16, _MM_SHUFFLE(3, 1, 2, 0));
            _mm256_storeu_si256((__m256i_u *) &dsty[x], y0ox16);

            r1ox16 = _mm256_lddqu_si256((const __m256i_u *)r1);
            g1ox16 = _mm256_lddqu_si256((const __m256i_u *)g1);
            b1ox16 = _mm256_lddqu_si256((const __m256i_u *)b1);

            r1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r1ox16));
            g1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(g1ox16));
            b1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1ox16));

            r1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r1ox16, 1));
            g1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(g1ox16, 1));
            b1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1ox16, 1));

            y1oax8 = _mm256_mullo_epi32(r1oax8, _mm256_set1_epi32(cry));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_mullo_epi32(g1oax8, _mm256_set1_epi32(cgy)));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_mullo_epi32(b1oax8, _mm256_set1_epi32(cby)));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_set1_epi32(out_rnd));
            y1oax8 = _mm256_srai_epi32(y1oax8, out_sh);
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_set1_epi32(params->out_yuv_off));

            y1obx8 = _mm256_mullo_epi32(r1obx8, _mm256_set1_epi32(cry));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_mullo_epi32(g1obx8, _mm256_set1_epi32(cgy)));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_mullo_epi32(b1obx8, _mm256_set1_epi32(cby)));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_set1_epi32(out_rnd));
            y1obx8 = _mm256_srai_epi32(y1obx8, out_sh);
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_set1_epi32(params->out_yuv_off));

            y1ox16 = _mm256_packus_epi32(y1oax8, y1obx8);
            y1ox16 = _mm256_permute4x64_epi64(y1ox16, _MM_SHUFFLE(3, 1, 2, 0));
            _mm256_storeu_si256((__m256i_u *) &dsty[x + dstlinesize[0] / 2], y1ox16);

            ravgx8 = _mm256_hadd_epi32(roax8, robx8);
            ravgx8 = _mm256_add_epi32(ravgx8, _mm256_hadd_epi32(r1oax8, r1obx8));
            ravgx8 = _mm256_permute4x64_epi64(ravgx8, _MM_SHUFFLE(3, 1, 2, 0));
            ravgx8 = _mm256_add_epi32(ravgx8, _mm256_set1_epi32(2));
            ravgx8 = _mm256_srai_epi32(ravgx8, 2);

            gavgx8 = _mm256_hadd_epi32(goax8, gobx8);
            gavgx8 = _mm256_add_epi32(gavgx8, _mm256_hadd_epi32(g1oax8, g1obx8));
            gavgx8 = _mm256_permute4x64_epi64(gavgx8, _MM_SHUFFLE(3, 1, 2, 0));
            gavgx8 = _mm256_add_epi32(gavgx8, _mm256_set1_epi32(2));
            gavgx8 = _mm256_srai_epi32(gavgx8, 2);

            bavgx8 = _mm256_hadd_epi32(boax8, bobx8);
            bavgx8 = _mm256_add_epi32(bavgx8, _mm256_hadd_epi32(b1oax8, b1obx8));
            bavgx8 = _mm256_permute4x64_epi64(bavgx8, _MM_SHUFFLE(3, 1, 2, 0));
            bavgx8 = _mm256_add_epi32(bavgx8, _mm256_set1_epi32(2));
            bavgx8 = _mm256_srai_epi32(bavgx8, 2);

            uox8 = _mm256_add_epi32(_mm256_set1_epi32(out_rnd), _mm256_mullo_epi32(ravgx8, _mm256_set1_epi32(cru)));
            uox8 = _mm256_add_epi32(uox8, _mm256_mullo_epi32(gavgx8, _mm256_set1_epi32(ocgu)));
            uox8 = _mm256_add_epi32(uox8, _mm256_mullo_epi32(bavgx8, _mm256_set1_epi32(cburv)));
            uox8 = _mm256_srai_epi32(uox8, out_sh);
            uox8 = _mm256_add_epi32(uox8, _mm256_set1_epi32(out_uv_offset));
            uox8 = _mm256_packus_epi32(uox8, _mm256_setzero_si256());
            uox8 = _mm256_permute4x64_epi64(uox8, _MM_SHUFFLE(3, 1, 2, 0));
            _mm_storeu_si128((__m128i_u *) &dstu[x >> 1], _mm256_castsi256_si128(uox8));

            vox8 = _mm256_add_epi32(_mm256_set1_epi32(out_rnd), _mm256_mullo_epi32(ravgx8, _mm256_set1_epi32(cburv)));
            vox8 = _mm256_add_epi32(vox8, _mm256_mullo_epi32(gavgx8, _mm256_set1_epi32(ocgv)));
            vox8 = _mm256_add_epi32(vox8, _mm256_mullo_epi32(bavgx8, _mm256_set1_epi32(cbv)));
            vox8 = _mm256_srai_epi32(vox8, out_sh);
            vox8 = _mm256_add_epi32(vox8, _mm256_set1_epi32(out_uv_offset));
            vox8 = _mm256_packus_epi32(vox8, _mm256_setzero_si256());
            vox8 = _mm256_permute4x64_epi64(vox8, _MM_SHUFFLE(3, 1, 2, 0));
            _mm_storeu_si128((__m128i_u *) &dstv[x >> 1], _mm256_castsi256_si128(vox8));
        }
    }

    // Process remaining pixels cannot fill the full simd register with scalar version
    if (remainw) {
        int offset = width & (int)0xfffffff0;
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
#endif // ENABLE_TONEMAPX_AVX_INTRINSICS
}

X86_64_V3 void tonemap_frame_dovi_2_420hdr_avx(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                               const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                               const int *dstlinesize, const int *srclinesize,
                                               int dstdepth, int srcdepth,
                                               int width, int height,
                                               const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_AVX_INTRINSICS
    uint16_t *rdsty = dsty;
    uint16_t *rdstu = dstu;
    uint16_t *rdstv = dstv;
    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcu = srcu;
    const uint16_t *rsrcv = srcv;
    int rheight = height;
    // not zero when not divisible by 8
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 14;

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

    __m256i ux8, vx8;
    __m256i y0x16, y1x16;
    __m256i y0x8a, y0x8b, y1x8a, y1x8b, ux8a, ux8b, vx8a, vx8b;
    __m256i r0x8a, g0x8a, b0x8a, r0x8b, g0x8b, b0x8b;
    __m256i r1x8a, g1x8a, b1x8a, r1x8b, g1x8b, b1x8b;

    __m256i y0ox16;
    __m256i roax8, robx8, goax8, gobx8, boax8, bobx8;
    __m256i yoax8, yobx8;

    __m256i y1ox16;
    __m256i r1oax8, r1obx8, g1oax8, g1obx8, b1oax8, b1obx8;
    __m256i y1oax8, y1obx8;
    __m256i uox8, vox8, ravgx8, gavgx8, bavgx8;

    __m128 ipt0, ipt1, ipt2, ipt3, ipt4, ipt5, ipt6, ipt7;
    __m256 ix8, px8, tx8;
    __m256 lx8, mx8, sx8;
    __m256 rx8a, gx8a, bx8a, rx8b, gx8b, bx8b;
    __m256 y0x8af, y0x8bf, y1x8af, y1x8bf, ux8af, ux8bf, vx8af, vx8bf;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstu += dstlinesize[1] / 2, dstv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 4; xx++) {
            int x = xx << 4;

            y0x16 = _mm256_lddqu_si256((__m256i*)(srcy + x));
            y1x16 = _mm256_lddqu_si256((__m256i*)(srcy + (srclinesize[0] / 2 + x)));
            ux8 = _mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i_u *)(srcu + (x >> 1))));
            vx8 = _mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i_u *)(srcv + (x >> 1))));

            y0x8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y0x16));
            y0x8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y0x16, 1));
            y1x8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y1x16));
            y1x8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y1x16, 1));

            ux8a = _mm256_permutevar8x32_epi32(ux8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
            ux8b = _mm256_permutevar8x32_epi32(ux8, _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4));
            vx8a = _mm256_permutevar8x32_epi32(vx8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
            vx8b = _mm256_permutevar8x32_epi32(vx8, _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4));

            y0x8af = _mm256_cvtepi32_ps(y0x8a);
            y0x8bf = _mm256_cvtepi32_ps(y0x8b);
            y1x8af = _mm256_cvtepi32_ps(y1x8a);
            y1x8bf = _mm256_cvtepi32_ps(y1x8b);
            ux8af = _mm256_cvtepi32_ps(ux8a);
            ux8bf = _mm256_cvtepi32_ps(ux8b);
            vx8af = _mm256_cvtepi32_ps(vx8a);
            vx8bf = _mm256_cvtepi32_ps(vx8b);

            y0x8af = _mm256_div_ps(y0x8af, _mm256_set1_ps(in_rng));
            y0x8bf = _mm256_div_ps(y0x8bf, _mm256_set1_ps(in_rng));
            y1x8af = _mm256_div_ps(y1x8af, _mm256_set1_ps(in_rng));
            y1x8bf = _mm256_div_ps(y1x8bf, _mm256_set1_ps(in_rng));
            ux8af = _mm256_div_ps(ux8af, _mm256_set1_ps(in_rng));
            ux8bf = _mm256_div_ps(ux8bf, _mm256_set1_ps(in_rng));
            vx8af = _mm256_div_ps(vx8af, _mm256_set1_ps(in_rng));
            vx8bf = _mm256_div_ps(vx8bf, _mm256_set1_ps(in_rng));

            // Reshape y0x8a
            reshapeiptx8(&ipt0, &ipt1, &ipt2, &ipt3,
                         &ipt4, &ipt5, &ipt6, &ipt7,
                         y0x8af, ux8af, vx8af, params);

            transpose_ipt8x4(ipt0, ipt1, ipt2, ipt3,
                             ipt4, ipt5, ipt6, ipt7,
                             &ix8, &px8, &tx8);

            ycc2rgbx8(&lx8, &mx8, &sx8, ix8, px8, tx8, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx8(&rx8a, &gx8a, &bx8a, lx8, mx8, sx8, *params->lms2rgb_matrix);

            rx8a = _mm256_mul_ps(rx8a, _mm256_set1_ps(JPEG_SCALE));
            gx8a = _mm256_mul_ps(gx8a, _mm256_set1_ps(JPEG_SCALE));
            bx8a = _mm256_mul_ps(bx8a, _mm256_set1_ps(JPEG_SCALE));

            r0x8a = _mm256_cvtps_epi32(rx8a);
            r0x8a = av_clip_int16_avx(r0x8a);
            g0x8a = _mm256_cvtps_epi32(gx8a);
            g0x8a = av_clip_int16_avx(g0x8a);
            b0x8a = _mm256_cvtps_epi32(bx8a);
            b0x8a = av_clip_int16_avx(b0x8a);

            // Reshape y1x8a
            reshapeiptx8(&ipt0, &ipt1, &ipt2, &ipt3,
                         &ipt4, &ipt5, &ipt6, &ipt7,
                         y1x8af, ux8af, vx8af, params);

            transpose_ipt8x4(ipt0, ipt1, ipt2, ipt3,
                             ipt4, ipt5, ipt6, ipt7,
                             &ix8, &px8, &tx8);

            ycc2rgbx8(&lx8, &mx8, &sx8, ix8, px8, tx8, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx8(&rx8a, &gx8a, &bx8a, lx8, mx8, sx8, *params->lms2rgb_matrix);

            rx8a = _mm256_mul_ps(rx8a, _mm256_set1_ps(JPEG_SCALE));
            gx8a = _mm256_mul_ps(gx8a, _mm256_set1_ps(JPEG_SCALE));
            bx8a = _mm256_mul_ps(bx8a, _mm256_set1_ps(JPEG_SCALE));

            r1x8a = _mm256_cvtps_epi32(rx8a);
            r1x8a = av_clip_int16_avx(r1x8a);
            g1x8a = _mm256_cvtps_epi32(gx8a);
            g1x8a = av_clip_int16_avx(g1x8a);
            b1x8a = _mm256_cvtps_epi32(bx8a);
            b1x8a = av_clip_int16_avx(b1x8a);

            // Reshape y0x8b
            reshapeiptx8(&ipt0, &ipt1, &ipt2, &ipt3,
                         &ipt4, &ipt5, &ipt6, &ipt7,
                         y0x8bf, ux8bf, vx8bf, params);

            transpose_ipt8x4(ipt0, ipt1, ipt2, ipt3,
                             ipt4, ipt5, ipt6, ipt7,
                             &ix8, &px8, &tx8);

            ycc2rgbx8(&lx8, &mx8, &sx8, ix8, px8, tx8, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx8(&rx8b, &gx8b, &bx8b, lx8, mx8, sx8, *params->lms2rgb_matrix);

            rx8b = _mm256_mul_ps(rx8b, _mm256_set1_ps(JPEG_SCALE));
            gx8b = _mm256_mul_ps(gx8b, _mm256_set1_ps(JPEG_SCALE));
            bx8b = _mm256_mul_ps(bx8b, _mm256_set1_ps(JPEG_SCALE));

            r0x8b = _mm256_cvtps_epi32(rx8b);
            r0x8b = av_clip_int16_avx(r0x8b);
            g0x8b = _mm256_cvtps_epi32(gx8b);
            g0x8b = av_clip_int16_avx(g0x8b);
            b0x8b = _mm256_cvtps_epi32(bx8b);
            b0x8b = av_clip_int16_avx(b0x8b);

            // Reshape y1x8b
            reshapeiptx8(&ipt0, &ipt1, &ipt2, &ipt3,
                         &ipt4, &ipt5, &ipt6, &ipt7,
                         y1x8bf, ux8bf, vx8bf, params);

            transpose_ipt8x4(ipt0, ipt1, ipt2, ipt3,
                             ipt4, ipt5, ipt6, ipt7,
                             &ix8, &px8, &tx8);

            ycc2rgbx8(&lx8, &mx8, &sx8, ix8, px8, tx8, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx8(&rx8b, &gx8b, &bx8b, lx8, mx8, sx8, *params->lms2rgb_matrix);

            rx8b = _mm256_mul_ps(rx8b, _mm256_set1_ps(JPEG_SCALE));
            gx8b = _mm256_mul_ps(gx8b, _mm256_set1_ps(JPEG_SCALE));
            bx8b = _mm256_mul_ps(bx8b, _mm256_set1_ps(JPEG_SCALE));

            r1x8b = _mm256_cvtps_epi32(rx8b);
            r1x8b = av_clip_int16_avx(r1x8b);
            g1x8b = _mm256_cvtps_epi32(gx8b);
            g1x8b = av_clip_int16_avx(g1x8b);
            b1x8b = _mm256_cvtps_epi32(bx8b);
            b1x8b = av_clip_int16_avx(b1x8b);

            roax8 = r0x8a;
            goax8 = g0x8a;
            boax8 = b0x8a;

            robx8 = r0x8b;
            gobx8 = g0x8b;
            bobx8 = b0x8b;

            yoax8 = _mm256_mullo_epi32(roax8, _mm256_set1_epi32(cry));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_mullo_epi32(goax8, _mm256_set1_epi32(cgy)));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_mullo_epi32(boax8, _mm256_set1_epi32(cby)));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_set1_epi32(out_rnd));
            yoax8 = _mm256_srai_epi32(yoax8, out_sh);
            yoax8 = _mm256_add_epi32(yoax8, _mm256_set1_epi32(params->out_yuv_off));

            yobx8 = _mm256_mullo_epi32(robx8, _mm256_set1_epi32(cry));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_mullo_epi32(gobx8, _mm256_set1_epi32(cgy)));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_mullo_epi32(bobx8, _mm256_set1_epi32(cby)));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_set1_epi32(out_rnd));
            yobx8 = _mm256_srai_epi32(yobx8, out_sh);
            yobx8 = _mm256_add_epi32(yobx8, _mm256_set1_epi32(params->out_yuv_off));

            y0ox16 = _mm256_packus_epi32(yoax8, yobx8);
            y0ox16 = _mm256_permute4x64_epi64(y0ox16, _MM_SHUFFLE(3, 1, 2, 0));
            _mm256_storeu_si256((__m256i_u *) &dsty[x], y0ox16);

            r1oax8 = r1x8a;
            g1oax8 = g1x8a;
            b1oax8 = b1x8a;

            r1obx8 = r1x8b;
            g1obx8 = g1x8b;
            b1obx8 = b1x8b;

            y1oax8 = _mm256_mullo_epi32(r1oax8, _mm256_set1_epi32(cry));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_mullo_epi32(g1oax8, _mm256_set1_epi32(cgy)));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_mullo_epi32(b1oax8, _mm256_set1_epi32(cby)));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_set1_epi32(out_rnd));
            y1oax8 = _mm256_srai_epi32(y1oax8, out_sh);
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_set1_epi32(params->out_yuv_off));

            y1obx8 = _mm256_mullo_epi32(r1obx8, _mm256_set1_epi32(cry));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_mullo_epi32(g1obx8, _mm256_set1_epi32(cgy)));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_mullo_epi32(b1obx8, _mm256_set1_epi32(cby)));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_set1_epi32(out_rnd));
            y1obx8 = _mm256_srai_epi32(y1obx8, out_sh);
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_set1_epi32(params->out_yuv_off));

            y1ox16 = _mm256_packus_epi32(y1oax8, y1obx8);
            y1ox16 = _mm256_permute4x64_epi64(y1ox16, _MM_SHUFFLE(3, 1, 2, 0));
            _mm256_storeu_si256((__m256i_u *) &dsty[x + dstlinesize[0] / 2], y1ox16);

            ravgx8 = _mm256_hadd_epi32(roax8, robx8);
            ravgx8 = _mm256_add_epi32(ravgx8, _mm256_hadd_epi32(r1oax8, r1obx8));
            ravgx8 = _mm256_permute4x64_epi64(ravgx8, _MM_SHUFFLE(3, 1, 2, 0));
            ravgx8 = _mm256_add_epi32(ravgx8, _mm256_set1_epi32(2));
            ravgx8 = _mm256_srai_epi32(ravgx8, 2);

            gavgx8 = _mm256_hadd_epi32(goax8, gobx8);
            gavgx8 = _mm256_add_epi32(gavgx8, _mm256_hadd_epi32(g1oax8, g1obx8));
            gavgx8 = _mm256_permute4x64_epi64(gavgx8, _MM_SHUFFLE(3, 1, 2, 0));
            gavgx8 = _mm256_add_epi32(gavgx8, _mm256_set1_epi32(2));
            gavgx8 = _mm256_srai_epi32(gavgx8, 2);

            bavgx8 = _mm256_hadd_epi32(boax8, bobx8);
            bavgx8 = _mm256_add_epi32(bavgx8, _mm256_hadd_epi32(b1oax8, b1obx8));
            bavgx8 = _mm256_permute4x64_epi64(bavgx8, _MM_SHUFFLE(3, 1, 2, 0));
            bavgx8 = _mm256_add_epi32(bavgx8, _mm256_set1_epi32(2));
            bavgx8 = _mm256_srai_epi32(bavgx8, 2);

            uox8 = _mm256_add_epi32(_mm256_set1_epi32(out_rnd), _mm256_mullo_epi32(ravgx8, _mm256_set1_epi32(cru)));
            uox8 = _mm256_add_epi32(uox8, _mm256_mullo_epi32(gavgx8, _mm256_set1_epi32(ocgu)));
            uox8 = _mm256_add_epi32(uox8, _mm256_mullo_epi32(bavgx8, _mm256_set1_epi32(cburv)));
            uox8 = _mm256_srai_epi32(uox8, out_sh);
            uox8 = _mm256_add_epi32(uox8, _mm256_set1_epi32(out_uv_offset));
            uox8 = _mm256_packus_epi32(uox8, _mm256_setzero_si256());
            uox8 = _mm256_permute4x64_epi64(uox8, _MM_SHUFFLE(3, 1, 2, 0));
            _mm_storeu_si128((__m128i_u *) &dstu[x >> 1], _mm256_castsi256_si128(uox8));

            vox8 = _mm256_add_epi32(_mm256_set1_epi32(out_rnd), _mm256_mullo_epi32(ravgx8, _mm256_set1_epi32(cburv)));
            vox8 = _mm256_add_epi32(vox8, _mm256_mullo_epi32(gavgx8, _mm256_set1_epi32(ocgv)));
            vox8 = _mm256_add_epi32(vox8, _mm256_mullo_epi32(bavgx8, _mm256_set1_epi32(cbv)));
            vox8 = _mm256_srai_epi32(vox8, out_sh);
            vox8 = _mm256_add_epi32(vox8, _mm256_set1_epi32(out_uv_offset));
            vox8 = _mm256_packus_epi32(vox8, _mm256_setzero_si256());
            vox8 = _mm256_permute4x64_epi64(vox8, _MM_SHUFFLE(3, 1, 2, 0));
            _mm_storeu_si128((__m128i_u *) &dstv[x >> 1], _mm256_castsi256_si128(vox8));
        }
    }

    // Process remaining pixels cannot fill the full simd register with scalar version
    if (remainw) {
        int offset = width & (int)0xfffffff0;
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
#endif // ENABLE_TONEMAPX_AVX_INTRINSICS
}

X86_64_V3 void tonemap_frame_420p10_2_420p_avx(uint8_t *dsty, uint8_t *dstu, uint8_t *dstv,
                                               const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                               const int *dstlinesize, const int *srclinesize,
                                               int dstdepth, int srcdepth,
                                               int width, int height,
                                               const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_AVX_INTRINSICS
    uint8_t *rdsty = dsty;
    uint8_t *rdstu = dstu;
    uint8_t *rdstv = dstv;
    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcu = srcu;
    const uint16_t *rsrcv = srcv;
    int rheight = height;
    // not zero when not divisible by 16
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 14;

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

    int16_t r[16], g[16], b[16];
    int16_t r1[16], g1[16], b1[16];
    __m256i in_yuv_offx8 = _mm256_set1_epi32(params->in_yuv_off);
    __m256i in_uv_offx8 = _mm256_set1_epi32(in_uv_offset);
    __m256i cyx8 = _mm256_set1_epi32(cy);
    __m256i rndx8 = _mm256_set1_epi32(in_rnd);

    __m256i ux8, vx8;
    __m256i y0x16, y1x16;
    __m256i y0x8a, y0x8b, y1x8a, y1x8b, ux8a, ux8b, vx8a, vx8b;
    __m256i r0x8a, g0x8a, b0x8a, r0x8b, g0x8b, b0x8b;
    __m256i r1x8a, g1x8a, b1x8a, r1x8b, g1x8b, b1x8b;

    __m256i r0ox16, g0ox16, b0ox16;
    __m256i y0ox16;
    __m256i roax8, robx8, goax8, gobx8, boax8, bobx8;
    __m256i yoax8, yobx8;

    __m256i r1ox16, g1ox16, b1ox16;
    __m256i y1ox16;
    __m256i r1oax8, r1obx8, g1oax8, g1obx8, b1oax8, b1obx8;
    __m256i y1oax8, y1obx8;
    __m256i uox8, vox8, ravgx8, gavgx8, bavgx8;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0] * 2, dstu += dstlinesize[1], dstv += dstlinesize[2],
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[2] / 2) {
        for (int xx = 0; xx < width >> 4; xx++) {
            int x = xx << 4;

            y0x16 = _mm256_lddqu_si256((__m256i*)(srcy + x));
            y1x16 = _mm256_lddqu_si256((__m256i*)(srcy + (srclinesize[0] / 2 + x)));
            ux8 = _mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i_u *)(srcu + (x >> 1))));
            vx8 = _mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i_u *)(srcv + (x >> 1))));

            y0x8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y0x16));
            y0x8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y0x16, 1));
            y1x8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y1x16));
            y1x8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y1x16, 1));

            y0x8a = _mm256_sub_epi32(y0x8a, in_yuv_offx8);
            y1x8a = _mm256_sub_epi32(y1x8a, in_yuv_offx8);
            y0x8b = _mm256_sub_epi32(y0x8b, in_yuv_offx8);
            y1x8b = _mm256_sub_epi32(y1x8b, in_yuv_offx8);
            ux8 = _mm256_sub_epi32(ux8, in_uv_offx8);
            vx8 = _mm256_sub_epi32(vx8, in_uv_offx8);

            ux8a = _mm256_permutevar8x32_epi32(ux8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
            ux8b = _mm256_permutevar8x32_epi32(ux8, _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4));
            vx8a = _mm256_permutevar8x32_epi32(vx8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
            vx8b = _mm256_permutevar8x32_epi32(vx8, _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4));

            // r = av_clip_int16((y * cy + crv * v + in_rnd) >> in_sh);
            r0x8a = g0x8a = b0x8a = _mm256_mullo_epi32(y0x8a, cyx8);
            r0x8a = _mm256_add_epi32(r0x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(crv)));
            r0x8a = _mm256_add_epi32(r0x8a, rndx8);
            r0x8a = _mm256_srai_epi32(r0x8a, in_sh);
            r0x8a = av_clip_int16_avx(r0x8a);

            r1x8a = g1x8a = b1x8a = _mm256_mullo_epi32(y1x8a, cyx8);
            r1x8a = _mm256_add_epi32(r1x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(crv)));
            r1x8a = _mm256_add_epi32(r1x8a, rndx8);
            r1x8a = _mm256_srai_epi32(r1x8a, in_sh);
            r1x8a = av_clip_int16_avx(r1x8a);

            // g = av_clip_int16((y * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g0x8a = _mm256_add_epi32(g0x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cgu)));
            g0x8a = _mm256_add_epi32(g0x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(cgv)));
            g0x8a = _mm256_add_epi32(g0x8a, rndx8);
            g0x8a = _mm256_srai_epi32(g0x8a, in_sh);
            g0x8a = av_clip_int16_avx(g0x8a);

            g1x8a = _mm256_add_epi32(g1x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cgu)));
            g1x8a = _mm256_add_epi32(g1x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(cgv)));
            g1x8a = _mm256_add_epi32(g1x8a, rndx8);
            g1x8a = _mm256_srai_epi32(g1x8a, in_sh);
            g1x8a = av_clip_int16_avx(g1x8a);

            // b = av_clip_int16((y * cy + cbu * u + in_rnd) >> in_sh);
            b0x8a = _mm256_add_epi32(b0x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cbu)));
            b0x8a = _mm256_add_epi32(b0x8a, rndx8);
            b0x8a = _mm256_srai_epi32(b0x8a, in_sh);
            b0x8a = av_clip_int16_avx(b0x8a);

            b1x8a = _mm256_add_epi32(b1x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cbu)));
            b1x8a = _mm256_add_epi32(b1x8a, rndx8);
            b1x8a = _mm256_srai_epi32(b1x8a, in_sh);
            b1x8a = av_clip_int16_avx(b1x8a);

            r0x8b = g0x8b = b0x8b = _mm256_mullo_epi32(y0x8b, cyx8);
            r0x8b = _mm256_add_epi32(r0x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(crv)));
            r0x8b = _mm256_add_epi32(r0x8b, rndx8);
            r0x8b = _mm256_srai_epi32(r0x8b, in_sh);
            r0x8b = av_clip_int16_avx(r0x8b);

            r1x8b = g1x8b = b1x8b = _mm256_mullo_epi32(y1x8b, cyx8);
            r1x8b = _mm256_add_epi32(r1x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(crv)));
            r1x8b = _mm256_add_epi32(r1x8b, rndx8);
            r1x8b = _mm256_srai_epi32(r1x8b, in_sh);
            r1x8b = av_clip_int16_avx(r1x8b);

            g0x8b = _mm256_add_epi32(g0x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cgu)));
            g0x8b = _mm256_add_epi32(g0x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(cgv)));
            g0x8b = _mm256_add_epi32(g0x8b, rndx8);
            g0x8b = _mm256_srai_epi32(g0x8b, in_sh);
            g0x8b = av_clip_int16_avx(g0x8b);

            g1x8b = _mm256_add_epi32(g1x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cgu)));
            g1x8b = _mm256_add_epi32(g1x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(cgv)));
            g1x8b = _mm256_add_epi32(g1x8b, rndx8);
            g1x8b = _mm256_srai_epi32(g1x8b, in_sh);
            g1x8b = av_clip_int16_avx(g1x8b);

            b0x8b = _mm256_add_epi32(b0x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cbu)));
            b0x8b = _mm256_add_epi32(b0x8b, rndx8);
            b0x8b = _mm256_srai_epi32(b0x8b, in_sh);
            b0x8b = av_clip_int16_avx(b0x8b);

            b1x8b = _mm256_add_epi32(b1x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cbu)));
            b1x8b = _mm256_add_epi32(b1x8b, rndx8);
            b1x8b = _mm256_srai_epi32(b1x8b, in_sh);
            b1x8b = av_clip_int16_avx(b1x8b);

            tonemap_int32x8_avx(r0x8a, g0x8a, b0x8a, r, g, b,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r1x8a, g1x8a, b1x8a, r1, g1, b1,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r0x8b, g0x8b, b0x8b, &r[8], &g[8], &b[8],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r1x8b, g1x8b, b1x8b, &r1[8], &g1[8], &b1[8],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);

            r0ox16 = _mm256_lddqu_si256((const __m256i_u *)r);
            g0ox16 = _mm256_lddqu_si256((const __m256i_u *)g);
            b0ox16 = _mm256_lddqu_si256((const __m256i_u *)b);

            roax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r0ox16));
            goax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(g0ox16));
            boax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0ox16));

            robx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r0ox16, 1));
            gobx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(g0ox16, 1));
            bobx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0ox16, 1));

            yoax8 = _mm256_mullo_epi32(roax8, _mm256_set1_epi32(cry));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_mullo_epi32(goax8, _mm256_set1_epi32(cgy)));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_mullo_epi32(boax8, _mm256_set1_epi32(cby)));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_set1_epi32(out_rnd));
            yoax8 = _mm256_srai_epi32(yoax8, out_sh);
            yoax8 = _mm256_add_epi32(yoax8, _mm256_set1_epi32(params->out_yuv_off));

            yobx8 = _mm256_mullo_epi32(robx8, _mm256_set1_epi32(cry));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_mullo_epi32(gobx8, _mm256_set1_epi32(cgy)));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_mullo_epi32(bobx8, _mm256_set1_epi32(cby)));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_set1_epi32(out_rnd));
            yobx8 = _mm256_srai_epi32(yobx8, out_sh);
            yobx8 = _mm256_add_epi32(yobx8, _mm256_set1_epi32(params->out_yuv_off));

            y0ox16 = _mm256_packs_epi32(yoax8, yobx8);
            y0ox16 = _mm256_permute4x64_epi64(y0ox16, _MM_SHUFFLE(3, 1, 2, 0));
            _mm_storeu_si128((__m128i_u *) &dsty[x], _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(y0ox16, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0))));

            r1ox16 = _mm256_lddqu_si256((const __m256i_u *)r1);
            g1ox16 = _mm256_lddqu_si256((const __m256i_u *)g1);
            b1ox16 = _mm256_lddqu_si256((const __m256i_u *)b1);

            r1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r1ox16));
            g1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(g1ox16));
            b1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1ox16));

            r1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r1ox16, 1));
            g1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(g1ox16, 1));
            b1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1ox16, 1));

            y1oax8 = _mm256_mullo_epi32(r1oax8, _mm256_set1_epi32(cry));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_mullo_epi32(g1oax8, _mm256_set1_epi32(cgy)));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_mullo_epi32(b1oax8, _mm256_set1_epi32(cby)));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_set1_epi32(out_rnd));
            y1oax8 = _mm256_srai_epi32(y1oax8, out_sh);
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_set1_epi32(params->out_yuv_off));

            y1obx8 = _mm256_mullo_epi32(r1obx8, _mm256_set1_epi32(cry));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_mullo_epi32(g1obx8, _mm256_set1_epi32(cgy)));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_mullo_epi32(b1obx8, _mm256_set1_epi32(cby)));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_set1_epi32(out_rnd));
            y1obx8 = _mm256_srai_epi32(y1obx8, out_sh);
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_set1_epi32(params->out_yuv_off));

            y1ox16 = _mm256_packs_epi32(y1oax8, y1obx8);
            y1ox16 = _mm256_permute4x64_epi64(y1ox16, _MM_SHUFFLE(3, 1, 2, 0));
            _mm_storeu_si128((__m128i_u *) &dsty[x + dstlinesize[0]], _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(y1ox16, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0))));

            ravgx8 = _mm256_hadd_epi32(roax8, robx8);
            ravgx8 = _mm256_add_epi32(ravgx8, _mm256_hadd_epi32(r1oax8, r1obx8));
            ravgx8 = _mm256_permute4x64_epi64(ravgx8, _MM_SHUFFLE(3, 1, 2, 0));
            ravgx8 = _mm256_add_epi32(ravgx8, _mm256_set1_epi32(2));
            ravgx8 = _mm256_srai_epi32(ravgx8, 2);

            gavgx8 = _mm256_hadd_epi32(goax8, gobx8);
            gavgx8 = _mm256_add_epi32(gavgx8, _mm256_hadd_epi32(g1oax8, g1obx8));
            gavgx8 = _mm256_permute4x64_epi64(gavgx8, _MM_SHUFFLE(3, 1, 2, 0));
            gavgx8 = _mm256_add_epi32(gavgx8, _mm256_set1_epi32(2));
            gavgx8 = _mm256_srai_epi32(gavgx8, 2);

            bavgx8 = _mm256_hadd_epi32(boax8, bobx8);
            bavgx8 = _mm256_add_epi32(bavgx8, _mm256_hadd_epi32(b1oax8, b1obx8));
            bavgx8 = _mm256_permute4x64_epi64(bavgx8, _MM_SHUFFLE(3, 1, 2, 0));
            bavgx8 = _mm256_add_epi32(bavgx8, _mm256_set1_epi32(2));
            bavgx8 = _mm256_srai_epi32(bavgx8, 2);

            uox8 = _mm256_add_epi32(_mm256_set1_epi32(out_rnd), _mm256_mullo_epi32(ravgx8, _mm256_set1_epi32(cru)));
            uox8 = _mm256_add_epi32(uox8, _mm256_mullo_epi32(gavgx8, _mm256_set1_epi32(ocgu)));
            uox8 = _mm256_add_epi32(uox8, _mm256_mullo_epi32(bavgx8, _mm256_set1_epi32(cburv)));
            uox8 = _mm256_srai_epi32(uox8, out_sh);
            uox8 = _mm256_add_epi32(uox8, _mm256_set1_epi32(out_uv_offset));
            uox8 = _mm256_packs_epi32(uox8, _mm256_setzero_si256());
            uox8 = _mm256_permute4x64_epi64(uox8, _MM_SHUFFLE(3, 1, 2, 0));
            uox8 = _mm256_packus_epi16(uox8, _mm256_setzero_si256());
            _mm_storeu_si64(&dstu[x >> 1], _mm256_castsi256_si128(uox8));

            vox8 = _mm256_add_epi32(_mm256_set1_epi32(out_rnd), _mm256_mullo_epi32(ravgx8, _mm256_set1_epi32(cburv)));
            vox8 = _mm256_add_epi32(vox8, _mm256_mullo_epi32(gavgx8, _mm256_set1_epi32(ocgv)));
            vox8 = _mm256_add_epi32(vox8, _mm256_mullo_epi32(bavgx8, _mm256_set1_epi32(cbv)));
            vox8 = _mm256_srai_epi32(vox8, out_sh);
            vox8 = _mm256_add_epi32(vox8, _mm256_set1_epi32(out_uv_offset));
            vox8 = _mm256_packs_epi32(vox8, _mm256_setzero_si256());
            vox8 = _mm256_permute4x64_epi64(vox8, _MM_SHUFFLE(3, 1, 2, 0));
            vox8 = _mm256_packus_epi16(vox8, _mm256_setzero_si256());
            _mm_storeu_si64(&dstv[x >> 1], _mm256_castsi256_si128(vox8));
        }
    }

    // Process remaining pixels cannot fill the full simd register with scalar version
    if (remainw) {
        int offset = width & (int)0xfffffff0;
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
#endif // ENABLE_TONEMAPX_AVX_INTRINSICS
}

X86_64_V3 void tonemap_frame_420p10_2_420p10_avx(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                                 const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                                 const int *dstlinesize, const int *srclinesize,
                                                 int dstdepth, int srcdepth,
                                                 int width, int height,
                                                 const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_AVX_INTRINSICS
    uint16_t *rdsty = dsty;
    uint16_t *rdstu = dstu;
    uint16_t *rdstv = dstv;
    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcu = srcu;
    const uint16_t *rsrcv = srcv;
    int rheight = height;
    // not zero when not divisible by 8
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 14;

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

    int16_t r[16], g[16], b[16];
    int16_t r1[16], g1[16], b1[16];
    __m256i in_yuv_offx8 = _mm256_set1_epi32(params->in_yuv_off);
    __m256i in_uv_offx8 = _mm256_set1_epi32(in_uv_offset);
    __m256i cyx8 = _mm256_set1_epi32(cy);
    __m256i rndx8 = _mm256_set1_epi32(in_rnd);

    __m256i r0ox16, g0ox16, b0ox16;
    __m256i y0ox16;
    __m256i roax8, robx8, goax8, gobx8, boax8, bobx8;
    __m256i yoax8, yobx8;
    __m256i ux8, vx8;
    __m256i y0x16, y1x16;
    __m256i y0x8a, y0x8b, y1x8a, y1x8b, ux8a, ux8b, vx8a, vx8b;
    __m256i r0x8a, g0x8a, b0x8a, r0x8b, g0x8b, b0x8b;
    __m256i r1x8a, g1x8a, b1x8a, r1x8b, g1x8b, b1x8b;

    __m256i r1ox16, g1ox16, b1ox16;
    __m256i y1ox16;
    __m256i r1oax8, r1obx8, g1oax8, g1obx8, b1oax8, b1obx8;
    __m256i y1oax8, y1obx8;
    __m256i uox8, vox8, ravgx8, gavgx8, bavgx8;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstu += dstlinesize[1] / 2, dstv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 4; xx++) {
            int x = xx << 4;

            y0x16 = _mm256_lddqu_si256((__m256i*)(srcy + x));
            y1x16 = _mm256_lddqu_si256((__m256i*)(srcy + (srclinesize[0] / 2 + x)));
            ux8 = _mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i_u *)(srcu + (x >> 1))));
            vx8 = _mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i_u *)(srcv + (x >> 1))));

            y0x8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y0x16));
            y0x8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y0x16, 1));
            y1x8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y1x16));
            y1x8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y1x16, 1));

            y0x8a = _mm256_sub_epi32(y0x8a, in_yuv_offx8);
            y1x8a = _mm256_sub_epi32(y1x8a, in_yuv_offx8);
            y0x8b = _mm256_sub_epi32(y0x8b, in_yuv_offx8);
            y1x8b = _mm256_sub_epi32(y1x8b, in_yuv_offx8);
            ux8 = _mm256_sub_epi32(ux8, in_uv_offx8);
            vx8 = _mm256_sub_epi32(vx8, in_uv_offx8);

            ux8a = _mm256_permutevar8x32_epi32(ux8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
            ux8b = _mm256_permutevar8x32_epi32(ux8, _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4));
            vx8a = _mm256_permutevar8x32_epi32(vx8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
            vx8b = _mm256_permutevar8x32_epi32(vx8, _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4));

            // r = av_clip_int16((y * cy + crv * v + in_rnd) >> in_sh);
            r0x8a = g0x8a = b0x8a = _mm256_mullo_epi32(y0x8a, cyx8);
            r0x8a = _mm256_add_epi32(r0x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(crv)));
            r0x8a = _mm256_add_epi32(r0x8a, rndx8);
            r0x8a = _mm256_srai_epi32(r0x8a, in_sh);
            r0x8a = av_clip_int16_avx(r0x8a);

            r1x8a = g1x8a = b1x8a = _mm256_mullo_epi32(y1x8a, cyx8);
            r1x8a = _mm256_add_epi32(r1x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(crv)));
            r1x8a = _mm256_add_epi32(r1x8a, rndx8);
            r1x8a = _mm256_srai_epi32(r1x8a, in_sh);
            r1x8a = av_clip_int16_avx(r1x8a);

            // g = av_clip_int16((y * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g0x8a = _mm256_add_epi32(g0x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cgu)));
            g0x8a = _mm256_add_epi32(g0x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(cgv)));
            g0x8a = _mm256_add_epi32(g0x8a, rndx8);
            g0x8a = _mm256_srai_epi32(g0x8a, in_sh);
            g0x8a = av_clip_int16_avx(g0x8a);

            g1x8a = _mm256_add_epi32(g1x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cgu)));
            g1x8a = _mm256_add_epi32(g1x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(cgv)));
            g1x8a = _mm256_add_epi32(g1x8a, rndx8);
            g1x8a = _mm256_srai_epi32(g1x8a, in_sh);
            g1x8a = av_clip_int16_avx(g1x8a);

            // b = av_clip_int16((y * cy + cbu * u + in_rnd) >> in_sh);
            b0x8a = _mm256_add_epi32(b0x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cbu)));
            b0x8a = _mm256_add_epi32(b0x8a, rndx8);
            b0x8a = _mm256_srai_epi32(b0x8a, in_sh);
            b0x8a = av_clip_int16_avx(b0x8a);

            b1x8a = _mm256_add_epi32(b1x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cbu)));
            b1x8a = _mm256_add_epi32(b1x8a, rndx8);
            b1x8a = _mm256_srai_epi32(b1x8a, in_sh);
            b1x8a = av_clip_int16_avx(b1x8a);

            r0x8b = g0x8b = b0x8b = _mm256_mullo_epi32(y0x8b, cyx8);
            r0x8b = _mm256_add_epi32(r0x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(crv)));
            r0x8b = _mm256_add_epi32(r0x8b, rndx8);
            r0x8b = _mm256_srai_epi32(r0x8b, in_sh);
            r0x8b = av_clip_int16_avx(r0x8b);

            r1x8b = g1x8b = b1x8b = _mm256_mullo_epi32(y1x8b, cyx8);
            r1x8b = _mm256_add_epi32(r1x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(crv)));
            r1x8b = _mm256_add_epi32(r1x8b, rndx8);
            r1x8b = _mm256_srai_epi32(r1x8b, in_sh);
            r1x8b = av_clip_int16_avx(r1x8b);

            g0x8b = _mm256_add_epi32(g0x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cgu)));
            g0x8b = _mm256_add_epi32(g0x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(cgv)));
            g0x8b = _mm256_add_epi32(g0x8b, rndx8);
            g0x8b = _mm256_srai_epi32(g0x8b, in_sh);
            g0x8b = av_clip_int16_avx(g0x8b);

            g1x8b = _mm256_add_epi32(g1x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cgu)));
            g1x8b = _mm256_add_epi32(g1x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(cgv)));
            g1x8b = _mm256_add_epi32(g1x8b, rndx8);
            g1x8b = _mm256_srai_epi32(g1x8b, in_sh);
            g1x8b = av_clip_int16_avx(g1x8b);

            b0x8b = _mm256_add_epi32(b0x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cbu)));
            b0x8b = _mm256_add_epi32(b0x8b, rndx8);
            b0x8b = _mm256_srai_epi32(b0x8b, in_sh);
            b0x8b = av_clip_int16_avx(b0x8b);

            b1x8b = _mm256_add_epi32(b1x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cbu)));
            b1x8b = _mm256_add_epi32(b1x8b, rndx8);
            b1x8b = _mm256_srai_epi32(b1x8b, in_sh);
            b1x8b = av_clip_int16_avx(b1x8b);

            tonemap_int32x8_avx(r0x8a, g0x8a, b0x8a, r, g, b,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r1x8a, g1x8a, b1x8a, r1, g1, b1,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r0x8b, g0x8b, b0x8b, &r[8], &g[8], &b[8],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r1x8b, g1x8b, b1x8b, &r1[8], &g1[8], &b1[8],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);

            r0ox16 = _mm256_lddqu_si256((const __m256i_u *)r);
            g0ox16 = _mm256_lddqu_si256((const __m256i_u *)g);
            b0ox16 = _mm256_lddqu_si256((const __m256i_u *)b);

            roax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r0ox16));
            goax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(g0ox16));
            boax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0ox16));

            robx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r0ox16, 1));
            gobx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(g0ox16, 1));
            bobx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0ox16, 1));

            yoax8 = _mm256_mullo_epi32(roax8, _mm256_set1_epi32(cry));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_mullo_epi32(goax8, _mm256_set1_epi32(cgy)));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_mullo_epi32(boax8, _mm256_set1_epi32(cby)));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_set1_epi32(out_rnd));
            yoax8 = _mm256_srai_epi32(yoax8, out_sh);
            yoax8 = _mm256_add_epi32(yoax8, _mm256_set1_epi32(params->out_yuv_off));

            yobx8 = _mm256_mullo_epi32(robx8, _mm256_set1_epi32(cry));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_mullo_epi32(gobx8, _mm256_set1_epi32(cgy)));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_mullo_epi32(bobx8, _mm256_set1_epi32(cby)));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_set1_epi32(out_rnd));
            yobx8 = _mm256_srai_epi32(yobx8, out_sh);
            yobx8 = _mm256_add_epi32(yobx8, _mm256_set1_epi32(params->out_yuv_off));

            y0ox16 = _mm256_packus_epi32(yoax8, yobx8);
            y0ox16 = _mm256_permute4x64_epi64(y0ox16, _MM_SHUFFLE(3, 1, 2, 0));
            _mm256_storeu_si256((__m256i_u *) &dsty[x], y0ox16);

            r1ox16 = _mm256_lddqu_si256((const __m256i_u *)r1);
            g1ox16 = _mm256_lddqu_si256((const __m256i_u *)g1);
            b1ox16 = _mm256_lddqu_si256((const __m256i_u *)b1);

            r1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r1ox16));
            g1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(g1ox16));
            b1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1ox16));

            r1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r1ox16, 1));
            g1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(g1ox16, 1));
            b1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1ox16, 1));

            y1oax8 = _mm256_mullo_epi32(r1oax8, _mm256_set1_epi32(cry));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_mullo_epi32(g1oax8, _mm256_set1_epi32(cgy)));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_mullo_epi32(b1oax8, _mm256_set1_epi32(cby)));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_set1_epi32(out_rnd));
            y1oax8 = _mm256_srai_epi32(y1oax8, out_sh);
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_set1_epi32(params->out_yuv_off));

            y1obx8 = _mm256_mullo_epi32(r1obx8, _mm256_set1_epi32(cry));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_mullo_epi32(g1obx8, _mm256_set1_epi32(cgy)));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_mullo_epi32(b1obx8, _mm256_set1_epi32(cby)));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_set1_epi32(out_rnd));
            y1obx8 = _mm256_srai_epi32(y1obx8, out_sh);
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_set1_epi32(params->out_yuv_off));

            y1ox16 = _mm256_packus_epi32(y1oax8, y1obx8);
            y1ox16 = _mm256_permute4x64_epi64(y1ox16, _MM_SHUFFLE(3, 1, 2, 0));
            _mm256_storeu_si256((__m256i_u *) &dsty[x + dstlinesize[0] / 2], y1ox16);

            ravgx8 = _mm256_hadd_epi32(roax8, robx8);
            ravgx8 = _mm256_add_epi32(ravgx8, _mm256_hadd_epi32(r1oax8, r1obx8));
            ravgx8 = _mm256_permute4x64_epi64(ravgx8, _MM_SHUFFLE(3, 1, 2, 0));
            ravgx8 = _mm256_add_epi32(ravgx8, _mm256_set1_epi32(2));
            ravgx8 = _mm256_srai_epi32(ravgx8, 2);

            gavgx8 = _mm256_hadd_epi32(goax8, gobx8);
            gavgx8 = _mm256_add_epi32(gavgx8, _mm256_hadd_epi32(g1oax8, g1obx8));
            gavgx8 = _mm256_permute4x64_epi64(gavgx8, _MM_SHUFFLE(3, 1, 2, 0));
            gavgx8 = _mm256_add_epi32(gavgx8, _mm256_set1_epi32(2));
            gavgx8 = _mm256_srai_epi32(gavgx8, 2);

            bavgx8 = _mm256_hadd_epi32(boax8, bobx8);
            bavgx8 = _mm256_add_epi32(bavgx8, _mm256_hadd_epi32(b1oax8, b1obx8));
            bavgx8 = _mm256_permute4x64_epi64(bavgx8, _MM_SHUFFLE(3, 1, 2, 0));
            bavgx8 = _mm256_add_epi32(bavgx8, _mm256_set1_epi32(2));
            bavgx8 = _mm256_srai_epi32(bavgx8, 2);

            uox8 = _mm256_add_epi32(_mm256_set1_epi32(out_rnd), _mm256_mullo_epi32(ravgx8, _mm256_set1_epi32(cru)));
            uox8 = _mm256_add_epi32(uox8, _mm256_mullo_epi32(gavgx8, _mm256_set1_epi32(ocgu)));
            uox8 = _mm256_add_epi32(uox8, _mm256_mullo_epi32(bavgx8, _mm256_set1_epi32(cburv)));
            uox8 = _mm256_srai_epi32(uox8, out_sh);
            uox8 = _mm256_add_epi32(uox8, _mm256_set1_epi32(out_uv_offset));
            uox8 = _mm256_packus_epi32(uox8, _mm256_setzero_si256());
            uox8 = _mm256_permute4x64_epi64(uox8, _MM_SHUFFLE(3, 1, 2, 0));
            _mm_storeu_si128((__m128i_u *) &dstu[x >> 1], _mm256_castsi256_si128(uox8));

            vox8 = _mm256_add_epi32(_mm256_set1_epi32(out_rnd), _mm256_mullo_epi32(ravgx8, _mm256_set1_epi32(cburv)));
            vox8 = _mm256_add_epi32(vox8, _mm256_mullo_epi32(gavgx8, _mm256_set1_epi32(ocgv)));
            vox8 = _mm256_add_epi32(vox8, _mm256_mullo_epi32(bavgx8, _mm256_set1_epi32(cbv)));
            vox8 = _mm256_srai_epi32(vox8, out_sh);
            vox8 = _mm256_add_epi32(vox8, _mm256_set1_epi32(out_uv_offset));
            vox8 = _mm256_packus_epi32(vox8, _mm256_setzero_si256());
            vox8 = _mm256_permute4x64_epi64(vox8, _MM_SHUFFLE(3, 1, 2, 0));
            _mm_storeu_si128((__m128i_u *) &dstv[x >> 1], _mm256_castsi256_si128(vox8));
        }
    }

    // Process remaining pixels cannot fill the full simd register with scalar version
    if (remainw) {
        int offset = width & (int)0xfffffff0;
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
#endif // ENABLE_TONEMAPX_AVX_INTRINSICS
}

X86_64_V3 void tonemap_frame_p010_2_nv12_avx(uint8_t *dsty, uint8_t *dstuv,
                                             const uint16_t *srcy, const uint16_t *srcuv,
                                             const int *dstlinesize, const int *srclinesize,
                                             int dstdepth, int srcdepth,
                                             int width, int height,
                                             const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_AVX_INTRINSICS
    uint8_t *rdsty = dsty;
    uint8_t *rdstuv = dstuv;
    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcuv = srcuv;
    int rheight = height;
    // not zero when not divisible by 16
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 14;

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

    int16_t r[16], g[16], b[16];
    int16_t r1[16], g1[16], b1[16];
    __m256i in_yuv_offx8 = _mm256_set1_epi32(params->in_yuv_off);
    __m256i in_uv_offx8 = _mm256_set1_epi32(in_uv_offset);
    __m256i cyx8 = _mm256_set1_epi32(cy);
    __m256i rndx8 = _mm256_set1_epi32(in_rnd);

    __m256i uvx16, uvx8a, uvx8b;
    __m256i y0x16, y1x16;
    __m256i y0x8a, y0x8b, y1x8a, y1x8b, ux8a, ux8b, vx8a, vx8b;
    __m256i r0x8a, g0x8a, b0x8a, r0x8b, g0x8b, b0x8b;
    __m256i r1x8a, g1x8a, b1x8a, r1x8b, g1x8b, b1x8b;

    __m256i r0ox16, g0ox16, b0ox16;
    __m256i y0ox16;
    __m256i roax8, robx8, goax8, gobx8, boax8, bobx8;
    __m256i yoax8, yobx8;

    __m256i r1ox16, g1ox16, b1ox16;
    __m256i y1ox16;
    __m256i r1oax8, r1obx8, g1oax8, g1obx8, b1oax8, b1obx8;
    __m256i y1oax8, y1obx8, uvoax8, uvobx8, uvox16;
    __m256i uox8, vox8, ravgx8, gavgx8, bavgx8;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0] * 2, dstuv += dstlinesize[1],
                       srcy += srclinesize[0], srcuv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 4; xx++) {
            int x = xx << 4;

            y0x16 = _mm256_lddqu_si256((__m256i*)(srcy + x));
            y1x16 = _mm256_lddqu_si256((__m256i*)(srcy + (srclinesize[0] / 2 + x)));
            uvx16 = _mm256_lddqu_si256((__m256i*)(srcuv + x));

            // shift to low10bits for 10bit input
            y0x16 = _mm256_srli_epi16(y0x16, TEN_BIT_BIPLANAR_SHIFT);
            y1x16 = _mm256_srli_epi16(y1x16, TEN_BIT_BIPLANAR_SHIFT);
            uvx16 = _mm256_srli_epi16(uvx16, TEN_BIT_BIPLANAR_SHIFT);

            y0x8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y0x16));
            y0x8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y0x16, 1));
            y1x8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y1x16));
            y1x8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y1x16, 1));
            uvx8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(uvx16));
            uvx8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(uvx16, 1));
            y0x8a = _mm256_sub_epi32(y0x8a, in_yuv_offx8);
            y1x8a = _mm256_sub_epi32(y1x8a, in_yuv_offx8);
            y0x8b = _mm256_sub_epi32(y0x8b, in_yuv_offx8);
            y1x8b = _mm256_sub_epi32(y1x8b, in_yuv_offx8);
            uvx8a = _mm256_sub_epi32(uvx8a, in_uv_offx8);
            uvx8b = _mm256_sub_epi32(uvx8b, in_uv_offx8);

            ux8a = _mm256_shuffle_epi32(uvx8a, _MM_SHUFFLE(2, 2, 0, 0));
            ux8b = _mm256_shuffle_epi32(uvx8b, _MM_SHUFFLE(2, 2, 0, 0));
            vx8a = _mm256_shuffle_epi32(uvx8a, _MM_SHUFFLE(3, 3, 1, 1));
            vx8b = _mm256_shuffle_epi32(uvx8b, _MM_SHUFFLE(3, 3, 1, 1));

            // r = av_clip_int16((y * cy + crv * v + in_rnd) >> in_sh);
            r0x8a = g0x8a = b0x8a = _mm256_mullo_epi32(y0x8a, cyx8);
            r0x8a = _mm256_add_epi32(r0x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(crv)));
            r0x8a = _mm256_add_epi32(r0x8a, rndx8);
            r0x8a = _mm256_srai_epi32(r0x8a, in_sh);
            r0x8a = av_clip_int16_avx(r0x8a);

            r1x8a = g1x8a = b1x8a = _mm256_mullo_epi32(y1x8a, cyx8);
            r1x8a = _mm256_add_epi32(r1x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(crv)));
            r1x8a = _mm256_add_epi32(r1x8a, rndx8);
            r1x8a = _mm256_srai_epi32(r1x8a, in_sh);
            r1x8a = av_clip_int16_avx(r1x8a);

            // g = av_clip_int16((y * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g0x8a = _mm256_add_epi32(g0x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cgu)));
            g0x8a = _mm256_add_epi32(g0x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(cgv)));
            g0x8a = _mm256_add_epi32(g0x8a, rndx8);
            g0x8a = _mm256_srai_epi32(g0x8a, in_sh);
            g0x8a = av_clip_int16_avx(g0x8a);

            g1x8a = _mm256_add_epi32(g1x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cgu)));
            g1x8a = _mm256_add_epi32(g1x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(cgv)));
            g1x8a = _mm256_add_epi32(g1x8a, rndx8);
            g1x8a = _mm256_srai_epi32(g1x8a, in_sh);
            g1x8a = av_clip_int16_avx(g1x8a);

            // b = av_clip_int16((y * cy + cbu * u + in_rnd) >> in_sh);
            b0x8a = _mm256_add_epi32(b0x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cbu)));
            b0x8a = _mm256_add_epi32(b0x8a, rndx8);
            b0x8a = _mm256_srai_epi32(b0x8a, in_sh);
            b0x8a = av_clip_int16_avx(b0x8a);

            b1x8a = _mm256_add_epi32(b1x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cbu)));
            b1x8a = _mm256_add_epi32(b1x8a, rndx8);
            b1x8a = _mm256_srai_epi32(b1x8a, in_sh);
            b1x8a = av_clip_int16_avx(b1x8a);

            r0x8b = g0x8b = b0x8b = _mm256_mullo_epi32(y0x8b, cyx8);
            r0x8b = _mm256_add_epi32(r0x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(crv)));
            r0x8b = _mm256_add_epi32(r0x8b, rndx8);
            r0x8b = _mm256_srai_epi32(r0x8b, in_sh);
            r0x8b = av_clip_int16_avx(r0x8b);

            r1x8b = g1x8b = b1x8b = _mm256_mullo_epi32(y1x8b, cyx8);
            r1x8b = _mm256_add_epi32(r1x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(crv)));
            r1x8b = _mm256_add_epi32(r1x8b, rndx8);
            r1x8b = _mm256_srai_epi32(r1x8b, in_sh);
            r1x8b = av_clip_int16_avx(r1x8b);

            g0x8b = _mm256_add_epi32(g0x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cgu)));
            g0x8b = _mm256_add_epi32(g0x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(cgv)));
            g0x8b = _mm256_add_epi32(g0x8b, rndx8);
            g0x8b = _mm256_srai_epi32(g0x8b, in_sh);
            g0x8b = av_clip_int16_avx(g0x8b);

            g1x8b = _mm256_add_epi32(g1x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cgu)));
            g1x8b = _mm256_add_epi32(g1x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(cgv)));
            g1x8b = _mm256_add_epi32(g1x8b, rndx8);
            g1x8b = _mm256_srai_epi32(g1x8b, in_sh);
            g1x8b = av_clip_int16_avx(g1x8b);

            b0x8b = _mm256_add_epi32(b0x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cbu)));
            b0x8b = _mm256_add_epi32(b0x8b, rndx8);
            b0x8b = _mm256_srai_epi32(b0x8b, in_sh);
            b0x8b = av_clip_int16_avx(b0x8b);

            b1x8b = _mm256_add_epi32(b1x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cbu)));
            b1x8b = _mm256_add_epi32(b1x8b, rndx8);
            b1x8b = _mm256_srai_epi32(b1x8b, in_sh);
            b1x8b = av_clip_int16_avx(b1x8b);

            tonemap_int32x8_avx(r0x8a, g0x8a, b0x8a, r, g, b,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r1x8a, g1x8a, b1x8a, r1, g1, b1,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r0x8b, g0x8b, b0x8b, &r[8], &g[8], &b[8],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r1x8b, g1x8b, b1x8b, &r1[8], &g1[8], &b1[8],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);

            r0ox16 = _mm256_lddqu_si256((const __m256i_u *)r);
            g0ox16 = _mm256_lddqu_si256((const __m256i_u *)g);
            b0ox16 = _mm256_lddqu_si256((const __m256i_u *)b);

            roax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r0ox16));
            goax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(g0ox16));
            boax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0ox16));

            robx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r0ox16, 1));
            gobx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(g0ox16, 1));
            bobx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0ox16, 1));

            yoax8 = _mm256_mullo_epi32(roax8, _mm256_set1_epi32(cry));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_mullo_epi32(goax8, _mm256_set1_epi32(cgy)));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_mullo_epi32(boax8, _mm256_set1_epi32(cby)));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_set1_epi32(out_rnd));
            yoax8 = _mm256_srai_epi32(yoax8, out_sh);
            yoax8 = _mm256_add_epi32(yoax8, _mm256_set1_epi32(params->out_yuv_off));

            yobx8 = _mm256_mullo_epi32(robx8, _mm256_set1_epi32(cry));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_mullo_epi32(gobx8, _mm256_set1_epi32(cgy)));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_mullo_epi32(bobx8, _mm256_set1_epi32(cby)));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_set1_epi32(out_rnd));
            yobx8 = _mm256_srai_epi32(yobx8, out_sh);
            yobx8 = _mm256_add_epi32(yobx8, _mm256_set1_epi32(params->out_yuv_off));

            y0ox16 = _mm256_packs_epi32(yoax8, yobx8);
            y0ox16 = _mm256_permute4x64_epi64(y0ox16, _MM_SHUFFLE(3, 1, 2, 0));
            _mm_storeu_si128((__m128i_u *) &dsty[x], _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(y0ox16, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0))));

            r1ox16 = _mm256_lddqu_si256((const __m256i_u *)r1);
            g1ox16 = _mm256_lddqu_si256((const __m256i_u *)g1);
            b1ox16 = _mm256_lddqu_si256((const __m256i_u *)b1);

            r1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r1ox16));
            g1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(g1ox16));
            b1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1ox16));

            r1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r1ox16, 1));
            g1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(g1ox16, 1));
            b1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1ox16, 1));

            y1oax8 = _mm256_mullo_epi32(r1oax8, _mm256_set1_epi32(cry));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_mullo_epi32(g1oax8, _mm256_set1_epi32(cgy)));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_mullo_epi32(b1oax8, _mm256_set1_epi32(cby)));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_set1_epi32(out_rnd));
            y1oax8 = _mm256_srai_epi32(y1oax8, out_sh);
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_set1_epi32(params->out_yuv_off));

            y1obx8 = _mm256_mullo_epi32(r1obx8, _mm256_set1_epi32(cry));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_mullo_epi32(g1obx8, _mm256_set1_epi32(cgy)));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_mullo_epi32(b1obx8, _mm256_set1_epi32(cby)));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_set1_epi32(out_rnd));
            y1obx8 = _mm256_srai_epi32(y1obx8, out_sh);
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_set1_epi32(params->out_yuv_off));

            y1ox16 = _mm256_packs_epi32(y1oax8, y1obx8);
            y1ox16 = _mm256_permute4x64_epi64(y1ox16, _MM_SHUFFLE(3, 1, 2, 0));
            _mm_storeu_si128((__m128i_u *) &dsty[x + dstlinesize[0]], _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(y1ox16, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0))));

            ravgx8 = _mm256_hadd_epi32(roax8, robx8);
            ravgx8 = _mm256_add_epi32(ravgx8, _mm256_hadd_epi32(r1oax8, r1obx8));
            ravgx8 = _mm256_permute4x64_epi64(ravgx8, _MM_SHUFFLE(3, 1, 2, 0));
            ravgx8 = _mm256_add_epi32(ravgx8, _mm256_set1_epi32(2));
            ravgx8 = _mm256_srai_epi32(ravgx8, 2);

            gavgx8 = _mm256_hadd_epi32(goax8, gobx8);
            gavgx8 = _mm256_add_epi32(gavgx8, _mm256_hadd_epi32(g1oax8, g1obx8));
            gavgx8 = _mm256_permute4x64_epi64(gavgx8, _MM_SHUFFLE(3, 1, 2, 0));
            gavgx8 = _mm256_add_epi32(gavgx8, _mm256_set1_epi32(2));
            gavgx8 = _mm256_srai_epi32(gavgx8, 2);

            bavgx8 = _mm256_hadd_epi32(boax8, bobx8);
            bavgx8 = _mm256_add_epi32(bavgx8, _mm256_hadd_epi32(b1oax8, b1obx8));
            bavgx8 = _mm256_permute4x64_epi64(bavgx8, _MM_SHUFFLE(3, 1, 2, 0));
            bavgx8 = _mm256_add_epi32(bavgx8, _mm256_set1_epi32(2));
            bavgx8 = _mm256_srai_epi32(bavgx8, 2);

            uox8 = _mm256_add_epi32(_mm256_set1_epi32(out_rnd), _mm256_mullo_epi32(ravgx8, _mm256_set1_epi32(cru)));
            uox8 = _mm256_add_epi32(uox8, _mm256_mullo_epi32(gavgx8, _mm256_set1_epi32(ocgu)));
            uox8 = _mm256_add_epi32(uox8, _mm256_mullo_epi32(bavgx8, _mm256_set1_epi32(cburv)));
            uox8 = _mm256_srai_epi32(uox8, out_sh);
            uox8 = _mm256_add_epi32(uox8, _mm256_set1_epi32(out_uv_offset));

            vox8 = _mm256_add_epi32(_mm256_set1_epi32(out_rnd), _mm256_mullo_epi32(ravgx8, _mm256_set1_epi32(cburv)));
            vox8 = _mm256_add_epi32(vox8, _mm256_mullo_epi32(gavgx8, _mm256_set1_epi32(ocgv)));
            vox8 = _mm256_add_epi32(vox8, _mm256_mullo_epi32(bavgx8, _mm256_set1_epi32(cbv)));
            vox8 = _mm256_srai_epi32(vox8, out_sh);
            vox8 = _mm256_add_epi32(vox8, _mm256_set1_epi32(out_uv_offset));

            uvoax8 = _mm256_unpacklo_epi32(uox8, vox8);
            uvobx8 = _mm256_unpackhi_epi32(uox8, vox8);
            uvox16 = _mm256_packs_epi32(uvoax8, uvobx8);
            _mm_storeu_si128((__m128i_u *) &dstuv[x], _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(uvox16, _mm256_setzero_si256()), _MM_SHUFFLE(3, 1, 2, 0))));
        }
    }

    // Process remaining pixels cannot fill the full simd register with scalar version
    if (remainw) {
        int offset = width & (int)0xfffffff0;
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
#endif // ENABLE_TONEMAPX_AVX_INTRINSICS
}

X86_64_V3 void tonemap_frame_p010_2_p010_avx(uint16_t *dsty, uint16_t *dstuv,
                                             const uint16_t *srcy, const uint16_t *srcuv,
                                             const int *dstlinesize, const int *srclinesize,
                                             int dstdepth, int srcdepth,
                                             int width, int height,
                                             const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_AVX_INTRINSICS
    uint16_t *rdsty = dsty;
    uint16_t *rdstuv = dstuv;
    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcuv = srcuv;
    int rheight = height;
    // not zero when not divisible by 8
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 14;

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

    int16_t r[16], g[16], b[16];
    int16_t r1[16], g1[16], b1[16];
    __m256i in_yuv_offx8 = _mm256_set1_epi32(params->in_yuv_off);
    __m256i in_uv_offx8 = _mm256_set1_epi32(in_uv_offset);
    __m256i cyx8 = _mm256_set1_epi32(cy);
    __m256i rndx8 = _mm256_set1_epi32(in_rnd);

    __m256i r0ox16, g0ox16, b0ox16;
    __m256i y0ox16;
    __m256i roax8, robx8, goax8, gobx8, boax8, bobx8;
    __m256i yoax8, yobx8;
    __m256i uvx16, uvx8a, uvx8b;
    __m256i y0x16, y1x16;
    __m256i y0x8a, y0x8b, y1x8a, y1x8b, ux8a, ux8b, vx8a, vx8b;
    __m256i r0x8a, g0x8a, b0x8a, r0x8b, g0x8b, b0x8b;
    __m256i r1x8a, g1x8a, b1x8a, r1x8b, g1x8b, b1x8b;

    __m256i r1ox16, g1ox16, b1ox16;
    __m256i y1ox16;
    __m256i r1oax8, r1obx8, g1oax8, g1obx8, b1oax8, b1obx8;
    __m256i y1oax8, y1obx8, uvoax8, uvobx8, uvox16;
    __m256i uox8, vox8, ravgx8, gavgx8, bavgx8;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstuv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcuv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 4; xx++) {
            int x = xx << 4;

            y0x16 = _mm256_lddqu_si256((__m256i*)(srcy + x));
            y1x16 = _mm256_lddqu_si256((__m256i*)(srcy + (srclinesize[0] / 2 + x)));
            uvx16 = _mm256_lddqu_si256((__m256i*)(srcuv + x));

            // shift to low10bits for 10bit input
            y0x16 = _mm256_srli_epi16(y0x16, TEN_BIT_BIPLANAR_SHIFT);
            y1x16 = _mm256_srli_epi16(y1x16, TEN_BIT_BIPLANAR_SHIFT);
            uvx16 = _mm256_srli_epi16(uvx16, TEN_BIT_BIPLANAR_SHIFT);

            y0x8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y0x16));
            y0x8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y0x16, 1));
            y1x8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y1x16));
            y1x8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y1x16, 1));
            uvx8a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(uvx16));
            uvx8b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(uvx16, 1));
            y0x8a = _mm256_sub_epi32(y0x8a, in_yuv_offx8);
            y1x8a = _mm256_sub_epi32(y1x8a, in_yuv_offx8);
            y0x8b = _mm256_sub_epi32(y0x8b, in_yuv_offx8);
            y1x8b = _mm256_sub_epi32(y1x8b, in_yuv_offx8);
            uvx8a = _mm256_sub_epi32(uvx8a, in_uv_offx8);
            uvx8b = _mm256_sub_epi32(uvx8b, in_uv_offx8);

            ux8a = _mm256_shuffle_epi32(uvx8a, _MM_SHUFFLE(2, 2, 0, 0));
            ux8b = _mm256_shuffle_epi32(uvx8b, _MM_SHUFFLE(2, 2, 0, 0));
            vx8a = _mm256_shuffle_epi32(uvx8a, _MM_SHUFFLE(3, 3, 1, 1));
            vx8b = _mm256_shuffle_epi32(uvx8b, _MM_SHUFFLE(3, 3, 1, 1));

            // r = av_clip_int16((y * cy + crv * v + in_rnd) >> in_sh);
            r0x8a = g0x8a = b0x8a = _mm256_mullo_epi32(y0x8a, cyx8);
            r0x8a = _mm256_add_epi32(r0x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(crv)));
            r0x8a = _mm256_add_epi32(r0x8a, rndx8);
            r0x8a = _mm256_srai_epi32(r0x8a, in_sh);
            r0x8a = av_clip_int16_avx(r0x8a);

            r1x8a = g1x8a = b1x8a = _mm256_mullo_epi32(y1x8a, cyx8);
            r1x8a = _mm256_add_epi32(r1x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(crv)));
            r1x8a = _mm256_add_epi32(r1x8a, rndx8);
            r1x8a = _mm256_srai_epi32(r1x8a, in_sh);
            r1x8a = av_clip_int16_avx(r1x8a);

            // g = av_clip_int16((y * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g0x8a = _mm256_add_epi32(g0x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cgu)));
            g0x8a = _mm256_add_epi32(g0x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(cgv)));
            g0x8a = _mm256_add_epi32(g0x8a, rndx8);
            g0x8a = _mm256_srai_epi32(g0x8a, in_sh);
            g0x8a = av_clip_int16_avx(g0x8a);

            g1x8a = _mm256_add_epi32(g1x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cgu)));
            g1x8a = _mm256_add_epi32(g1x8a, _mm256_mullo_epi32(vx8a, _mm256_set1_epi32(cgv)));
            g1x8a = _mm256_add_epi32(g1x8a, rndx8);
            g1x8a = _mm256_srai_epi32(g1x8a, in_sh);
            g1x8a = av_clip_int16_avx(g1x8a);

            // b = av_clip_int16((y * cy + cbu * u + in_rnd) >> in_sh);
            b0x8a = _mm256_add_epi32(b0x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cbu)));
            b0x8a = _mm256_add_epi32(b0x8a, rndx8);
            b0x8a = _mm256_srai_epi32(b0x8a, in_sh);
            b0x8a = av_clip_int16_avx(b0x8a);

            b1x8a = _mm256_add_epi32(b1x8a, _mm256_mullo_epi32(ux8a, _mm256_set1_epi32(cbu)));
            b1x8a = _mm256_add_epi32(b1x8a, rndx8);
            b1x8a = _mm256_srai_epi32(b1x8a, in_sh);
            b1x8a = av_clip_int16_avx(b1x8a);

            r0x8b = g0x8b = b0x8b = _mm256_mullo_epi32(y0x8b, cyx8);
            r0x8b = _mm256_add_epi32(r0x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(crv)));
            r0x8b = _mm256_add_epi32(r0x8b, rndx8);
            r0x8b = _mm256_srai_epi32(r0x8b, in_sh);
            r0x8b = av_clip_int16_avx(r0x8b);

            r1x8b = g1x8b = b1x8b = _mm256_mullo_epi32(y1x8b, cyx8);
            r1x8b = _mm256_add_epi32(r1x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(crv)));
            r1x8b = _mm256_add_epi32(r1x8b, rndx8);
            r1x8b = _mm256_srai_epi32(r1x8b, in_sh);
            r1x8b = av_clip_int16_avx(r1x8b);

            g0x8b = _mm256_add_epi32(g0x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cgu)));
            g0x8b = _mm256_add_epi32(g0x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(cgv)));
            g0x8b = _mm256_add_epi32(g0x8b, rndx8);
            g0x8b = _mm256_srai_epi32(g0x8b, in_sh);
            g0x8b = av_clip_int16_avx(g0x8b);

            g1x8b = _mm256_add_epi32(g1x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cgu)));
            g1x8b = _mm256_add_epi32(g1x8b, _mm256_mullo_epi32(vx8b, _mm256_set1_epi32(cgv)));
            g1x8b = _mm256_add_epi32(g1x8b, rndx8);
            g1x8b = _mm256_srai_epi32(g1x8b, in_sh);
            g1x8b = av_clip_int16_avx(g1x8b);

            b0x8b = _mm256_add_epi32(b0x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cbu)));
            b0x8b = _mm256_add_epi32(b0x8b, rndx8);
            b0x8b = _mm256_srai_epi32(b0x8b, in_sh);
            b0x8b = av_clip_int16_avx(b0x8b);

            b1x8b = _mm256_add_epi32(b1x8b, _mm256_mullo_epi32(ux8b, _mm256_set1_epi32(cbu)));
            b1x8b = _mm256_add_epi32(b1x8b, rndx8);
            b1x8b = _mm256_srai_epi32(b1x8b, in_sh);
            b1x8b = av_clip_int16_avx(b1x8b);

            tonemap_int32x8_avx(r0x8a, g0x8a, b0x8a, r, g, b,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r1x8a, g1x8a, b1x8a, r1, g1, b1,
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r0x8b, g0x8b, b0x8b, &r[8], &g[8], &b[8],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);
            tonemap_int32x8_avx(r1x8b, g1x8b, b1x8b, &r1[8], &g1[8], &b1[8],
                                params->lin_lut, params->tonemap_lut, params->delin_lut,
                                params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                params->rgb2rgb_passthrough);

            r0ox16 = _mm256_lddqu_si256((const __m256i_u *)r);
            g0ox16 = _mm256_lddqu_si256((const __m256i_u *)g);
            b0ox16 = _mm256_lddqu_si256((const __m256i_u *)b);

            roax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r0ox16));
            goax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(g0ox16));
            boax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0ox16));

            robx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r0ox16, 1));
            gobx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(g0ox16, 1));
            bobx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0ox16, 1));

            yoax8 = _mm256_mullo_epi32(roax8, _mm256_set1_epi32(cry));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_mullo_epi32(goax8, _mm256_set1_epi32(cgy)));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_mullo_epi32(boax8, _mm256_set1_epi32(cby)));
            yoax8 = _mm256_add_epi32(yoax8, _mm256_set1_epi32(out_rnd));
            yoax8 = _mm256_srai_epi32(yoax8, out_sh);
            yoax8 = _mm256_add_epi32(yoax8, _mm256_set1_epi32(params->out_yuv_off));
            yoax8 = _mm256_slli_epi32(yoax8, out_sh2);

            yobx8 = _mm256_mullo_epi32(robx8, _mm256_set1_epi32(cry));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_mullo_epi32(gobx8, _mm256_set1_epi32(cgy)));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_mullo_epi32(bobx8, _mm256_set1_epi32(cby)));
            yobx8 = _mm256_add_epi32(yobx8, _mm256_set1_epi32(out_rnd));
            yobx8 = _mm256_srai_epi32(yobx8, out_sh);
            yobx8 = _mm256_add_epi32(yobx8, _mm256_set1_epi32(params->out_yuv_off));
            yobx8 = _mm256_slli_epi32(yobx8, out_sh2);

            y0ox16 = _mm256_packus_epi32(yoax8, yobx8);
            y0ox16 = _mm256_permute4x64_epi64(y0ox16, _MM_SHUFFLE(3, 1, 2, 0));
            _mm256_storeu_si256((__m256i_u *) &dsty[x], y0ox16);

            r1ox16 = _mm256_lddqu_si256((const __m256i_u *)r1);
            g1ox16 = _mm256_lddqu_si256((const __m256i_u *)g1);
            b1ox16 = _mm256_lddqu_si256((const __m256i_u *)b1);

            r1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r1ox16));
            g1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(g1ox16));
            b1oax8 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1ox16));

            r1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(r1ox16, 1));
            g1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(g1ox16, 1));
            b1obx8 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1ox16, 1));

            y1oax8 = _mm256_mullo_epi32(r1oax8, _mm256_set1_epi32(cry));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_mullo_epi32(g1oax8, _mm256_set1_epi32(cgy)));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_mullo_epi32(b1oax8, _mm256_set1_epi32(cby)));
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_set1_epi32(out_rnd));
            y1oax8 = _mm256_srai_epi32(y1oax8, out_sh);
            y1oax8 = _mm256_add_epi32(y1oax8, _mm256_set1_epi32(params->out_yuv_off));
            y1oax8 = _mm256_slli_epi32(y1oax8, out_sh2);

            y1obx8 = _mm256_mullo_epi32(r1obx8, _mm256_set1_epi32(cry));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_mullo_epi32(g1obx8, _mm256_set1_epi32(cgy)));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_mullo_epi32(b1obx8, _mm256_set1_epi32(cby)));
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_set1_epi32(out_rnd));
            y1obx8 = _mm256_srai_epi32(y1obx8, out_sh);
            y1obx8 = _mm256_add_epi32(y1obx8, _mm256_set1_epi32(params->out_yuv_off));
            y1obx8 = _mm256_slli_epi32(y1obx8, out_sh2);

            y1ox16 = _mm256_packus_epi32(y1oax8, y1obx8);
            y1ox16 = _mm256_permute4x64_epi64(y1ox16, _MM_SHUFFLE(3, 1, 2, 0));
            _mm256_storeu_si256((__m256i_u *) &dsty[x + dstlinesize[0] / 2], y1ox16);

            ravgx8 = _mm256_hadd_epi32(roax8, robx8);
            ravgx8 = _mm256_add_epi32(ravgx8, _mm256_hadd_epi32(r1oax8, r1obx8));
            ravgx8 = _mm256_permute4x64_epi64(ravgx8, _MM_SHUFFLE(3, 1, 2, 0));
            ravgx8 = _mm256_add_epi32(ravgx8, _mm256_set1_epi32(2));
            ravgx8 = _mm256_srai_epi32(ravgx8, 2);

            gavgx8 = _mm256_hadd_epi32(goax8, gobx8);
            gavgx8 = _mm256_add_epi32(gavgx8, _mm256_hadd_epi32(g1oax8, g1obx8));
            gavgx8 = _mm256_permute4x64_epi64(gavgx8, _MM_SHUFFLE(3, 1, 2, 0));
            gavgx8 = _mm256_add_epi32(gavgx8, _mm256_set1_epi32(2));
            gavgx8 = _mm256_srai_epi32(gavgx8, 2);

            bavgx8 = _mm256_hadd_epi32(boax8, bobx8);
            bavgx8 = _mm256_add_epi32(bavgx8, _mm256_hadd_epi32(b1oax8, b1obx8));
            bavgx8 = _mm256_permute4x64_epi64(bavgx8, _MM_SHUFFLE(3, 1, 2, 0));
            bavgx8 = _mm256_add_epi32(bavgx8, _mm256_set1_epi32(2));
            bavgx8 = _mm256_srai_epi32(bavgx8, 2);

            uox8 = _mm256_add_epi32(_mm256_set1_epi32(out_rnd), _mm256_mullo_epi32(ravgx8, _mm256_set1_epi32(cru)));
            uox8 = _mm256_add_epi32(uox8, _mm256_mullo_epi32(gavgx8, _mm256_set1_epi32(ocgu)));
            uox8 = _mm256_add_epi32(uox8, _mm256_mullo_epi32(bavgx8, _mm256_set1_epi32(cburv)));
            uox8 = _mm256_srai_epi32(uox8, out_sh);
            uox8 = _mm256_add_epi32(uox8, _mm256_set1_epi32(out_uv_offset));

            vox8 = _mm256_add_epi32(_mm256_set1_epi32(out_rnd), _mm256_mullo_epi32(ravgx8, _mm256_set1_epi32(cburv)));
            vox8 = _mm256_add_epi32(vox8, _mm256_mullo_epi32(gavgx8, _mm256_set1_epi32(ocgv)));
            vox8 = _mm256_add_epi32(vox8, _mm256_mullo_epi32(bavgx8, _mm256_set1_epi32(cbv)));
            vox8 = _mm256_srai_epi32(vox8, out_sh);
            vox8 = _mm256_add_epi32(vox8, _mm256_set1_epi32(out_uv_offset));

            uvoax8 = _mm256_unpacklo_epi32(uox8, vox8);
            uvobx8 = _mm256_unpackhi_epi32(uox8, vox8);
            uvoax8 = _mm256_slli_epi32(uvoax8, out_sh2);
            uvobx8 = _mm256_slli_epi32(uvobx8, out_sh2);
            uvox16 = _mm256_packus_epi32(uvoax8, uvobx8);
            _mm256_storeu_si256((__m256i_u *) &dstuv[x], uvox16);
        }
    }

    // Process remaining pixels cannot fill the full simd register with scalar version
    if (remainw) {
        int offset = width & (int)0xfffffff0;
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
#endif // ENABLE_TONEMAPX_AVX_INTRINSICS
}
