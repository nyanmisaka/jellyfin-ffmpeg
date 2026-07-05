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

#include "vf_tonemapx_intrin_neon.h"

#ifdef ENABLE_TONEMAPX_NEON_INTRINSICS
#    include <arm_neon.h>
#endif // ENABLE_TONEMAPX_NEON_INTRINSICS

#ifdef ENABLE_TONEMAPX_NEON_INTRINSICS

static inline float reshape_poly(float s, float32x4_t coeffs)
{
    float32x4_t ps = vdupq_n_f32(0.0f);
    ps = vsetq_lane_f32(1.0f, ps, 0);
    ps = vsetq_lane_f32(s, ps, 1);
    ps = vsetq_lane_f32(s * s, ps, 2);
    ps = vmulq_f32(ps, coeffs);
    return vaddvq_f32(ps);
}

inline static float reshape_mmr(float32x4_t sig, float32x4_t coeffs, const float* mmr,
                                int mmr_single, int min_order, int max_order)
{
    int mmr_idx = mmr_single ? 0 : (int)vgetq_lane_f32(coeffs, 1);
    int order = (int)vgetq_lane_f32(coeffs, 3);
    float s = vgetq_lane_f32(coeffs, 0);
    float32x4_t mmr_coeffs, ps;
    float32x4_t sigX01 = vmulq_laneq_f32(sig, sig, 1); // {sig[0]*sig[1], sig[1]*sig[1], sig[2]*sig[1], sig[3]*sig[1]}
    float32x4_t sigX02 = vmulq_laneq_f32(sig, sig, 2); // {sig[0]*sig[2], sig[1]*sig[2], sig[2]*sig[2], sig[3]*sig[2]}
    float32x4_t sigX12 = vmulq_laneq_f32(sigX01, sig, 2); // {sig[0]*sig[1]*sig[2], sig[1]*sig[1]*sig[2], sig[2]*sig[1]*sig[2], sig[3]*sig[1]*sig[2]}
    float32x4_t sigX = sigX01; // sig[0]*sig[1] now positioned at 0
    sigX = vsetq_lane_f32(vgetq_lane_f32(sigX02, 0), sigX, 1); // sig[0]*sig[2] at 1
    sigX = vsetq_lane_f32(vgetq_lane_f32(sigX02, 1), sigX, 2); // sig[1]*sig[2] at 2
    sigX = vsetq_lane_f32(vgetq_lane_f32(sigX12, 0), sigX, 3); // sig[0]*sig[1]*sig[2] at 3

    // dot first order
    mmr_coeffs = vld1q_f32(&mmr[mmr_idx + 0*4]);
    ps = vmulq_f32(sig, mmr_coeffs);
    s += vaddvq_f32(ps);
    mmr_coeffs = vld1q_f32(&mmr[mmr_idx + 1*4]);
    ps = vmulq_f32(sigX, mmr_coeffs);
    s += vaddvq_f32(ps);

    if (max_order >= 2 && (min_order >= 2 || order >= 2)) {
        float32x4_t sig2 = vmulq_f32(sig, sig);
        float32x4_t sigX2 = vmulq_f32(sigX, sigX);

        mmr_coeffs = vld1q_f32(&mmr[mmr_idx + 2*4]);
        ps = vmulq_f32(sig2, mmr_coeffs);
        s += vaddvq_f32(ps);
        mmr_coeffs = vld1q_f32(&mmr[mmr_idx + 3*4]);
        ps = vmulq_f32(sigX2, mmr_coeffs);
        s += vaddvq_f32(ps);

        if (max_order == 3 && (min_order == 3 || order >= 3)) {
            float32x4_t sig3 = vmulq_f32(sig2, sig);
            float32x4_t sigX3 = vmulq_f32(sigX2, sigX);

            mmr_coeffs = vld1q_f32(&mmr[mmr_idx + 4*4]);
            ps = vmulq_f32(sig3, mmr_coeffs);
            s += vaddvq_f32(ps);
            mmr_coeffs = vld1q_f32(&mmr[mmr_idx + 5*4]);
            ps = vmulq_f32(sigX3, mmr_coeffs);
            s += vaddvq_f32(ps);
        }
    }

    return s;
}

#define CLAMP(a, b, c) (FFMIN(FFMAX((a), (b)), (c)))
inline static float32x4_t reshape_dovi_iptpqc2(float32x4_t sig, const TonemapIntParams *ctx)
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

    float32x4_t coeffs, result;

    // reshape I
    s = vgetq_lane_f32(sig, 0);
    result = sig;
    if (dovi_num_pivots_i > 2) {
        float32x4_t m01 = s >= dovi_pivots_i[0] ? vld1q_f32(dovi_coeffs_i + 4) : vld1q_f32(dovi_coeffs_i);
        float32x4_t m23 = s >= dovi_pivots_i[2] ? vld1q_f32(dovi_coeffs_i + 3*4) : vld1q_f32(dovi_coeffs_i + 2*4);
        float32x4_t m0123 = s >= dovi_pivots_i[1] ? m23 : m01;
        float32x4_t m45 = s >= dovi_pivots_i[4] ? vld1q_f32(dovi_coeffs_i + 5*4) : vld1q_f32(dovi_coeffs_i + 4*4);
        float32x4_t m67 = s >= dovi_pivots_i[6] ? vld1q_f32(dovi_coeffs_i + 7*4) : vld1q_f32(dovi_coeffs_i + 6*4);
        float32x4_t m4567 = s >= dovi_pivots_i[5] ? m67 : m45;
        coeffs = s >= dovi_pivots_i[3] ? m4567 : m0123;
    } else {
        coeffs = vld1q_f32(dovi_coeffs_i);
    }

    has_mmr_poly = dovi_has_mmr_i && dovi_has_poly_i;

    if ((has_mmr_poly && vgetq_lane_f32(coeffs, 3) == 0.0f) || (!has_mmr_poly && dovi_has_poly_i))
        s = reshape_poly(s, coeffs);
    else
        s = reshape_mmr(result, coeffs, dovi_mmr_i,
                        dovi_mmr_single_i, dovi_min_order_i, dovi_max_order_i);

    result = vsetq_lane_f32(CLAMP(s, dovi_lo_i, dovi_hi_i), result, 0);

    // reshape P
    s = vgetq_lane_f32(sig, 1);
    coeffs = vld1q_f32(dovi_coeffs_p);
    has_mmr_poly = dovi_has_mmr_p && dovi_has_poly_p;

    if ((has_mmr_poly && vgetq_lane_f32(coeffs, 3) == 0.0f) || (!has_mmr_poly && dovi_has_poly_p))
        s = reshape_poly(s, coeffs);
    else
        s = reshape_mmr(result, coeffs, dovi_mmr_p,
                        dovi_mmr_single_p, dovi_min_order_p, dovi_max_order_p);

    result = vsetq_lane_f32(CLAMP(s, dovi_lo_p, dovi_hi_p), result, 1);

    // reshape T
    s = vgetq_lane_f32(sig, 2);
    coeffs = vld1q_f32(dovi_coeffs_t);
    has_mmr_poly = dovi_has_mmr_t && dovi_has_poly_t;

    if ((has_mmr_poly && vgetq_lane_f32(coeffs, 3) == 0.0f) || (!has_mmr_poly && dovi_has_poly_t))
        s = reshape_poly(s, coeffs);
    else
        s = reshape_mmr(result, coeffs, dovi_mmr_t,
                        dovi_mmr_single_t, dovi_min_order_t, dovi_max_order_t);

    result = vsetq_lane_f32(CLAMP(s, dovi_lo_t, dovi_hi_t), result, 2);

    return result;
}

inline static void ycc2rgbx4(float32x4_t* dy, float32x4_t* dcb, float32x4_t* dcr,
                             float32x4_t y, float32x4_t cb, float32x4_t cr,
                             const double nonlinear[3][3], const float ycc_offset[3])
{
    *dy = vmulq_n_f32(y, (float)nonlinear[0][0]);
    *dy = vfmaq_n_f32(*dy, cb, (float)nonlinear[0][1]);
    *dy = vfmaq_n_f32(*dy, cr, (float)nonlinear[0][2]);
    *dy = vsubq_f32(*dy, vdupq_n_f32(ycc_offset[0]));

    *dcb = vmulq_n_f32(y, (float)nonlinear[1][0]);
    *dcb = vfmaq_n_f32(*dcb, cb, (float)nonlinear[1][1]);
    *dcb = vfmaq_n_f32(*dcb, cr, (float)nonlinear[1][2]);
    *dcb = vsubq_f32(*dcb, vdupq_n_f32(ycc_offset[1]));

    *dcr = vmulq_n_f32(y, (float)nonlinear[2][0]);
    *dcr = vfmaq_n_f32(*dcr, cb, (float)nonlinear[2][1]);
    *dcr = vfmaq_n_f32(*dcr, cr, (float)nonlinear[2][2]);
    *dcr = vsubq_f32(*dcr, vdupq_n_f32(ycc_offset[2]));
}

inline static void lms2rgbx4(float32x4_t* dl, float32x4_t* dm, float32x4_t* ds,
                             float32x4_t l, float32x4_t m, float32x4_t s,
                             const double lms2rgb_matrix[3][3])
{
    *dl = vmulq_n_f32(l, (float)lms2rgb_matrix[0][0]);
    *dl = vfmaq_n_f32(*dl, m, (float)lms2rgb_matrix[0][1]);
    *dl = vfmaq_n_f32(*dl, s, (float)lms2rgb_matrix[0][2]);

    *dm = vmulq_n_f32(l, (float)lms2rgb_matrix[1][0]);
    *dm = vfmaq_n_f32(*dm, m, (float)lms2rgb_matrix[1][1]);
    *dm = vfmaq_n_f32(*dm, s, (float)lms2rgb_matrix[1][2]);

    *ds = vmulq_n_f32(l, (float)lms2rgb_matrix[2][0]);
    *ds = vfmaq_n_f32(*ds, m, (float)lms2rgb_matrix[2][1]);
    *ds = vfmaq_n_f32(*ds, s, (float)lms2rgb_matrix[2][2]);
}

// Hardcoded for 10bit inputs
inline static void yuv2rgbx8(uint16x8_t *rx8, uint16x8_t *gx8, uint16x8_t *bx8,
                             uint16x8_t yx8, uint16x8_t ux8, uint16x8_t vx8,
                             int cy, int crv, int cgu, int cgv, int cbu)
{
    int32x4_t yx4a = vmovl_s16(vget_low_s16(vreinterpretq_s16_u16(yx8)));
    int32x4_t yx4b = vmovl_s16(vget_high_s16(vreinterpretq_s16_u16(yx8)));
    int32x4_t ux4a = vmovl_s16(vget_low_s16(vreinterpretq_s16_u16(ux8)));
    int32x4_t ux4b = vmovl_s16(vget_high_s16(vreinterpretq_s16_u16(ux8)));
    int32x4_t vx4a = vmovl_s16(vget_low_s16(vreinterpretq_s16_u16(vx8)));
    int32x4_t vx4b = vmovl_s16(vget_high_s16(vreinterpretq_s16_u16(vx8)));

    int32x4_t rx4a, gx4a, bx4a, rx4b, gx4b, bx4b;

    rx4a = gx4a = bx4a = vmlaq_n_s32(vdupq_n_s32(TEN_BIT_ROUNDING), yx4a,  cy);
    rx4a = vmlaq_n_s32(rx4a, vx4a, crv);
    rx4a = vshrq_n_s32(rx4a, 9); // 9 = 10bit - 1

    gx4a = vmlaq_n_s32(gx4a, ux4a, cgu);
    gx4a = vmlaq_n_s32(gx4a, vx4a, cgv);
    gx4a = vshrq_n_s32(gx4a, 9);

    bx4a = vmlaq_n_s32(bx4a, ux4a, cbu);
    bx4a = vshrq_n_s32(bx4a, 9);

    rx4b = gx4b = bx4b = vmlaq_n_s32(vdupq_n_s32(TEN_BIT_ROUNDING), yx4b,  cy);
    rx4b = vmlaq_n_s32(rx4b, vx4b, crv);
    rx4b = vshrq_n_s32(rx4b, 9);

    gx4b = vmlaq_n_s32(gx4b, ux4b, cgu);
    gx4b = vmlaq_n_s32(gx4b, vx4b, cgv);
    gx4b = vshrq_n_s32(gx4b, 9);

    bx4b = vmlaq_n_s32(bx4b, ux4b, cbu);
    bx4b = vshrq_n_s32(bx4b, 9);

    *rx8 = vreinterpretq_u16_s16(vcombine_s16(vqmovn_s32(rx4a), vqmovn_s32(rx4b)));
    *gx8 = vreinterpretq_u16_s16(vcombine_s16(vqmovn_s32(gx4a), vqmovn_s32(gx4b)));
    *bx8 = vreinterpretq_u16_s16(vcombine_s16(vqmovn_s32(bx4a), vqmovn_s32(bx4b)));
}

static inline void tonemap_int16x8_neon(uint16x8_t r_in, uint16x8_t g_in, uint16x8_t b_in,
                                        int16_t *r_out, int16_t *g_out, int16_t *b_out,
                                        float *lin_lut, float *tonemap_lut, uint16_t *delin_lut,
                                        const AVLumaCoefficients *coeffs,
                                        const AVLumaCoefficients *ocoeffs, double desat,
                                        double (*rgb2rgb)[3][3],
                                        int rgb2rgb_passthrough)
{
    int16x8_t sig8;
    float32x4_t mapvalx4a, mapvalx4b;
    float32x4_t r_linx4a, r_linx4b, g_linx4a, g_linx4b, b_linx4a, b_linx4b;
    float32x4_t offset = vdupq_n_f32(0.5f);
    int32x4_t output_upper_bound = vdupq_n_s32(INT16_MAX);
    int32x4_t zerox4 = vdupq_n_s32(0);
    int16x8_t r, g, b;
    int32x4_t rx4a, gx4a, bx4a, rx4b, gx4b, bx4b;

    float mapval4a[4], mapval4b[4], r_lin4a[4], r_lin4b[4], g_lin4a[4], g_lin4b[4], b_lin4a[4], b_lin4b[4];

    r = vreinterpretq_s16_u16(r_in);
    g = vreinterpretq_s16_u16(g_in);
    b = vreinterpretq_s16_u16(b_in);

    r = vmaxq_s16(r, vreinterpretq_s16_s32(zerox4));
    g = vmaxq_s16(g, vreinterpretq_s16_s32(zerox4));
    b = vmaxq_s16(b, vreinterpretq_s16_s32(zerox4));

    sig8 = vmaxq_s16(r, vmaxq_s16(g, b));

    // Cannot use loop here as the lane has to be compile-time constant
#define LOAD_LUT(i) mapval4a[i] = tonemap_lut[vget_lane_s16(vget_low_s16(sig8), i)]; \
mapval4b[i] = tonemap_lut[vget_lane_s16(vget_high_s16(sig8), i)];                    \
r_lin4a[i] = lin_lut[vget_lane_s16(vget_low_s16(r), i)];                             \
r_lin4b[i] = lin_lut[vget_lane_s16(vget_high_s16(r), i)];                            \
g_lin4a[i] = lin_lut[vget_lane_s16(vget_low_s16(g), i)];                             \
g_lin4b[i] = lin_lut[vget_lane_s16(vget_high_s16(g), i)];                            \
b_lin4a[i] = lin_lut[vget_lane_s16(vget_low_s16(b), i)];                             \
b_lin4b[i] = lin_lut[vget_lane_s16(vget_high_s16(b), i)];

    LOAD_LUT(0)
    LOAD_LUT(1)
    LOAD_LUT(2)
    LOAD_LUT(3)

#undef  LOAD_LUT

    mapvalx4a = vld1q_f32(mapval4a);
    mapvalx4b = vld1q_f32(mapval4b);
    r_linx4a = vld1q_f32(r_lin4a);
    r_linx4b = vld1q_f32(r_lin4b);
    g_linx4a = vld1q_f32(g_lin4a);
    g_linx4b = vld1q_f32(g_lin4b);
    b_linx4a = vld1q_f32(b_lin4a);
    b_linx4b = vld1q_f32(b_lin4b);

    if (!rgb2rgb_passthrough) {
        float32x4_t r_tmpx4a, g_tmpx4a, b_tmpx4a;
        float32x4_t r_tmpx4b, g_tmpx4b, b_tmpx4b;

        r_tmpx4a = vmulq_n_f32(r_linx4a, (float)(*rgb2rgb)[0][0]);
        r_tmpx4a = vfmaq_n_f32(r_tmpx4a, g_linx4a, (float)(*rgb2rgb)[0][1]);
        r_tmpx4a = vfmaq_n_f32(r_tmpx4a, b_linx4a, (float)(*rgb2rgb)[0][2]);
        r_tmpx4b = vmulq_n_f32(r_linx4b, (float)(*rgb2rgb)[0][0]);
        r_tmpx4b = vfmaq_n_f32(r_tmpx4b, g_linx4b, (float)(*rgb2rgb)[0][1]);
        r_tmpx4b = vfmaq_n_f32(r_tmpx4b, b_linx4b, (float)(*rgb2rgb)[0][2]);

        g_tmpx4a = vmulq_n_f32(g_linx4a, (float)(*rgb2rgb)[1][1]);
        g_tmpx4a = vfmaq_n_f32(g_tmpx4a, r_linx4a, (float)(*rgb2rgb)[1][0]);
        g_tmpx4a = vfmaq_n_f32(g_tmpx4a, b_linx4a, (float)(*rgb2rgb)[1][2]);
        g_tmpx4b = vmulq_n_f32(g_linx4b, (float)(*rgb2rgb)[1][1]);
        g_tmpx4b = vfmaq_n_f32(g_tmpx4b, r_linx4b, (float)(*rgb2rgb)[1][0]);
        g_tmpx4b = vfmaq_n_f32(g_tmpx4b, b_linx4b, (float)(*rgb2rgb)[1][2]);

        b_tmpx4a = vmulq_n_f32(b_linx4a, (float)(*rgb2rgb)[2][2]);
        b_tmpx4a = vfmaq_n_f32(b_tmpx4a, r_linx4a, (float)(*rgb2rgb)[2][0]);
        b_tmpx4a = vfmaq_n_f32(b_tmpx4a, g_linx4a, (float)(*rgb2rgb)[2][1]);
        b_tmpx4b = vmulq_n_f32(b_linx4b, (float)(*rgb2rgb)[2][2]);
        b_tmpx4b = vfmaq_n_f32(b_tmpx4b, r_linx4b, (float)(*rgb2rgb)[2][0]);
        b_tmpx4b = vfmaq_n_f32(b_tmpx4b, g_linx4b, (float)(*rgb2rgb)[2][1]);

        r_linx4a = r_tmpx4a; r_linx4b = r_tmpx4b;
        g_linx4a = g_tmpx4a; g_linx4b = g_tmpx4b;
        b_linx4a = b_tmpx4a; b_linx4b = b_tmpx4b;
    }

    if (desat > 0) {
        float32x4_t eps_x4 = vdupq_n_f32(FLOAT_EPS);
        float32x4_t desat4 = vdupq_n_f32((float)desat);
        float32x4_t luma4 = vdupq_n_f32(0);
        float32x4_t overbright4;
        // Group A
        luma4 = vfmaq_n_f32(luma4, r_linx4a, (float)av_q2d(coeffs->cr));
        luma4 = vfmaq_n_f32(luma4, g_linx4a, (float)av_q2d(coeffs->cg));
        luma4 = vfmaq_n_f32(luma4, b_linx4a, (float)av_q2d(coeffs->cb));
        overbright4 = vdivq_f32(vmaxq_f32(vsubq_f32(luma4, desat4), eps_x4), vmaxq_f32(luma4, eps_x4));
        r_linx4a = vfmsq_f32(r_linx4a, r_linx4a, overbright4);
        r_linx4a = vfmaq_f32(r_linx4a, luma4, overbright4);
        g_linx4a = vfmsq_f32(g_linx4a, g_linx4a, overbright4);
        g_linx4a = vfmaq_f32(g_linx4a, luma4, overbright4);
        b_linx4a = vfmsq_f32(b_linx4a, b_linx4a, overbright4);
        b_linx4a = vfmaq_f32(b_linx4a, luma4, overbright4);
        // Group B
        luma4 = vdupq_n_f32(0);
        luma4 = vfmaq_n_f32(luma4, r_linx4b, (float)av_q2d(coeffs->cr));
        luma4 = vfmaq_n_f32(luma4, g_linx4b, (float)av_q2d(coeffs->cg));
        luma4 = vfmaq_n_f32(luma4, b_linx4b, (float)av_q2d(coeffs->cb));
        overbright4 = vdivq_f32(vmaxq_f32(vsubq_f32(luma4, desat4), eps_x4), vmaxq_f32(luma4, eps_x4));
        r_linx4b = vfmsq_f32(r_linx4b, r_linx4b, overbright4);
        r_linx4b = vfmaq_f32(r_linx4b, luma4, overbright4);
        g_linx4b = vfmsq_f32(g_linx4b, g_linx4b, overbright4);
        g_linx4b = vfmaq_f32(g_linx4b, luma4, overbright4);
        b_linx4b = vfmsq_f32(b_linx4b, b_linx4b, overbright4);
        b_linx4b = vfmaq_f32(b_linx4b, luma4, overbright4);
    }

    r_linx4a = vmulq_f32(r_linx4a, mapvalx4a);
    g_linx4a = vmulq_f32(g_linx4a, mapvalx4a);
    b_linx4a = vmulq_f32(b_linx4a, mapvalx4a);

    r_linx4b = vmulq_f32(r_linx4b, mapvalx4b);
    g_linx4b = vmulq_f32(g_linx4b, mapvalx4b);
    b_linx4b = vmulq_f32(b_linx4b, mapvalx4b);

    r_linx4a = vfmaq_n_f32(offset, r_linx4a, INT16_MAX);
    r_linx4b = vfmaq_n_f32(offset, r_linx4b, INT16_MAX);
    g_linx4a = vfmaq_n_f32(offset, g_linx4a, INT16_MAX);
    g_linx4b = vfmaq_n_f32(offset, g_linx4b, INT16_MAX);
    b_linx4a = vfmaq_n_f32(offset, b_linx4a, INT16_MAX);
    b_linx4b = vfmaq_n_f32(offset, b_linx4b, INT16_MAX);

    rx4a = vcvtq_s32_f32(r_linx4a);
    rx4a = vminq_s32(rx4a, output_upper_bound);
    rx4a = vmaxq_s32(rx4a, zerox4);
    gx4a = vcvtq_s32_f32(g_linx4a);
    gx4a = vminq_s32(gx4a, output_upper_bound);
    gx4a = vmaxq_s32(gx4a, zerox4);
    bx4a = vcvtq_s32_f32(b_linx4a);
    bx4a = vminq_s32(bx4a, output_upper_bound);
    bx4a = vmaxq_s32(bx4a, zerox4);
    rx4b = vcvtq_s32_f32(r_linx4b);
    rx4b = vminq_s32(rx4b, output_upper_bound);
    rx4b = vmaxq_s32(rx4b, zerox4);
    gx4b = vcvtq_s32_f32(g_linx4b);
    gx4b = vminq_s32(gx4b, output_upper_bound);
    gx4b = vmaxq_s32(gx4b, zerox4);
    bx4b = vcvtq_s32_f32(b_linx4b);
    bx4b = vminq_s32(bx4b, output_upper_bound);
    bx4b = vmaxq_s32(bx4b, zerox4);

    r_out[0] = delin_lut[vget_lane_s32(vget_low_s32(rx4a), 0)];
    r_out[1] = delin_lut[vget_lane_s32(vget_low_s32(rx4a), 1)];
    r_out[2] = delin_lut[vget_lane_s32(vget_high_s32(rx4a), 0)];
    r_out[3] = delin_lut[vget_lane_s32(vget_high_s32(rx4a), 1)];
    r_out[4] = delin_lut[vget_lane_s32(vget_low_s32(rx4b), 0)];
    r_out[5] = delin_lut[vget_lane_s32(vget_low_s32(rx4b), 1)];
    r_out[6] = delin_lut[vget_lane_s32(vget_high_s32(rx4b), 0)];
    r_out[7] = delin_lut[vget_lane_s32(vget_high_s32(rx4b), 1)];

    g_out[0] = delin_lut[vget_lane_s32(vget_low_s32(gx4a), 0)];
    g_out[1] = delin_lut[vget_lane_s32(vget_low_s32(gx4a), 1)];
    g_out[2] = delin_lut[vget_lane_s32(vget_high_s32(gx4a), 0)];
    g_out[3] = delin_lut[vget_lane_s32(vget_high_s32(gx4a), 1)];
    g_out[4] = delin_lut[vget_lane_s32(vget_low_s32(gx4b), 0)];
    g_out[5] = delin_lut[vget_lane_s32(vget_low_s32(gx4b), 1)];
    g_out[6] = delin_lut[vget_lane_s32(vget_high_s32(gx4b), 0)];
    g_out[7] = delin_lut[vget_lane_s32(vget_high_s32(gx4b), 1)];

    b_out[0] = delin_lut[vget_lane_s32(vget_low_s32(bx4a), 0)];
    b_out[1] = delin_lut[vget_lane_s32(vget_low_s32(bx4a), 1)];
    b_out[2] = delin_lut[vget_lane_s32(vget_high_s32(bx4a), 0)];
    b_out[3] = delin_lut[vget_lane_s32(vget_high_s32(bx4a), 1)];
    b_out[4] = delin_lut[vget_lane_s32(vget_low_s32(bx4b), 0)];
    b_out[5] = delin_lut[vget_lane_s32(vget_low_s32(bx4b), 1)];
    b_out[6] = delin_lut[vget_lane_s32(vget_high_s32(bx4b), 0)];
    b_out[7] = delin_lut[vget_lane_s32(vget_high_s32(bx4b), 1)];
}
#endif // ENABLE_TONEMAPX_NEON_INTRINSICS

void tonemap_frame_dovi_2_420p_neon(uint8_t *dsty, uint8_t *dstu, uint8_t *dstv,
                                    const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                    const int *dstlinesize, const int *srclinesize,
                                    int dstdepth, int srcdepth,
                                    int width, int height,
                                    const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_NEON_INTRINSICS
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
    uint16x8_t y0x8, y1x8, ux8, vx8;
    uint16x8_t r0x8, g0x8, b0x8;
    uint16x8_t r1x8, g1x8, b1x8;
    uint16x4_t ux4, vx4;

    int16x8_t r0ox8, g0ox8, b0ox8;
    int16x8_t y0ox8;
    int32x4_t r0oax4, r0obx4, g0oax4, g0obx4, b0oax4, b0obx4;
    int32x4_t y0oax4, y0obx4;

    int16x8_t r1ox8, g1ox8, b1ox8;
    int16x8_t y1ox8;
    int32x4_t r1oax4, r1obx4, g1oax4, g1obx4, b1oax4, b1obx4;
    int32x4_t y1oax4, y1obx4;
    int32x2_t ravgax2, gavgax2, bavgax2, ravgbx2, gavgbx2, bavgbx2;
    int32x4_t ravgx4, gavgx4, bavgx4, uox4, vox4;
    int32x4_t out_yuv_offx4 = vdupq_n_s32(params->out_yuv_off);
    int32x4_t out_rndx4 = vdupq_n_s32(EIGHT_BIT_ROUNDING);
    int32x4_t out_uv_offsetx4 = vdupq_n_s32(EIGHT_BIT_UV_OFFSET);
    int32x4_t rgb_avg_rndx4 = vdupq_n_s32(CHROMA_AVG_ROUNDING);
    float32x4_t ipt0, ipt1, ipt2, ipt3;
    float32x4_t ia1, ib1, ia2, ib2;
    float32x4_t ix4, px4, tx4;
    float32x4_t lx4, mx4, sx4;
    float32x4_t rx4a, gx4a, bx4a, rx4b, gx4b, bx4b;
    float32x4_t y0x4a, y0x4b, y1x4a, y1x4b, ux4a, ux4b, vx4a, vx4b;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0] * 2, dstu += dstlinesize[1], dstv += dstlinesize[2],
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[2] / 2) {
        for (int xx = 0; xx < width >> 3; xx++) {
            int x = xx << 3;

            y0x8 = vld1q_u16(srcy + x);
            y1x8 = vld1q_u16(srcy + (srclinesize[0] / 2 + x));
            ux4 = vld1_u16(srcu + (x >> 1));
            vx4 = vld1_u16(srcv + (x >> 1));

            ux8 = vcombine_u16(vzip1_u16(ux4, ux4), vzip2_u16(ux4, ux4));
            vx8 = vcombine_u16(vzip1_u16(vx4, vx4), vzip2_u16(vx4, vx4));

            y0x4a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(y0x8)));
            y0x4b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(y0x8)));
            y1x4a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(y1x8)));
            y1x4b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(y1x8)));

            ux4a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(ux8)));
            ux4b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(ux8)));
            vx4a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vx8)));
            vx4b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vx8)));

            y0x4a = vdivq_f32(y0x4a, vdupq_n_f32(TEN_BIT_SCALE));
            y0x4b = vdivq_f32(y0x4b, vdupq_n_f32(TEN_BIT_SCALE));
            y1x4a = vdivq_f32(y1x4a, vdupq_n_f32(TEN_BIT_SCALE));
            y1x4b = vdivq_f32(y1x4b, vdupq_n_f32(TEN_BIT_SCALE));
            ux4a = vdivq_f32(ux4a, vdupq_n_f32(TEN_BIT_SCALE));
            ux4b = vdivq_f32(ux4b, vdupq_n_f32(TEN_BIT_SCALE));
            vx4a = vdivq_f32(vx4a, vdupq_n_f32(TEN_BIT_SCALE));
            vx4b = vdivq_f32(vx4b, vdupq_n_f32(TEN_BIT_SCALE));

            // Reshape y0x4a
            ia1 = vzip1q_f32(y0x4a, ux4a);
            ia2 = vzip2q_f32(y0x4a, ux4a);
            ib1 = vzip1q_f32(vx4a, vdupq_n_f32(0.0f));
            ib2 = vzip2q_f32(vx4a, vdupq_n_f32(0.0f));
            ipt0 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ib1));
            ipt1 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ib1));
            ipt2 = vcombine_f32(vget_low_f32(ia2), vget_low_f32(ib2));
            ipt3 = vcombine_f32(vget_high_f32(ia2), vget_high_f32(ib2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ia1 = vtrn1q_f32(ipt0, ipt1);
            ia2 = vtrn1q_f32(ipt2, ipt3);
            ib1 = vtrn2q_f32(ipt0, ipt1);
            ib2 = vtrn2q_f32(ipt2, ipt3);

            ix4 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ia2));
            px4 = vcombine_f32(vget_low_f32(ib1), vget_low_f32(ib2));
            tx4 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ia2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4a, &gx4a, &bx4a, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4a = vmulq_n_f32(rx4a, JPEG_SCALE);
            gx4a = vmulq_n_f32(gx4a, JPEG_SCALE);
            bx4a = vmulq_n_f32(bx4a, JPEG_SCALE);

            // Reshape y0x4b
            ia1 = vzip1q_f32(y0x4b, ux4b);
            ia2 = vzip2q_f32(y0x4b, ux4b);
            ib1 = vzip1q_f32(vx4b, vdupq_n_f32(0.0f));
            ib2 = vzip2q_f32(vx4b, vdupq_n_f32(0.0f));
            ipt0 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ib1));
            ipt1 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ib1));
            ipt2 = vcombine_f32(vget_low_f32(ia2), vget_low_f32(ib2));
            ipt3 = vcombine_f32(vget_high_f32(ia2), vget_high_f32(ib2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ia1 = vtrn1q_f32(ipt0, ipt1);
            ia2 = vtrn1q_f32(ipt2, ipt3);
            ib1 = vtrn2q_f32(ipt0, ipt1);
            ib2 = vtrn2q_f32(ipt2, ipt3);

            ix4 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ia2));
            px4 = vcombine_f32(vget_low_f32(ib1), vget_low_f32(ib2));
            tx4 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ia2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4b, &gx4b, &bx4b, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4b = vmulq_n_f32(rx4b, JPEG_SCALE);
            gx4b = vmulq_n_f32(gx4b, JPEG_SCALE);
            bx4b = vmulq_n_f32(bx4b, JPEG_SCALE);

            r0x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(rx4a)), vqmovn_u32(vcvtq_u32_f32(rx4b)));
            g0x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(gx4a)), vqmovn_u32(vcvtq_u32_f32(gx4b)));
            b0x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(bx4a)), vqmovn_u32(vcvtq_u32_f32(bx4b)));

            r0x8 = vminq_u16(r0x8, vdupq_n_u16(INT16_MAX));
            g0x8 = vminq_u16(g0x8, vdupq_n_u16(INT16_MAX));
            b0x8 = vminq_u16(b0x8, vdupq_n_u16(INT16_MAX));

            // Reshape y1x4a
            ia1 = vzip1q_f32(y1x4a, ux4a);
            ia2 = vzip2q_f32(y1x4a, ux4a);
            ib1 = vzip1q_f32(vx4a, vdupq_n_f32(0.0f));
            ib2 = vzip2q_f32(vx4a, vdupq_n_f32(0.0f));
            ipt0 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ib1));
            ipt1 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ib1));
            ipt2 = vcombine_f32(vget_low_f32(ia2), vget_low_f32(ib2));
            ipt3 = vcombine_f32(vget_high_f32(ia2), vget_high_f32(ib2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ia1 = vtrn1q_f32(ipt0, ipt1);
            ia2 = vtrn1q_f32(ipt2, ipt3);
            ib1 = vtrn2q_f32(ipt0, ipt1);
            ib2 = vtrn2q_f32(ipt2, ipt3);

            ix4 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ia2));
            px4 = vcombine_f32(vget_low_f32(ib1), vget_low_f32(ib2));
            tx4 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ia2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4a, &gx4a, &bx4a, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4a = vmulq_n_f32(rx4a, JPEG_SCALE);
            gx4a = vmulq_n_f32(gx4a, JPEG_SCALE);
            bx4a = vmulq_n_f32(bx4a, JPEG_SCALE);

            // Reshape y1x4b
            ia1 = vzip1q_f32(y1x4b, ux4b);
            ia2 = vzip2q_f32(y1x4b, ux4b);
            ib1 = vzip1q_f32(vx4b, vdupq_n_f32(0.0f));
            ib2 = vzip2q_f32(vx4b, vdupq_n_f32(0.0f));
            ipt0 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ib1));
            ipt1 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ib1));
            ipt2 = vcombine_f32(vget_low_f32(ia2), vget_low_f32(ib2));
            ipt3 = vcombine_f32(vget_high_f32(ia2), vget_high_f32(ib2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ia1 = vtrn1q_f32(ipt0, ipt1);
            ia2 = vtrn1q_f32(ipt2, ipt3);
            ib1 = vtrn2q_f32(ipt0, ipt1);
            ib2 = vtrn2q_f32(ipt2, ipt3);

            ix4 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ia2));
            px4 = vcombine_f32(vget_low_f32(ib1), vget_low_f32(ib2));
            tx4 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ia2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4b, &gx4b, &bx4b, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4b = vmulq_n_f32(rx4b, JPEG_SCALE);
            gx4b = vmulq_n_f32(gx4b, JPEG_SCALE);
            bx4b = vmulq_n_f32(bx4b, JPEG_SCALE);

            r1x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(rx4a)), vqmovn_u32(vcvtq_u32_f32(rx4b)));
            g1x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(gx4a)), vqmovn_u32(vcvtq_u32_f32(gx4b)));
            b1x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(bx4a)), vqmovn_u32(vcvtq_u32_f32(bx4b)));

            r1x8 = vminq_u16(r1x8, vdupq_n_u16(INT16_MAX));
            g1x8 = vminq_u16(g1x8, vdupq_n_u16(INT16_MAX));
            b1x8 = vminq_u16(b1x8, vdupq_n_u16(INT16_MAX));


            tonemap_int16x8_neon(r0x8, g0x8, b0x8, (int16_t *) &r, (int16_t *) &g, (int16_t *) &b,
                                 params->lin_lut, params->tonemap_lut, params->delin_lut,
                                 params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                 params->rgb2rgb_passthrough);
            tonemap_int16x8_neon(r1x8, g1x8, b1x8, (int16_t *) &r1, (int16_t *) &g1, (int16_t *) &b1,
                                 params->lin_lut, params->tonemap_lut, params->delin_lut,
                                 params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                 params->rgb2rgb_passthrough);

            r0ox8 = vld1q_s16(r);
            g0ox8 = vld1q_s16(g);
            b0ox8 = vld1q_s16(b);

            r0oax4 = vmovl_s16(vget_low_s16(r0ox8));
            g0oax4 = vmovl_s16(vget_low_s16(g0ox8));
            b0oax4 = vmovl_s16(vget_low_s16(b0ox8));

            r0obx4 = vmovl_s16(vget_high_s16(r0ox8));
            g0obx4 = vmovl_s16(vget_high_s16(g0ox8));
            b0obx4 = vmovl_s16(vget_high_s16(b0ox8));

            y0oax4 = vmulq_n_s32(r0oax4, cry);
            y0oax4 = vmlaq_n_s32(y0oax4, g0oax4, cgy);
            y0oax4 = vmlaq_n_s32(y0oax4, b0oax4, cby);
            y0oax4 = vaddq_s32(y0oax4, out_rndx4);
            // output shift bits for 8bit outputs is 29 - 8 = 21
            y0oax4 = vshrq_n_s32(y0oax4, EIGHT_BIT_SCALE_SHIFT);
            y0oax4 = vaddq_s32(y0oax4, out_yuv_offx4);

            y0obx4 = vmulq_n_s32(r0obx4, cry);
            y0obx4 = vmlaq_n_s32(y0obx4, g0obx4, cgy);
            y0obx4 = vmlaq_n_s32(y0obx4, b0obx4, cby);
            y0obx4 = vaddq_s32(y0obx4, out_rndx4);
            y0obx4 = vshrq_n_s32(y0obx4, EIGHT_BIT_SCALE_SHIFT);
            y0obx4 = vaddq_s32(y0obx4, out_yuv_offx4);

            y0ox8 = vcombine_s16(vqmovn_s32(y0oax4), vqmovn_s32(y0obx4));
            vst1_u8(&dsty[x], vqmovun_s16(y0ox8));

            r1ox8 = vld1q_s16(r1);
            g1ox8 = vld1q_s16(g1);
            b1ox8 = vld1q_s16(b1);

            r1oax4 = vmovl_s16(vget_low_s16(r1ox8));
            g1oax4 = vmovl_s16(vget_low_s16(g1ox8));
            b1oax4 = vmovl_s16(vget_low_s16(b1ox8));

            r1obx4 = vmovl_s16(vget_high_s16(r1ox8));
            g1obx4 = vmovl_s16(vget_high_s16(g1ox8));
            b1obx4 = vmovl_s16(vget_high_s16(b1ox8));

            y1oax4 = vmulq_n_s32(r1oax4, cry);
            y1oax4 = vmlaq_n_s32(y1oax4, g1oax4, cgy);
            y1oax4 = vmlaq_n_s32(y1oax4, b1oax4, cby);
            y1oax4 = vaddq_s32(y1oax4, out_rndx4);
            y1oax4 = vshrq_n_s32(y1oax4, EIGHT_BIT_SCALE_SHIFT);
            y1oax4 = vaddq_s32(y1oax4, out_yuv_offx4);

            y1obx4 = vmulq_n_s32(r1obx4, cry);
            y1obx4 = vmlaq_n_s32(y1obx4, g1obx4, cgy);
            y1obx4 = vmlaq_n_s32(y1obx4, b1obx4, cby);
            y1obx4 = vaddq_s32(y1obx4, out_rndx4);
            y1obx4 = vshrq_n_s32(y1obx4, EIGHT_BIT_SCALE_SHIFT);
            y1obx4 = vaddq_s32(y1obx4, out_yuv_offx4);

            y1ox8 = vcombine_s16(vqmovn_s32(y1oax4), vqmovn_s32(y1obx4));
            vst1_u8(&dsty[x + dstlinesize[0]], vqmovun_s16(y1ox8));

            ravgax2 = vpadd_s32(vget_low_s32(r0oax4), vget_high_s32(r0oax4));
            ravgbx2 = vpadd_s32(vget_low_s32(r0obx4), vget_high_s32(r0obx4));
            ravgx4 = vcombine_s32(ravgax2, ravgbx2);
            ravgax2 = vpadd_s32(vget_low_s32(r1oax4), vget_high_s32(r1oax4));
            ravgbx2 = vpadd_s32(vget_low_s32(r1obx4), vget_high_s32(r1obx4));
            ravgx4 = vaddq_s32(ravgx4, vcombine_s32(ravgax2, ravgbx2));
            ravgx4 = vaddq_s32(ravgx4, rgb_avg_rndx4);
            ravgx4 = vshrq_n_s32(ravgx4, CHROMA_AVG_ROUNDING);

            gavgax2 = vpadd_s32(vget_low_s32(g0oax4), vget_high_s32(g0oax4));
            gavgbx2 = vpadd_s32(vget_low_s32(g0obx4), vget_high_s32(g0obx4));
            gavgx4 = vcombine_s32(gavgax2, gavgbx2);
            gavgax2 = vpadd_s32(vget_low_s32(g1oax4), vget_high_s32(g1oax4));
            gavgbx2 = vpadd_s32(vget_low_s32(g1obx4), vget_high_s32(g1obx4));
            gavgx4 = vaddq_s32(gavgx4, vcombine_s32(gavgax2, gavgbx2));
            gavgx4 = vaddq_s32(gavgx4, rgb_avg_rndx4);
            gavgx4 = vshrq_n_s32(gavgx4, CHROMA_AVG_ROUNDING);

            bavgax2 = vpadd_s32(vget_low_s32(b0oax4), vget_high_s32(b0oax4));
            bavgbx2 = vpadd_s32(vget_low_s32(b0obx4), vget_high_s32(b0obx4));
            bavgx4 = vcombine_s32(bavgax2, bavgbx2);
            bavgax2 = vpadd_s32(vget_low_s32(b1oax4), vget_high_s32(b1oax4));
            bavgbx2 = vpadd_s32(vget_low_s32(b1obx4), vget_high_s32(b1obx4));
            bavgx4 = vaddq_s32(bavgx4, vcombine_s32(bavgax2, bavgbx2));
            bavgx4 = vaddq_s32(bavgx4, rgb_avg_rndx4);
            bavgx4 = vshrq_n_s32(bavgx4, CHROMA_AVG_ROUNDING);

            uox4 = vmlaq_n_s32(out_rndx4, ravgx4, cru);
            uox4 = vmlaq_n_s32(uox4, gavgx4, ocgu);
            uox4 = vmlaq_n_s32(uox4, bavgx4, cburv);
            uox4 = vshrq_n_s32(uox4, EIGHT_BIT_SCALE_SHIFT);
            uox4 = vaddq_s32(uox4, out_uv_offsetx4);
            vst1_lane_u32((uint32_t *) &dstu[x >> 1], vreinterpret_u32_u8(vqmovun_s16(vcombine_s16(vmovn_s32(uox4), vdup_n_s16(0)))), 0);

            vox4 = vmlaq_n_s32(out_rndx4, ravgx4, cburv);
            vox4 = vmlaq_n_s32(vox4, gavgx4, ocgv);
            vox4 = vmlaq_n_s32(vox4, bavgx4, cbv);
            vox4 = vshrq_n_s32(vox4, EIGHT_BIT_SCALE_SHIFT);
            vox4 = vaddq_s32(vox4, out_uv_offsetx4);
            vst1_lane_u32((uint32_t *) &dstv[x >> 1], vreinterpret_u32_u8(vqmovun_s16(vcombine_s16(vmovn_s32(vox4), vdup_n_s16(0)))), 0);
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
#endif // ENABLE_TONEMAPX_NEON_INTRINSICS
}

void tonemap_frame_420p10_2_420p_neon(uint8_t *dsty, uint8_t *dstu, uint8_t *dstv,
                                      const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                      const int *dstlinesize, const int *srclinesize,
                                      int dstdepth, int srcdepth,
                                      int width, int height,
                                      const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_NEON_INTRINSICS
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
    uint16x8_t in_yuv_offx8 = vdupq_n_u16(params->in_yuv_off);
    uint16x8_t in_uv_offx8 = vdupq_n_u16(TEN_BIT_UV_OFFSET);
    uint16x8_t y0x8, y1x8, ux8, vx8;
    uint16x8_t r0x8, g0x8, b0x8;
    uint16x8_t r1x8, g1x8, b1x8;
    uint16x4_t ux4, vx4;

    int16x8_t r0ox8, g0ox8, b0ox8;
    int16x8_t y0ox8;
    int32x4_t r0oax4, r0obx4, g0oax4, g0obx4, b0oax4, b0obx4;
    int32x4_t y0oax4, y0obx4;

    int16x8_t r1ox8, g1ox8, b1ox8;
    int16x8_t y1ox8;
    int32x4_t r1oax4, r1obx4, g1oax4, g1obx4, b1oax4, b1obx4;
    int32x4_t y1oax4, y1obx4;
    int32x2_t ravgax2, gavgax2, bavgax2, ravgbx2, gavgbx2, bavgbx2;
    int32x4_t ravgx4, gavgx4, bavgx4, uox4, vox4;
    int32x4_t out_yuv_offx4 = vdupq_n_s32(params->out_yuv_off);
    int32x4_t out_rndx4 = vdupq_n_s32(EIGHT_BIT_ROUNDING);
    int32x4_t out_uv_offsetx4 = vdupq_n_s32(EIGHT_BIT_UV_OFFSET);
    int32x4_t rgb_avg_rndx4 = vdupq_n_s32(CHROMA_AVG_ROUNDING);
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0] * 2, dstu += dstlinesize[1], dstv += dstlinesize[2],
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[2] / 2) {
        for (int xx = 0; xx < width >> 3; xx++) {
            int x = xx << 3;

            y0x8 = vld1q_u16(srcy + x);
            y1x8 = vld1q_u16(srcy + (srclinesize[0] / 2 + x));
            ux4 = vld1_u16(srcu + (x >> 1));
            vx4 = vld1_u16(srcv + (x >> 1));

            y0x8 = vsubq_u16(y0x8, in_yuv_offx8);
            y0x8 = vreinterpretq_u16_s16(vmaxq_s16(vreinterpretq_s16_u16(y0x8), vdupq_n_s16(0)));
            y1x8 = vsubq_u16(y1x8, in_yuv_offx8);
            y1x8 = vreinterpretq_u16_s16(vmaxq_s16(vreinterpretq_s16_u16(y1x8), vdupq_n_s16(0)));
            ux8 = vcombine_u16(vzip1_u16(ux4, ux4), vzip2_u16(ux4, ux4));
            ux8 = vsubq_u16(ux8, in_uv_offx8);
            vx8 = vcombine_u16(vzip1_u16(vx4, vx4), vzip2_u16(vx4, vx4));
            vx8 = vsubq_u16(vx8, in_uv_offx8);

            yuv2rgbx8(&r0x8, &g0x8, &b0x8, y0x8, ux8, vx8, cy, crv, cgu, cgv, cbu);
            yuv2rgbx8(&r1x8, &g1x8, &b1x8, y1x8, ux8, vx8, cy, crv, cgu, cgv, cbu);

            tonemap_int16x8_neon(r0x8, g0x8, b0x8, (int16_t *) &r, (int16_t *) &g, (int16_t *) &b,
                                 params->lin_lut, params->tonemap_lut, params->delin_lut,
                                 params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                 params->rgb2rgb_passthrough);
            tonemap_int16x8_neon(r1x8, g1x8, b1x8, (int16_t *) &r1, (int16_t *) &g1, (int16_t *) &b1,
                                 params->lin_lut, params->tonemap_lut, params->delin_lut,
                                 params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                 params->rgb2rgb_passthrough);

            r0ox8 = vld1q_s16(r);
            g0ox8 = vld1q_s16(g);
            b0ox8 = vld1q_s16(b);

            r0oax4 = vmovl_s16(vget_low_s16(r0ox8));
            g0oax4 = vmovl_s16(vget_low_s16(g0ox8));
            b0oax4 = vmovl_s16(vget_low_s16(b0ox8));

            r0obx4 = vmovl_s16(vget_high_s16(r0ox8));
            g0obx4 = vmovl_s16(vget_high_s16(g0ox8));
            b0obx4 = vmovl_s16(vget_high_s16(b0ox8));

            y0oax4 = vmulq_n_s32(r0oax4, cry);
            y0oax4 = vmlaq_n_s32(y0oax4, g0oax4, cgy);
            y0oax4 = vmlaq_n_s32(y0oax4, b0oax4, cby);
            y0oax4 = vaddq_s32(y0oax4, out_rndx4);
            // output shift bits for 8bit outputs is 29 - 8 = 21
            y0oax4 = vshrq_n_s32(y0oax4, EIGHT_BIT_SCALE_SHIFT);
            y0oax4 = vaddq_s32(y0oax4, out_yuv_offx4);

            y0obx4 = vmulq_n_s32(r0obx4, cry);
            y0obx4 = vmlaq_n_s32(y0obx4, g0obx4, cgy);
            y0obx4 = vmlaq_n_s32(y0obx4, b0obx4, cby);
            y0obx4 = vaddq_s32(y0obx4, out_rndx4);
            y0obx4 = vshrq_n_s32(y0obx4, EIGHT_BIT_SCALE_SHIFT);
            y0obx4 = vaddq_s32(y0obx4, out_yuv_offx4);

            y0ox8 = vcombine_s16(vqmovn_s32(y0oax4), vqmovn_s32(y0obx4));
            vst1_u8(&dsty[x], vqmovun_s16(y0ox8));

            r1ox8 = vld1q_s16(r1);
            g1ox8 = vld1q_s16(g1);
            b1ox8 = vld1q_s16(b1);

            r1oax4 = vmovl_s16(vget_low_s16(r1ox8));
            g1oax4 = vmovl_s16(vget_low_s16(g1ox8));
            b1oax4 = vmovl_s16(vget_low_s16(b1ox8));

            r1obx4 = vmovl_s16(vget_high_s16(r1ox8));
            g1obx4 = vmovl_s16(vget_high_s16(g1ox8));
            b1obx4 = vmovl_s16(vget_high_s16(b1ox8));

            y1oax4 = vmulq_n_s32(r1oax4, cry);
            y1oax4 = vmlaq_n_s32(y1oax4, g1oax4, cgy);
            y1oax4 = vmlaq_n_s32(y1oax4, b1oax4, cby);
            y1oax4 = vaddq_s32(y1oax4, out_rndx4);
            y1oax4 = vshrq_n_s32(y1oax4, EIGHT_BIT_SCALE_SHIFT);
            y1oax4 = vaddq_s32(y1oax4, out_yuv_offx4);

            y1obx4 = vmulq_n_s32(r1obx4, cry);
            y1obx4 = vmlaq_n_s32(y1obx4, g1obx4, cgy);
            y1obx4 = vmlaq_n_s32(y1obx4, b1obx4, cby);
            y1obx4 = vaddq_s32(y1obx4, out_rndx4);
            y1obx4 = vshrq_n_s32(y1obx4, EIGHT_BIT_SCALE_SHIFT);
            y1obx4 = vaddq_s32(y1obx4, out_yuv_offx4);

            y1ox8 = vcombine_s16(vqmovn_s32(y1oax4), vqmovn_s32(y1obx4));
            vst1_u8(&dsty[x + dstlinesize[0]], vqmovun_s16(y1ox8));

            ravgax2 = vpadd_s32(vget_low_s32(r0oax4), vget_high_s32(r0oax4));
            ravgbx2 = vpadd_s32(vget_low_s32(r0obx4), vget_high_s32(r0obx4));
            ravgx4 = vcombine_s32(ravgax2, ravgbx2);
            ravgax2 = vpadd_s32(vget_low_s32(r1oax4), vget_high_s32(r1oax4));
            ravgbx2 = vpadd_s32(vget_low_s32(r1obx4), vget_high_s32(r1obx4));
            ravgx4 = vaddq_s32(ravgx4, vcombine_s32(ravgax2, ravgbx2));
            ravgx4 = vaddq_s32(ravgx4, rgb_avg_rndx4);
            ravgx4 = vshrq_n_s32(ravgx4, CHROMA_AVG_ROUNDING);

            gavgax2 = vpadd_s32(vget_low_s32(g0oax4), vget_high_s32(g0oax4));
            gavgbx2 = vpadd_s32(vget_low_s32(g0obx4), vget_high_s32(g0obx4));
            gavgx4 = vcombine_s32(gavgax2, gavgbx2);
            gavgax2 = vpadd_s32(vget_low_s32(g1oax4), vget_high_s32(g1oax4));
            gavgbx2 = vpadd_s32(vget_low_s32(g1obx4), vget_high_s32(g1obx4));
            gavgx4 = vaddq_s32(gavgx4, vcombine_s32(gavgax2, gavgbx2));
            gavgx4 = vaddq_s32(gavgx4, rgb_avg_rndx4);
            gavgx4 = vshrq_n_s32(gavgx4, CHROMA_AVG_ROUNDING);

            bavgax2 = vpadd_s32(vget_low_s32(b0oax4), vget_high_s32(b0oax4));
            bavgbx2 = vpadd_s32(vget_low_s32(b0obx4), vget_high_s32(b0obx4));
            bavgx4 = vcombine_s32(bavgax2, bavgbx2);
            bavgax2 = vpadd_s32(vget_low_s32(b1oax4), vget_high_s32(b1oax4));
            bavgbx2 = vpadd_s32(vget_low_s32(b1obx4), vget_high_s32(b1obx4));
            bavgx4 = vaddq_s32(bavgx4, vcombine_s32(bavgax2, bavgbx2));
            bavgx4 = vaddq_s32(bavgx4, rgb_avg_rndx4);
            bavgx4 = vshrq_n_s32(bavgx4, CHROMA_AVG_ROUNDING);

            uox4 = vmlaq_n_s32(out_rndx4, ravgx4, cru);
            uox4 = vmlaq_n_s32(uox4, gavgx4, ocgu);
            uox4 = vmlaq_n_s32(uox4, bavgx4, cburv);
            uox4 = vshrq_n_s32(uox4, EIGHT_BIT_SCALE_SHIFT);
            uox4 = vaddq_s32(uox4, out_uv_offsetx4);
            vst1_lane_u32((uint32_t *) &dstu[x >> 1], vreinterpret_u32_u8(vqmovun_s16(vcombine_s16(vmovn_s32(uox4), vdup_n_s16(0)))), 0);

            vox4 = vmlaq_n_s32(out_rndx4, ravgx4, cburv);
            vox4 = vmlaq_n_s32(vox4, gavgx4, ocgv);
            vox4 = vmlaq_n_s32(vox4, bavgx4, cbv);
            vox4 = vshrq_n_s32(vox4, EIGHT_BIT_SCALE_SHIFT);
            vox4 = vaddq_s32(vox4, out_uv_offsetx4);
            vst1_lane_u32((uint32_t *) &dstv[x >> 1], vreinterpret_u32_u8(vqmovun_s16(vcombine_s16(vmovn_s32(vox4), vdup_n_s16(0)))), 0);
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
#endif // ENABLE_TONEMAPX_NEON_INTRINSICS
}

void tonemap_frame_p010_2_nv12_neon(uint8_t *dsty, uint8_t *dstuv,
                                    const uint16_t *srcy, const uint16_t *srcuv,
                                    const int *dstlinesize, const int *srclinesize,
                                    int dstdepth, int srcdepth,
                                    int width, int height,
                                    const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_NEON_INTRINSICS
    uint8_t *rdsty = dsty;
    uint8_t *rdstuv = dstuv;
    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcuv = srcuv;
    int rheight = height;
    // not zero when not divisible by 8
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 6;

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
    uint16x8_t in_yuv_offx8 = vdupq_n_u16(params->in_yuv_off);
    uint16x8_t in_uv_offx8 = vdupq_n_u16(TEN_BIT_UV_OFFSET);
    uint16x8_t uvx8;
    uint16x4_t ux2a, vx2a, ux2b, vx2b;
    uint16x8_t y0x8, y1x8, ux8, vx8;
    uint16x8_t r0x8, g0x8, b0x8;
    uint16x8_t r1x8, g1x8, b1x8;

    int16x8_t r0ox8, g0ox8, b0ox8;
    int16x8_t y0ox8;
    int32x4_t r0oax4, r0obx4, g0oax4, g0obx4, b0oax4, b0obx4;
    int32x4_t y0oax4, y0obx4;

    int16x8_t r1ox8, g1ox8, b1ox8;
    int16x8_t y1ox8;
    int32x4_t r1oax4, r1obx4, g1oax4, g1obx4, b1oax4, b1obx4;
    int32x4_t y1oax4, y1obx4;
    int32x4_t uvoax4, uvobx4;
    int32x2_t ravgax2, gavgax2, bavgax2, ravgbx2, gavgbx2, bavgbx2;
    int32x4_t ravgx4, gavgx4, bavgx4, uox4, vox4;
    int32x4_t out_yuv_offx4 = vdupq_n_s32(params->out_yuv_off);
    int32x4_t out_rndx4 = vdupq_n_s32(EIGHT_BIT_ROUNDING);
    int32x4_t out_uv_offsetx4 = vdupq_n_s32(EIGHT_BIT_UV_OFFSET);
    int32x4_t rgb_avg_rndx4 = vdupq_n_s32(CHROMA_AVG_ROUNDING);
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0] * 2, dstuv += dstlinesize[1],
                       srcy += srclinesize[0], srcuv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 3; xx++) {
            int x = xx << 3;

            y0x8 = vld1q_u16(srcy + x);
            y1x8 = vld1q_u16(srcy + (srclinesize[0] / 2 + x));
            uvx8 = vld1q_u16(srcuv + x);
            // shift to low10bits for 10bit input
            // shift bit has to be compile-time constant
            y0x8 = vshrq_n_u16(y0x8, 6);
            y1x8 = vshrq_n_u16(y1x8, 6);
            uvx8 = vshrq_n_u16(uvx8, 6);
            y0x8 = vsubq_u16(y0x8, in_yuv_offx8);
            y0x8 = vreinterpretq_u16_s16(vmaxq_s16(vreinterpretq_s16_u16(y0x8), vdupq_n_s16(0)));
            y1x8 = vsubq_u16(y1x8, in_yuv_offx8);
            y1x8 = vreinterpretq_u16_s16(vmaxq_s16(vreinterpretq_s16_u16(y1x8), vdupq_n_s16(0)));
            uvx8 = vsubq_u16(uvx8, in_uv_offx8);

            ux2a = vext_u16(vdup_lane_u16(vget_low_u16(uvx8), 0), vdup_lane_u16(vget_low_u16(uvx8), 2), 2);
            vx2a = vext_u16(vdup_lane_u16(vget_low_u16(uvx8), 1), vdup_lane_u16(vget_low_u16(uvx8), 3), 2);
            ux2b = vext_u16(vdup_lane_u16(vget_high_u16(uvx8), 0), vdup_lane_u16(vget_high_u16(uvx8), 2), 2);
            vx2b = vext_u16(vdup_lane_u16(vget_high_u16(uvx8), 1), vdup_lane_u16(vget_high_u16(uvx8), 3), 2);

            ux8 = vcombine_u16(ux2a, ux2b);
            vx8 = vcombine_u16(vx2a, vx2b);

            yuv2rgbx8(&r0x8, &g0x8, &b0x8, y0x8, ux8, vx8, cy, crv, cgu, cgv, cbu);
            yuv2rgbx8(&r1x8, &g1x8, &b1x8, y1x8, ux8, vx8, cy, crv, cgu, cgv, cbu);

            tonemap_int16x8_neon(r0x8, g0x8, b0x8, (int16_t *) &r, (int16_t *) &g, (int16_t *) &b,
                                 params->lin_lut, params->tonemap_lut, params->delin_lut,
                                 params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                 params->rgb2rgb_passthrough);
            tonemap_int16x8_neon(r1x8, g1x8, b1x8, (int16_t *) &r1, (int16_t *) &g1, (int16_t *) &b1,
                                 params->lin_lut, params->tonemap_lut, params->delin_lut,
                                 params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                 params->rgb2rgb_passthrough);

            r0ox8 = vld1q_s16(r);
            g0ox8 = vld1q_s16(g);
            b0ox8 = vld1q_s16(b);

            r0oax4 = vmovl_s16(vget_low_s16(r0ox8));
            g0oax4 = vmovl_s16(vget_low_s16(g0ox8));
            b0oax4 = vmovl_s16(vget_low_s16(b0ox8));

            r0obx4 = vmovl_s16(vget_high_s16(r0ox8));
            g0obx4 = vmovl_s16(vget_high_s16(g0ox8));
            b0obx4 = vmovl_s16(vget_high_s16(b0ox8));

            y0oax4 = vmulq_n_s32(r0oax4, cry);
            y0oax4 = vmlaq_n_s32(y0oax4, g0oax4, cgy);
            y0oax4 = vmlaq_n_s32(y0oax4, b0oax4, cby);
            y0oax4 = vaddq_s32(y0oax4, out_rndx4);
            // output shift bits for 8bit outputs is 29 - 8 = 21
            y0oax4 = vshrq_n_s32(y0oax4, EIGHT_BIT_SCALE_SHIFT);
            y0oax4 = vaddq_s32(y0oax4, out_yuv_offx4);

            y0obx4 = vmulq_n_s32(r0obx4, cry);
            y0obx4 = vmlaq_n_s32(y0obx4, g0obx4, cgy);
            y0obx4 = vmlaq_n_s32(y0obx4, b0obx4, cby);
            y0obx4 = vaddq_s32(y0obx4, out_rndx4);
            y0obx4 = vshrq_n_s32(y0obx4, EIGHT_BIT_SCALE_SHIFT);
            y0obx4 = vaddq_s32(y0obx4, out_yuv_offx4);

            y0ox8 = vcombine_s16(vqmovn_s32(y0oax4), vqmovn_s32(y0obx4));
            vst1_u8(&dsty[x], vqmovun_s16(y0ox8));

            r1ox8 = vld1q_s16(r1);
            g1ox8 = vld1q_s16(g1);
            b1ox8 = vld1q_s16(b1);

            r1oax4 = vmovl_s16(vget_low_s16(r1ox8));
            g1oax4 = vmovl_s16(vget_low_s16(g1ox8));
            b1oax4 = vmovl_s16(vget_low_s16(b1ox8));

            r1obx4 = vmovl_s16(vget_high_s16(r1ox8));
            g1obx4 = vmovl_s16(vget_high_s16(g1ox8));
            b1obx4 = vmovl_s16(vget_high_s16(b1ox8));

            y1oax4 = vmulq_n_s32(r1oax4, cry);
            y1oax4 = vmlaq_n_s32(y1oax4, g1oax4, cgy);
            y1oax4 = vmlaq_n_s32(y1oax4, b1oax4, cby);
            y1oax4 = vaddq_s32(y1oax4, out_rndx4);
            y1oax4 = vshrq_n_s32(y1oax4, EIGHT_BIT_SCALE_SHIFT);
            y1oax4 = vaddq_s32(y1oax4, out_yuv_offx4);

            y1obx4 = vmulq_n_s32(r1obx4, cry);
            y1obx4 = vmlaq_n_s32(y1obx4, g1obx4, cgy);
            y1obx4 = vmlaq_n_s32(y1obx4, b1obx4, cby);
            y1obx4 = vaddq_s32(y1obx4, out_rndx4);
            y1obx4 = vshrq_n_s32(y1obx4, EIGHT_BIT_SCALE_SHIFT);
            y1obx4 = vaddq_s32(y1obx4, out_yuv_offx4);

            y1ox8 = vcombine_s16(vqmovn_s32(y1oax4), vqmovn_s32(y1obx4));
            vst1_u8(&dsty[x + dstlinesize[0]], vqmovun_s16(y1ox8));

            ravgax2 = vpadd_s32(vget_low_s32(r0oax4), vget_high_s32(r0oax4));
            ravgbx2 = vpadd_s32(vget_low_s32(r0obx4), vget_high_s32(r0obx4));
            ravgx4 = vcombine_s32(ravgax2, ravgbx2);
            ravgax2 = vpadd_s32(vget_low_s32(r1oax4), vget_high_s32(r1oax4));
            ravgbx2 = vpadd_s32(vget_low_s32(r1obx4), vget_high_s32(r1obx4));
            ravgx4 = vaddq_s32(ravgx4, vcombine_s32(ravgax2, ravgbx2));
            ravgx4 = vaddq_s32(ravgx4, rgb_avg_rndx4);
            ravgx4 = vshrq_n_s32(ravgx4, CHROMA_AVG_ROUNDING);

            gavgax2 = vpadd_s32(vget_low_s32(g0oax4), vget_high_s32(g0oax4));
            gavgbx2 = vpadd_s32(vget_low_s32(g0obx4), vget_high_s32(g0obx4));
            gavgx4 = vcombine_s32(gavgax2, gavgbx2);
            gavgax2 = vpadd_s32(vget_low_s32(g1oax4), vget_high_s32(g1oax4));
            gavgbx2 = vpadd_s32(vget_low_s32(g1obx4), vget_high_s32(g1obx4));
            gavgx4 = vaddq_s32(gavgx4, vcombine_s32(gavgax2, gavgbx2));
            gavgx4 = vaddq_s32(gavgx4, rgb_avg_rndx4);
            gavgx4 = vshrq_n_s32(gavgx4, CHROMA_AVG_ROUNDING);

            bavgax2 = vpadd_s32(vget_low_s32(b0oax4), vget_high_s32(b0oax4));
            bavgbx2 = vpadd_s32(vget_low_s32(b0obx4), vget_high_s32(b0obx4));
            bavgx4 = vcombine_s32(bavgax2, bavgbx2);
            bavgax2 = vpadd_s32(vget_low_s32(b1oax4), vget_high_s32(b1oax4));
            bavgbx2 = vpadd_s32(vget_low_s32(b1obx4), vget_high_s32(b1obx4));
            bavgx4 = vaddq_s32(bavgx4, vcombine_s32(bavgax2, bavgbx2));
            bavgx4 = vaddq_s32(bavgx4, rgb_avg_rndx4);
            bavgx4 = vshrq_n_s32(bavgx4, CHROMA_AVG_ROUNDING);

            uox4 = vmlaq_n_s32(out_rndx4, ravgx4, cru);
            uox4 = vmlaq_n_s32(uox4, gavgx4, ocgu);
            uox4 = vmlaq_n_s32(uox4, bavgx4, cburv);
            uox4 = vshrq_n_s32(uox4, EIGHT_BIT_SCALE_SHIFT);
            uox4 = vaddq_s32(uox4, out_uv_offsetx4);

            vox4 = vmlaq_n_s32(out_rndx4, ravgx4, cburv);
            vox4 = vmlaq_n_s32(vox4, gavgx4, ocgv);
            vox4 = vmlaq_n_s32(vox4, bavgx4, cbv);
            vox4 = vshrq_n_s32(vox4, EIGHT_BIT_SCALE_SHIFT);
            vox4 = vaddq_s32(vox4, out_uv_offsetx4);

            uvoax4 = vzip1q_s32(uox4, vox4);
            uvobx4 = vzip2q_s32(uox4, vox4);

            vst1_u8(&dstuv[x], vqmovun_s16(vcombine_s16(vmovn_s32(uvoax4), vmovn_s32(uvobx4))));
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
#endif // ENABLE_TONEMAPX_NEON_INTRINSICS
}

void tonemap_frame_dovi_2_420p10_neon(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                      const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                      const int *dstlinesize, const int *srclinesize,
                                      int dstdepth, int srcdepth,
                                      int width, int height,
                                      const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_NEON_INTRINSICS
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
    uint16x4_t ux4, vx4;
    uint16x8_t y0x8, y1x8, ux8, vx8;
    uint16x8_t r0x8, g0x8, b0x8;
    uint16x8_t r1x8, g1x8, b1x8;

    int16x8_t r0ox8, g0ox8, b0ox8;
    uint16x8_t y0ox8;
    int32x4_t r0oax4, r0obx4, g0oax4, g0obx4, b0oax4, b0obx4;
    int32x4_t y0oax4, y0obx4;

    int16x8_t r1ox8, g1ox8, b1ox8;
    uint16x8_t y1ox8;
    int32x4_t r1oax4, r1obx4, g1oax4, g1obx4, b1oax4, b1obx4;
    int32x4_t y1oax4, y1obx4;
    int32x2_t ravgax2, gavgax2, bavgax2, ravgbx2, gavgbx2, bavgbx2;
    int32x4_t ravgx4, gavgx4, bavgx4, uox4, vox4;
    int32x4_t out_yuv_offx4 = vdupq_n_s32(params->out_yuv_off);
    int32x4_t out_rndx4 = vdupq_n_s32(TEN_BIT_ROUNDING);
    int32x4_t out_uv_offsetx4 = vdupq_n_s32(TEN_BIT_UV_OFFSET);
    int32x4_t rgb_avg_rndx4 = vdupq_n_s32(CHROMA_AVG_ROUNDING);
    float32x4_t ipt0, ipt1, ipt2, ipt3;
    float32x4_t ia1, ib1, ia2, ib2;
    float32x4_t ix4, px4, tx4;
    float32x4_t lx4, mx4, sx4;
    float32x4_t rx4a, gx4a, bx4a, rx4b, gx4b, bx4b;
    float32x4_t y0x4a, y0x4b, y1x4a, y1x4b, ux4a, ux4b, vx4a, vx4b;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstu += dstlinesize[1] / 2, dstv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 3; xx++) {
            int x = xx << 3;

            y0x8 = vld1q_u16(srcy + x);
            y1x8 = vld1q_u16(srcy + (srclinesize[0] / 2 + x));
            ux4 = vld1_u16(srcu + (x >> 1));
            vx4 = vld1_u16(srcv + (x >> 1));

            ux8 = vcombine_u16(vzip1_u16(ux4, ux4), vzip2_u16(ux4, ux4));
            vx8 = vcombine_u16(vzip1_u16(vx4, vx4), vzip2_u16(vx4, vx4));

            y0x4a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(y0x8)));
            y0x4b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(y0x8)));
            y1x4a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(y1x8)));
            y1x4b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(y1x8)));

            ux4a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(ux8)));
            ux4b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(ux8)));
            vx4a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vx8)));
            vx4b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vx8)));

            y0x4a = vdivq_f32(y0x4a, vdupq_n_f32(TEN_BIT_SCALE));
            y0x4b = vdivq_f32(y0x4b, vdupq_n_f32(TEN_BIT_SCALE));
            y1x4a = vdivq_f32(y1x4a, vdupq_n_f32(TEN_BIT_SCALE));
            y1x4b = vdivq_f32(y1x4b, vdupq_n_f32(TEN_BIT_SCALE));
            ux4a = vdivq_f32(ux4a, vdupq_n_f32(TEN_BIT_SCALE));
            ux4b = vdivq_f32(ux4b, vdupq_n_f32(TEN_BIT_SCALE));
            vx4a = vdivq_f32(vx4a, vdupq_n_f32(TEN_BIT_SCALE));
            vx4b = vdivq_f32(vx4b, vdupq_n_f32(TEN_BIT_SCALE));

            // Reshape y0x4a
            ia1 = vzip1q_f32(y0x4a, ux4a);
            ia2 = vzip2q_f32(y0x4a, ux4a);
            ib1 = vzip1q_f32(vx4a, vdupq_n_f32(0.0f));
            ib2 = vzip2q_f32(vx4a, vdupq_n_f32(0.0f));
            ipt0 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ib1));
            ipt1 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ib1));
            ipt2 = vcombine_f32(vget_low_f32(ia2), vget_low_f32(ib2));
            ipt3 = vcombine_f32(vget_high_f32(ia2), vget_high_f32(ib2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ia1 = vtrn1q_f32(ipt0, ipt1);
            ia2 = vtrn1q_f32(ipt2, ipt3);
            ib1 = vtrn2q_f32(ipt0, ipt1);
            ib2 = vtrn2q_f32(ipt2, ipt3);

            ix4 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ia2));
            px4 = vcombine_f32(vget_low_f32(ib1), vget_low_f32(ib2));
            tx4 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ia2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4a, &gx4a, &bx4a, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4a = vmulq_n_f32(rx4a, JPEG_SCALE);
            gx4a = vmulq_n_f32(gx4a, JPEG_SCALE);
            bx4a = vmulq_n_f32(bx4a, JPEG_SCALE);

            // Reshape y0x4b
            ia1 = vzip1q_f32(y0x4b, ux4b);
            ia2 = vzip2q_f32(y0x4b, ux4b);
            ib1 = vzip1q_f32(vx4b, vdupq_n_f32(0.0f));
            ib2 = vzip2q_f32(vx4b, vdupq_n_f32(0.0f));
            ipt0 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ib1));
            ipt1 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ib1));
            ipt2 = vcombine_f32(vget_low_f32(ia2), vget_low_f32(ib2));
            ipt3 = vcombine_f32(vget_high_f32(ia2), vget_high_f32(ib2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ia1 = vtrn1q_f32(ipt0, ipt1);
            ia2 = vtrn1q_f32(ipt2, ipt3);
            ib1 = vtrn2q_f32(ipt0, ipt1);
            ib2 = vtrn2q_f32(ipt2, ipt3);

            ix4 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ia2));
            px4 = vcombine_f32(vget_low_f32(ib1), vget_low_f32(ib2));
            tx4 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ia2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4b, &gx4b, &bx4b, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4b = vmulq_n_f32(rx4b, JPEG_SCALE);
            gx4b = vmulq_n_f32(gx4b, JPEG_SCALE);
            bx4b = vmulq_n_f32(bx4b, JPEG_SCALE);

            r0x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(rx4a)), vqmovn_u32(vcvtq_u32_f32(rx4b)));
            g0x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(gx4a)), vqmovn_u32(vcvtq_u32_f32(gx4b)));
            b0x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(bx4a)), vqmovn_u32(vcvtq_u32_f32(bx4b)));
            r0x8 = vminq_u16(r0x8, vdupq_n_u16(INT16_MAX));
            g0x8 = vminq_u16(g0x8, vdupq_n_u16(INT16_MAX));
            b0x8 = vminq_u16(b0x8, vdupq_n_u16(INT16_MAX));

            // Reshape y1x4a
            ia1 = vzip1q_f32(y1x4a, ux4a);
            ia2 = vzip2q_f32(y1x4a, ux4a);
            ib1 = vzip1q_f32(vx4a, vdupq_n_f32(0.0f));
            ib2 = vzip2q_f32(vx4a, vdupq_n_f32(0.0f));
            ipt0 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ib1));
            ipt1 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ib1));
            ipt2 = vcombine_f32(vget_low_f32(ia2), vget_low_f32(ib2));
            ipt3 = vcombine_f32(vget_high_f32(ia2), vget_high_f32(ib2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ia1 = vtrn1q_f32(ipt0, ipt1);
            ia2 = vtrn1q_f32(ipt2, ipt3);
            ib1 = vtrn2q_f32(ipt0, ipt1);
            ib2 = vtrn2q_f32(ipt2, ipt3);

            ix4 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ia2));
            px4 = vcombine_f32(vget_low_f32(ib1), vget_low_f32(ib2));
            tx4 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ia2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4a, &gx4a, &bx4a, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4a = vmulq_n_f32(rx4a, JPEG_SCALE);
            gx4a = vmulq_n_f32(gx4a, JPEG_SCALE);
            bx4a = vmulq_n_f32(bx4a, JPEG_SCALE);

            // Reshape y1x4b
            ia1 = vzip1q_f32(y1x4b, ux4b);
            ia2 = vzip2q_f32(y1x4b, ux4b);
            ib1 = vzip1q_f32(vx4b, vdupq_n_f32(0.0f));
            ib2 = vzip2q_f32(vx4b, vdupq_n_f32(0.0f));
            ipt0 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ib1));
            ipt1 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ib1));
            ipt2 = vcombine_f32(vget_low_f32(ia2), vget_low_f32(ib2));
            ipt3 = vcombine_f32(vget_high_f32(ia2), vget_high_f32(ib2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ia1 = vtrn1q_f32(ipt0, ipt1);
            ia2 = vtrn1q_f32(ipt2, ipt3);
            ib1 = vtrn2q_f32(ipt0, ipt1);
            ib2 = vtrn2q_f32(ipt2, ipt3);

            ix4 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ia2));
            px4 = vcombine_f32(vget_low_f32(ib1), vget_low_f32(ib2));
            tx4 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ia2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4b, &gx4b, &bx4b, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4b = vmulq_n_f32(rx4b, JPEG_SCALE);
            gx4b = vmulq_n_f32(gx4b, JPEG_SCALE);
            bx4b = vmulq_n_f32(bx4b, JPEG_SCALE);

            r1x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(rx4a)), vqmovn_u32(vcvtq_u32_f32(rx4b)));
            g1x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(gx4a)), vqmovn_u32(vcvtq_u32_f32(gx4b)));
            b1x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(bx4a)), vqmovn_u32(vcvtq_u32_f32(bx4b)));
            r1x8 = vminq_u16(r1x8, vdupq_n_u16(INT16_MAX));
            g1x8 = vminq_u16(g1x8, vdupq_n_u16(INT16_MAX));
            b1x8 = vminq_u16(b1x8, vdupq_n_u16(INT16_MAX));

            tonemap_int16x8_neon(r0x8, g0x8, b0x8, (int16_t *) &r, (int16_t *) &g, (int16_t *) &b,
                                 params->lin_lut, params->tonemap_lut, params->delin_lut,
                                 params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                 params->rgb2rgb_passthrough);
            tonemap_int16x8_neon(r1x8, g1x8, b1x8, (int16_t *) &r1, (int16_t *) &g1, (int16_t *) &b1,
                                 params->lin_lut, params->tonemap_lut, params->delin_lut,
                                 params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                 params->rgb2rgb_passthrough);

            r0ox8 = vld1q_s16(r);
            g0ox8 = vld1q_s16(g);
            b0ox8 = vld1q_s16(b);

            r0oax4 = vmovl_s16(vget_low_s16(r0ox8));
            g0oax4 = vmovl_s16(vget_low_s16(g0ox8));
            b0oax4 = vmovl_s16(vget_low_s16(b0ox8));

            r0obx4 = vmovl_s16(vget_high_s16(r0ox8));
            g0obx4 = vmovl_s16(vget_high_s16(g0ox8));
            b0obx4 = vmovl_s16(vget_high_s16(b0ox8));

            y0oax4 = vmulq_n_s32(r0oax4, cry);
            y0oax4 = vmlaq_n_s32(y0oax4, g0oax4, cgy);
            y0oax4 = vmlaq_n_s32(y0oax4, b0oax4, cby);
            y0oax4 = vaddq_s32(y0oax4, out_rndx4);
            y0oax4 = vshrq_n_s32(y0oax4, TEN_BIT_SCALE_SHIFT);
            y0oax4 = vaddq_s32(y0oax4, out_yuv_offx4);

            y0obx4 = vmulq_n_s32(r0obx4, cry);
            y0obx4 = vmlaq_n_s32(y0obx4, g0obx4, cgy);
            y0obx4 = vmlaq_n_s32(y0obx4, b0obx4, cby);
            y0obx4 = vaddq_s32(y0obx4, out_rndx4);
            y0obx4 = vshrq_n_s32(y0obx4, TEN_BIT_SCALE_SHIFT);
            y0obx4 = vaddq_s32(y0obx4, out_yuv_offx4);

            y0ox8 = vcombine_u16(vqmovun_s32(y0oax4), vqmovun_s32(y0obx4));
            vst1q_u16(&dsty[x], y0ox8);

            r1ox8 = vld1q_s16(r1);
            g1ox8 = vld1q_s16(g1);
            b1ox8 = vld1q_s16(b1);

            r1oax4 = vmovl_s16(vget_low_s16(r1ox8));
            g1oax4 = vmovl_s16(vget_low_s16(g1ox8));
            b1oax4 = vmovl_s16(vget_low_s16(b1ox8));

            r1obx4 = vmovl_s16(vget_high_s16(r1ox8));
            g1obx4 = vmovl_s16(vget_high_s16(g1ox8));
            b1obx4 = vmovl_s16(vget_high_s16(b1ox8));

            y1oax4 = vmulq_n_s32(r1oax4, cry);
            y1oax4 = vmlaq_n_s32(y1oax4, g1oax4, cgy);
            y1oax4 = vmlaq_n_s32(y1oax4, b1oax4, cby);
            y1oax4 = vaddq_s32(y1oax4, out_rndx4);
            y1oax4 = vshrq_n_s32(y1oax4, TEN_BIT_SCALE_SHIFT);
            y1oax4 = vaddq_s32(y1oax4, out_yuv_offx4);

            y1obx4 = vmulq_n_s32(r1obx4, cry);
            y1obx4 = vmlaq_n_s32(y1obx4, g1obx4, cgy);
            y1obx4 = vmlaq_n_s32(y1obx4, b1obx4, cby);
            y1obx4 = vaddq_s32(y1obx4, out_rndx4);
            y1obx4 = vshrq_n_s32(y1obx4, TEN_BIT_SCALE_SHIFT);
            y1obx4 = vaddq_s32(y1obx4, out_yuv_offx4);

            y1ox8 = vcombine_u16(vqmovun_s32(y1oax4), vqmovun_s32(y1obx4));
            vst1q_u16(&dsty[x + dstlinesize[0] / 2], y1ox8);

            ravgax2 = vpadd_s32(vget_low_s32(r0oax4), vget_high_s32(r0oax4));
            ravgbx2 = vpadd_s32(vget_low_s32(r0obx4), vget_high_s32(r0obx4));
            ravgx4 = vcombine_s32(ravgax2, ravgbx2);
            ravgax2 = vpadd_s32(vget_low_s32(r1oax4), vget_high_s32(r1oax4));
            ravgbx2 = vpadd_s32(vget_low_s32(r1obx4), vget_high_s32(r1obx4));
            ravgx4 = vaddq_s32(ravgx4, vcombine_s32(ravgax2, ravgbx2));
            ravgx4 = vaddq_s32(ravgx4, rgb_avg_rndx4);
            ravgx4 = vshrq_n_s32(ravgx4, CHROMA_AVG_ROUNDING);

            gavgax2 = vpadd_s32(vget_low_s32(g0oax4), vget_high_s32(g0oax4));
            gavgbx2 = vpadd_s32(vget_low_s32(g0obx4), vget_high_s32(g0obx4));
            gavgx4 = vcombine_s32(gavgax2, gavgbx2);
            gavgax2 = vpadd_s32(vget_low_s32(g1oax4), vget_high_s32(g1oax4));
            gavgbx2 = vpadd_s32(vget_low_s32(g1obx4), vget_high_s32(g1obx4));
            gavgx4 = vaddq_s32(gavgx4, vcombine_s32(gavgax2, gavgbx2));
            gavgx4 = vaddq_s32(gavgx4, rgb_avg_rndx4);
            gavgx4 = vshrq_n_s32(gavgx4, CHROMA_AVG_ROUNDING);

            bavgax2 = vpadd_s32(vget_low_s32(b0oax4), vget_high_s32(b0oax4));
            bavgbx2 = vpadd_s32(vget_low_s32(b0obx4), vget_high_s32(b0obx4));
            bavgx4 = vcombine_s32(bavgax2, bavgbx2);
            bavgax2 = vpadd_s32(vget_low_s32(b1oax4), vget_high_s32(b1oax4));
            bavgbx2 = vpadd_s32(vget_low_s32(b1obx4), vget_high_s32(b1obx4));
            bavgx4 = vaddq_s32(bavgx4, vcombine_s32(bavgax2, bavgbx2));
            bavgx4 = vaddq_s32(bavgx4, rgb_avg_rndx4);
            bavgx4 = vshrq_n_s32(bavgx4, CHROMA_AVG_ROUNDING);

            uox4 = vmlaq_n_s32(out_rndx4, ravgx4, cru);
            uox4 = vmlaq_n_s32(uox4, gavgx4, ocgu);
            uox4 = vmlaq_n_s32(uox4, bavgx4, cburv);
            uox4 = vshrq_n_s32(uox4, TEN_BIT_SCALE_SHIFT);
            uox4 = vaddq_s32(uox4, out_uv_offsetx4);
            vst1_u16(&dstu[x >> 1], vqmovun_s32(uox4));

            vox4 = vmlaq_n_s32(out_rndx4, ravgx4, cburv);
            vox4 = vmlaq_n_s32(vox4, gavgx4, ocgv);
            vox4 = vmlaq_n_s32(vox4, bavgx4, cbv);
            vox4 = vshrq_n_s32(vox4, TEN_BIT_SCALE_SHIFT);
            vox4 = vaddq_s32(vox4, out_uv_offsetx4);
            vst1_u16(&dstv[x >> 1], vqmovun_s32(vox4));
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
#endif // ENABLE_TONEMAPX_NEON_INTRINSICS
}

void tonemap_frame_dovi_2_420hdr_neon(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                      const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                      const int *dstlinesize, const int *srclinesize,
                                      int dstdepth, int srcdepth,
                                      int width, int height,
                                      const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_NEON_INTRINSICS
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

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    uint16x4_t ux4, vx4;
    uint16x8_t y0x8, y1x8, ux8, vx8;
    uint16x8_t r0x8, g0x8, b0x8;
    uint16x8_t r1x8, g1x8, b1x8;

    int16x8_t r0ox8, g0ox8, b0ox8;
    uint16x8_t y0ox8;
    int32x4_t r0oax4, r0obx4, g0oax4, g0obx4, b0oax4, b0obx4;
    int32x4_t y0oax4, y0obx4;

    int16x8_t r1ox8, g1ox8, b1ox8;
    uint16x8_t y1ox8;
    int32x4_t r1oax4, r1obx4, g1oax4, g1obx4, b1oax4, b1obx4;
    int32x4_t y1oax4, y1obx4;
    int32x2_t ravgax2, gavgax2, bavgax2, ravgbx2, gavgbx2, bavgbx2;
    int32x4_t ravgx4, gavgx4, bavgx4, uox4, vox4;
    int32x4_t out_yuv_offx4 = vdupq_n_s32(params->out_yuv_off);
    int32x4_t out_rndx4 = vdupq_n_s32(TEN_BIT_ROUNDING);
    int32x4_t out_uv_offsetx4 = vdupq_n_s32(TEN_BIT_UV_OFFSET);
    int32x4_t rgb_avg_rndx4 = vdupq_n_s32(CHROMA_AVG_ROUNDING);
    float32x4_t ipt0, ipt1, ipt2, ipt3;
    float32x4_t ia1, ib1, ia2, ib2;
    float32x4_t ix4, px4, tx4;
    float32x4_t lx4, mx4, sx4;
    float32x4_t rx4a, gx4a, bx4a, rx4b, gx4b, bx4b;
    float32x4_t y0x4a, y0x4b, y1x4a, y1x4b, ux4a, ux4b, vx4a, vx4b;
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstu += dstlinesize[1] / 2, dstv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 3; xx++) {
            int x = xx << 3;

            y0x8 = vld1q_u16(srcy + x);
            y1x8 = vld1q_u16(srcy + (srclinesize[0] / 2 + x));
            ux4 = vld1_u16(srcu + (x >> 1));
            vx4 = vld1_u16(srcv + (x >> 1));

            ux8 = vcombine_u16(vzip1_u16(ux4, ux4), vzip2_u16(ux4, ux4));
            vx8 = vcombine_u16(vzip1_u16(vx4, vx4), vzip2_u16(vx4, vx4));

            y0x4a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(y0x8)));
            y0x4b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(y0x8)));
            y1x4a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(y1x8)));
            y1x4b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(y1x8)));

            ux4a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(ux8)));
            ux4b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(ux8)));
            vx4a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vx8)));
            vx4b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vx8)));

            y0x4a = vdivq_f32(y0x4a, vdupq_n_f32(TEN_BIT_SCALE));
            y0x4b = vdivq_f32(y0x4b, vdupq_n_f32(TEN_BIT_SCALE));
            y1x4a = vdivq_f32(y1x4a, vdupq_n_f32(TEN_BIT_SCALE));
            y1x4b = vdivq_f32(y1x4b, vdupq_n_f32(TEN_BIT_SCALE));
            ux4a = vdivq_f32(ux4a, vdupq_n_f32(TEN_BIT_SCALE));
            ux4b = vdivq_f32(ux4b, vdupq_n_f32(TEN_BIT_SCALE));
            vx4a = vdivq_f32(vx4a, vdupq_n_f32(TEN_BIT_SCALE));
            vx4b = vdivq_f32(vx4b, vdupq_n_f32(TEN_BIT_SCALE));

            // Reshape y0x4a
            ia1 = vzip1q_f32(y0x4a, ux4a);
            ia2 = vzip2q_f32(y0x4a, ux4a);
            ib1 = vzip1q_f32(vx4a, vdupq_n_f32(0.0f));
            ib2 = vzip2q_f32(vx4a, vdupq_n_f32(0.0f));
            ipt0 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ib1));
            ipt1 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ib1));
            ipt2 = vcombine_f32(vget_low_f32(ia2), vget_low_f32(ib2));
            ipt3 = vcombine_f32(vget_high_f32(ia2), vget_high_f32(ib2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ia1 = vtrn1q_f32(ipt0, ipt1);
            ia2 = vtrn1q_f32(ipt2, ipt3);
            ib1 = vtrn2q_f32(ipt0, ipt1);
            ib2 = vtrn2q_f32(ipt2, ipt3);

            ix4 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ia2));
            px4 = vcombine_f32(vget_low_f32(ib1), vget_low_f32(ib2));
            tx4 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ia2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4a, &gx4a, &bx4a, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4a = vmulq_n_f32(rx4a, JPEG_SCALE);
            gx4a = vmulq_n_f32(gx4a, JPEG_SCALE);
            bx4a = vmulq_n_f32(bx4a, JPEG_SCALE);

            // Reshape y0x4b
            ia1 = vzip1q_f32(y0x4b, ux4b);
            ia2 = vzip2q_f32(y0x4b, ux4b);
            ib1 = vzip1q_f32(vx4b, vdupq_n_f32(0.0f));
            ib2 = vzip2q_f32(vx4b, vdupq_n_f32(0.0f));
            ipt0 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ib1));
            ipt1 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ib1));
            ipt2 = vcombine_f32(vget_low_f32(ia2), vget_low_f32(ib2));
            ipt3 = vcombine_f32(vget_high_f32(ia2), vget_high_f32(ib2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ia1 = vtrn1q_f32(ipt0, ipt1);
            ia2 = vtrn1q_f32(ipt2, ipt3);
            ib1 = vtrn2q_f32(ipt0, ipt1);
            ib2 = vtrn2q_f32(ipt2, ipt3);

            ix4 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ia2));
            px4 = vcombine_f32(vget_low_f32(ib1), vget_low_f32(ib2));
            tx4 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ia2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4b, &gx4b, &bx4b, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4b = vmulq_n_f32(rx4b, JPEG_SCALE);
            gx4b = vmulq_n_f32(gx4b, JPEG_SCALE);
            bx4b = vmulq_n_f32(bx4b, JPEG_SCALE);

            r0x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(rx4a)), vqmovn_u32(vcvtq_u32_f32(rx4b)));
            g0x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(gx4a)), vqmovn_u32(vcvtq_u32_f32(gx4b)));
            b0x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(bx4a)), vqmovn_u32(vcvtq_u32_f32(bx4b)));
            r0x8 = vminq_u16(r0x8, vdupq_n_u16(INT16_MAX));
            g0x8 = vminq_u16(g0x8, vdupq_n_u16(INT16_MAX));
            b0x8 = vminq_u16(b0x8, vdupq_n_u16(INT16_MAX));

            // Reshape y1x4a
            ia1 = vzip1q_f32(y1x4a, ux4a);
            ia2 = vzip2q_f32(y1x4a, ux4a);
            ib1 = vzip1q_f32(vx4a, vdupq_n_f32(0.0f));
            ib2 = vzip2q_f32(vx4a, vdupq_n_f32(0.0f));
            ipt0 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ib1));
            ipt1 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ib1));
            ipt2 = vcombine_f32(vget_low_f32(ia2), vget_low_f32(ib2));
            ipt3 = vcombine_f32(vget_high_f32(ia2), vget_high_f32(ib2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ia1 = vtrn1q_f32(ipt0, ipt1);
            ia2 = vtrn1q_f32(ipt2, ipt3);
            ib1 = vtrn2q_f32(ipt0, ipt1);
            ib2 = vtrn2q_f32(ipt2, ipt3);

            ix4 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ia2));
            px4 = vcombine_f32(vget_low_f32(ib1), vget_low_f32(ib2));
            tx4 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ia2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4a, &gx4a, &bx4a, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4a = vmulq_n_f32(rx4a, JPEG_SCALE);
            gx4a = vmulq_n_f32(gx4a, JPEG_SCALE);
            bx4a = vmulq_n_f32(bx4a, JPEG_SCALE);

            // Reshape y1x4b
            ia1 = vzip1q_f32(y1x4b, ux4b);
            ia2 = vzip2q_f32(y1x4b, ux4b);
            ib1 = vzip1q_f32(vx4b, vdupq_n_f32(0.0f));
            ib2 = vzip2q_f32(vx4b, vdupq_n_f32(0.0f));
            ipt0 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ib1));
            ipt1 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ib1));
            ipt2 = vcombine_f32(vget_low_f32(ia2), vget_low_f32(ib2));
            ipt3 = vcombine_f32(vget_high_f32(ia2), vget_high_f32(ib2));

            ipt0 = reshape_dovi_iptpqc2(ipt0, params);
            ipt1 = reshape_dovi_iptpqc2(ipt1, params);
            ipt2 = reshape_dovi_iptpqc2(ipt2, params);
            ipt3 = reshape_dovi_iptpqc2(ipt3, params);

            ia1 = vtrn1q_f32(ipt0, ipt1);
            ia2 = vtrn1q_f32(ipt2, ipt3);
            ib1 = vtrn2q_f32(ipt0, ipt1);
            ib2 = vtrn2q_f32(ipt2, ipt3);

            ix4 = vcombine_f32(vget_low_f32(ia1), vget_low_f32(ia2));
            px4 = vcombine_f32(vget_low_f32(ib1), vget_low_f32(ib2));
            tx4 = vcombine_f32(vget_high_f32(ia1), vget_high_f32(ia2));

            ycc2rgbx4(&lx4, &mx4, &sx4, ix4, px4, tx4, params->dovi->nonlinear, *params->ycc_offset);
            lms2rgbx4(&rx4b, &gx4b, &bx4b, lx4, mx4, sx4, *params->lms2rgb_matrix);

            rx4b = vmulq_n_f32(rx4b, JPEG_SCALE);
            gx4b = vmulq_n_f32(gx4b, JPEG_SCALE);
            bx4b = vmulq_n_f32(bx4b, JPEG_SCALE);

            r1x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(rx4a)), vqmovn_u32(vcvtq_u32_f32(rx4b)));
            g1x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(gx4a)), vqmovn_u32(vcvtq_u32_f32(gx4b)));
            b1x8 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(bx4a)), vqmovn_u32(vcvtq_u32_f32(bx4b)));
            r1x8 = vminq_u16(r1x8, vdupq_n_u16(INT16_MAX));
            g1x8 = vminq_u16(g1x8, vdupq_n_u16(INT16_MAX));
            b1x8 = vminq_u16(b1x8, vdupq_n_u16(INT16_MAX));

            r0ox8 = vreinterpretq_s16_u16(r0x8);
            g0ox8 = vreinterpretq_s16_u16(g0x8);
            b0ox8 = vreinterpretq_s16_u16(b0x8);

            r0oax4 = vmovl_s16(vget_low_s16(r0ox8));
            g0oax4 = vmovl_s16(vget_low_s16(g0ox8));
            b0oax4 = vmovl_s16(vget_low_s16(b0ox8));

            r0obx4 = vmovl_s16(vget_high_s16(r0ox8));
            g0obx4 = vmovl_s16(vget_high_s16(g0ox8));
            b0obx4 = vmovl_s16(vget_high_s16(b0ox8));

            y0oax4 = vmulq_n_s32(r0oax4, cry);
            y0oax4 = vmlaq_n_s32(y0oax4, g0oax4, cgy);
            y0oax4 = vmlaq_n_s32(y0oax4, b0oax4, cby);
            y0oax4 = vaddq_s32(y0oax4, out_rndx4);
            y0oax4 = vshrq_n_s32(y0oax4, TEN_BIT_SCALE_SHIFT);
            y0oax4 = vaddq_s32(y0oax4, out_yuv_offx4);

            y0obx4 = vmulq_n_s32(r0obx4, cry);
            y0obx4 = vmlaq_n_s32(y0obx4, g0obx4, cgy);
            y0obx4 = vmlaq_n_s32(y0obx4, b0obx4, cby);
            y0obx4 = vaddq_s32(y0obx4, out_rndx4);
            y0obx4 = vshrq_n_s32(y0obx4, TEN_BIT_SCALE_SHIFT);
            y0obx4 = vaddq_s32(y0obx4, out_yuv_offx4);

            y0ox8 = vcombine_u16(vqmovun_s32(y0oax4), vqmovun_s32(y0obx4));
            vst1q_u16(&dsty[x], y0ox8);

            r1ox8 = vreinterpretq_s16_u16(r1x8);
            g1ox8 = vreinterpretq_s16_u16(g1x8);
            b1ox8 = vreinterpretq_s16_u16(b1x8);

            r1oax4 = vmovl_s16(vget_low_s16(r1ox8));
            g1oax4 = vmovl_s16(vget_low_s16(g1ox8));
            b1oax4 = vmovl_s16(vget_low_s16(b1ox8));

            r1obx4 = vmovl_s16(vget_high_s16(r1ox8));
            g1obx4 = vmovl_s16(vget_high_s16(g1ox8));
            b1obx4 = vmovl_s16(vget_high_s16(b1ox8));

            y1oax4 = vmulq_n_s32(r1oax4, cry);
            y1oax4 = vmlaq_n_s32(y1oax4, g1oax4, cgy);
            y1oax4 = vmlaq_n_s32(y1oax4, b1oax4, cby);
            y1oax4 = vaddq_s32(y1oax4, out_rndx4);
            y1oax4 = vshrq_n_s32(y1oax4, TEN_BIT_SCALE_SHIFT);
            y1oax4 = vaddq_s32(y1oax4, out_yuv_offx4);

            y1obx4 = vmulq_n_s32(r1obx4, cry);
            y1obx4 = vmlaq_n_s32(y1obx4, g1obx4, cgy);
            y1obx4 = vmlaq_n_s32(y1obx4, b1obx4, cby);
            y1obx4 = vaddq_s32(y1obx4, out_rndx4);
            y1obx4 = vshrq_n_s32(y1obx4, TEN_BIT_SCALE_SHIFT);
            y1obx4 = vaddq_s32(y1obx4, out_yuv_offx4);

            y1ox8 = vcombine_u16(vqmovun_s32(y1oax4), vqmovun_s32(y1obx4));
            vst1q_u16(&dsty[x + dstlinesize[0] / 2], y1ox8);

            ravgax2 = vpadd_s32(vget_low_s32(r0oax4), vget_high_s32(r0oax4));
            ravgbx2 = vpadd_s32(vget_low_s32(r0obx4), vget_high_s32(r0obx4));
            ravgx4 = vcombine_s32(ravgax2, ravgbx2);
            ravgax2 = vpadd_s32(vget_low_s32(r1oax4), vget_high_s32(r1oax4));
            ravgbx2 = vpadd_s32(vget_low_s32(r1obx4), vget_high_s32(r1obx4));
            ravgx4 = vaddq_s32(ravgx4, vcombine_s32(ravgax2, ravgbx2));
            ravgx4 = vaddq_s32(ravgx4, rgb_avg_rndx4);
            ravgx4 = vshrq_n_s32(ravgx4, CHROMA_AVG_ROUNDING);

            gavgax2 = vpadd_s32(vget_low_s32(g0oax4), vget_high_s32(g0oax4));
            gavgbx2 = vpadd_s32(vget_low_s32(g0obx4), vget_high_s32(g0obx4));
            gavgx4 = vcombine_s32(gavgax2, gavgbx2);
            gavgax2 = vpadd_s32(vget_low_s32(g1oax4), vget_high_s32(g1oax4));
            gavgbx2 = vpadd_s32(vget_low_s32(g1obx4), vget_high_s32(g1obx4));
            gavgx4 = vaddq_s32(gavgx4, vcombine_s32(gavgax2, gavgbx2));
            gavgx4 = vaddq_s32(gavgx4, rgb_avg_rndx4);
            gavgx4 = vshrq_n_s32(gavgx4, CHROMA_AVG_ROUNDING);

            bavgax2 = vpadd_s32(vget_low_s32(b0oax4), vget_high_s32(b0oax4));
            bavgbx2 = vpadd_s32(vget_low_s32(b0obx4), vget_high_s32(b0obx4));
            bavgx4 = vcombine_s32(bavgax2, bavgbx2);
            bavgax2 = vpadd_s32(vget_low_s32(b1oax4), vget_high_s32(b1oax4));
            bavgbx2 = vpadd_s32(vget_low_s32(b1obx4), vget_high_s32(b1obx4));
            bavgx4 = vaddq_s32(bavgx4, vcombine_s32(bavgax2, bavgbx2));
            bavgx4 = vaddq_s32(bavgx4, rgb_avg_rndx4);
            bavgx4 = vshrq_n_s32(bavgx4, CHROMA_AVG_ROUNDING);

            uox4 = vmlaq_n_s32(out_rndx4, ravgx4, cru);
            uox4 = vmlaq_n_s32(uox4, gavgx4, ocgu);
            uox4 = vmlaq_n_s32(uox4, bavgx4, cburv);
            uox4 = vshrq_n_s32(uox4, TEN_BIT_SCALE_SHIFT);
            uox4 = vaddq_s32(uox4, out_uv_offsetx4);
            vst1_u16(&dstu[x >> 1], vqmovun_s32(uox4));

            vox4 = vmlaq_n_s32(out_rndx4, ravgx4, cburv);
            vox4 = vmlaq_n_s32(vox4, gavgx4, ocgv);
            vox4 = vmlaq_n_s32(vox4, bavgx4, cbv);
            vox4 = vshrq_n_s32(vox4, TEN_BIT_SCALE_SHIFT);
            vox4 = vaddq_s32(vox4, out_uv_offsetx4);
            vst1_u16(&dstv[x >> 1], vqmovun_s32(vox4));
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
#endif // ENABLE_TONEMAPX_NEON_INTRINSICS
}

void tonemap_frame_420p10_2_420p10_neon(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                        const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                        const int *dstlinesize, const int *srclinesize,
                                        int dstdepth, int srcdepth,
                                        int width, int height,
                                        const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_NEON_INTRINSICS
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
    uint16x8_t in_yuv_offx8 = vdupq_n_u16(params->in_yuv_off);
    uint16x8_t in_uv_offx8 = vdupq_n_u16(512);
    uint16x4_t ux4, vx4;
    uint16x8_t y0x8, y1x8, ux8, vx8;
    uint16x8_t r0x8, g0x8, b0x8;
    uint16x8_t r1x8, g1x8, b1x8;

    int16x8_t r0ox8, g0ox8, b0ox8;
    uint16x8_t y0ox8;
    int32x4_t r0oax4, r0obx4, g0oax4, g0obx4, b0oax4, b0obx4;
    int32x4_t y0oax4, y0obx4;

    int16x8_t r1ox8, g1ox8, b1ox8;
    uint16x8_t y1ox8;
    int32x4_t r1oax4, r1obx4, g1oax4, g1obx4, b1oax4, b1obx4;
    int32x4_t y1oax4, y1obx4;
    int32x2_t ravgax2, gavgax2, bavgax2, ravgbx2, gavgbx2, bavgbx2;
    int32x4_t ravgx4, gavgx4, bavgx4, uox4, vox4;
    int32x4_t out_yuv_offx4 = vdupq_n_s32(params->out_yuv_off);
    int32x4_t out_rndx4 = vdupq_n_s32(TEN_BIT_ROUNDING);
    int32x4_t out_uv_offsetx4 = vdupq_n_s32(TEN_BIT_UV_OFFSET);
    int32x4_t rgb_avg_rndx4 = vdupq_n_s32(CHROMA_AVG_ROUNDING);
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstu += dstlinesize[1] / 2, dstv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 3; xx++) {
            int x = xx << 3;

            y0x8 = vld1q_u16(srcy + x);
            y1x8 = vld1q_u16(srcy + (srclinesize[0] / 2 + x));
            ux4 = vld1_u16(srcu + (x >> 1));
            vx4 = vld1_u16(srcv + (x >> 1));
            y0x8 = vsubq_u16(y0x8, in_yuv_offx8);
            y0x8 = vreinterpretq_u16_s16(vmaxq_s16(vreinterpretq_s16_u16(y0x8), vdupq_n_s16(0)));
            y1x8 = vsubq_u16(y1x8, in_yuv_offx8);
            y1x8 = vreinterpretq_u16_s16(vmaxq_s16(vreinterpretq_s16_u16(y1x8), vdupq_n_s16(0)));

            ux8 = vcombine_u16(vzip1_u16(ux4, ux4), vzip2_u16(ux4, ux4));
            ux8 = vsubq_u16(ux8, in_uv_offx8);
            vx8 = vcombine_u16(vzip1_u16(vx4, vx4), vzip2_u16(vx4, vx4));
            vx8 = vsubq_u16(vx8, in_uv_offx8);

            yuv2rgbx8(&r0x8, &g0x8, &b0x8, y0x8, ux8, vx8, cy, crv, cgu, cgv, cbu);
            yuv2rgbx8(&r1x8, &g1x8, &b1x8, y1x8, ux8, vx8, cy, crv, cgu, cgv, cbu);

            tonemap_int16x8_neon(r0x8, g0x8, b0x8, (int16_t *) &r, (int16_t *) &g, (int16_t *) &b,
                                 params->lin_lut, params->tonemap_lut, params->delin_lut,
                                 params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                 params->rgb2rgb_passthrough);
            tonemap_int16x8_neon(r1x8, g1x8, b1x8, (int16_t *) &r1, (int16_t *) &g1, (int16_t *) &b1,
                                 params->lin_lut, params->tonemap_lut, params->delin_lut,
                                 params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                 params->rgb2rgb_passthrough);

            r0ox8 = vld1q_s16(r);
            g0ox8 = vld1q_s16(g);
            b0ox8 = vld1q_s16(b);

            r0oax4 = vmovl_s16(vget_low_s16(r0ox8));
            g0oax4 = vmovl_s16(vget_low_s16(g0ox8));
            b0oax4 = vmovl_s16(vget_low_s16(b0ox8));

            r0obx4 = vmovl_s16(vget_high_s16(r0ox8));
            g0obx4 = vmovl_s16(vget_high_s16(g0ox8));
            b0obx4 = vmovl_s16(vget_high_s16(b0ox8));

            y0oax4 = vmulq_n_s32(r0oax4, cry);
            y0oax4 = vmlaq_n_s32(y0oax4, g0oax4, cgy);
            y0oax4 = vmlaq_n_s32(y0oax4, b0oax4, cby);
            y0oax4 = vaddq_s32(y0oax4, out_rndx4);
            y0oax4 = vshrq_n_s32(y0oax4, TEN_BIT_SCALE_SHIFT);
            y0oax4 = vaddq_s32(y0oax4, out_yuv_offx4);

            y0obx4 = vmulq_n_s32(r0obx4, cry);
            y0obx4 = vmlaq_n_s32(y0obx4, g0obx4, cgy);
            y0obx4 = vmlaq_n_s32(y0obx4, b0obx4, cby);
            y0obx4 = vaddq_s32(y0obx4, out_rndx4);
            y0obx4 = vshrq_n_s32(y0obx4, TEN_BIT_SCALE_SHIFT);
            y0obx4 = vaddq_s32(y0obx4, out_yuv_offx4);

            y0ox8 = vcombine_u16(vqmovun_s32(y0oax4), vqmovun_s32(y0obx4));
            vst1q_u16(&dsty[x], y0ox8);

            r1ox8 = vld1q_s16(r1);
            g1ox8 = vld1q_s16(g1);
            b1ox8 = vld1q_s16(b1);

            r1oax4 = vmovl_s16(vget_low_s16(r1ox8));
            g1oax4 = vmovl_s16(vget_low_s16(g1ox8));
            b1oax4 = vmovl_s16(vget_low_s16(b1ox8));

            r1obx4 = vmovl_s16(vget_high_s16(r1ox8));
            g1obx4 = vmovl_s16(vget_high_s16(g1ox8));
            b1obx4 = vmovl_s16(vget_high_s16(b1ox8));

            y1oax4 = vmulq_n_s32(r1oax4, cry);
            y1oax4 = vmlaq_n_s32(y1oax4, g1oax4, cgy);
            y1oax4 = vmlaq_n_s32(y1oax4, b1oax4, cby);
            y1oax4 = vaddq_s32(y1oax4, out_rndx4);
            y1oax4 = vshrq_n_s32(y1oax4, TEN_BIT_SCALE_SHIFT);
            y1oax4 = vaddq_s32(y1oax4, out_yuv_offx4);

            y1obx4 = vmulq_n_s32(r1obx4, cry);
            y1obx4 = vmlaq_n_s32(y1obx4, g1obx4, cgy);
            y1obx4 = vmlaq_n_s32(y1obx4, b1obx4, cby);
            y1obx4 = vaddq_s32(y1obx4, out_rndx4);
            y1obx4 = vshrq_n_s32(y1obx4, TEN_BIT_SCALE_SHIFT);
            y1obx4 = vaddq_s32(y1obx4, out_yuv_offx4);

            y1ox8 = vcombine_u16(vqmovun_s32(y1oax4), vqmovun_s32(y1obx4));
            vst1q_u16(&dsty[x + dstlinesize[0] / 2], y1ox8);

            ravgax2 = vpadd_s32(vget_low_s32(r0oax4), vget_high_s32(r0oax4));
            ravgbx2 = vpadd_s32(vget_low_s32(r0obx4), vget_high_s32(r0obx4));
            ravgx4 = vcombine_s32(ravgax2, ravgbx2);
            ravgax2 = vpadd_s32(vget_low_s32(r1oax4), vget_high_s32(r1oax4));
            ravgbx2 = vpadd_s32(vget_low_s32(r1obx4), vget_high_s32(r1obx4));
            ravgx4 = vaddq_s32(ravgx4, vcombine_s32(ravgax2, ravgbx2));
            ravgx4 = vaddq_s32(ravgx4, rgb_avg_rndx4);
            ravgx4 = vshrq_n_s32(ravgx4, CHROMA_AVG_ROUNDING);

            gavgax2 = vpadd_s32(vget_low_s32(g0oax4), vget_high_s32(g0oax4));
            gavgbx2 = vpadd_s32(vget_low_s32(g0obx4), vget_high_s32(g0obx4));
            gavgx4 = vcombine_s32(gavgax2, gavgbx2);
            gavgax2 = vpadd_s32(vget_low_s32(g1oax4), vget_high_s32(g1oax4));
            gavgbx2 = vpadd_s32(vget_low_s32(g1obx4), vget_high_s32(g1obx4));
            gavgx4 = vaddq_s32(gavgx4, vcombine_s32(gavgax2, gavgbx2));
            gavgx4 = vaddq_s32(gavgx4, rgb_avg_rndx4);
            gavgx4 = vshrq_n_s32(gavgx4, CHROMA_AVG_ROUNDING);

            bavgax2 = vpadd_s32(vget_low_s32(b0oax4), vget_high_s32(b0oax4));
            bavgbx2 = vpadd_s32(vget_low_s32(b0obx4), vget_high_s32(b0obx4));
            bavgx4 = vcombine_s32(bavgax2, bavgbx2);
            bavgax2 = vpadd_s32(vget_low_s32(b1oax4), vget_high_s32(b1oax4));
            bavgbx2 = vpadd_s32(vget_low_s32(b1obx4), vget_high_s32(b1obx4));
            bavgx4 = vaddq_s32(bavgx4, vcombine_s32(bavgax2, bavgbx2));
            bavgx4 = vaddq_s32(bavgx4, rgb_avg_rndx4);
            bavgx4 = vshrq_n_s32(bavgx4, CHROMA_AVG_ROUNDING);

            uox4 = vmlaq_n_s32(out_rndx4, ravgx4, cru);
            uox4 = vmlaq_n_s32(uox4, gavgx4, ocgu);
            uox4 = vmlaq_n_s32(uox4, bavgx4, cburv);
            uox4 = vshrq_n_s32(uox4, TEN_BIT_SCALE_SHIFT);
            uox4 = vaddq_s32(uox4, out_uv_offsetx4);
            vst1_u16(&dstu[x >> 1], vqmovun_s32(uox4));

            vox4 = vmlaq_n_s32(out_rndx4, ravgx4, cburv);
            vox4 = vmlaq_n_s32(vox4, gavgx4, ocgv);
            vox4 = vmlaq_n_s32(vox4, bavgx4, cbv);
            vox4 = vshrq_n_s32(vox4, TEN_BIT_SCALE_SHIFT);
            vox4 = vaddq_s32(vox4, out_uv_offsetx4);
            vst1_u16(&dstv[x >> 1], vqmovun_s32(vox4));
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
#endif // ENABLE_TONEMAPX_NEON_INTRINSICS
}

void tonemap_frame_p010_2_p010_neon(uint16_t *dsty, uint16_t *dstuv,
                                    const uint16_t *srcy, const uint16_t *srcuv,
                                    const int *dstlinesize, const int *srclinesize,
                                    int dstdepth, int srcdepth,
                                    int width, int height,
                                    const struct TonemapIntParams *params)
{
#ifdef ENABLE_TONEMAPX_NEON_INTRINSICS
    uint16_t *rdsty = dsty;
    uint16_t *rdstuv = dstuv;
    const uint16_t *rsrcy = srcy;
    const uint16_t *rsrcuv = srcuv;
    int rheight = height;
    // not zero when not divisible by 8
    // intentionally leave last pixel emtpy when input is odd
    int remainw = width & 6;

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
    uint16x8_t in_yuv_offx8 = vdupq_n_u16(params->in_yuv_off);
    uint16x8_t in_uv_offx8 = vdupq_n_u16(TEN_BIT_UV_OFFSET);
    uint16x8_t uvx8;
    uint16x4_t ux2a, vx2a, ux2b, vx2b;
    uint16x8_t y0x8, y1x8, ux8, vx8;
    uint16x8_t r0x8, g0x8, b0x8;
    uint16x8_t r1x8, g1x8, b1x8;

    int16x8_t r0ox8, g0ox8, b0ox8;
    uint16x8_t y0ox8;
    int32x4_t r0oax4, r0obx4, g0oax4, g0obx4, b0oax4, b0obx4;
    int32x4_t y0oax4, y0obx4;

    int16x8_t r1ox8, g1ox8, b1ox8;
    uint16x8_t y1ox8;
    int32x4_t r1oax4, r1obx4, g1oax4, g1obx4, b1oax4, b1obx4;
    int32x4_t y1oax4, y1obx4;
    int32x4_t uvoax4, uvobx4;
    int32x2_t ravgax2, gavgax2, bavgax2, ravgbx2, gavgbx2, bavgbx2;
    int32x4_t ravgx4, gavgx4, bavgx4, uox4, vox4;
    int32x4_t out_yuv_offx4 = vdupq_n_s32(params->out_yuv_off);
    int32x4_t out_rndx4 = vdupq_n_s32(TEN_BIT_ROUNDING);
    int32x4_t out_uv_offsetx4 = vdupq_n_s32(TEN_BIT_UV_OFFSET);
    int32x4_t rgb_avg_rndx4 = vdupq_n_s32(CHROMA_AVG_ROUNDING);
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstuv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcuv += srclinesize[1] / 2) {
        for (int xx = 0; xx < width >> 3; xx++) {
            int x = xx << 3;

            y0x8 = vld1q_u16(srcy + x);
            y1x8 = vld1q_u16(srcy + (srclinesize[0] / 2 + x));
            uvx8 = vld1q_u16(srcuv + x);
            // shift to low10bits for 10bit input
            // shift bit has to be compile-time constant
            y0x8 = vshrq_n_u16(y0x8, TEN_BIT_BIPLANAR_SHIFT);
            y1x8 = vshrq_n_u16(y1x8, TEN_BIT_BIPLANAR_SHIFT);
            uvx8 = vshrq_n_u16(uvx8, TEN_BIT_BIPLANAR_SHIFT);
            y0x8 = vsubq_u16(y0x8, in_yuv_offx8);
            y0x8 = vreinterpretq_u16_s16(vmaxq_s16(vreinterpretq_s16_u16(y0x8), vdupq_n_s16(0)));
            y1x8 = vsubq_u16(y1x8, in_yuv_offx8);
            y1x8 = vreinterpretq_u16_s16(vmaxq_s16(vreinterpretq_s16_u16(y1x8), vdupq_n_s16(0)));
            uvx8 = vsubq_u16(uvx8, in_uv_offx8);

            ux2a = vext_u16(vdup_lane_u16(vget_low_u16(uvx8), 0), vdup_lane_u16(vget_low_u16(uvx8), 2), 2);
            vx2a = vext_u16(vdup_lane_u16(vget_low_u16(uvx8), 1), vdup_lane_u16(vget_low_u16(uvx8), 3), 2);
            ux2b = vext_u16(vdup_lane_u16(vget_high_u16(uvx8), 0), vdup_lane_u16(vget_high_u16(uvx8), 2), 2);
            vx2b = vext_u16(vdup_lane_u16(vget_high_u16(uvx8), 1), vdup_lane_u16(vget_high_u16(uvx8), 3), 2);

            ux8 = vcombine_u16(ux2a, ux2b);
            vx8 = vcombine_u16(vx2a, vx2b);

            yuv2rgbx8(&r0x8, &g0x8, &b0x8, y0x8, ux8, vx8, cy, crv, cgu, cgv, cbu);
            yuv2rgbx8(&r1x8, &g1x8, &b1x8, y1x8, ux8, vx8, cy, crv, cgu, cgv, cbu);

            tonemap_int16x8_neon(r0x8, g0x8, b0x8, (int16_t *) &r, (int16_t *) &g, (int16_t *) &b,
                                 params->lin_lut, params->tonemap_lut, params->delin_lut,
                                 params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                 params->rgb2rgb_passthrough);
            tonemap_int16x8_neon(r1x8, g1x8, b1x8, (int16_t *) &r1, (int16_t *) &g1, (int16_t *) &b1,
                                 params->lin_lut, params->tonemap_lut, params->delin_lut,
                                 params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs,
                                 params->rgb2rgb_passthrough);

            r0ox8 = vld1q_s16(r);
            g0ox8 = vld1q_s16(g);
            b0ox8 = vld1q_s16(b);

            r0oax4 = vmovl_s16(vget_low_s16(r0ox8));
            g0oax4 = vmovl_s16(vget_low_s16(g0ox8));
            b0oax4 = vmovl_s16(vget_low_s16(b0ox8));

            r0obx4 = vmovl_s16(vget_high_s16(r0ox8));
            g0obx4 = vmovl_s16(vget_high_s16(g0ox8));
            b0obx4 = vmovl_s16(vget_high_s16(b0ox8));

            y0oax4 = vmulq_n_s32(r0oax4, cry);
            y0oax4 = vmlaq_n_s32(y0oax4, g0oax4, cgy);
            y0oax4 = vmlaq_n_s32(y0oax4, b0oax4, cby);
            y0oax4 = vaddq_s32(y0oax4, out_rndx4);

            y0obx4 = vmulq_n_s32(r0obx4, cry);
            y0obx4 = vmlaq_n_s32(y0obx4, g0obx4, cgy);
            y0obx4 = vmlaq_n_s32(y0obx4, b0obx4, cby);
            y0obx4 = vaddq_s32(y0obx4, out_rndx4);

            r1ox8 = vld1q_s16(r1);
            g1ox8 = vld1q_s16(g1);
            b1ox8 = vld1q_s16(b1);

            r1oax4 = vmovl_s16(vget_low_s16(r1ox8));
            g1oax4 = vmovl_s16(vget_low_s16(g1ox8));
            b1oax4 = vmovl_s16(vget_low_s16(b1ox8));

            r1obx4 = vmovl_s16(vget_high_s16(r1ox8));
            g1obx4 = vmovl_s16(vget_high_s16(g1ox8));
            b1obx4 = vmovl_s16(vget_high_s16(b1ox8));

            y1oax4 = vmulq_n_s32(r1oax4, cry);
            y1oax4 = vmlaq_n_s32(y1oax4, g1oax4, cgy);
            y1oax4 = vmlaq_n_s32(y1oax4, b1oax4, cby);
            y1oax4 = vaddq_s32(y1oax4, out_rndx4);

            y1obx4 = vmulq_n_s32(r1obx4, cry);
            y1obx4 = vmlaq_n_s32(y1obx4, g1obx4, cgy);
            y1obx4 = vmlaq_n_s32(y1obx4, b1obx4, cby);
            y1obx4 = vaddq_s32(y1obx4, out_rndx4);

            ravgax2 = vpadd_s32(vget_low_s32(r0oax4), vget_high_s32(r0oax4));
            ravgbx2 = vpadd_s32(vget_low_s32(r0obx4), vget_high_s32(r0obx4));
            ravgx4 = vcombine_s32(ravgax2, ravgbx2);
            ravgax2 = vpadd_s32(vget_low_s32(r1oax4), vget_high_s32(r1oax4));
            ravgbx2 = vpadd_s32(vget_low_s32(r1obx4), vget_high_s32(r1obx4));
            ravgx4 = vaddq_s32(ravgx4, vcombine_s32(ravgax2, ravgbx2));
            ravgx4 = vaddq_s32(ravgx4, rgb_avg_rndx4);
            ravgx4 = vshrq_n_s32(ravgx4, CHROMA_AVG_ROUNDING);

            gavgax2 = vpadd_s32(vget_low_s32(g0oax4), vget_high_s32(g0oax4));
            gavgbx2 = vpadd_s32(vget_low_s32(g0obx4), vget_high_s32(g0obx4));
            gavgx4 = vcombine_s32(gavgax2, gavgbx2);
            gavgax2 = vpadd_s32(vget_low_s32(g1oax4), vget_high_s32(g1oax4));
            gavgbx2 = vpadd_s32(vget_low_s32(g1obx4), vget_high_s32(g1obx4));
            gavgx4 = vaddq_s32(gavgx4, vcombine_s32(gavgax2, gavgbx2));
            gavgx4 = vaddq_s32(gavgx4, rgb_avg_rndx4);
            gavgx4 = vshrq_n_s32(gavgx4, CHROMA_AVG_ROUNDING);

            bavgax2 = vpadd_s32(vget_low_s32(b0oax4), vget_high_s32(b0oax4));
            bavgbx2 = vpadd_s32(vget_low_s32(b0obx4), vget_high_s32(b0obx4));
            bavgx4 = vcombine_s32(bavgax2, bavgbx2);
            bavgax2 = vpadd_s32(vget_low_s32(b1oax4), vget_high_s32(b1oax4));
            bavgbx2 = vpadd_s32(vget_low_s32(b1obx4), vget_high_s32(b1obx4));
            bavgx4 = vaddq_s32(bavgx4, vcombine_s32(bavgax2, bavgbx2));
            bavgx4 = vaddq_s32(bavgx4, rgb_avg_rndx4);
            bavgx4 = vshrq_n_s32(bavgx4, CHROMA_AVG_ROUNDING);

            uox4 = vmlaq_n_s32(out_rndx4, ravgx4, cru);
            uox4 = vmlaq_n_s32(uox4, gavgx4, ocgu);
            uox4 = vmlaq_n_s32(uox4, bavgx4, cburv);

            vox4 = vmlaq_n_s32(out_rndx4, ravgx4, cburv);
            vox4 = vmlaq_n_s32(vox4, gavgx4, ocgv);
            vox4 = vmlaq_n_s32(vox4, bavgx4, cbv);

            y0oax4 = vshrq_n_s32(y0oax4, TEN_BIT_SCALE_SHIFT);
            y0obx4 = vshrq_n_s32(y0obx4, TEN_BIT_SCALE_SHIFT);
            y1oax4 = vshrq_n_s32(y1oax4, TEN_BIT_SCALE_SHIFT);
            y1obx4 = vshrq_n_s32(y1obx4, TEN_BIT_SCALE_SHIFT);
            uox4 = vshrq_n_s32(uox4, TEN_BIT_SCALE_SHIFT);
            vox4 = vshrq_n_s32(vox4, TEN_BIT_SCALE_SHIFT);

            y0oax4 = vaddq_s32(y0oax4, out_yuv_offx4);
            y0oax4 = vshlq_n_s32(y0oax4, TEN_BIT_BIPLANAR_SHIFT);
            y0obx4 = vaddq_s32(y0obx4, out_yuv_offx4);
            y0obx4 = vshlq_n_s32(y0obx4, TEN_BIT_BIPLANAR_SHIFT);
            y1oax4 = vaddq_s32(y1oax4, out_yuv_offx4);
            y1oax4 = vshlq_n_s32(y1oax4, TEN_BIT_BIPLANAR_SHIFT);
            y1obx4 = vaddq_s32(y1obx4, out_yuv_offx4);
            y1obx4 = vshlq_n_s32(y1obx4, TEN_BIT_BIPLANAR_SHIFT);
            uox4 = vaddq_s32(uox4, out_uv_offsetx4);
            uox4 = vshlq_n_s32(uox4, TEN_BIT_BIPLANAR_SHIFT);
            vox4 = vaddq_s32(vox4, out_uv_offsetx4);
            vox4 = vshlq_n_s32(vox4, TEN_BIT_BIPLANAR_SHIFT);

            y0ox8 = vcombine_u16(vqmovun_s32(y0oax4), vqmovun_s32(y0obx4));
            vst1q_u16(&dsty[x], y0ox8);

            y1ox8 = vcombine_u16(vqmovun_s32(y1oax4), vqmovun_s32(y1obx4));
            vst1q_u16(&dsty[x + dstlinesize[0] / 2], y1ox8);

            uvoax4 = vzip1q_s32(uox4, vox4);
            uvobx4 = vzip2q_s32(uox4, vox4);

            vst1q_u16(&dstuv[x], vcombine_u16(vqmovun_s32(uvoax4), vqmovun_s32(uvobx4)));
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
#endif // ENABLE_TONEMAPX_NEON_INTRINSICS
}
