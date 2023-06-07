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

#define FLOAT_EPS 1e-6f

extern float3 lrgb2yuv(float3);
extern float  lrgb2y(float3);
extern float3 yuv2lrgb(float3);
extern float3 lrgb2lrgb(float3);
extern float  eotf_st2084(float);
extern float  inverse_eotf_st2084(float);
extern float4 get_luma_dst4(float4, float4, float4);
extern float3 get_chroma_sample(float3, float3, float3, float3);
#ifdef DOVI_RESHAPE
extern float3 rgb2lrgb(float3);
extern float3 ycc2rgb(float, float, float);
extern float3 lms2rgb(float, float, float);
#endif

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

    float s_pq = inverse_eotf_st2084(s) * scale;
    float max_lum = target_peak_inv_pq * scale;

    float ks = 1.5f * max_lum - 0.5f;
    float tb = (s_pq - ks) / (1.0f - ks);
    float tb2 = tb * tb;
    float tb3 = tb2 * tb;
    float pb = (2.0f * tb3 - 3.0f * tb2 + 1.0f) * ks +
               (tb3 - 2.0f * tb2 + tb) * (1.0f - ks) +
               (-2.0f * tb3 + 3.0f * tb2) * max_lum;
    float sig = mix(pb, s_pq, s_pq < ks);

    return eotf_st2084(sig * peak_pq);
}

void map_four_pixels_rgb(float4 *r4, float4 *g4, float4 *b4, float peak) {
#ifdef TONE_MODE_RGB
    float4 sig_r = fmax(*r4, FLOAT_EPS), sig_ro = sig_r;
    float4 sig_g = fmax(*g4, FLOAT_EPS), sig_go = sig_g;
    float4 sig_b = fmax(*b4, FLOAT_EPS), sig_bo = sig_b;
#else
    float4 sig = fmax(fmax(*r4, fmax(*g4, *b4)), FLOAT_EPS);
    float4 sig_o = sig;
#endif

    // Desaturate the color using a coefficient dependent on the signal level
    if (desat_param > 0.0f) {
#ifdef TONE_MODE_RGB
        float4 sig = fmax(fmax(*r4, fmax(*g4, *b4)), FLOAT_EPS);
#endif
        float4 luma = get_luma_dst4(*r4, *g4, *b4);
        float4 coeff = fmax(sig - 0.18f, FLOAT_EPS) / fmax(sig, FLOAT_EPS);
        coeff = native_powr(coeff, 10.0f / desat_param);
        *r4 = mix(*r4, luma, coeff);
        *g4 = mix(*g4, luma, coeff);
        *b4 = mix(*b4, luma, coeff);
    }

#define MAP_FOUR_PIXELS(sig, peak, target_peak) \
{ \
    sig.x = TONE_FUNC(sig.x, peak, target_peak); \
    sig.y = TONE_FUNC(sig.y, peak, target_peak); \
    sig.z = TONE_FUNC(sig.z, peak, target_peak); \
    sig.w = TONE_FUNC(sig.w, peak, target_peak); \
}

#ifdef TONE_FUNC_BT2390
    float src_peak_delin_pq = inverse_eotf_st2084(peak);
    float dst_peak_delin_pq = inverse_eotf_st2084(1.0f);
  #ifdef TONE_MODE_RGB
    MAP_FOUR_PIXELS(sig_r, src_peak_delin_pq, dst_peak_delin_pq)
    MAP_FOUR_PIXELS(sig_g, src_peak_delin_pq, dst_peak_delin_pq)
    MAP_FOUR_PIXELS(sig_b, src_peak_delin_pq, dst_peak_delin_pq)
  #else
    MAP_FOUR_PIXELS(sig, src_peak_delin_pq, dst_peak_delin_pq)
  #endif
#else
  #ifdef TONE_MODE_RGB
    MAP_FOUR_PIXELS(sig_r, peak, 1.0f)
    MAP_FOUR_PIXELS(sig_g, peak, 1.0f)
    MAP_FOUR_PIXELS(sig_b, peak, 1.0f)
  #else
    MAP_FOUR_PIXELS(sig, peak, 1.0f)
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

// Map from source space YUV to destination space RGB
float3 map_to_dst_space_from_yuv(float3 yuv) {
#ifdef DOVI_RESHAPE
    float3 c = ycc2rgb(yuv.x, yuv.y, yuv.z);
    c = lms2rgb(c.x, c.y, c.z);
    c = rgb2lrgb(c);
#else
    float3 c = yuv2lrgb(yuv);
    c = lrgb2lrgb(c);
#endif
    return c;
}

#ifdef DOVI_RESHAPE
float reshape_poly(float s, float4 coeffs) {
    return (coeffs.z * s + coeffs.y) * s + coeffs.x;
}

float reshape_mmr(float3 sig,
                  float4 coeffs,
                  __global float4 *dovi_mmr,
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
                        __global float *src_dovi_params,
                        __global float *src_dovi_pivots,
                        __global float4 *src_dovi_coeffs,
                        __global float4 *src_dovi_mmr)
{
    int i;
    float s;
    float3 sig = clamp(yuv.xyz, 0.0f, 1.0f);
    float sig_arr[3] = {sig.x, sig.y, sig.z};
    float4 coeffs;
    int dovi_num_pivots, dovi_has_mmr, dovi_has_poly;
    int dovi_mmr_single, dovi_min_order, dovi_max_order;
    float dovi_lo, dovi_hi;
    __global float *dovi_params;
    __global float *dovi_pivots;
    __global float4 *dovi_coeffs, *dovi_mmr;

#pragma unroll
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

    return (float3)(sig_arr[0], sig_arr[1], sig_arr[2]);
}
#endif

__constant sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                                CLK_ADDRESS_CLAMP_TO_EDGE   |
                                CLK_FILTER_NEAREST);

__constant sampler_t l_sampler = (CLK_NORMALIZED_COORDS_TRUE |
                                  CLK_ADDRESS_CLAMP_TO_EDGE  |
                                  CLK_FILTER_LINEAR);

__constant sampler_t d_sampler = (CLK_NORMALIZED_COORDS_TRUE |
                                  CLK_ADDRESS_REPEAT         |
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
                      __global float *dovi_buf,
#endif
                      float peak)
{
    int xi = get_global_id(0);
    int yi = get_global_id(1);
    // each work item process four pixels
    int x = 2 * xi;
    int y = 2 * yi;

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

    yuv0.x = read_imagef(src1, sampler, (int2)(x,     y)).x;
    yuv1.x = read_imagef(src1, sampler, (int2)(x + 1, y)).x;
    yuv2.x = read_imagef(src1, sampler, (int2)(x,     y + 1)).x;
    yuv3.x = read_imagef(src1, sampler, (int2)(x + 1, y + 1)).x;

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
    yuv0.yz = read_imagef(src2, l_sampler, ncoords_yuv0).xy;
    yuv1.yz = read_imagef(src2, l_sampler, ncoords_yuv1).xy;
    yuv2.yz = read_imagef(src2, l_sampler, ncoords_yuv2).xy;
    yuv3.yz = read_imagef(src2, l_sampler, ncoords_yuv3).xy;
#endif

#ifdef DOVI_RESHAPE
    __global float *dovi_params = dovi_buf;
    __global float *dovi_pivots = dovi_buf + 24;
    __global float4 *dovi_coeffs = (__global float4 *)(dovi_buf + 48);
    __global float4 *dovi_mmr = (__global float4 *)(dovi_buf + 144);
    yuv0 = reshape_dovi_yuv(yuv0, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
    yuv1 = reshape_dovi_yuv(yuv1, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
    yuv2 = reshape_dovi_yuv(yuv2, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
    yuv3 = reshape_dovi_yuv(yuv3, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);
#endif

    float3 c0 = map_to_dst_space_from_yuv(yuv0);
    float3 c1 = map_to_dst_space_from_yuv(yuv1);
    float3 c2 = map_to_dst_space_from_yuv(yuv2);
    float3 c3 = map_to_dst_space_from_yuv(yuv3);

#ifndef SKIP_TONEMAP
    float4 r4 = (float4)(c0.x, c1.x, c2.x, c3.x);
    float4 g4 = (float4)(c0.y, c1.y, c2.y, c3.y);
    float4 b4 = (float4)(c0.z, c1.z, c2.z, c3.z);
    map_four_pixels_rgb(&r4, &g4, &b4, peak);
    c0 = (float3)(r4.x, g4.x, b4.x);
    c1 = (float3)(r4.y, g4.y, b4.y);
    c2 = (float3)(r4.z, g4.z, b4.z);
    c3 = (float3)(r4.w, g4.w, b4.w);
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
