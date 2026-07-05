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

#include "colorspace_common.h"
#include "pixfmt.h"
#include "tonemap.h"
#include "util.h"

extern __constant__ const enum TonemapAlgorithm tonemap_func;
extern __constant__ const float tone_param;
extern __constant__ const float desat_param;
extern __constant__ const int lut_size;
extern __constant__ const int enable_dither;
extern __constant__ const float dither_size;
extern __constant__ const float dither_quantization;

#define clamp(a, b, c) min(max((a), (b)), (c))
#define mix(x, y, a) ((x) + ((y) - (x)) * (a))
#define select(a, b, c) ((c) ? (b) : (a))
#define dot3(a, b) ((a).z * (b).z + ((a).y * (b).y + (a).x * (b).x))
#define dot4(a, b) ((a).w * (b).w + ((a).z * (b).z + ((a).y * (b).y + (a).x * (b).x)))

template <typename T, typename S>
static __inline__ __device__
T clamp3(const T a, const S min_val, const S max_val) {
    T result;
    result.x = clamp(a.x, min_val, max_val);
    result.y = clamp(a.y, min_val, max_val);
    result.z = clamp(a.z, min_val, max_val);
    return result;
}

static __inline__ __device__
float get_dithered_y(float y, float d) {
    return floor(y * dither_quantization + d + 0.5f / (dither_size * dither_size)) * 1.0f / dither_quantization;
}

static __inline__ __device__
float hable_f(float in) {
    float a = 0.15f, b = 0.50f, c = 0.10f, d = 0.20f, e = 0.02f, f = 0.30f;
    return (in * (in * a + b * c) + d * e) / (in * (in * a + b) + d * f) - e / f;
}

static __inline__ __device__
float direct(float s, float peak) {
    return s;
}

static __inline__ __device__
float linear(float s, float peak) {
    return s * tone_param / peak;
}

static __inline__ __device__
float gamma(float s, float peak) {
    float p = s > 0.05f ? s / peak : 0.05f / peak;
    float v = __powf(p, 1.0f / tone_param);
    return s > 0.05f ? v : (s * v / 0.05f);
}

static __inline__ __device__
float clip(float s, float peak) {
    return clamp(s * tone_param, 0.0f, 1.0f);
}

static __inline__ __device__
float reinhard(float s, float peak) {
    return s / (s + tone_param) * (peak + tone_param) / peak;
}

static __inline__ __device__
float hable(float s, float peak) {
    return hable_f(s) / hable_f(peak);
}

static __inline__ __device__
float mobius(float s, float peak) {
    float j = tone_param;
    float a, b;

    if (s <= j)
        return s;

    a = -j * j * (peak - 1.0f) / (j * j - 2.0f * j + peak);
    b = (j * j - 2.0f * j * peak + peak) / max(peak - 1.0f, FLOAT_EPS);

    return (b * b + 2.0f * b * j + j * j) / (b - a) * (s + a) / (s + b);
}

static __inline__ __device__
float bt2390_common(float s, float peak, float dst_peak) {
    float peak_pq = inverse_eotf_st2084(peak);
    float scale = peak_pq > 0.0f ? (1.0f / peak_pq) : 1.0f;

    float s_pq = s * scale;
    float max_lum = inverse_eotf_st2084(dst_peak) * scale;

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

static __inline__ __device__
float bt2390(float s, float peak, float dst_peak) {
    float s_clipped = min(s, peak);
    float s_pq = inverse_eotf_st2084(s_clipped);
    return eotf_st2084(bt2390_common(s_pq, peak, dst_peak));
}

static __inline__ __device__
float map(float s, float peak, float dst_peak)
{
    switch (tonemap_func) {
    case TONEMAP_NONE:
    default:
        return direct(s, peak);
    case TONEMAP_LINEAR:
        return linear(s, peak);
    case TONEMAP_GAMMA:
        return gamma(s, peak);
    case TONEMAP_CLIP:
        return clip(s, peak);
    case TONEMAP_REINHARD:
        return reinhard(s, peak);
    case TONEMAP_HABLE:
        return hable(s, peak);
    case TONEMAP_MOBIUS:
        return mobius(s, peak);
    case TONEMAP_BT2390:
        return bt2390(s, peak, dst_peak);
    }
}

static __inline__ __device__
float map_itp(float s, float peak, float dst_peak)
{
    switch (tonemap_func) {
    default:
        return inverse_eotf_st2084(map(eotf_st2084(s), peak, dst_peak));
    case TONEMAP_BT2390:
        return bt2390_common(s, peak, dst_peak);
    }
}

static __inline__ __device__
float3 map_one_pixel_rgb_mode_max(float3 rgb, float peak, float dst_peak) {
    float sig = max(max(rgb.x, max(rgb.y, rgb.z)), FLOAT_EPS);
    float sig_old = sig;

    // Desaturate the color using a coefficient dependent on the signal level
    if (desat_param > 0.0f) {
        float luma = get_luma_dst(rgb, luma_dst);
        float coeff = max(sig - 0.18f, FLOAT_EPS) / max(sig, FLOAT_EPS);
        coeff = __powf(coeff, 10.0f / desat_param);
        rgb = mix(rgb, make_float3(luma, luma, luma), make_float3(coeff, coeff, coeff));
    }

    sig = map(sig, peak, dst_peak);
    sig = min(sig, 1.0f);
    rgb = rgb * (sig / sig_old);

    return rgb;
}

static __inline__ __device__
float3 map_one_pixel_rgb_mode_rgb(float3 rgb, float peak, float dst_peak) {
    float3 sig;
    sig.x = max(rgb.x, FLOAT_EPS);
    sig.y = max(rgb.y, FLOAT_EPS);
    sig.z = max(rgb.z, FLOAT_EPS);
    float3 sig_old = sig;

    // Desaturate the color using a coefficient dependent on the signal level
    if (desat_param > 0.0f) {
        float sig_max = max(max(rgb.x, max(rgb.y, rgb.z)), FLOAT_EPS);
        float luma = get_luma_dst(rgb, luma_dst);
        float coeff = max(sig_max - 0.18f, FLOAT_EPS) / max(sig_max, FLOAT_EPS);
        coeff = __powf(coeff, 10.0f / desat_param);
        rgb = mix(rgb, make_float3(luma, luma, luma), make_float3(coeff, coeff, coeff));
    }

    sig.x = map(sig.x, peak, dst_peak);
    sig.y = map(sig.y, peak, dst_peak);
    sig.z = map(sig.z, peak, dst_peak);
    sig.x = min(sig.x, 1.0f);
    sig.y = min(sig.y, 1.0f);
    sig.z = min(sig.z, 1.0f);
    rgb = rgb * (sig / sig_old);

    return rgb;
}

static __inline__ __device__
float3 map_one_pixel_rgb_mode_lum(float3 rgb, float peak, float dst_peak) {
    float sig = max((rgb.x * 0.2627f + rgb.y * 0.678f + rgb.z * 0.0593f), FLOAT_EPS);
    sig = min(sig, peak);
    float sig_old = sig;

    // Desaturate the color using a coefficient dependent on the signal level
    if (desat_param > 0.0f) {
        float coeff = max(sig - 0.18f, FLOAT_EPS) / max(sig, FLOAT_EPS);
        coeff = __powf(coeff, 10.0f / desat_param);
        rgb = mix(rgb, make_float3(sig, sig, sig), make_float3(coeff, coeff, coeff));
    }

    sig = map(sig, peak, dst_peak);
    rgb = rgb * (sig / sig_old);

    return rgb;
}

static __inline__ __device__
float3 map_one_pixel_itp_mode(float3 rgb, float peak, float dst_peak) {
    float r = mix(rgb.x, min(rgb.x, peak), tonemap_func == TONEMAP_BT2390);
    float g = mix(rgb.y, min(rgb.y, peak), tonemap_func == TONEMAP_BT2390);
    float b = mix(rgb.z, min(rgb.z, peak), tonemap_func == TONEMAP_BT2390);
    float3 ictcp = lrgb2ictcp(r, g, b);
    ictcp.x = max(ictcp.x, FLOAT_EPS);
    float i_o = ictcp.x;

    if (desat_param > 0.0f) {
        float p = eotf_st2084(ictcp.x) - (dst_peak - desat_param) * 0.5f;
        float coeff = __expf(-(p * p) / (2.0f * peak));
        ictcp.y *= coeff;
        ictcp.z *= coeff;
    }

    ictcp.x = map_itp(ictcp.x, peak, dst_peak);
    ictcp.x = min(ictcp.x, 1.0f);
    float factor = min(ictcp.x / i_o, i_o / ictcp.x);
    ictcp.y *= factor;
    ictcp.z *= factor;

    return ictcp2lrgb(ictcp.x, ictcp.y, ictcp.z);
}

// Map from source space YUV to source space RGB
static __inline__ __device__
float3 map_to_src_space_from_yuv(float3 yuv) {
    float3 c = yuv2lrgb(yuv);
    return c;
}

static __inline__ __device__
float3 map_to_src_space_from_yuv_dovi(float3 yuv) {
    float3 c = ycc2rgb(yuv.x, yuv.y, yuv.z);
    c = lms2rgb(c.x, c.y, c.z);
    return c;
}

// Map from source space YUV to destination space RGB
static __inline__ __device__
float3 map_to_dst_space_from_yuv(float3 yuv) {
    float3 c = yuv2lrgb(yuv);
    return lrgb2lrgb(c);
}

static __inline__ __device__
float3 map_to_dst_space_from_yuv_dovi(float3 yuv) {
    float3 c = ycc2rgb(yuv.x, yuv.y, yuv.z);
    c = lms2rgb(c.x, c.y, c.z);
    c = lrgb2lrgb(c);
    return c;
}

static __inline__ __device__
float reshape_poly(float s, float4 coeffs) {
    return (coeffs.z * s + coeffs.y) * s + coeffs.x;
}

static __inline__ __device__
float reshape_mmr(float3 sig, float4 coeffs, const float4 *dovi_mmr,
                  int dovi_mmr_single, int dovi_min_order, int dovi_max_order)
{
    int mmr_idx = dovi_mmr_single ? 0 : (int)coeffs.y;
    int order = (int)coeffs.w;
    float3 sigXxyz = make_float3(sig.x, sig.x, sig.y) * make_float3(sig.y, sig.z, sig.z);
    float4 sigX = make_float4(sigXxyz.x, sigXxyz.y, sigXxyz.z, sigXxyz.x * sig.z);
    float4 mmr;

    float s = coeffs.x;
    mmr = dovi_mmr[mmr_idx + 0];
    s += dot3(make_float3(mmr.x, mmr.y, mmr.z), sig);
    mmr = dovi_mmr[mmr_idx + 1];
    s += dot4(mmr, sigX);

    int t = dovi_max_order >= 2 && (dovi_min_order >= 2 || order >= 2);
    if (t) {
        float3 sig2 = sig * sig;
        float4 sigX2 = sigX * sigX;
        mmr = dovi_mmr[mmr_idx + 2];
        s += dot3(make_float3(mmr.x, mmr.y, mmr.z), sig2);
        mmr = dovi_mmr[mmr_idx + 3];
        s += dot4(mmr, sigX2);
        t = dovi_max_order == 3 && (dovi_min_order == 3 || order >= 3);
        if (t) {
            mmr = dovi_mmr[mmr_idx + 4];
            s += dot3(make_float3(mmr.x, mmr.y, mmr.z), sig2 * sig);
            mmr = dovi_mmr[mmr_idx + 5];
            s += dot4(mmr, sigX2 * sigX);
        }
    }

    return s;
}

static __inline__ __device__
float3 reshape_dovi_yuv(float3 yuv,
                        const float *src_dovi_params, const float *src_dovi_pivots,
                        const float4 *src_dovi_coeffs, const float4 *src_dovi_mmr)
{
    int i;
    float s;
    float3 sig = make_float3(clamp(yuv.x, 0.0f, 1.0f),
                             clamp(yuv.y, 0.0f, 1.0f),
                             clamp(yuv.z, 0.0f, 1.0f));
    float sig_arr[3] = {sig.x, sig.y, sig.z};
    float4 coeffs;
    int dovi_num_pivots, dovi_has_mmr, dovi_has_poly;
    int dovi_mmr_single, dovi_min_order, dovi_max_order;
    float dovi_lo, dovi_hi;
    const float *dovi_params;
    const float *dovi_pivots;
    const float4 *dovi_coeffs, *dovi_mmr;

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
            float t0 = s >= dovi_pivots[0], t1 = s >= dovi_pivots[1];
            float t2 = s >= dovi_pivots[2], t3 = s >= dovi_pivots[3];
            float t4 = s >= dovi_pivots[4], t5 = s >= dovi_pivots[5], t6 = s >= dovi_pivots[6];

            coeffs = mix(mix(mix(dovi_coeffs[0], dovi_coeffs[1], make_float4(t0, t0, t0, t0)),
                             mix(dovi_coeffs[2], dovi_coeffs[3], make_float4(t2, t2, t2, t2)),
                             make_float4(t1, t1, t1, t1)),
                         mix(mix(dovi_coeffs[4], dovi_coeffs[5], make_float4(t4, t4, t4, t4)),
                             mix(dovi_coeffs[6], dovi_coeffs[7], make_float4(t6, t6, t6, t6)),
                             make_float4(t5, t5, t5, t5)),
                         make_float4(t3, t3, t3, t3));
        }

        int has_mmr_poly = dovi_has_mmr && dovi_has_poly;

        if ((has_mmr_poly && coeffs.w == 0.0f) || (!has_mmr_poly && dovi_has_poly))
            s = reshape_poly(s, coeffs);
        else
            s = reshape_mmr(sig, coeffs, dovi_mmr,
                            dovi_mmr_single, dovi_min_order, dovi_max_order);

        sig_arr[i] = clamp(s, dovi_lo, dovi_hi);
    }

    return make_float3(sig_arr[0], sig_arr[1], sig_arr[2]);
}

static __inline__ __device__
float3 apply_lut3d(const float4 *lut, float3 color)
{
    float4 lut_val;
    color = clamp3(color, 0.0f, 1.0f);

    // Scale the color to the LUT grid
    float3 pos = color * (float)(lut_size - 1);

    // Get the integer base indices in the LUT
    int3 base = clamp3(make_int3((int)floorf(pos.x),
                                 (int)floorf(pos.y),
                                 (int)floorf(pos.z)), 0, lut_size - 2);

    // Compute the fractional part within the cell
    float3 f = pos - make_float3((float)base.x, (float)base.y, (float)base.z);

    // Sort the fraction offsets, so that we always have f_max>=f_mid>=f_min
    float f_max = max(f.x, max(f.y, f.z));
    float f_min = min(f.x, min(f.y, f.z));
    float f_mid = f.x + f.y + f.z - f_max - f_min;

    // Compute the base linear index
    unsigned base_idx = base.x + base.y * lut_size + base.z * lut_size * lut_size;
    unsigned last_idx = base_idx + 1 + lut_size + lut_size * lut_size;

    // The initial and the last corner values of current cube will always be fetched
    lut_val = lut[base_idx];
    float3 c000 = make_float3(lut_val.x, lut_val.y, lut_val.z);
    lut_val = lut[last_idx];
    float3 c111 = make_float3(lut_val.x, lut_val.y, lut_val.z);

    // Select the index for vertices of the tetrahedron
    unsigned idx100 = base_idx + 1;
    unsigned idx010 = base_idx + lut_size;
    unsigned idx110 = base_idx + 1 + lut_size;
    unsigned idx001 = base_idx + lut_size * lut_size;
    unsigned idx101 = base_idx + 1 + lut_size * lut_size;
    unsigned idx011 = base_idx + lut_size + lut_size * lut_size;

    // Although we have a and c as max and min value, we cannot use them in the
    // following selection as float equality comparison is not accurate on GPU
    unsigned y_max = f.y >= f.z && f.y >= f.x;
    unsigned x_max = f.x >= f.y && f.x >= f.z;
    unsigned idx0 = select(select(idx001, idx010, y_max), idx100, x_max);
    unsigned y_min = f.y <= f.z && f.y <= f.x;
    unsigned x_min = f.x <= f.y && f.x <= f.z;
    unsigned idx1 = select(select(idx110, idx101, y_min), idx011, x_min);

    // Fetch LUT value with determined tetrahedron
    lut_val = lut[idx0];
    float3 c0 = make_float3(lut_val.x, lut_val.y, lut_val.z);
    lut_val = lut[idx1];
    float3 c1 = make_float3(lut_val.x, lut_val.y, lut_val.z);

    return clamp3(c000 + f_max * (c0   - c000)
                       + f_mid * (c1   - c0)
                       + f_min * (c111 - c1), 0.0f, 1.0f);
}

extern "C" {

#define _READER \
    int xi = blockIdx.x * blockDim.x + threadIdx.x; \
    int yi = blockIdx.y * blockDim.y + threadIdx.y; \
    int x = xi << 1; \
    int y = yi << 1; \
    if (y + 1 >= src.height || x + 1 >= src.width) \
        return; \
    float3 yuv0 = read_tex_px_flt(src, x,     y); \
    float3 yuv1 = read_tex_px_flt(src, x + 1, y); \
    float3 yuv2 = read_tex_px_flt(src, x,     y + 1); \
    float3 yuv3 = read_tex_px_flt(src, x + 1, y + 1);

#define _RESHAPE \
    const float *dovi_params = dovi_buf; \
    const float *dovi_pivots = dovi_buf + 24; \
    const float4 *dovi_coeffs = (const float4 *)(dovi_buf + 48); \
    const float4 *dovi_mmr = (const float4 *)(dovi_buf + 144); \
    yuv0 = reshape_dovi_yuv(yuv0, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr); \
    yuv1 = reshape_dovi_yuv(yuv1, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr); \
    yuv2 = reshape_dovi_yuv(yuv2, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr); \
    yuv3 = reshape_dovi_yuv(yuv3, dovi_params, dovi_pivots, dovi_coeffs, dovi_mmr);

#define _YUV2RGB \
    float3 c0 = map_to_dst_space_from_yuv(yuv0); \
    float3 c1 = map_to_dst_space_from_yuv(yuv1); \
    float3 c2 = map_to_dst_space_from_yuv(yuv2); \
    float3 c3 = map_to_dst_space_from_yuv(yuv3);

#define _YUV2RGB_SRC \
    float3 c0 = map_to_src_space_from_yuv(yuv0); \
    float3 c1 = map_to_src_space_from_yuv(yuv1); \
    float3 c2 = map_to_src_space_from_yuv(yuv2); \
    float3 c3 = map_to_src_space_from_yuv(yuv3);

#define _YCC2RGB \
    float3 c0 = map_to_dst_space_from_yuv_dovi(yuv0); \
    float3 c1 = map_to_dst_space_from_yuv_dovi(yuv1); \
    float3 c2 = map_to_dst_space_from_yuv_dovi(yuv2); \
    float3 c3 = map_to_dst_space_from_yuv_dovi(yuv3);

#define _YCC2RGB_SRC \
    float3 c0 = map_to_src_space_from_yuv_dovi(yuv0); \
    float3 c1 = map_to_src_space_from_yuv_dovi(yuv1); \
    float3 c2 = map_to_src_space_from_yuv_dovi(yuv2); \
    float3 c3 = map_to_src_space_from_yuv_dovi(yuv3);

#define _TONEMAP_MAX \
    c0 = map_one_pixel_rgb_mode_max(c0, src.peak, dst.peak); \
    c1 = map_one_pixel_rgb_mode_max(c1, src.peak, dst.peak); \
    c2 = map_one_pixel_rgb_mode_max(c2, src.peak, dst.peak); \
    c3 = map_one_pixel_rgb_mode_max(c3, src.peak, dst.peak);

#define _TONEMAP_RGB \
    c0 = map_one_pixel_rgb_mode_rgb(c0, src.peak, dst.peak); \
    c1 = map_one_pixel_rgb_mode_rgb(c1, src.peak, dst.peak); \
    c2 = map_one_pixel_rgb_mode_rgb(c2, src.peak, dst.peak); \
    c3 = map_one_pixel_rgb_mode_rgb(c3, src.peak, dst.peak);

#define _TONEMAP_LUM \
    c0 = map_one_pixel_rgb_mode_lum(c0, src.peak, dst.peak); \
    c1 = map_one_pixel_rgb_mode_lum(c1, src.peak, dst.peak); \
    c2 = map_one_pixel_rgb_mode_lum(c2, src.peak, dst.peak); \
    c3 = map_one_pixel_rgb_mode_lum(c3, src.peak, dst.peak);

#define _TONEMAP_ITP \
    c0 = map_one_pixel_itp_mode(c0, src.peak, dst.peak); \
    c1 = map_one_pixel_itp_mode(c1, src.peak, dst.peak); \
    c2 = map_one_pixel_itp_mode(c2, src.peak, dst.peak); \
    c3 = map_one_pixel_itp_mode(c3, src.peak, dst.peak);

#define _RGB2YUV \
    yuv0 = lrgb2yuv(c0); \
    yuv1 = lrgb2yuv(c1); \
    yuv2 = lrgb2yuv(c2); \
    yuv3 = lrgb2yuv(c3);

#define _RGB2YUV_SRC \
    c0 = lrgb2lrgb(c0); \
    c1 = lrgb2lrgb(c1); \
    c2 = lrgb2lrgb(c2); \
    c3 = lrgb2lrgb(c3); \
    if (!rgb2rgb_passthrough) {  \
        c0 = gamut_compress(c0); \
        c1 = gamut_compress(c1); \
        c2 = gamut_compress(c2); \
        c3 = gamut_compress(c3); \
    } \
    yuv0 = lrgb2yuv(clamp3(c0, 0.0f, 1.0f)); \
    yuv1 = lrgb2yuv(clamp3(c1, 0.0f, 1.0f)); \
    yuv2 = lrgb2yuv(clamp3(c2, 0.0f, 1.0f)); \
    yuv3 = lrgb2yuv(clamp3(c3, 0.0f, 1.0f));

#define _DITHER \
    float d = read_dither(dither_tex, dither_size, xi, yi); \
    yuv0.x = get_dithered_y(yuv0.x, d); \
    yuv1.x = get_dithered_y(yuv1.x, d); \
    yuv2.x = get_dithered_y(yuv2.x, d); \
    yuv3.x = get_dithered_y(yuv3.x, d);

#define _LUT3D \
    yuv0 = apply_lut3d(lut, yuv0); \
    yuv1 = apply_lut3d(lut, yuv1); \
    yuv2 = apply_lut3d(lut, yuv2); \
    yuv3 = apply_lut3d(lut, yuv3);

#define _WRITER \
    write_2x2_flt(dst, x, y, yuv0, yuv1, yuv2, yuv3);

#define TONEMAP_VARIANT(NAME, READER, RESHAPE, YUV2RGB, TONEMAP, RGB2YUV, DITHER, WRITER) \
__global__ void tonemap ## NAME( \
    FFCUDAFrame src, FFCUDAFrame dst, \
    cudaTextureObject_t dither_tex, \
    const float *__restrict__ dovi_buf) \
{ \
    READER \
    RESHAPE \
    YUV2RGB \
    TONEMAP \
    RGB2YUV \
    DITHER \
    WRITER \
}

TONEMAP_VARIANT(_max,        _READER,         , _YUV2RGB,     _TONEMAP_MAX, _RGB2YUV,            , _WRITER)
TONEMAP_VARIANT(_max_d,      _READER,         , _YUV2RGB,     _TONEMAP_MAX, _RGB2YUV,     _DITHER, _WRITER)
TONEMAP_VARIANT(_rgb,        _READER,         , _YUV2RGB,     _TONEMAP_RGB, _RGB2YUV,            , _WRITER)
TONEMAP_VARIANT(_rgb_d,      _READER,         , _YUV2RGB,     _TONEMAP_RGB, _RGB2YUV,     _DITHER, _WRITER)
TONEMAP_VARIANT(_lum,        _READER,         , _YUV2RGB_SRC, _TONEMAP_LUM, _RGB2YUV_SRC,        , _WRITER)
TONEMAP_VARIANT(_lum_d,      _READER,         , _YUV2RGB_SRC, _TONEMAP_LUM, _RGB2YUV_SRC, _DITHER, _WRITER)
TONEMAP_VARIANT(_itp,        _READER,         , _YUV2RGB_SRC, _TONEMAP_ITP, _RGB2YUV_SRC,        , _WRITER)
TONEMAP_VARIANT(_itp_d,      _READER,         , _YUV2RGB_SRC, _TONEMAP_ITP, _RGB2YUV_SRC, _DITHER, _WRITER)
TONEMAP_VARIANT(_dovi_max,   _READER, _RESHAPE, _YCC2RGB,     _TONEMAP_MAX, _RGB2YUV,            , _WRITER)
TONEMAP_VARIANT(_dovi_max_d, _READER, _RESHAPE, _YCC2RGB,     _TONEMAP_MAX, _RGB2YUV,     _DITHER, _WRITER)
TONEMAP_VARIANT(_dovi_rgb,   _READER, _RESHAPE, _YCC2RGB,     _TONEMAP_RGB, _RGB2YUV,            , _WRITER)
TONEMAP_VARIANT(_dovi_rgb_d, _READER, _RESHAPE, _YCC2RGB,     _TONEMAP_RGB, _RGB2YUV,     _DITHER, _WRITER)
TONEMAP_VARIANT(_dovi_lum,   _READER, _RESHAPE, _YCC2RGB_SRC, _TONEMAP_LUM, _RGB2YUV_SRC,        , _WRITER)
TONEMAP_VARIANT(_dovi_lum_d, _READER, _RESHAPE, _YCC2RGB_SRC, _TONEMAP_LUM, _RGB2YUV_SRC, _DITHER, _WRITER)
TONEMAP_VARIANT(_dovi_itp,   _READER, _RESHAPE, _YCC2RGB_SRC, _TONEMAP_ITP, _RGB2YUV_SRC,        , _WRITER)
TONEMAP_VARIANT(_dovi_itp_d, _READER, _RESHAPE, _YCC2RGB_SRC, _TONEMAP_ITP, _RGB2YUV_SRC, _DITHER, _WRITER)
TONEMAP_VARIANT(_dovi_pq,    _READER, _RESHAPE, _YCC2RGB,                 , _RGB2YUV,            , _WRITER)

__global__ void build_lut(float4 *__restrict__ lut,
                          float peak,
                          float dst_peak,
                          int tonemap_mode,
                          int skip_tonemap,
                          int dovi_reshape)
{
    const int total_entries = lut_size * lut_size * lut_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_entries) return;
    int z = idx / (lut_size * lut_size);
    int rem = idx - (z * lut_size * lut_size);
    int y = rem / lut_size;
    int x = rem % lut_size;
    float fx = (float)x / (lut_size - 1);
    float fy = (float)y / (lut_size - 1);
    float fz = (float)z / (lut_size - 1);
    float3 c = make_float3(fx, fy, fz);
    if (tonemap_mode == TONEMAP_MODE_ITP) {
        c = dovi_reshape
            ? map_to_src_space_from_yuv_dovi(c)
            : map_to_src_space_from_yuv(c);
    } else {
        c = dovi_reshape
            ? map_to_dst_space_from_yuv_dovi(c)
            : map_to_dst_space_from_yuv(c);
    }
    switch (tonemap_mode) {
    case TONEMAP_MODE_MAX:
        c = skip_tonemap ? c : map_one_pixel_rgb_mode_max(c, peak, dst_peak);
        break;
    case TONEMAP_MODE_RGB:
        c = skip_tonemap ? c : map_one_pixel_rgb_mode_rgb(c, peak, dst_peak);
        break;
    case TONEMAP_MODE_LUM:
        c = skip_tonemap ? c : map_one_pixel_rgb_mode_lum(c, peak, dst_peak);
        break;
    case TONEMAP_MODE_ITP:
        c = skip_tonemap ? c : map_one_pixel_itp_mode(c, peak, dst_peak);
        break;
    }
    if (tonemap_mode == TONEMAP_MODE_ITP) {
        c = lrgb2lrgb(c);
        c = rgb2rgb_passthrough ? c : gamut_compress(c);
        c = clamp3(c, 0.0f, 1.0f);
    }
    c = lrgb2yuv(c);
    c = clamp3(c, 0.0f, 1.0f);
    lut[idx] = make_float4(c.x, c.y, c.z, 0.0f);
}

#define TONEMAP_LUT_VARIANT(NAME, READER, RESHAPE, LUT3D, WRITER) \
__global__ void tonemap_lut ## NAME( \
    FFCUDAFrame src, FFCUDAFrame dst, \
    const float4 *__restrict__ lut, \
    const float *__restrict__ dovi_buf) \
{ \
    READER \
    RESHAPE \
    LUT3D \
    WRITER \
}

TONEMAP_LUT_VARIANT(     , _READER,         , _LUT3D, _WRITER)
TONEMAP_LUT_VARIANT(_dovi, _READER, _RESHAPE, _LUT3D, _WRITER)

}
