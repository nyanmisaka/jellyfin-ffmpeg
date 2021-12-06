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

#define mix(x, y, a) ((x) + ((y) - (x)) * (a))

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
float bt2390(float s, float peak, float dst_peak) {
    float peak_pq = inverse_eotf_st2084(peak);
    float scale = peak_pq > 0.0f ? (1.0f / peak_pq) : 1.0f;

    float s_pq = inverse_eotf_st2084(s) * scale;
    float max_lum = inverse_eotf_st2084(dst_peak) * scale;

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
float3 map_one_pixel_rgb(float3 rgb, const FFCUDAFrame& src, const FFCUDAFrame& dst) {
    float sig = max(max(rgb.x, max(rgb.y, rgb.z)), FLOAT_EPS);
    float peak = src.peak;
    float dst_peak = dst.peak;

    // Rescale the variables in order to bring it into a representation where
    // 1.0 represents the dst_peak. This is because all of the tone mapping
    // algorithms are defined in such a way that they map to the range [0.0, 1.0].
    if (dst.peak > 1.0f) {
        sig *= 1.0f / dst.peak;
        peak *= 1.0f / dst.peak;
    }

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

// Map from source space YUV to destination space RGB
static __inline__ __device__
float3 map_to_dst_space_from_yuv(float3 yuv) {
    float3 c = yuv2lrgb(yuv);
    c = lrgb2lrgb(c);
    return c;
}

extern "C" {

__global__ void tonemap(FFCUDAFrame src, FFCUDAFrame dst)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    // each work item process four pixels
    int x = 2 * xi;
    int y = 2 * yi;

    if (y + 1 < src.height && x + 1 < src.width)
    {
        float3 yuv0 = read_px_flt(src, x,     y);
        float3 yuv1 = read_px_flt(src, x + 1, y);
        float3 yuv2 = read_px_flt(src, x,     y + 1);
        float3 yuv3 = read_px_flt(src, x + 1, y + 1);

        float3 c0 = map_to_dst_space_from_yuv(yuv0);
        float3 c1 = map_to_dst_space_from_yuv(yuv1);
        float3 c2 = map_to_dst_space_from_yuv(yuv2);
        float3 c3 = map_to_dst_space_from_yuv(yuv3);

        c0 = map_one_pixel_rgb(c0, src, dst);
        c1 = map_one_pixel_rgb(c1, src, dst);
        c2 = map_one_pixel_rgb(c2, src, dst);
        c3 = map_one_pixel_rgb(c3, src, dst);

        yuv0 = lrgb2yuv(c0);
        yuv1 = lrgb2yuv(c1);
        yuv2 = lrgb2yuv(c2);
        yuv3 = lrgb2yuv(c3);

        write_2x2_flt(dst, x, y, yuv0, yuv1, yuv2, yuv3);
    }
}

}
