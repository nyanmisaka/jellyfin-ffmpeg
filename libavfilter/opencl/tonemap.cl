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

#define FLOAT_EPS 1.175494351e-38f

extern float3 lrgb2yuv(float3);
extern float  lrgb2y(float3);
extern float3 yuv2lrgb(float3);
extern float3 lrgb2lrgb(float3);
extern float  linearize_pq(float);
extern float  delinearize_pq(float);
extern float  inverse_eotf_st2084(float);
extern float  get_luma_src(float3);
extern float  get_luma_dst(float3);
extern float3 get_chroma_sample(float3, float3, float3, float3);

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
    b = (j * j - 2.0f * j * peak + peak) / max(peak - 1.0f, FLOAT_EPS);

    return (b * b + 2.0f * b * j + j * j) / (b - a) * (s + a) / (s + b);
}

float bt2390(float s, float peak, float target_peak) {
    float peak_pq = inverse_eotf_st2084(peak);
    float scale = peak_pq > 0.0f ? (1.0f / peak_pq) : 1.0f;

    float s_pq = inverse_eotf_st2084(s) * scale;
    float max_lum = inverse_eotf_st2084(target_peak) * scale;

    float ks = 1.5f * max_lum - 0.5f;
    float tb = (s_pq - ks) / (1.0f - ks);
    float tb2 = tb * tb;
    float tb3 = tb2 * tb;
    float pb = (2.0f * tb3 - 3.0f * tb2 + 1.0f) * ks +
               (tb3 - 2.0f * tb2 + tb) * (1.0f - ks) +
               (-2.0f * tb3 + 3.0f * tb2) * max_lum;
    float sig = mix(pb, s_pq, s_pq < ks);

    return linearize_pq(sig * peak_pq);
}

float3 map_one_pixel_rgb(float3 rgb, float peak) {
    float sig = max(max(rgb.x, max(rgb.y, rgb.z)), FLOAT_EPS);

    // Rescale the variables in order to bring it into a representation where
    // 1.0 represents the dst_peak. This is because all of the tone mapping
    // algorithms are defined in such a way that they map to the range [0.0, 1.0].
    if (target_peak > 1.0f) {
        sig *= 1.0f / target_peak;
        peak *= 1.0f / target_peak;
    }

    float sig_old = sig;

    // Desaturate the color using a coefficient dependent on the signal level
    if (desat_param > 0.0f) {
        float luma = get_luma_dst(rgb);
        float coeff = max(sig - 0.18f, FLOAT_EPS) / max(sig, FLOAT_EPS);
        coeff = native_powr(coeff, 10.0f / desat_param);
        rgb = mix(rgb, (float3)luma, (float3)coeff);
    }

    sig = TONE_FUNC(sig, peak, target_peak);
    sig = min(sig, 1.0f);
    rgb *= (sig / sig_old);

    return rgb;
}

// Map from source space YUV to destination space RGB
float3 map_to_dst_space_from_yuv(float3 yuv) {
    float3 c = yuv2lrgb(yuv);
    c = lrgb2lrgb(c);
    return c;
}

__constant sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                                CLK_ADDRESS_CLAMP_TO_EDGE   |
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
                      float peak
                      )
{
    int xi = get_global_id(0);
    int yi = get_global_id(1);
    // each work item process four pixels
    int x = 2 * xi;
    int y = 2 * yi;

    if (xi < get_image_width(dst2) && yi < get_image_height(dst2)) {
        float y0 = read_imagef(src1, sampler, (int2)(x,     y)).x;
        float y1 = read_imagef(src1, sampler, (int2)(x + 1, y)).x;
        float y2 = read_imagef(src1, sampler, (int2)(x,     y + 1)).x;
        float y3 = read_imagef(src1, sampler, (int2)(x + 1, y + 1)).x;
#ifdef NON_SEMI_PLANAR_IN
        float u = read_imagef(src2, sampler, (int2)(xi, yi)).x;
        float v = read_imagef(src3, sampler, (int2)(xi, yi)).x;
        float2 uv = (float2)(u, v);
#else
        float2 uv = read_imagef(src2, sampler, (int2)(xi, yi)).xy;
#endif

        float3 c0 = map_to_dst_space_from_yuv((float3)(y0, uv.x, uv.y));
        float3 c1 = map_to_dst_space_from_yuv((float3)(y1, uv.x, uv.y));
        float3 c2 = map_to_dst_space_from_yuv((float3)(y2, uv.x, uv.y));
        float3 c3 = map_to_dst_space_from_yuv((float3)(y3, uv.x, uv.y));

        c0 = map_one_pixel_rgb(c0, peak);
        c1 = map_one_pixel_rgb(c1, peak);
        c2 = map_one_pixel_rgb(c2, peak);
        c3 = map_one_pixel_rgb(c3, peak);

        y0 = lrgb2y(c0);
        y1 = lrgb2y(c1);
        y2 = lrgb2y(c2);
        y3 = lrgb2y(c3);

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
}
