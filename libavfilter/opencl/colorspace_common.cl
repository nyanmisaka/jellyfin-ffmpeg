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

#if chroma_loc == 1
    #define chroma_sample(a,b,c,d) (((a) + (c)) * 0.5f)
#elif chroma_loc == 3
    #define chroma_sample(a,b,c,d) (a)
#elif chroma_loc == 4
    #define chroma_sample(a,b,c,d) (((a) + (b)) * 0.5f)
#elif chroma_loc == 5
    #define chroma_sample(a,b,c,d) (c)
#elif chroma_loc == 6
    #define chroma_sample(a,b,c,d) (((c) + (d)) * 0.5f)
#else
    #define chroma_sample(a,b,c,d) (((a) + (b) + (c) + (d)) * 0.25f)
#endif

float get_luma_dst(float3 c) {
    return luma_dst.x * c.x + luma_dst.y * c.y + luma_dst.z * c.z;
}

float4 get_luma_dst4(float4 r4, float4 g4, float4 b4) {
    return luma_dst.x * r4 + luma_dst.y * g4 + luma_dst.z * b4;
}

/*
float get_luma_src(float3 c) {
    return luma_src.x * c.x + luma_src.y * c.y + luma_src.z * c.z;
}

float4 get_luma_src4(float4 r4, float4 g4, float4 b4) {
    return luma_src.x * r4 + luma_src.y * g4 + luma_src.z * b4;
}
*/

float3 get_chroma_sample(float3 a, float3 b, float3 c, float3 d) {
    return chroma_sample(a, b, c, d);
}

// linearizer for PQ/ST2084
float eotf_st2084_common(float x) {
    x = fmax(x, 0.0f);
    float xpow = native_powr(x, 1.0f / ST2084_M2);
    float num = fmax(xpow - ST2084_C1, 0.0f);
    float den = fmax(ST2084_C2 - ST2084_C3 * xpow, FLOAT_EPS);
    x = native_powr(num / den, 1.0f / ST2084_M1);
    return x;
}

float eotf_st2084(float x) {
    return eotf_st2084_common(x) * pq_max_lum_div_ref_white;
}

// delinearizer for PQ/ST2084
float inverse_eotf_st2084_common(float x) {
    x = fmax(x, 0.0f);
    float xpow = native_powr(x, ST2084_M1);
#if 0
    // Original formulation from SMPTE ST 2084:2014 publication.
    float num = ST2084_C1 + ST2084_C2 * xpow;
    float den = 1.0f + ST2084_C3 * xpow;
    return native_powr(num / den, ST2084_M2);
#else
    // More stable arrangement that avoids some cancellation error.
    float num = (ST2084_C1 - 1.0f) + (ST2084_C2 - ST2084_C3) * xpow;
    float den = 1.0f + ST2084_C3 * xpow;
    return native_powr(1.0f + num / den, ST2084_M2);
#endif
}

float inverse_eotf_st2084(float x) {
    x *= ref_white_div_pq_max_lum;
    return inverse_eotf_st2084_common(x);
}

float ootf_1_2(float x) {
    return x > 0.0f ? native_powr(x, 1.2f) : x;
}

float inverse_ootf_1_2(float x) {
    return x > 0.0f ? native_powr(x, 1.0f / 1.2f) : x;
}

float oetf_arib_b67(float x) {
    x = fmax(x, 0.0f);
    return x <= (1.0f / 12.0f)
           ? native_sqrt(3.0f * x)
           : (ARIB_B67_A * native_log(12.0f * x - ARIB_B67_B) + ARIB_B67_C);
}

float inverse_oetf_arib_b67(float x) {
    x = fmax(x, 0.0f);
    return x <= 0.5f
           ? (x * x) * (1.0f / 3.0f)
           : (native_exp((x - ARIB_B67_C) / ARIB_B67_A) + ARIB_B67_B) * (1.0f / 12.0f);
}

// linearizer for HLG/ARIB-B67
float eotf_arib_b67(float x) {
    return ootf_1_2(inverse_oetf_arib_b67(x));
}

// delinearizer for HLG/ARIB-B67
float inverse_eotf_arib_b67(float x) {
    return oetf_arib_b67(inverse_ootf_1_2(x));
}

// delinearizer for BT709, BT2020-10
float inverse_eotf_bt1886(float x) {
    return x > 0.0f ? native_powr(x, 1.0f / 2.4f) : 0.0f;
}

#ifdef LUT_TRC
float linearize_lut(float x) {
    return lin_lut[clamp((int)(x * LUT_TRC), 0, LUT_TRC)];
}
#endif

float3 yuv2rgb(float y, float u, float v) {
#ifdef FULL_RANGE_IN
    u -= 0.5f; v -= 0.5f;
#else
    y = (y * 255.0f -  16.0f) / 219.0f;
    u = (u * 255.0f - 128.0f) / 224.0f;
    v = (v * 255.0f - 128.0f) / 224.0f;
#endif
    float r = y * rgb_matrix[0] + u * rgb_matrix[1] + v * rgb_matrix[2];
    float g = y * rgb_matrix[3] + u * rgb_matrix[4] + v * rgb_matrix[5];
    float b = y * rgb_matrix[6] + u * rgb_matrix[7] + v * rgb_matrix[8];
    return (float3)(r, g, b);
}

float3 yuv2lrgb(float3 yuv) {
    float3 rgb = yuv2rgb(yuv.x, yuv.y, yuv.z);
#ifdef linearize
    float r = linearize(rgb.x);
    float g = linearize(rgb.y);
    float b = linearize(rgb.z);
    return (float3)(r, g, b);
#else
    return rgb;
#endif
}

float3 rgb2yuv(float r, float g, float b) {
    float y = r*yuv_matrix[0] + g*yuv_matrix[1] + b*yuv_matrix[2];
    float u = r*yuv_matrix[3] + g*yuv_matrix[4] + b*yuv_matrix[5];
    float v = r*yuv_matrix[6] + g*yuv_matrix[7] + b*yuv_matrix[8];
#ifdef FULL_RANGE_OUT
    u += 0.5f; v += 0.5f;
#else
    y = (219.0f * y + 16.0f) / 255.0f;
    u = (224.0f * u + 128.0f) / 255.0f;
    v = (224.0f * v + 128.0f) / 255.0f;
#endif
    return (float3)(y, u, v);
}

float rgb2y(float r, float g, float b) {
    float y = r*yuv_matrix[0] + g*yuv_matrix[1] + b*yuv_matrix[2];
    y = (219.0f * y + 16.0f) / 255.0f;
    return y;
}

float3 lrgb2yuv(float3 c) {
#ifdef delinearize
    float r = delinearize(c.x);
    float g = delinearize(c.y);
    float b = delinearize(c.z);
    return rgb2yuv(r, g, b);
#else
    return rgb2yuv(c.x, c.y, c.z);
#endif
}

float lrgb2y(float3 c) {
#ifdef delinearize
    float r = delinearize(c.x);
    float g = delinearize(c.y);
    float b = delinearize(c.z);
    return rgb2y(r, g, b);
#else
    return rgb2y(c.x, c.y, c.z);
#endif
}

float3 lrgb2lrgb(float3 c) {
#ifdef RGB2RGB_PASSTHROUGH
    return c;
#else
    float r = c.x, g = c.y, b = c.z;
    float rr = rgb2rgb[0] * r + rgb2rgb[1] * g + rgb2rgb[2] * b;
    float gg = rgb2rgb[3] * r + rgb2rgb[4] * g + rgb2rgb[5] * b;
    float bb = rgb2rgb[6] * r + rgb2rgb[7] * g + rgb2rgb[8] * b;
    return (float3)(rr, gg, bb);
#endif
}

float3 rgb2lrgb(float3 c) {
#ifdef linearize
    float r = linearize(c.x);
    float g = linearize(c.y);
    float b = linearize(c.z);
    return lrgb2lrgb((float3)(r, g, b));
#else
    return lrgb2lrgb((float3)(c.x, c.y, c.z));
#endif
}

#ifdef DOVI_RESHAPE
float3 ycc2rgb(float y, float cb, float cr) {
    float r = y * rgb_matrix[0] + cb * rgb_matrix[1] + cr * rgb_matrix[2];
    float g = y * rgb_matrix[3] + cb * rgb_matrix[4] + cr * rgb_matrix[5];
    float b = y * rgb_matrix[6] + cb * rgb_matrix[7] + cr * rgb_matrix[8];
    return (float3)(r, g, b) + ycc2rgb_offset;
}

float3 lms2rgb(float r, float g, float b) {
  #ifndef DOVI_PERF_TRADEOFF
    r = eotf_st2084_common(r);
    g = eotf_st2084_common(g);
    b = eotf_st2084_common(b);
  #endif
    float rr = r * lms2rgb_matrix[0] + g * lms2rgb_matrix[1] + b * lms2rgb_matrix[2];
    float gg = r * lms2rgb_matrix[3] + g * lms2rgb_matrix[4] + b * lms2rgb_matrix[5];
    float bb = r * lms2rgb_matrix[6] + g * lms2rgb_matrix[7] + b * lms2rgb_matrix[8];
  #ifndef DOVI_PERF_TRADEOFF
    rr = inverse_eotf_st2084_common(rr);
    gg = inverse_eotf_st2084_common(gg);
    bb = inverse_eotf_st2084_common(bb);
  #endif
    return (float3)(rr, gg, bb);
}
#endif
