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

float4 eotf_st2084x4(float4 x) {
    x.x = eotf_st2084_common(x.x);
    x.y = eotf_st2084_common(x.y);
    x.z = eotf_st2084_common(x.z);
    x.w = eotf_st2084_common(x.w);
    return x * pq_max_lum_div_ref_white;
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
    return ootf_1_2(inverse_oetf_arib_b67(x)) * 5.0f;
}

// delinearizer for HLG/ARIB-B67
float inverse_eotf_arib_b67(float x) {
    return oetf_arib_b67(inverse_ootf_1_2(x / 5.0f));
}

// delinearizer for BT709, BT2020-10
float inverse_eotf_bt1886(float x) {
    return x > 0.0f ? native_powr(x, 1.0f / 2.4f) : 0.0f;
}

float3 yuv2rgb(float y, float u, float v) {
    y += mix(0.0f, input_quantization_offset, y > 0.0f);
    u += mix(0.0f, input_quantization_offset, u > 0.0f);
    v += mix(0.0f, input_quantization_offset, v > 0.0f);
#ifndef FULL_RANGE_IN
    y = input_y_scale * y - 0.07305936073f;
    u = input_uv_scale * u - 0.5714285714f;
    v = input_uv_scale * v - 0.5714285714f;
#else
    u -= 0.5f; v -= 0.5f;
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
#ifndef FULL_RANGE_OUT
  #ifdef RESCALE_LIMITED_RANGE_OUTPUT
    y = floor(((219.0f * y + 16.0f) * 256.0f) + 0.5f) / 65535.0f;
    u = floor(((224.0f * u + 128.0f) * 256.0f) + 0.5f) / 65535.0f;
    v = floor(((224.0f * v + 128.0f) * 256.0f) + 0.5f) / 65535.0f;
  #else
    y = floor((219.0f * y + 16.0f) + 0.5f) / 255.0f;
    u = floor((224.0f * u + 128.0f) + 0.5f) / 255.0f;
    v = floor((224.0f * v + 128.0f) + 0.5f) / 255.0f;
  #endif
#else
    u += 0.5f; v += 0.5f;
#endif
    y -= mix(0.0f, output_quantization_offset, y > 0.0f);
    u -= mix(0.0f, output_quantization_offset, u > 0.0f);
    v -= mix(0.0f, output_quantization_offset, v > 0.0f);
    return (float3)(y, u, v);
}

float rgb2y(float r, float g, float b) {
    float y = r*yuv_matrix[0] + g*yuv_matrix[1] + b*yuv_matrix[2];
#ifndef FULL_RANGE_OUT
  #ifdef RESCALE_LIMITED_RANGE_OUTPUT
    y = floor(((219.0f * y + 16.0f) * 256.0f) + 0.5f) / 65535.0f;
  #else
    y = floor((219.0f * y + 16.0f) + 0.5f) / 255.0f;
  #endif
#endif
    y -= mix(0.0f, output_quantization_offset, y > 0.0f);
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
    return (float3)(r, g, b);
#else
    return c;
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
    r = eotf_st2084_common(r);
    g = eotf_st2084_common(g);
    b = eotf_st2084_common(b);
    float rr = r * lms2rgb_matrix[0] + g * lms2rgb_matrix[1] + b * lms2rgb_matrix[2];
    float gg = r * lms2rgb_matrix[3] + g * lms2rgb_matrix[4] + b * lms2rgb_matrix[5];
    float bb = r * lms2rgb_matrix[6] + g * lms2rgb_matrix[7] + b * lms2rgb_matrix[8];
    rr = inverse_eotf_st2084_common(rr);
    gg = inverse_eotf_st2084_common(gg);
    bb = inverse_eotf_st2084_common(bb);
    return (float3)(rr, gg, bb);
}
#endif

#ifdef TONE_MODE_ITP
// The following assumes bt2020
void lrgb2ictcp(float4 r4, float4 g4, float4 b4, float4* i4, float4* ct4, float4* cp4) {
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

void ictcp2lrgb(float4 i4, float4 ct4, float4 cp4, float4* r4, float4* g4, float4* b4) {
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
#endif

float parabolic(float x, float t0, float x0, float y0) {
    float s = (y0 - t0) / native_sqrt(x0 - y0);
    float ox = t0 - s * s * 0.25f;
    float oy = t0 - s * native_sqrt(s * s * 0.25f);
    return (x < t0 ? x : s * native_sqrt(x - ox) + oy);
}

float3 gamut_compress(float3 rgb) {
    // BT.709 boundary info
    #define cyan_limit 1.5187050250638159f
    #define magenta_limit 1.0750082769546088f
    #define yellow_limit 1.0887800403483898f
    #define cyan_threshold 1.050508660266247f
    #define magenta_threshold 0.940509816042432f
    #define yellow_threshold 0.9771607996420639f

    // Achromatic axis
    float ac = fmax(fmax(rgb.x, rgb.y), rgb.z);

    // Inverse RGB Ratios: distance from achromatic axis
    float3 d = ac == 0.0f ? (float3)(0.0f, 0.0f, 0.0f) : (ac - rgb) / fabs(ac);

    // Compressed distance
    float3 cd = (float3)(
        parabolic(d.x, cyan_threshold, cyan_limit, 1.0f),
        parabolic(d.y, magenta_threshold, magenta_limit, 1.0f),
        parabolic(d.z, yellow_threshold, yellow_limit, 1.0f)
    );

    // Inverse RGB Ratios to RGB
    float3 crgb = ac - cd * fabs(ac);

    return crgb;
}
