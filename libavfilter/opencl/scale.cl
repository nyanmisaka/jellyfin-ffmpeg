/*
 * Copyright (c) 2026 NyanMisaka
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

__constant sampler_t n_sampler = (CLK_NORMALIZED_COORDS_FALSE |
                                  CLK_ADDRESS_CLAMP_TO_EDGE   |
                                  CLK_FILTER_NEAREST);

__constant sampler_t l_sampler = (CLK_NORMALIZED_COORDS_FALSE |
                                  CLK_ADDRESS_CLAMP_TO_EDGE   |
                                  CLK_FILTER_LINEAR);

__constant sampler_t d_sampler = (CLK_NORMALIZED_COORDS_TRUE  |
                                  CLK_ADDRESS_REPEAT          |
                                  CLK_FILTER_NEAREST);

#ifdef ENABLE_DITHER
float get_dithered_y(float y, float d) {
    return floor(y * dither_quantization + d + 0.5f / dither_size2) * 1.0f / dither_quantization;
}
#endif

#ifdef SCALE_BUILTIN
__kernel void scale_builtin(__write_only image2d_t dst1,
                            __read_only  image2d_t src1,
                            __write_only image2d_t dst2,
                            __read_only  image2d_t src2,
  #ifdef NON_SEMI_PLANAR_OUT
                            __write_only image2d_t dst3,
  #endif
  #if defined(NON_SEMI_PLANAR_IN) && !defined(BLIT_NV15)
                            __read_only  image2d_t src3,
  #endif
  #ifdef ENABLE_DITHER
                            __read_only  image2d_t dither,
  #endif
                                         int4      crop_whxy)
{
    int2 uv_pos = (int2)(get_global_id(0), get_global_id(1));
    int2 y_pos_base = uv_pos << 1;
    float2 dst1_sz = (float2)(get_global_size(0) << 1, get_global_size(1) << 1);

    float2 scale_wh = convert_float2(crop_whxy.xy) / dst1_sz;
    float2 base_offset = 0.5f * scale_wh + convert_float2(crop_whxy.zw);

  #if defined(ENABLE_DITHER) && !defined(BLIT_NV15)
    float2 ncoords_d = convert_float2(uv_pos) *
        native_recip((float2)(get_image_width(dither), get_image_height(dither)));
  #endif
    float2 coord_y00 = convert_float2(y_pos_base + (int2)(0, 0)) * scale_wh + base_offset;
    float2 coord_y10 = convert_float2(y_pos_base + (int2)(1, 0)) * scale_wh + base_offset;
    float2 coord_y01 = convert_float2(y_pos_base + (int2)(0, 1)) * scale_wh + base_offset;
    float2 coord_y11 = convert_float2(y_pos_base + (int2)(1, 1)) * scale_wh + base_offset;
  #ifdef SCALE_BUILTIN_BILINEAR
    float res_y00 = read_imagef(src1, l_sampler, coord_y00).x;
    float res_y10 = read_imagef(src1, l_sampler, coord_y10).x;
    float res_y01 = read_imagef(src1, l_sampler, coord_y01).x;
    float res_y11 = read_imagef(src1, l_sampler, coord_y11).x;
  #else // SCALE_BUILTIN_NEIGHBOR
    float res_y00 = read_imagef(src1, n_sampler, coord_y00).x;
    float res_y10 = read_imagef(src1, n_sampler, coord_y10).x;
    float res_y01 = read_imagef(src1, n_sampler, coord_y01).x;
    float res_y11 = read_imagef(src1, n_sampler, coord_y11).x;
  #endif
  #if defined(ENABLE_DITHER) && !defined(BLIT_NV15)
    float d = read_imagef(dither, d_sampler, ncoords_d).x;
    res_y00 = get_dithered_y(res_y00, d);
    res_y10 = get_dithered_y(res_y10, d);
    res_y01 = get_dithered_y(res_y01, d);
    res_y11 = get_dithered_y(res_y11, d);
  #endif
    write_imagef(dst1, y_pos_base + (int2)(0, 0), (float4)(res_y00, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, y_pos_base + (int2)(1, 0), (float4)(res_y10, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, y_pos_base + (int2)(0, 1), (float4)(res_y01, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, y_pos_base + (int2)(1, 1), (float4)(res_y11, 0.0f, 0.0f, 1.0f));

    float2 coord_uv = ((convert_float2(y_pos_base) + 0.5f) * scale_wh + base_offset) * 0.5f;
  #if defined(NON_SEMI_PLANAR_IN) && !defined(BLIT_NV15)
    #ifdef SCALE_BUILTIN_BILINEAR
    float2 res_uv = { read_imagef(src2, l_sampler, coord_uv).x,
                      read_imagef(src3, l_sampler, coord_uv).x };
    #else // SCALE_BUILTIN_NEIGHBOR
    float2 res_uv = { read_imagef(src2, n_sampler, coord_uv).x,
                      read_imagef(src3, n_sampler, coord_uv).x };
    #endif
  #else
    #ifdef SCALE_BUILTIN_BILINEAR
    float2 res_uv = read_imagef(src2, l_sampler, coord_uv).xy;
    #else // SCALE_BUILTIN_NEIGHBOR
    float2 res_uv = read_imagef(src2, n_sampler, coord_uv).xy;
    #endif
  #endif
  #ifdef NON_SEMI_PLANAR_OUT
    write_imagef(dst2, uv_pos, (float4)(res_uv.x, 0.0f, 0.0f, 1.0f));
    write_imagef(dst3, uv_pos, (float4)(res_uv.y, 0.0f, 0.0f, 1.0f));
  #else
    write_imagef(dst2, uv_pos, (float4)(res_uv.x, res_uv.y, 0.0f, 1.0f));
  #endif
}
#endif

#ifdef SCALE_CONVOLVE
float4 bicubic_coeffs(float x)
{
  #ifdef SCALE_PARAM
    const float A = -scale_param;
  #else
    const float A = 0.0f;
  #endif

    float4 res;
    res.x = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    res.y = ((A + 2) * x - (A + 3)) * x * x + 1;
    res.z = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    res.w = 1.0f - res.x - res.y - res.z;

    return res;
}

float4 lanczos_coeffs(float x)
{
    float4 v = (float4)(x + 1.0f, x, x - 1.0f, x - 2.0f);
    float4 v_pi = v * M_PI_F;

    float4 s1 = native_sin(v_pi);
    float4 s2 = native_sin(v_pi * 0.5f);
    float4 num = s1 * s2 * 2.0f;
    float4 den = v_pi * v_pi;
    float4 res = select(num / den, (float4)(1.0f), isequal(v, (float4)(0.0f)));

    return res / (res.x + res.y + res.z + res.w);
}

float2 convolve1(__read_only image2d_t src,
                 sampler_t sampler, float2 coord)
{
    float2 pos = floor(coord);
    float2 fxy = coord - pos;

  #ifndef COEFFS_FUNCTION
    #define COEFFS_FUNCTION bicubic_coeffs
  #endif
    float4 cx = COEFFS_FUNCTION(fxy.x);
    float4 cy = COEFFS_FUNCTION(fxy.y);

    float2 r[4];
  #pragma unroll
    for (int i = 0; i < 4; i++) {
        float y_map = pos.y - 1 + i;
        r[i] = read_imagef(src, sampler, (int2)(pos.x - 1, y_map)).xy * cx.x +
               read_imagef(src, sampler, (int2)(pos.x,     y_map)).xy * cx.y +
               read_imagef(src, sampler, (int2)(pos.x + 1, y_map)).xy * cx.z +
               read_imagef(src, sampler, (int2)(pos.x + 2, y_map)).xy * cx.w;
    }
    return r[0] * cy.x + r[1] * cy.y + r[2] * cy.z + r[3] * cy.w;
}

float2 convolve2(__read_only image2d_t src1,
                 __read_only image2d_t src2,
                 sampler_t sampler, float2 coord)
{
    float2 pos = floor(coord);
    float2 fxy = coord - pos;

  #ifndef COEFFS_FUNCTION
    #define COEFFS_FUNCTION bicubic_coeffs
  #endif
    float4 cx = COEFFS_FUNCTION(fxy.x);
    float4 cy = COEFFS_FUNCTION(fxy.y);

    float2 r[4];
  #pragma unroll
    for (int i = 0; i < 4; i++) {
        float y_map = pos.y - 1 + i;
        r[i].x = read_imagef(src1, sampler, (int2)(pos.x - 1, y_map)).x * cx.x +
                 read_imagef(src1, sampler, (int2)(pos.x,     y_map)).x * cx.y +
                 read_imagef(src1, sampler, (int2)(pos.x + 1, y_map)).x * cx.z +
                 read_imagef(src1, sampler, (int2)(pos.x + 2, y_map)).x * cx.w;
        r[i].y = read_imagef(src2, sampler, (int2)(pos.x - 1, y_map)).x * cx.x +
                 read_imagef(src2, sampler, (int2)(pos.x,     y_map)).x * cx.y +
                 read_imagef(src2, sampler, (int2)(pos.x + 1, y_map)).x * cx.z +
                 read_imagef(src2, sampler, (int2)(pos.x + 2, y_map)).x * cx.w;
    }
    return r[0] * cy.x + r[1] * cy.y + r[2] * cy.z + r[3] * cy.w;
}

__kernel void scale_convolve(__write_only image2d_t dst1,
                             __read_only  image2d_t src1,
                             __write_only image2d_t dst2,
                             __read_only  image2d_t src2,
  #ifdef NON_SEMI_PLANAR_OUT
                             __write_only image2d_t dst3,
  #endif
  #if defined(NON_SEMI_PLANAR_IN) && !defined(BLIT_NV15)
                             __read_only  image2d_t src3,
  #endif
  #ifdef ENABLE_DITHER
                             __read_only  image2d_t dither,
  #endif
                                          int4      crop_whxy)
{
    int2 uv_pos = (int2)(get_global_id(0), get_global_id(1));
    int2 y_pos_base = uv_pos << 1;
    float2 dst1_sz = (float2)(get_global_size(0) << 1, get_global_size(1) << 1);

    float2 scale_wh = convert_float2(crop_whxy.xy) / dst1_sz;
    float2 base_offset = 0.5f * scale_wh - 0.5f + convert_float2(crop_whxy.zw);

  #if defined(ENABLE_DITHER) && !defined(BLIT_NV15)
    float2 ncoords_d = convert_float2(uv_pos) *
        native_recip((float2)(get_image_width(dither), get_image_height(dither)));
  #endif
    float2 coord_y00 = convert_float2(y_pos_base + (int2)(0, 0)) * scale_wh + base_offset;
    float2 coord_y10 = convert_float2(y_pos_base + (int2)(1, 0)) * scale_wh + base_offset;
    float2 coord_y01 = convert_float2(y_pos_base + (int2)(0, 1)) * scale_wh + base_offset;
    float2 coord_y11 = convert_float2(y_pos_base + (int2)(1, 1)) * scale_wh + base_offset;
    float res_y00 = convolve1(src1, n_sampler, coord_y00).x;
    float res_y10 = convolve1(src1, n_sampler, coord_y10).x;
    float res_y01 = convolve1(src1, n_sampler, coord_y01).x;
    float res_y11 = convolve1(src1, n_sampler, coord_y11).x;
  #if defined(ENABLE_DITHER) && !defined(BLIT_NV15)
    float d = read_imagef(dither, d_sampler, ncoords_d).x;
    res_y00 = get_dithered_y(res_y00, d);
    res_y10 = get_dithered_y(res_y10, d);
    res_y01 = get_dithered_y(res_y01, d);
    res_y11 = get_dithered_y(res_y11, d);
  #endif
    write_imagef(dst1, y_pos_base + (int2)(0, 0), (float4)(res_y00, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, y_pos_base + (int2)(1, 0), (float4)(res_y10, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, y_pos_base + (int2)(0, 1), (float4)(res_y01, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, y_pos_base + (int2)(1, 1), (float4)(res_y11, 0.0f, 0.0f, 1.0f));

    float2 coord_uv = (convert_float2(y_pos_base) * scale_wh + base_offset) * 0.5f;
  #if defined(NON_SEMI_PLANAR_IN) && !defined(BLIT_NV15)
    float2 res_uv = convolve2(src2, src3, n_sampler, coord_uv);
  #else
    float2 res_uv = convolve1(src2, n_sampler, coord_uv);
  #endif
  #ifdef NON_SEMI_PLANAR_OUT
    write_imagef(dst2, uv_pos, (float4)(res_uv.x, 0.0f, 0.0f, 1.0f));
    write_imagef(dst3, uv_pos, (float4)(res_uv.y, 0.0f, 0.0f, 1.0f));
  #else
    write_imagef(dst2, uv_pos, (float4)(res_uv.x, res_uv.y, 0.0f, 1.0f));
  #endif
}
#endif

#ifdef BLIT_NV15
__kernel void blit_nv15(__write_only image2d_t dst1,
                        __read_only  image2d_t src1,
                        __write_only image2d_t dst2,
                        __read_only  image2d_t src2
  #if defined(NON_SEMI_PLANAR_OUT) && !defined(SCALE_BUILTIN) && !defined(SCALE_CONVOLVE)
                       ,__write_only image2d_t dst3
  #endif
  #ifdef ENABLE_DITHER
                       ,__read_only  image2d_t dither
  #endif
                        )
{
    int xi = get_global_id(0);
    int yi = get_global_id(1);
    int x = xi << 1;
    int y = yi << 1;

  #ifdef ENABLE_DITHER
    float2 ncoords_d = convert_float2((int2)(xi, yi)) *
        native_recip((float2)(get_image_width(dither), get_image_height(dither)));
  #endif
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
    float y0 = px4f.x, y1 = px4f.y, y2 = px4f.z, y3 = px4f.w;
  #ifdef ENABLE_DITHER
    float d = read_imagef(dither, d_sampler, ncoords_d).x;
    y0 = get_dithered_y(y0, d);
    y1 = get_dithered_y(y1, d);
    y2 = get_dithered_y(y2, d);
    y3 = get_dithered_y(y3, d);
  #endif
    write_imagef(dst1, (int2)(x,     y), (float4)(y0, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, (int2)(x + 1, y), (float4)(y1, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, (int2)(x,     y + 1), (float4)(y2, 0.0f, 0.0f, 1.0f));
    write_imagef(dst1, (int2)(x + 1, y + 1), (float4)(y3, 0.0f, 0.0f, 1.0f));

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
    float2 uv = convert_float2((px4ui.xy & 0x3FF) << 6) / USHRT_MAX;
  #if defined(NON_SEMI_PLANAR_OUT) && !defined(SCALE_BUILTIN) && !defined(SCALE_CONVOLVE)
    write_imagef(dst2, (int2)(xi, yi), (float4)(uv.x, 0.0f, 0.0f, 1.0f));
    write_imagef(dst3, (int2)(xi, yi), (float4)(uv.y, 0.0f, 0.0f, 1.0f));
  #else
    write_imagef(dst2, (int2)(xi, yi), (float4)(uv.x, uv.y, 0.0f, 1.0f));
  #endif
}
#endif
