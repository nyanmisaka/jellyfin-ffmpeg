/*
 * Copyright (c) 2018 Gabriel Machado
 * Copyright (c) 2021 NyanMisaka
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

__constant sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                                CLK_ADDRESS_CLAMP_TO_EDGE   |
                                CLK_FILTER_NEAREST);

__constant sampler_t sampler2 = (CLK_NORMALIZED_COORDS_FALSE |
                                 CLK_ADDRESS_NONE            |
                                 CLK_FILTER_NEAREST);

__constant sampler_t d_sampler = (CLK_NORMALIZED_COORDS_TRUE |
                                  CLK_ADDRESS_REPEAT         |
                                  CLK_FILTER_NEAREST);

#ifdef ENABLE_DITHER
float get_dithered_y(float y, float d) {
    return floor(y * dither_quantization + d + 0.5f / dither_size2) * 1.0f / dither_quantization;
}
#endif

#ifdef CONV
__kernel void conv_yuv(__write_only image2d_t dst1,
                       __read_only  image2d_t src1,
                       __write_only image2d_t dst2,
                       __read_only  image2d_t src2
#ifdef NON_SEMI_PLANAR_OUT
                      ,__write_only image2d_t dst3
#endif
#ifdef NON_SEMI_PLANAR_IN
                      ,__read_only  image2d_t src3
#endif
#ifdef ENABLE_DITHER
                      ,__read_only  image2d_t dither
#endif
                       )
{
    int xi = get_global_id(0);
    int yi = get_global_id(1);
    // each work item process four pixels
    int x = 2 * xi;
    int y = 2 * yi;

#ifdef ENABLE_DITHER
    float2 ncoords = convert_float2((int2)(xi, yi)) *
        native_recip((float2)(get_image_width(dither), get_image_height(dither)));
#endif

    if (xi < get_image_width(dst2) && yi < get_image_height(dst2)) {
        float y0 = read_imagef(src1, sampler, (int2)(x,     y)).x;
        float y1 = read_imagef(src1, sampler, (int2)(x + 1, y)).x;
        float y2 = read_imagef(src1, sampler, (int2)(x,     y + 1)).x;
        float y3 = read_imagef(src1, sampler, (int2)(x + 1, y + 1)).x;
#ifdef NON_SEMI_PLANAR_IN
        float u = read_imagef(src2, sampler, (int2)(xi, yi)).x;
        float v = read_imagef(src3, sampler, (int2)(xi, yi)).x;
#else
        float2 uv = read_imagef(src2, sampler, (int2)(xi, yi)).xy;
        float u = uv.x;
        float v = uv.y;
#endif
#ifdef ENABLE_DITHER
        float d = read_imagef(dither, d_sampler, ncoords).x;
        y0 = get_dithered_y(y0, d);
        y1 = get_dithered_y(y1, d);
        y2 = get_dithered_y(y2, d);
        y3 = get_dithered_y(y3, d);
#endif
        write_imagef(dst1, (int2)(x,     y), (float4)(y0, 0.0f, 0.0f, 1.0f));
        write_imagef(dst1, (int2)(x + 1, y), (float4)(y1, 0.0f, 0.0f, 1.0f));
        write_imagef(dst1, (int2)(x,     y + 1), (float4)(y2, 0.0f, 0.0f, 1.0f));
        write_imagef(dst1, (int2)(x + 1, y + 1), (float4)(y3, 0.0f, 0.0f, 1.0f));
#ifdef NON_SEMI_PLANAR_OUT
        write_imagef(dst2, (int2)(xi, yi), (float4)(u, 0.0f, 0.0f, 1.0f));
        write_imagef(dst3, (int2)(xi, yi), (float4)(v, 0.0f, 0.0f, 1.0f));
#else
        write_imagef(dst2, (int2)(xi, yi), (float4)(u, v, 0.0f, 1.0f));
#endif
    }
}
#endif

#ifdef NEIGHBOR
__kernel void neighbor(__write_only image2d_t dst1,
                       __read_only  image2d_t src1,
#ifdef ENABLE_DITHER
                       __read_only  image2d_t dither,
#endif
                                    int2      src_size)
{
    int xi = get_global_id(0);
    int yi = get_global_id(1);

    int2 dst_pos = { xi, yi };
    float2 dst_size = { get_global_size(0), get_global_size(1) };

    float2 src_coord = (convert_float2(dst_pos) + 0.5f) * convert_float2(src_size) * native_recip(dst_size);
    int2 src_pos = convert_int2(floor(src_coord - 0.5f));

#ifdef ENABLE_DITHER
    float2 ncoords = convert_float2((int2)(xi, yi)) *
        native_recip((float2)(get_image_width(dither), get_image_height(dither)));
#endif

    int2 read_pos = clamp(src_pos, 0, src_size - 1);
    float y = read_imagef(src1, sampler2, read_pos).x;

#ifdef ENABLE_DITHER
    float d = read_imagef(dither, d_sampler, ncoords).x;
    y = get_dithered_y(y, d);
#endif

    write_imagef(dst1, dst_pos, (float4)(y, 0.0f, 0.0f, 1.0f));
}

__kernel void neighbor_uv(__write_only image2d_t dst2,
                          __read_only  image2d_t src2,
#ifdef NON_SEMI_PLANAR_OUT
                          __write_only image2d_t dst3,
#endif
#ifdef NON_SEMI_PLANAR_IN
                          __read_only  image2d_t src3,
#endif
                                       int2      src_size)
{
    int2 dst_pos = { get_global_id(0), get_global_id(1) };
    float2 dst_size = { get_global_size(0), get_global_size(1) };

    float2 src_coord = (convert_float2(dst_pos) + 0.5f) * convert_float2(src_size) * native_recip(dst_size);
    int2 src_pos = convert_int2(floor(src_coord - 0.5f));

    int2 read_pos = clamp(src_pos, 0, src_size - 1);
#ifdef NON_SEMI_PLANAR_IN
    float u = read_imagef(src2, sampler2, read_pos).x;
    float v = read_imagef(src3, sampler2, read_pos).x;
#else
    float2 uv = read_imagef(src2, sampler2, read_pos).xy;
    float u = uv.x;
    float v = uv.y;
#endif

#ifdef NON_SEMI_PLANAR_OUT
    write_imagef(dst2, dst_pos, (float4)(u, 0.0f, 0.0f, 1.0f));
    write_imagef(dst3, dst_pos, (float4)(v, 0.0f, 0.0f, 1.0f));
#else
    write_imagef(dst2, dst_pos, (float4)(u, v, 0.0f, 1.0f));
#endif
}
#endif

#ifdef SCALE
__kernel void scale(__write_only image2d_t dst1,
                    __read_only  image2d_t src1,
#ifdef ENABLE_DITHER
                    __read_only  image2d_t dither,
#endif
                    __constant   float    *cx,
                    __constant   float    *cy,
                                 int2      src_size)
{
    int xi = get_global_id(0);
    int yi = get_global_id(1);

    int2 dst_pos = { xi, yi };
    float2 dst_size = { get_global_size(0), get_global_size(1) };

    float2 src_coord = (convert_float2(dst_pos) + 0.5f) * convert_float2(src_size) * native_recip(dst_size);
    int2 src_pos = convert_int2(floor(src_coord - 0.5f));

#ifdef ENABLE_DITHER
    float2 ncoords = convert_float2((int2)(xi, yi)) *
        native_recip((float2)(get_image_width(dither), get_image_height(dither)));
#endif

    int i, j;
    int filterw2 = filterw >> 1;
    int filterh2 = filterh >> 1;
    int2 src_size_edge = src_size - 1;
    float4 col1 = 0.0f, s1 = 0.0f;

#pragma unroll
    for (i = 0; i < filterh; ++i, s1 = 0.0f) {
        #pragma unroll
        for (j = 0; j < filterw; ++j) {
            int2 read_pos = clamp(src_pos + (int2)(filterw2 - j, filterh2 - i), 0, src_size_edge);
            float4 c1 = read_imagef(src1, sampler2, read_pos);
            s1 += c1 * cx[dst_pos.x * filterw + j];
        }
        col1 += s1 * cy[dst_pos.y * filterh + i];
    }

    float y = col1.x;
#ifdef ENABLE_DITHER
    float d = read_imagef(dither, d_sampler, ncoords).x;
    y = get_dithered_y(y, d);
#endif

    write_imagef(dst1, dst_pos, (float4)(y, 0.0f, 0.0f, 1.0f));
}

__kernel void scale_uv(__write_only image2d_t dst2,
                       __read_only  image2d_t src2,
#ifdef NON_SEMI_PLANAR_OUT
                       __write_only image2d_t dst3,
#endif
#ifdef NON_SEMI_PLANAR_IN
                       __read_only  image2d_t src3,
#endif
                       __constant   float    *cx,
                       __constant   float    *cy,
                                    int2      src_size)
{
    int2 dst_pos = { get_global_id(0), get_global_id(1) };
    float2 dst_size = { get_global_size(0), get_global_size(1) };

    float2 src_coord = (convert_float2(dst_pos) + 0.5f) * convert_float2(src_size) * native_recip(dst_size);
    int2 src_pos = convert_int2(floor(src_coord - 0.5f));

    int i, j;
    int filterw2 = filterw >> 1;
    int filterh2 = filterh >> 1;
    int2 src_size_edge = src_size - 1;
    float4 col2 = 0.0f, col3 = 0.0f, s2 = 0.0f, s3 = 0.0f;

#pragma unroll
    for (i = 0; i < filterh; ++i, s2 = s3 = 0.0f) {
        #pragma unroll
        for (j = 0; j < filterw; ++j) {
            int2 read_pos = clamp(src_pos + (int2)(filterw2 - j, filterh2 - i), 0, src_size_edge);
            float4 c2 = read_imagef(src2, sampler2, read_pos);
            s2 += c2 * cx[dst_pos.x * filterw + j];
#ifdef NON_SEMI_PLANAR_IN
            float4 c3 = read_imagef(src3, sampler2, read_pos);
            s3 += c3 * cx[dst_pos.x * filterw + j];
#endif
        }
        col2 += s2 * cy[dst_pos.y * filterh + i];
#ifdef NON_SEMI_PLANAR_IN
        col3 += s3 * cy[dst_pos.y * filterh + i];
#endif
    }

#ifdef NON_SEMI_PLANAR_IN
    float u = col2.x;
    float v = col3.x;
#else
    float u = col2.x;
    float v = col2.y;
#endif

#ifdef NON_SEMI_PLANAR_OUT
    write_imagef(dst2, dst_pos, (float4)(u, 0.0f, 0.0f, 1.0f));
    write_imagef(dst3, dst_pos, (float4)(v, 0.0f, 0.0f, 1.0f));
#else
    write_imagef(dst2, dst_pos, (float4)(u, v, 0.0f, 1.0f));
#endif
}
#endif
