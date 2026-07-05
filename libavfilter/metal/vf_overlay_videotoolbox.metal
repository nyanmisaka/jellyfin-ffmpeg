/*
 * Copyright (C) 2024 Gnattu OC <gnattuoc@me.com>
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

#include <metal_stdlib>
#include <metal_integer>
#include <metal_texture>

using namespace metal;

struct mtlBlendParams {
    uint x_position;
    uint y_position;
};

/*
 * Blend shader for premultiplied alpha textures
 */
kernel void blend_shader(
                         texture2d<float, access::read> source [[ texture(0) ]],
                         texture2d<float, access::read> mask [[ texture(1) ]],
                         texture2d<float, access::write> dest [[ texture(2) ]],
                         constant mtlBlendParams& params [[ buffer(3) ]],
                         uint2 gid [[ thread_position_in_grid ]])
{
    const auto mask_size = uint2(mask.get_width(),
                                 mask.get_height());
    const auto loc_overlay = uint2(params.x_position, params.y_position);
    if (gid.x <  loc_overlay.x ||
        gid.y <  loc_overlay.y ||
        gid.x >= mask_size.x + loc_overlay.x ||
        gid.y >= mask_size.y + loc_overlay.y)
    {
        float4 source_color = source.read(gid);
        dest.write(source_color, gid);
    } else {
        float4 source_color = source.read(gid);
        float4 mask_color = mask.read((gid - loc_overlay));
        float4 result_color = source_color * (1.0f - mask_color.w) + (mask_color * mask_color.w);
        dest.write(result_color, gid);
    }
}

/*
 * Blend shader for sperated yuv main and bgra mask
 */
kernel void blend_shader_bgra_overlay(
                                      texture2d<float, access::read> source_y [[ texture(0) ]],
                                      texture2d<float, access::read> source_uv [[ texture(1) ]],
                                      texture2d<float, access::read> mask [[ texture(2) ]],
                                      texture2d<float, access::write> dest_y [[ texture(3) ]],
                                      texture2d<float, access::write> dest_uv [[ texture(4) ]],
                                      constant mtlBlendParams& params [[ buffer(5) ]],
                                      uint2 gid [[ thread_position_in_grid ]])
{
    const auto mask_size = uint2(mask.get_width(),
                                 mask.get_height());
    const auto loc_overlay = uint2(params.x_position, params.y_position);
    const auto loc_uv = gid >> 1;
    if (gid.x <  loc_overlay.x ||
        gid.y <  loc_overlay.y ||
        gid.x >= mask_size.x + loc_overlay.x ||
        gid.y >= mask_size.y + loc_overlay.y)
    {
        float4 source_color_y = source_y.read(gid);
        float4 source_color_uv = source_uv.read(loc_uv);
        dest_y.write(source_color_y, gid);
        dest_uv.write(source_color_uv, loc_uv);
    } else {
        float4 source_color_y = source_y.read(gid);
        float4 source_color_uv = source_uv.read(loc_uv);
        float4 mask_color = mask.read(gid - loc_overlay);
        float y_overlay = 0.183 * mask_color.r + 0.614 * mask_color.g + 0.062 * mask_color.b + 0.0625f;
        float u_overlay = -0.101 * mask_color.r - 0.339 * mask_color.g + 0.439 * mask_color.b + 0.5f;
        float v_overlay = 0.439 * mask_color.r - 0.399 * mask_color.g - 0.040 * mask_color.b + 0.5f;
        float alpha_color = mask_color.a;
        float3 main_color = float3(source_color_y.x, source_color_uv.x, source_color_uv.y);
        float3 overlay_color = float3(y_overlay, u_overlay, v_overlay);
        float3 result_color = main_color * (1.0f - alpha_color) + (overlay_color * alpha_color);
        dest_y.write(float4(result_color.x, 0.0f, 0.0f, 1.0f), gid);
        dest_uv.write(float4(result_color.y, result_color.z, 0.0f, 1.0f), loc_uv);
    }
}
