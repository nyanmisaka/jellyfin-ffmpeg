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
