/*
 * Copyright (C) 2019 Philip Langdale <philipl@overt.org>
 * Copyright (C) 2025 NyanMisaka
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

#define max3(a, b, c) max(max((a), (b)), (c))
#define min3(a, b, c) min(min((a), (b)), (c))

__constant sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                                CLK_ADDRESS_CLAMP_TO_EDGE   |
                                CLK_FILTER_NEAREST);

__constant float coef_lf[2] = { 4309.0f, 213.0f };
__constant float coef_hf[3] = { 5570.0f, 3801.0f, 1016.0f };
__constant float coef_sp[2] = { 5077.0f, 981.0f };

#define FILTER_INTRA(T) \
T filter_intra_##T(T cur_prefs3, T cur_prefs, \
                   T cur_mrefs, T cur_mrefs3) \
{ \
    T final = native_divide((coef_sp[0] * (cur_mrefs + cur_prefs) - \
                             coef_sp[1] * (cur_mrefs3 + cur_prefs3)), (float)(1 << 13)); \
    return clamp(final, 0.0f, 1.0f); \
}

FILTER_INTRA(float)
FILTER_INTRA(float2)

float filter_temp_float(float cur_prefs3, float cur_prefs, float cur_mrefs, float cur_mrefs3,
                        float prev2_prefs4, float prev2_prefs2, float prev2_0, float prev2_mrefs2, float prev2_mrefs4,
                        float prev_prefs, float prev_mrefs, float next_prefs, float next_mrefs,
                        float next2_prefs4, float next2_prefs2, float next2_0, float next2_mrefs2, float next2_mrefs4)
{
    float final;
    float c = cur_mrefs;
    float d = (prev2_0 + next2_0) * 0.5f;
    float e = cur_prefs;
    float temporal_diff0 = fabs(prev2_0 - next2_0);
    float temporal_diff1 = (fabs(prev_mrefs - c) + fabs(prev_prefs - e)) * 0.5f;
    float temporal_diff2 = (fabs(next_mrefs - c) + fabs(next_prefs - e)) * 0.5f;
    float diff = max3(temporal_diff0 * 0.5f, temporal_diff1, temporal_diff2);

    if (!diff) {
        final = d;
    } else {
        float b = ((prev2_mrefs2 + next2_mrefs2) * 0.5f) - c;
        float f = ((prev2_prefs2 + next2_prefs2) * 0.5f) - e;
        float dc = d - c;
        float de = d - e;
        float mmax = max3(de, dc, min(b, f));
        float mmin = min3(de, dc, max(b, f));
        diff = max3(diff, mmin, -mmax);

        float interpol;
        if (fabs(c - e) > temporal_diff0) {
            interpol = native_divide((((coef_hf[0] * (prev2_0 + next2_0)
                - coef_hf[1] * (prev2_mrefs2 + next2_mrefs2 + prev2_prefs2 + next2_prefs2)
                + coef_hf[2] * (prev2_mrefs4 + next2_mrefs4 + prev2_prefs4 + next2_prefs4)) * 0.25f)
                + coef_lf[0] * (c + e) - coef_lf[1] * (cur_mrefs3 + cur_prefs3)), (float)(1 << 13));
        } else {
            interpol = native_divide((coef_sp[0] * (c + e) - coef_sp[1] * (cur_mrefs3 + cur_prefs3)), (float)(1 << 13));
        }

        if (interpol > d + diff) {
            interpol = d + diff;
        } else if (interpol < d - diff) {
            interpol = d - diff;
        }
        final = clamp(interpol, 0.0f, 1.0f);
    }
    return final;
}

float2 filter_temp_float2(float2 cur_prefs3, float2 cur_prefs, float2 cur_mrefs, float2 cur_mrefs3,
                          float2 prev2_prefs4, float2 prev2_prefs2, float2 prev2_0, float2 prev2_mrefs2, float2 prev2_mrefs4,
                          float2 prev_prefs, float2 prev_mrefs, float2 next_prefs, float2 next_mrefs,
                          float2 next2_prefs4, float2 next2_prefs2, float2 next2_0, float2 next2_mrefs2, float2 next2_mrefs4)
{
    return (float2)(filter_temp_float(cur_prefs3.x, cur_prefs.x, cur_mrefs.x, cur_mrefs3.x,
                                      prev2_prefs4.x, prev2_prefs2.x, prev2_0.x, prev2_mrefs2.x, prev2_mrefs4.x,
                                      prev_prefs.x, prev_mrefs.x, next_prefs.x, next_mrefs.x,
                                      next2_prefs4.x, next2_prefs2.x, next2_0.x, next2_mrefs2.x, next2_mrefs4.x),
                    filter_temp_float(cur_prefs3.y, cur_prefs.y, cur_mrefs.y, cur_mrefs3.y,
                                      prev2_prefs4.y, prev2_prefs2.y, prev2_0.y, prev2_mrefs2.y, prev2_mrefs4.y,
                                      prev_prefs.y, prev_mrefs.y, next_prefs.y, next_mrefs.y,
                                      next2_prefs4.y, next2_prefs2.y, next2_0.y, next2_mrefs2.y, next2_mrefs4.y));
}

#define BWDIF_COMPUTE(T, XY) \
T bwdif_compute_##T(__write_only image2d_t dst, \
                    __read_only  image2d_t cur, \
                    __read_only  image2d_t prev2, \
                    __read_only  image2d_t prev1, \
                    __read_only  image2d_t next1, \
                    __read_only  image2d_t next2, \
                    int parity, \
                    bool is_field_end, \
                    int2 pos) \
{ \
    /* Don't modify the primary field */ \
    if (pos.y % 2 == parity) { \
        return read_imagef(cur, sampler, (int2)(pos.x, pos.y)).XY; \
    } \
    T cur_prefs3 = read_imagef(cur, sampler, (int2)(pos.x, pos.y + 3)).XY; \
    T cur_prefs  = read_imagef(cur, sampler, (int2)(pos.x, pos.y + 1)).XY; \
    T cur_mrefs  = read_imagef(cur, sampler, (int2)(pos.x, pos.y - 1)).XY; \
    T cur_mrefs3 = read_imagef(cur, sampler, (int2)(pos.x, pos.y - 3)).XY; \
    if (is_field_end) { \
        return filter_intra_##T(cur_prefs3, cur_prefs, cur_mrefs, cur_mrefs3); \
    } \
    /* Calculate temporal prediction */ \
    T prev2_prefs4 = read_imagef(prev2, sampler, (int2)(pos.x, pos.y + 4)).XY; \
    T prev2_prefs2 = read_imagef(prev2, sampler, (int2)(pos.x, pos.y + 2)).XY; \
    T prev2_0      = read_imagef(prev2, sampler, (int2)(pos.x, pos.y + 0)).XY; \
    T prev2_mrefs2 = read_imagef(prev2, sampler, (int2)(pos.x, pos.y - 2)).XY; \
    T prev2_mrefs4 = read_imagef(prev2, sampler, (int2)(pos.x, pos.y - 4)).XY; \
    T prev_prefs   = read_imagef(prev1, sampler, (int2)(pos.x, pos.y + 1)).XY; \
    T prev_mrefs   = read_imagef(prev1, sampler, (int2)(pos.x, pos.y - 1)).XY; \
    T next_prefs   = read_imagef(next1, sampler, (int2)(pos.x, pos.y + 1)).XY; \
    T next_mrefs   = read_imagef(next1, sampler, (int2)(pos.x, pos.y - 1)).XY; \
    T next2_prefs4 = read_imagef(next2, sampler, (int2)(pos.x, pos.y + 4)).XY; \
    T next2_prefs2 = read_imagef(next2, sampler, (int2)(pos.x, pos.y + 2)).XY; \
    T next2_0      = read_imagef(next2, sampler, (int2)(pos.x, pos.y + 0)).XY; \
    T next2_mrefs2 = read_imagef(next2, sampler, (int2)(pos.x, pos.y - 2)).XY; \
    T next2_mrefs4 = read_imagef(next2, sampler, (int2)(pos.x, pos.y - 4)).XY; \
    return filter_temp_##T(cur_prefs3, cur_prefs, cur_mrefs, cur_mrefs3, \
                           prev2_prefs4, prev2_prefs2, prev2_0, prev2_mrefs2, prev2_mrefs4, \
                           prev_prefs, prev_mrefs, next_prefs, next_mrefs, \
                           next2_prefs4, next2_prefs2, next2_0, next2_mrefs2, next2_mrefs4); \
}

BWDIF_COMPUTE(float, x)
BWDIF_COMPUTE(float2, xy)

__kernel void bwdif(__write_only image2d_t dst,
                    __read_only  image2d_t prev,
                    __read_only  image2d_t cur,
                    __read_only  image2d_t next,
                    int channels,
                    int parity,
                    int is_field_end,
                    int is_second_field)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    if (pos.x >= get_image_width(dst) ||
        pos.y >= get_image_height(dst))
        return;

    if (channels == 1) {
        float pred = is_second_field
            ? bwdif_compute_float(dst, cur, cur, prev, next, next,
                                  parity, is_field_end, pos)
            : bwdif_compute_float(dst, cur, prev, prev, next, cur,
                                  parity, is_field_end, pos);

        write_imagef(dst, pos, (float4)(pred, 0.0f, 0.0f, 1.0f));
    } else if (channels == 2) {
        float2 pred = is_second_field
            ? bwdif_compute_float2(dst, cur, cur, prev, next, next,
                                   parity, is_field_end, pos)
            : bwdif_compute_float2(dst, cur, prev, prev, next, cur,
                                   parity, is_field_end, pos);

        write_imagef(dst, pos, (float4)(pred.x, pred.y, 0.0f, 1.0f));
    }
}
