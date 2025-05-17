/*
 * Copyright (C) 2018 Philip Langdale <philipl@overt.org>
 *               2025 NyanMisaka
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

inline float spatial_predictor_float(float a, float b, float c, float d, float e, float f, float g,
                                     float h, float i, float j, float k, float l, float m, float n)
{
    float spatial_pred = (d + k) * 0.5f;
    float spatial_score = fabs(c - j) + fabs(d - k) + fabs(e - l);

    float score = fabs(b - k) + fabs(c - l) + fabs(d - m);
    if (score < spatial_score) {
        spatial_pred = (c + l) * 0.5f;
        spatial_score = score;
        score = fabs(a - l) + fabs(b - m) + fabs(c - n);
        if (score < spatial_score) {
            spatial_pred = (b + m) * 0.5f;
            spatial_score = score;
        }
    }
    score = fabs(d - i) + fabs(e - j) + fabs(f - k);
    if (score < spatial_score) {
        spatial_pred = (e + j) * 0.5f;
        spatial_score = score;
        score = fabs(e - h) + fabs(f - i) + fabs(g - j);
        if (score < spatial_score) {
            spatial_pred = (f + i) * 0.5f;
            spatial_score = score;
        }
    }
    return spatial_pred;
}

inline float2 spatial_predictor_float2(float2 a, float2 b, float2 c, float2 d, float2 e, float2 f, float2 g,
                                       float2 h, float2 i, float2 j, float2 k, float2 l, float2 m, float2 n)
{
    return (float2)(spatial_predictor_float(a.x, b.x, c.x, d.x, e.x, f.x, g.x,
                                            h.x, i.x, j.x, k.x, l.x, m.x, n.x),
                    spatial_predictor_float(a.y, b.y, c.y, d.y, e.y, f.y, g.y,
                                            h.y, i.y, j.y, k.y, l.y, m.y, n.y));
}


inline float temporal_predictor_float(float A, float B, float C, float D, float E, float F,
                                      float G, float H, float I, float J, float K, float L,
                                      float spatial_pred, bool skip_check)
{
    float p0 = (C + H) * 0.5f;
    float p1 = F;
    float p2 = (D + I) * 0.5f;
    float p3 = G;
    float p4 = (E + J) * 0.5f;

    float tdiff0 = fabs(D - I);
    float tdiff1 = (fabs(A - F) + fabs(B - G)) * 0.5f;
    float tdiff2 = (fabs(K - F) + fabs(G - L)) * 0.5f;

    float diff = max3(tdiff0, tdiff1, tdiff2);

    if (!skip_check) {
        float maxi = max3(p2 - p3, p2 - p1, min(p0 - p1, p4 - p3));
        float mini = min3(p2 - p3, p2 - p1, max(p0 - p1, p4 - p3));
        diff = max3(diff, mini, -maxi);
    }
    return clamp(spatial_pred, p2 - diff, p2 + diff);
}

inline float2 temporal_predictor_float2(float2 A, float2 B, float2 C, float2 D, float2 E, float2 F,
                                        float2 G, float2 H, float2 I, float2 J, float2 K, float2 L,
                                        float2 spatial_pred, bool skip_check)
{
    return (float2)(temporal_predictor_float(A.x, B.x, C.x, D.x, E.x, F.x,
                                             G.x, H.x, I.x, J.x, K.x, L.x,
                                             spatial_pred.x, skip_check),
                    temporal_predictor_float(A.y, B.y, C.y, D.y, E.y, F.y,
                                             G.y, H.y, I.y, J.y, K.y, L.y,
                                             spatial_pred.y, skip_check));
}

#define YADIF_COMPUTE_SPATIAL(T, XY) \
inline T yadif_compute_spatial_##T(__read_only image2d_t cur, int2 pos) \
{ \
    T a = read_imagef(cur, sampler, (int2)(pos.x - 3, pos.y - 1)).XY; \
    T b = read_imagef(cur, sampler, (int2)(pos.x - 2, pos.y - 1)).XY; \
    T c = read_imagef(cur, sampler, (int2)(pos.x - 1, pos.y - 1)).XY; \
    T d = read_imagef(cur, sampler, (int2)(pos.x - 0, pos.y - 1)).XY; \
    T e = read_imagef(cur, sampler, (int2)(pos.x + 1, pos.y - 1)).XY; \
    T f = read_imagef(cur, sampler, (int2)(pos.x + 2, pos.y - 1)).XY; \
    T g = read_imagef(cur, sampler, (int2)(pos.x + 3, pos.y - 1)).XY; \
    T h = read_imagef(cur, sampler, (int2)(pos.x - 3, pos.y + 1)).XY; \
    T i = read_imagef(cur, sampler, (int2)(pos.x - 2, pos.y + 1)).XY; \
    T j = read_imagef(cur, sampler, (int2)(pos.x - 1, pos.y + 1)).XY; \
    T k = read_imagef(cur, sampler, (int2)(pos.x - 0, pos.y + 1)).XY; \
    T l = read_imagef(cur, sampler, (int2)(pos.x + 1, pos.y + 1)).XY; \
    T m = read_imagef(cur, sampler, (int2)(pos.x + 2, pos.y + 1)).XY; \
    T n = read_imagef(cur, sampler, (int2)(pos.x + 3, pos.y + 1)).XY; \
    return spatial_predictor_##T(a, b, c, d, e, f, g, \
                                 h, i, j, k, l, m, n); \
}

#define YADIF_COMPUTE_TEMPORAL(T, XY) \
inline T yadif_compute_temporal_##T(__read_only image2d_t cur, \
                                    __read_only image2d_t prev2, \
                                    __read_only image2d_t prev1, \
                                    __read_only image2d_t next1, \
                                    __read_only image2d_t next2, \
                                    T spatial_pred, \
                                    bool skip_spatial_check, \
                                    int2 pos) \
{ \
    T A = read_imagef(prev2, sampler, (int2)(pos.x, pos.y - 1)).XY; \
    T B = read_imagef(prev2, sampler, (int2)(pos.x, pos.y + 1)).XY; \
    T C = read_imagef(prev1, sampler, (int2)(pos.x, pos.y - 2)).XY; \
    T D = read_imagef(prev1, sampler, (int2)(pos.x, pos.y + 0)).XY; \
    T E = read_imagef(prev1, sampler, (int2)(pos.x, pos.y + 2)).XY; \
    T F = read_imagef(cur,   sampler, (int2)(pos.x, pos.y - 1)).XY; \
    T G = read_imagef(cur,   sampler, (int2)(pos.x, pos.y + 1)).XY; \
    T H = read_imagef(next1, sampler, (int2)(pos.x, pos.y - 2)).XY; \
    T I = read_imagef(next1, sampler, (int2)(pos.x, pos.y + 0)).XY; \
    T J = read_imagef(next1, sampler, (int2)(pos.x, pos.y + 2)).XY; \
    T K = read_imagef(next2, sampler, (int2)(pos.x, pos.y - 1)).XY; \
    T L = read_imagef(next2, sampler, (int2)(pos.x, pos.y + 1)).XY; \
    return temporal_predictor_##T(A, B, C, D, E, F, G, H, I, J, K, L, \
                                  spatial_pred, skip_spatial_check); \
}

YADIF_COMPUTE_SPATIAL(float, x)
YADIF_COMPUTE_TEMPORAL(float, x)
YADIF_COMPUTE_SPATIAL(float2, xy)
YADIF_COMPUTE_TEMPORAL(float2, xy)

__kernel void yadif(__write_only image2d_t dst,
                    __read_only  image2d_t prev,
                    __read_only  image2d_t cur,
                    __read_only  image2d_t next,
                    int channels,
                    int parity,
                    int is_second_field,
                    int skip_spatial_check)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    if (pos.x >= get_image_width(dst) ||
        pos.y >= get_image_height(dst))
        return;

    // Don't modify the primary field
    if (pos.y % 2 == parity) {
        float4 in = read_imagef(cur, sampler, pos);
        write_imagef(dst, pos, in);
        return;
    }

    if (channels == 1) {
        float spatial_pred = yadif_compute_spatial_float(cur, pos);
        float pred = is_second_field
            ? yadif_compute_temporal_float(cur, prev, cur, next, next,
                                           spatial_pred, skip_spatial_check, pos)
            : yadif_compute_temporal_float(cur, prev, prev, cur, next,
                                           spatial_pred, skip_spatial_check, pos);

        write_imagef(dst, pos, (float4)(pred, 0.0f, 0.0f, 1.0f));
    } else if (channels == 2) {
        float2 spatial_pred = yadif_compute_spatial_float2(cur, pos);
        float2 pred = is_second_field
            ? yadif_compute_temporal_float2(cur, prev, cur, next, next,
                                            spatial_pred, skip_spatial_check, pos)
            : yadif_compute_temporal_float2(cur, prev, prev, cur, next,
                                            spatial_pred, skip_spatial_check, pos);

        write_imagef(dst, pos, (float4)(pred.x, pred.y, 0.0f, 1.0f));
    }
}
