/*
 * Copyright (c) 2024 Gnattu OC <gnattuoc@me.com>
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

#ifndef AVFILTER_X86_TONEMAPX_INTRIN_SSE_H
#define AVFILTER_X86_TONEMAPX_INTRIN_SSE_H

#include "libavfilter/vf_tonemapx.h"

X86_64_V2 void tonemap_frame_dovi_2_420p_sse(uint8_t *dsty, uint8_t *dstu, uint8_t *dstv,
                                             const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                             const int *dstlinesize, const int *srclinesize,
                                             int dstdepth, int srcdepth,
                                             int width, int height,
                                             const struct TonemapIntParams *params);

X86_64_V2 void tonemap_frame_dovi_2_420p10_sse(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                               const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                               const int *dstlinesize, const int *srclinesize,
                                               int dstdepth, int srcdepth,
                                               int width, int height,
                                               const struct TonemapIntParams *params);

X86_64_V2 void tonemap_frame_dovi_2_420hdr_sse(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                               const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                               const int *dstlinesize, const int *srclinesize,
                                               int dstdepth, int srcdepth,
                                               int width, int height,
                                               const struct TonemapIntParams *params);

X86_64_V2 void tonemap_frame_420p10_2_420p_sse(uint8_t *dsty, uint8_t *dstu, uint8_t *dstv,
                                               const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                               const int *dstlinesize, const int *srclinesize,
                                               int dstdepth, int srcdepth,
                                               int width, int height,
                                               const struct TonemapIntParams *params);

X86_64_V2 void tonemap_frame_420p10_2_420p10_sse(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                                 const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                                 const int *dstlinesize, const int *srclinesize,
                                                 int dstdepth, int srcdepth,
                                                 int width, int height,
                                                 const struct TonemapIntParams *params);

X86_64_V2 void tonemap_frame_p010_2_nv12_sse(uint8_t *dsty, uint8_t *dstuv,
                                             const uint16_t *srcy, const uint16_t *srcuv,
                                             const int *dstlinesize, const int *srclinesize,
                                             int dstdepth, int srcdepth,
                                             int width, int height,
                                             const struct TonemapIntParams *params);

X86_64_V2 void tonemap_frame_p010_2_p010_sse(uint16_t *dsty, uint16_t *dstuv,
                                             const uint16_t *srcy, const uint16_t *srcuv,
                                             const int *dstlinesize, const int *srclinesize,
                                             int dstdepth, int srcdepth,
                                             int width, int height,
                                             const struct TonemapIntParams *params);

#endif // AVFILTER_X86_TONEMAPX_INTRIN_SSE_H
