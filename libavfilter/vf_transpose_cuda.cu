/*
 * Copyright (C) 2024 NyanMisaka
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

template<typename T>
__inline__ __device__ void transpose_func(
    T* dst, int dst_width, int dst_height, int dst_pitch,
    T* src, int src_width, int src_height, int src_pitch,
    int pix_step, int pix_offset, int dir)
{
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;
    if (xo >= dst_width || yo >= dst_height)
        return;

    int xi = (dir < 4) ? ((dir &  2) ? (dst_height - 1 - yo) : yo)
                       : ((dir == 6) ? xo : (dst_width  - 1 - xo));
    int yi = (dir < 4) ? ((dir &  1) ? (dst_width  - 1 - xo) : xo)
                       : ((dir == 5) ? yo : (dst_height - 1 - yo));
    if (xi >= src_width || yi >= src_height)
        return;

    int dst_pos = xo*pix_step + yo*dst_pitch + pix_offset;
    int src_pos = xi*pix_step + yi*src_pitch + pix_offset;
    dst[dst_pos] = src[src_pos];
}

extern "C" {

#define TRANSPOSE_VARIANT(NAME, TYPE) \
__global__ void Transpose_Cuda_ ## NAME( \
    TYPE* dst, int dst_width, int dst_height, int dst_pitch, \
    TYPE* src, int src_width, int src_height, int src_pitch, \
    int pix_step, int pix_offset, int dir) \
{ \
    transpose_func( \
        dst, dst_width, dst_height, dst_pitch, \
        src, src_width, src_height, src_pitch, \
        pix_step, pix_offset, dir); \
}

TRANSPOSE_VARIANT(uchar, unsigned char)
TRANSPOSE_VARIANT(ushort, unsigned short)

} /* extern "C" */
