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

#ifndef AVFILTER_CUDA_PIXFMT_H
#define AVFILTER_CUDA_PIXFMT_H

#include "shared.h"

extern __constant__ const enum AVPixelFormat fmt_src, fmt_dst;
extern __constant__ const int depth_src, depth_dst;

// Single-sample read function
template<class T, int p>
static __inline__ __device__ T read_sample(const FFCUDAFrame& frame, int x, int y)
{
    T* ptr = (T*)(frame.data[p] + (y * frame.linesize[p]));
    return ptr[x];
}

// Per-format read functions
static __inline__ __device__ ushort3 read_p016(const FFCUDAFrame& frame, int x, int y)
{
    return make_ushort3(read_sample<unsigned short, 0>(frame, x,          y),
                        read_sample<unsigned short, 1>(frame, (x & ~1),     y / 2),
                        read_sample<unsigned short, 1>(frame, (x & ~1) + 1, y / 2));
}

static __inline__ __device__ ushort3 read_p010(const FFCUDAFrame& frame, int x, int y)
{
    ushort3 val = read_p016(frame, x, y);
    return make_ushort3(val.x >> 6,
                        val.y >> 6,
                        val.z >> 6);
}

static __inline__ __device__ ushort3 read_yuv420p16(const FFCUDAFrame& frame, int x, int y)
{
    return make_ushort3(read_sample<unsigned short, 0>(frame, x,      y),
                        read_sample<unsigned short, 1>(frame, x / 2, y / 2),
                        read_sample<unsigned short, 2>(frame, x / 2, y / 2));
}

static __inline__ __device__ ushort3 read_yuv420p10(const FFCUDAFrame& frame, int x, int y)
{
    ushort3 val = read_yuv420p16(frame, x, y);
    return make_ushort3(val.x >> 6,
                        val.y >> 6,
                        val.z >> 6);
}

// Generic read functions
static __inline__ __device__ ushort3 read_px(const FFCUDAFrame& frame, int x, int y)
{
    if (fmt_src == AV_PIX_FMT_P016)
        return read_p016(frame, x, y);
    else if (fmt_src == AV_PIX_FMT_P010)
        return read_p010(frame, x, y);
    else
        return make_ushort3(0, 0, 0);
}

static __inline__ __device__ float sample_to_float(unsigned short i)
{
    return (float)i / ((1 << depth_src) - 1);
}

static __inline__ __device__ float3 pixel_to_float3(ushort3 flt)
{
    return make_float3(sample_to_float(flt.x),
                       sample_to_float(flt.y),
                       sample_to_float(flt.z));
}

static __inline__ __device__ float3 read_px_flt(const FFCUDAFrame& frame, int x, int y)
{
    return pixel_to_float3(read_px(frame, x, y));
}

// Single-sample write function
template<int p, class T>
static __inline__ __device__ void write_sample(const FFCUDAFrame& frame, int x, int y, T sample)
{
    T* ptr = (T*)(frame.data[p] + (y * frame.linesize[p]));
    ptr[x] = sample;
}

// Per-format write functions
static __inline__ __device__ void write_nv12_2x2(const FFCUDAFrame& frame, int x, int y, ushort3 a, ushort3 b, ushort3 c, ushort3 d, ushort3 chroma)
{
    write_sample<0>(frame, x,     y,     (unsigned char)a.x);
    write_sample<0>(frame, x + 1, y,     (unsigned char)b.x);
    write_sample<0>(frame, x,     y + 1, (unsigned char)c.x);
    write_sample<0>(frame, x + 1, y + 1, (unsigned char)d.x);

    write_sample<1>(frame, (x & ~1),     y / 2, (unsigned char)chroma.y);
    write_sample<1>(frame, (x & ~1) + 1, y / 2, (unsigned char)chroma.z);
}

static __inline__ __device__ void write_yuv420p_2x2(const FFCUDAFrame& frame, int x, int y, ushort3 a, ushort3 b, ushort3 c, ushort3 d, ushort3 chroma)
{
    write_sample<0>(frame, x,     y,     (unsigned char)a.x);
    write_sample<0>(frame, x + 1, y,     (unsigned char)b.x);
    write_sample<0>(frame, x,     y + 1, (unsigned char)c.x);
    write_sample<0>(frame, x + 1, y + 1, (unsigned char)d.x);

    write_sample<1>(frame, x / 2, y / 2, (unsigned char)chroma.y);
    write_sample<2>(frame, x / 2, y / 2, (unsigned char)chroma.z);
}

static __inline__ __device__ void write_p016_2x2(const FFCUDAFrame& frame, int x, int y, ushort3 a, ushort3 b, ushort3 c, ushort3 d, ushort3 chroma)
{
    write_sample<0>(frame, x,     y,     (unsigned short)a.x);
    write_sample<0>(frame, x + 1, y,     (unsigned short)b.x);
    write_sample<0>(frame, x,     y + 1, (unsigned short)c.x);
    write_sample<0>(frame, x + 1, y + 1, (unsigned short)d.x);

    write_sample<1>(frame, (x & ~1),     y / 2, (unsigned short)chroma.y);
    write_sample<1>(frame, (x & ~1) + 1, y / 2, (unsigned short)chroma.z);
}

static __inline__ __device__ void write_p010_2x2(const FFCUDAFrame& frame, int x, int y, ushort3 a, ushort3 b, ushort3 c, ushort3 d, ushort3 chroma)
{
    write_sample<0>(frame, x,     y,     (unsigned short)(a.x << 6));
    write_sample<0>(frame, x + 1, y,     (unsigned short)(b.x << 6));
    write_sample<0>(frame, x,     y + 1, (unsigned short)(c.x << 6));
    write_sample<0>(frame, x + 1, y + 1, (unsigned short)(d.x << 6));

    write_sample<1>(frame, (x & ~1),     y / 2, (unsigned short)(chroma.y << 6));
    write_sample<1>(frame, (x & ~1) + 1, y / 2, (unsigned short)(chroma.z << 6));
}

static __inline__ __device__ void write_yuv420p16_2x2(const FFCUDAFrame& frame, int x, int y, ushort3 a, ushort3 b, ushort3 c, ushort3 d, ushort3 chroma)
{
    write_sample<0>(frame, x,     y,     (unsigned short)a.x);
    write_sample<0>(frame, x + 1, y,     (unsigned short)b.x);
    write_sample<0>(frame, x,     y + 1, (unsigned short)c.x);
    write_sample<0>(frame, x + 1, y + 1, (unsigned short)d.x);

    write_sample<1>(frame, x / 2, y / 2, (unsigned short)chroma.y);
    write_sample<2>(frame, x / 2, y / 2, (unsigned short)chroma.z);
}

static __inline__ __device__ void write_yuv420p10_2x2(const FFCUDAFrame& frame, int x, int y, ushort3 a, ushort3 b, ushort3 c, ushort3 d, ushort3 chroma)
{
    write_sample<0>(frame, x,     y,     (unsigned short)(a.x << 6));
    write_sample<0>(frame, x + 1, y,     (unsigned short)(b.x << 6));
    write_sample<0>(frame, x,     y + 1, (unsigned short)(c.x << 6));
    write_sample<0>(frame, x + 1, y + 1, (unsigned short)(d.x << 6));

    write_sample<1>(frame, x / 2, y / 2, (unsigned short)(chroma.y << 6));
    write_sample<2>(frame, x / 2, y / 2, (unsigned short)(chroma.z << 6));
}

// Generic write functions
static __inline__ __device__ void write_2x2(const FFCUDAFrame& frame, int x, int y, ushort3 a, ushort3 b, ushort3 c, ushort3 d, ushort3 chroma)
{
    if (fmt_dst == AV_PIX_FMT_YUV420P)
        write_yuv420p_2x2(frame, x, y, a, b, c, d, chroma);
    else if (fmt_dst == AV_PIX_FMT_NV12)
        write_nv12_2x2(frame, x, y, a, b, c, d, chroma);
    else if (fmt_dst == AV_PIX_FMT_P016)
        write_p016_2x2(frame, x, y, a, b, c, d, chroma);
    else if (fmt_dst == AV_PIX_FMT_P010)
        write_p010_2x2(frame, x, y, a, b, c, d, chroma);
}

static __inline__ __device__ unsigned short sample_to_ushort(float flt)
{
    return (unsigned short)(flt * ((1 << depth_dst) - 1));
}

static __inline__ __device__ ushort3 pixel_to_ushort3(float3 flt)
{
    return make_ushort3(sample_to_ushort(flt.x),
                        sample_to_ushort(flt.y),
                        sample_to_ushort(flt.z));
}

static __inline__ __device__ void write_2x2_flt(const FFCUDAFrame& frame, int x, int y, float3 a, float3 b, float3 c, float3 d)
{
    float3 chroma = get_chroma_sample(a, b, c, d);

    ushort3 ia = pixel_to_ushort3(a);
    ushort3 ib = pixel_to_ushort3(b);
    ushort3 ic = pixel_to_ushort3(c);
    ushort3 id = pixel_to_ushort3(d);

    ushort3 ichroma = pixel_to_ushort3(chroma);

    write_2x2(frame, x, y, ia, ib, ic, id, ichroma);
}

#endif /* AVFILTER_CUDA_PIXFMT_H */

