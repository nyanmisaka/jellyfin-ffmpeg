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

#ifndef AVFILTER_CUDA_UTIL_H
#define AVFILTER_CUDA_UTIL_H

static inline __device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static inline __device__ float3 operator+(const float3 &a, float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

static inline __device__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static inline __device__ float3 operator-(const float3 &a, float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}

static inline __device__ float3 operator*(const float3 &a, const float3 &b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static inline __device__ float3 operator*(const float3 &a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

static inline __device__ float3 operator/(const float3 &a, const float3 &b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

static inline __device__ float3 operator/(const float3 &a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

#endif /* AVFILTER_CUDA_UTIL_H */

