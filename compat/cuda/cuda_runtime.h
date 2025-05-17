/*
 * Minimum CUDA compatibility definitions header
 *
 * Copyright (c) 2019 rcombs
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

#ifndef COMPAT_CUDA_CUDA_RUNTIME_H
#define COMPAT_CUDA_CUDA_RUNTIME_H

// Common macros
#define __constant__ __attribute__((constant))
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __device_builtin__ __attribute__((device_builtin))
#define __align__(N) __attribute__((aligned(N)))
#define __inline__ __inline__ __attribute__((always_inline))

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define abs(x) ((x) < 0 ? -(x) : (x))

#define atomicAdd(a, b) (__atomic_fetch_add(a, b, __ATOMIC_SEQ_CST))

// Basic typedefs
typedef __device_builtin__ unsigned long long cudaTextureObject_t;

#define MAKE_VECTORS(type, base) \
typedef struct __device_builtin__ type##1 { \
    base x; \
} type##1; \
static __inline__ __device__ type##1 make_##type##1(base x) { \
    type##1 ret; \
    ret.x = x; \
    return ret; \
} \
typedef struct __device_builtin__ __align__(sizeof(base) * 2) type##2 { \
    base x, y; \
} type##2; \
static __inline__ __device__ type##2 make_##type##2(base x, base y) { \
    type##2 ret; \
    ret.x = x; \
    ret.y = y; \
    return ret; \
} \
typedef struct __device_builtin__ type##3 { \
    base x, y, z; \
} type##3; \
static __inline__ __device__ type##3 make_##type##3(base x, base y, base z) { \
    type##3 ret; \
    ret.x = x; \
    ret.y = y; \
    ret.z = z; \
    return ret; \
} \
typedef struct __device_builtin__ __align__(sizeof(base) * 4) type##4 { \
    base x, y, z, w; \
} type##4; \
static __inline__ __device__ type##4 make_##type##4(base x, base y, base z, base w) { \
    type##4 ret; \
    ret.x = x; \
    ret.y = y; \
    ret.z = z; \
    ret.w = w; \
    return ret; \
}

// Basic initializers
MAKE_VECTORS(uchar, unsigned char)
MAKE_VECTORS(ushort, unsigned short)
MAKE_VECTORS(int, int)
MAKE_VECTORS(uint, unsigned int)
MAKE_VECTORS(float, float)

// Accessors for special registers
#define GETCOMP(reg, comp) \
    asm("mov.u32 %0, %%" #reg "." #comp ";" : "=r"(tmp)); \
    ret.comp = tmp;

#define GET(name, reg) static __inline__ __device__ uint3 name() {\
    uint3 ret; \
    unsigned tmp; \
    GETCOMP(reg, x) \
    GETCOMP(reg, y) \
    GETCOMP(reg, z) \
    return ret; \
}

GET(getBlockIdx, ctaid)
GET(getBlockDim, ntid)
GET(getThreadIdx, tid)

// Instead of externs for these registers, we turn access to them into calls into trivial ASM
#define blockIdx (getBlockIdx())
#define blockDim (getBlockDim())
#define threadIdx (getThreadIdx())

// Conversions from the tex instruction's 4-register output to various types
#define TEX2D(type, ret) static __inline__ __device__ void conv(type* out, unsigned a, unsigned b, unsigned c, unsigned d) {*out = (ret);}

TEX2D(unsigned char, a & 0xFF)
TEX2D(unsigned short, a & 0xFFFF)
TEX2D(float, a)
TEX2D(uchar2, make_uchar2(a & 0xFF, b & 0xFF))
TEX2D(ushort2, make_ushort2(a & 0xFFFF, b & 0xFFFF))
TEX2D(float2, make_float2(a, b))
TEX2D(uchar4, make_uchar4(a & 0xFF, b & 0xFF, c & 0xFF, d & 0xFF))
TEX2D(ushort4, make_ushort4(a & 0xFFFF, b & 0xFFFF, c & 0xFFFF, d & 0xFFFF))
TEX2D(float4, make_float4(a, b, c, d))

// Template calling tex instruction and converting the output to the selected type
template<typename T>
__inline__ __device__ T tex2D(cudaTextureObject_t texObject, float x, float y)
{
    T ret;
    unsigned ret1, ret2, ret3, ret4;
    asm("tex.2d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];" :
        "=r"(ret1), "=r"(ret2), "=r"(ret3), "=r"(ret4) :
        "l"(texObject), "f"(x), "f"(y));
    conv(&ret, ret1, ret2, ret3, ret4);
    return ret;
}

template<>
__inline__ __device__ float4 tex2D<float4>(cudaTextureObject_t texObject, float x, float y)
{
    float4 ret;
    asm("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];" :
        "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) :
        "l"(texObject), "f"(x), "f"(y));
    return ret;
}

template<>
__inline__ __device__ float tex2D<float>(cudaTextureObject_t texObject, float x, float y)
{
    return tex2D<float4>(texObject, x, y).x;
}

template<>
__inline__ __device__ float2 tex2D<float2>(cudaTextureObject_t texObject, float x, float y)
{
    float4 ret = tex2D<float4>(texObject, x, y);
    return make_float2(ret.x, ret.y);
}

// Math helper functions
static __inline__ __device__ float floorf(float a) { return __builtin_floorf(a); }
static __inline__ __device__ float floor(float a) { return __builtin_floorf(a); }
static __inline__ __device__ double floor(double a) { return __builtin_floor(a); }
static __inline__ __device__ float ceilf(float a) { return __builtin_ceilf(a); }
static __inline__ __device__ float ceil(float a) { return __builtin_ceilf(a); }
static __inline__ __device__ double ceil(double a) { return __builtin_ceil(a); }
static __inline__ __device__ float truncf(float a) { return __builtin_truncf(a); }
static __inline__ __device__ float trunc(float a) { return __builtin_truncf(a); }
static __inline__ __device__ double trunc(double a) { return __builtin_trunc(a); }
static __inline__ __device__ float fabsf(float a) { return __builtin_fabsf(a); }
static __inline__ __device__ float fabs(float a) { return __builtin_fabsf(a); }
static __inline__ __device__ double fabs(double a) { return __builtin_fabs(a); }
static __inline__ __device__ float sqrtf(float a) { return __builtin_sqrtf(a); }

static __inline__ __device__ float __saturatef(float a) { return __nvvm_saturate_f(a); }
static __inline__ __device__ float __sinf(float a) { return __nvvm_sin_approx_f(a); }
static __inline__ __device__ float __cosf(float a) { return __nvvm_cos_approx_f(a); }
static __inline__ __device__ float __exp2f(float a) { return __nvvm_ex2_approx_f(a); }
static __inline__ __device__ float __expf(float a) { return __nvvm_ex2_approx_f(a * 1.4427f); }
static __inline__ __device__ float __log2f(float a) { return __nvvm_lg2_approx_f(a); }
static __inline__ __device__ float __logf(float a) { return __nvvm_lg2_approx_f(a) * 0.693147f; }
static __inline__ __device__ float __log10f(float a) { return __nvvm_lg2_approx_f(a) * 0.30103f; }
static __inline__ __device__ float __powf(float a, float b) { return __nvvm_ex2_approx_f(__nvvm_lg2_approx_f(a) * b); }

#endif /* COMPAT_CUDA_CUDA_RUNTIME_H */
