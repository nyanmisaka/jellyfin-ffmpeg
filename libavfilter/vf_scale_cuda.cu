/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

typedef unsigned char uchar;
typedef unsigned short ushort;

#define SHIFTDOWN(val) (dstbase)(val >> abs(2 + shift))
#define SHIFTUP(val)   (dstbase)(val << abs(-shift - 2))

template<class SRC, class DST, int shift, int dither> struct add_conv_shift1_d
{
    typedef DST dstbase;

    __inline__ __device__ DST operator()(SRC i1, SRC i2, SRC i3, SRC i4, ushort d)
    {
        unsigned ret = (unsigned)i1 + (unsigned)i2 + (unsigned)i3 + (unsigned)i4 + ((1 + d) >> (sizeof(SRC) * 8 - dither + 3));

        if (shift > -2)
            return SHIFTDOWN(ret);
        else
            return SHIFTUP(ret);
    }
};

template<class SRC, class DST, int shift, int dither> struct add_conv_shift1
{
    typedef DST dstbase;

    __inline__ __device__ DST operator()(SRC i1, SRC i2, SRC i3, SRC i4, ushort d)
    {
        unsigned ret = (unsigned)i1 + (unsigned)i2 + (unsigned)i3 + (unsigned)i4 + 2;

        if (shift > -2)
            return SHIFTDOWN(ret);
        else
            return SHIFTUP(ret);
    }
};

template<class SRC, class DST, int shift, int dither> struct add_conv_shift2
{
    typedef decltype(DST::x) dstbase;

    __inline__ __device__ DST operator()(SRC i1, SRC i2, SRC i3, SRC i4, ushort d)
    {
        unsigned retx = (unsigned)i1.x + (unsigned)i2.x + (unsigned)i3.x + (unsigned)i4.x + 2;
        unsigned rety = (unsigned)i1.y + (unsigned)i2.y + (unsigned)i3.y + (unsigned)i4.y + 2;

        if (shift > -2)
            return { SHIFTDOWN(retx), SHIFTDOWN(rety) };
        else
            return { SHIFTUP(retx),   SHIFTUP(rety)   };
    }
};

template<class SRC, class DST, int shift, int dither> struct add_conv_shift2_x
{
    __inline__ __device__ DST operator()(SRC i1, SRC i2, SRC i3, SRC i4, ushort d)
    {
        return add_conv_shift1<unsigned, DST, shift, dither>()(i1.x, i2.x, i3.x, i4.x, d);
    }
};

template<class SRC, class DST, int shift, int dither> struct add_conv_shift2_y
{
    __inline__ __device__ DST operator()(SRC i1, SRC i2, SRC i3, SRC i4, ushort d)
    {
        return add_conv_shift1<unsigned, DST, shift, dither>()(i1.y, i2.y, i3.y, i4.y, d);
    }
};

template<class SRC, class DST, int shift, int dither> struct add_conv_shift3
{
    typedef decltype(DST::x) dstbase;

    __inline__ __device__ DST operator()(SRC i1, SRC i2, SRC i3, SRC i4, ushort d)
    {
        unsigned retx = (unsigned)i1.x + (unsigned)i2.x + (unsigned)i3.x + (unsigned)i4.x + 2;
        unsigned rety = (unsigned)i1.y + (unsigned)i2.y + (unsigned)i3.y + (unsigned)i4.y + 2;
        unsigned retz = (unsigned)i1.z + (unsigned)i2.z + (unsigned)i3.z + (unsigned)i4.z + 2;

        if (shift > -2)
            return { SHIFTDOWN(retx), SHIFTDOWN(rety), SHIFTDOWN(retz) };
        else
            return { SHIFTUP(retx),   SHIFTUP(rety),   SHIFTUP(retz)   };
    }
};

template<class SRC, class DST, int shift, int dither> struct add_conv_shift4
{
    typedef decltype(DST::x) dstbase;

    __inline__ __device__ DST operator()(SRC i1, SRC i2, SRC i3, SRC i4, ushort d)
    {
        unsigned retx = (unsigned)i1.x + (unsigned)i2.x + (unsigned)i3.x + (unsigned)i4.x + 2;
        unsigned rety = (unsigned)i1.y + (unsigned)i2.y + (unsigned)i3.y + (unsigned)i4.y + 2;
        unsigned retz = (unsigned)i1.z + (unsigned)i2.z + (unsigned)i3.z + (unsigned)i4.z + 2;
        unsigned retw = (unsigned)i1.w + (unsigned)i2.w + (unsigned)i3.w + (unsigned)i4.w + 2;

        if (shift > -2)
            return { SHIFTDOWN(retx), SHIFTDOWN(rety), SHIFTDOWN(retz), SHIFTDOWN(retw) };
        else
            return { SHIFTUP(retx),   SHIFTUP(rety),   SHIFTUP(retz),   SHIFTUP(retw)   };
    }
};

template<class SRC, class DST, template<class, class, int, int> class conv, int pitch, int shift, int dither>
__inline__ __device__ void Subsample_Bilinear(cudaTextureObject_t tex,
                                   DST *dst,
                                   int dst_width, int dst_height, int dst_pitch,
                                   int src_width, int src_height,
                                   cudaTextureObject_t ditherTex)
{
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;

    if (yo < dst_height && xo < dst_width)
    {
        float hscale = (float)src_width / (float)dst_width;
        float vscale = (float)src_height / (float)dst_height;
        float xi = (xo + 0.5f) * hscale;
        float yi = (yo + 0.5f) * vscale;
        // 3-tap filter weights are {wh,1.0,wh} and {wv,1.0,wv}
        float wh = min(max(0.5f * (hscale - 1.0f), 0.0f), 1.0f);
        float wv = min(max(0.5f * (vscale - 1.0f), 0.0f), 1.0f);
        // Convert weights to two bilinear weights -> {wh,1.0,wh} -> {wh,0.5,0} + {0,0.5,wh}
        float dx = wh / (0.5f + wh);
        float dy = wv / (0.5f + wv);

        SRC i0 = tex2D<SRC>(tex, xi-dx, yi-dy);
        SRC i1 = tex2D<SRC>(tex, xi+dx, yi-dy);
        SRC i2 = tex2D<SRC>(tex, xi-dx, yi+dy);
        SRC i3 = tex2D<SRC>(tex, xi+dx, yi+dy);

        ushort ditherVal = dither ? tex2D<ushort>(ditherTex, xo, yo) : 0;

        dst[yo*(dst_pitch / sizeof(DST))+xo*pitch] = conv<SRC, DST, shift, dither>()(i0, i1, i2, i3, ditherVal);
    }
}

extern "C" {

#define VARIANT(SRC, DST, CONV, SHIFT, PITCH, DITHER, NAME) \
__global__ void Subsample_Bilinear_ ## NAME(cudaTextureObject_t tex, \
                                    DST *dst, \
                                    int dst_width, int dst_height, int dst_pitch, \
                                    int src_width, int src_height, \
                                    cudaTextureObject_t ditherTex) \
{ \
    Subsample_Bilinear<SRC, DST, CONV, PITCH, SHIFT, DITHER>(tex, dst, dst_width, dst_height, dst_pitch, \
                                                             src_width, src_height, ditherTex); \
}

#define VARIANTSET2(SRC, DST, SHIFT, NAME) \
    VARIANT(SRC,      DST,      add_conv_shift1_d, SHIFT, 1, (sizeof(DST) < sizeof(SRC)) ? sizeof(DST) : 0, NAME) \
    VARIANT(SRC,      DST,      add_conv_shift1,   SHIFT, 1, 0, NAME ## _c) \
    VARIANT(SRC,      DST,      add_conv_shift1,   SHIFT, 2, 0, NAME ## _p2) \
    VARIANT(SRC ## 2, DST ## 2, add_conv_shift2,   SHIFT, 1, 0, NAME ## _2) \
    VARIANT(SRC ## 2, DST,      add_conv_shift2_x, SHIFT, 1, 0, NAME ## _2_u) \
    VARIANT(SRC ## 2, DST,      add_conv_shift2_y, SHIFT, 1, 0, NAME ## _2_v) \
    VARIANT(SRC ## 4, DST ## 4, add_conv_shift4,   SHIFT, 1, 0, NAME ## _4)

#define VARIANTSET(SRC, DST, SRCSIZE, DSTSIZE) \
    VARIANTSET2(SRC, DST, (SRCSIZE - DSTSIZE), SRCSIZE ## _ ## DSTSIZE)

// Straight no-conversion
VARIANTSET(uchar,  uchar,  8,  8)
VARIANTSET(ushort, ushort, 16, 16)

// Conversion between 8- and 16-bit
VARIANTSET(uchar,  ushort, 8,  16)
VARIANTSET(ushort, uchar,  16, 8)

}
