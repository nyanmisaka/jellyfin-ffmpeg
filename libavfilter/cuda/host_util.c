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

#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "libavfilter/colorspace.h"
#include "host_util.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, cu, x)
#define DEPTH_BYTES(depth) (((depth) + 7) / 8)

int ff_make_cuda_frame(AVFilterContext *ctx, CudaFunctions *cu, int make_cuTex,
                       FFCUDAFrame *dst, const AVFrame *src, const AVPixFmtDescriptor *src_desc)
{
    int i, ret = 0;
    for (i = 0, dst->planes = 0; i < src_desc->nb_components; i++)
        dst->planes = FFMAX(dst->planes, src_desc->comp[i].plane + 1);

    for (i = 0; i < dst->planes; i++) {
        dst->data[i] = src->data[i];
        dst->linesize[i] = src->linesize[i];
        dst->tex[i] = 0;

        if (!make_cuTex)
            continue;

#ifndef CU_TRSF_NORMALIZED_COORDINATES
  #define CU_TRSF_NORMALIZED_COORDINATES 2
#endif
        CUDA_TEXTURE_DESC tex_desc = {
            .addressMode = CU_TR_ADDRESS_MODE_CLAMP,
            .filterMode = i == 0 ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR,
            .flags = i == 0 ? 0 : CU_TRSF_NORMALIZED_COORDINATES,
        };

        CUDA_RESOURCE_DESC res_desc = {
            .resType = CU_RESOURCE_TYPE_PITCH2D,
            .res.pitch2D.format = DEPTH_BYTES(src_desc->comp[i].depth) == 1 ?
                                  CU_AD_FORMAT_UNSIGNED_INT8 :
                                  CU_AD_FORMAT_UNSIGNED_INT16,
            .res.pitch2D.numChannels = i == 0 ? 1 : (dst->planes == 2 ? 2 : 1),
            .res.pitch2D.width = i == 0 ? src->width : AV_CEIL_RSHIFT(src->width, src_desc->log2_chroma_w),
            .res.pitch2D.height = i == 0 ? src->height : AV_CEIL_RSHIFT(src->height, src_desc->log2_chroma_h),
            .res.pitch2D.pitchInBytes = src->linesize[i],
            .res.pitch2D.devPtr = (CUdeviceptr)src->data[i],
        };

        if ((ret = CHECK_CU(cu->cuTexObjectCreate(&dst->tex[i], &res_desc, &tex_desc, NULL))) < 0)
            goto fail;
    }

    dst->width  = src->width;
    dst->height = src->height;

    return ret;

fail:
    for (i = 0; i < dst->planes; i++)
        if (dst->tex[i])
            CHECK_CU(cu->cuTexObjectDestroy(dst->tex[i]));

    return ret;
}
