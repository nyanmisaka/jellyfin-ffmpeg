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

#ifndef AVFILTER_CUDA_HOST_UTIL_H
#define AVFILTER_CUDA_HOST_UTIL_H

#include "libavutil/frame.h"
#include "libavutil/pixdesc.h"
#include "libavfilter/avfilter.h"
#include "shared.h"

int ff_make_cuda_frame(AVFilterContext *ctx, CudaFunctions *cu, int make_cuTex,
                       FFCUDAFrame *dst, const AVFrame *src, const AVPixFmtDescriptor *src_desc);

#endif /* AVFILTER_CUDA_HOST_UTIL_H */
