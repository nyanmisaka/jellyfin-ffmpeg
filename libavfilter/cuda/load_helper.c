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

#include "config.h"

#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "libavutil/mem.h"

#if CONFIG_SHADER_COMPRESSION
#include "libavutil/zlib_utils.h"
#endif

#include "load_helper.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(avctx, cu, x)

static int decompress_cuda_ptx(void *avctx, uint8_t **data_out, size_t *length_out,
                               const unsigned char *data, const unsigned int length)
{
#if CONFIG_SHADER_COMPRESSION
    uint8_t *out;
    size_t out_len;
    int ret = ff_zlib_expand(avctx, &out, &out_len,
                             data, length);
    if (ret < 0)
        return ret;

    *data_out   = out;
    *length_out = out_len;
#else
    *data_out   = NULL;
    *length_out = 0;
#endif
    return 0;
}

int ff_cuda_load_module(void *avctx, AVCUDADeviceContext *hwctx, CUmodule *cu_module,
                        const unsigned char *data, const unsigned int length)
{
    CudaFunctions *cu = hwctx->internal->cuda_dl;
    uint8_t *data_out = NULL;
    size_t length_out = 0;
    int ret;

    if ((ret = decompress_cuda_ptx(avctx, &data_out, &length_out, data, length)) < 0)
        goto exit;

    ret = CHECK_CU(cu->cuModuleLoadData(cu_module, (data_out ? data_out : data)));
exit:
    if (data_out)
        av_free(data_out);
    return ret;
}

int ff_cuda_link_add_data(void *avctx, AVCUDADeviceContext *hwctx,
                          CUlinkState state, const char* name,
                          const unsigned char *data, const unsigned int length)
{
    CudaFunctions *cu = hwctx->internal->cuda_dl;
    uint8_t *data_out = NULL;
    size_t length_out = 0;
    int ret;

    if ((ret = decompress_cuda_ptx(avctx, &data_out, &length_out, data, length)) < 0)
        goto exit;

    ret = CHECK_CU(cu->cuLinkAddData(state, CU_JIT_INPUT_PTX,
                                     (void *)(data_out ? data_out : data),
                                     (size_t)(data_out ? length_out : length),
                                     name, 0, NULL, NULL));
exit:
    if (data_out)
        av_free(data_out);
    return ret;
}
