/*
 * Copyright (c) 2023 NyanMisaka
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ffhwinfo_gpu.h"
#include "libavutil/hwcontext.h"

#if CONFIG_CUDA
#   define CHECK_CU(x) FF_CUDA_CHECK_DL(NULL, cu, x)
#   include "libavutil/cuda_check.h"
#   include "libavutil/hwcontext_cuda_internal.h"
#endif

/* CUDA */
int create_cuda_devices(HwDeviceRefs *refs)
{
#if CONFIG_CUDA
    int i, j, n = 0, ret = 0;
    char ibuf[4];
    CudaFunctions *cu = NULL;

    ret = cuda_load_functions(&cu, NULL);
    if (ret < 0)
        goto exit;

    ret = CHECK_CU(cu->cuInit(0));
    if (ret < 0)
        goto exit;

    ret = CHECK_CU(cu->cuDeviceGetCount(&n));
    if (ret < 0)
        goto exit;

    if (n <= 0) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    n = FFMIN(n, MAX_HW_DEVICE_NUM);
    for (i = 0, j = 0; i < n && refs; i++) {
        snprintf(ibuf, sizeof(ibuf), "%d", i);
        ret = av_hwdevice_ctx_create(&refs[j].cuda_ref, AV_HWDEVICE_TYPE_CUDA,
                                     ibuf, NULL, 0);
        if (ret < 0)
            continue;

        refs[j++].device_index = i;
    }

    ret = 0;

exit:
    cuda_free_functions(&cu);
    return ret;
#else
    return AVERROR(ENOSYS);
#endif
}
