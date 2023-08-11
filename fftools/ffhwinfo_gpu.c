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

static int show_vaapi_info(HwDeviceRefs *refs, const WriterContext *wctx, int accel_flags)
{
    if (!refs)
        return AVERROR(ENOMEM);

    create_drm_devices(refs);
    create_derive_vaapi_devices_from_drm(refs);

    return 0;
}

static int show_qsv_info(HwDeviceRefs *refs, const WriterContext *wctx, int accel_flags)
{
    if (!refs)
        return AVERROR(ENOMEM);
#if CONFIG_D3D11VA
    create_d3d11va_devices(refs);
    create_derive_qsv_devices_from_d3d11va(refs);
#elif (CONFIG_DRM)
    create_drm_devices(refs);
    create_derive_vaapi_devices_from_drm(refs);
#endif
    return 0;
}

static int show_nv_info(HwDeviceRefs *refs, const WriterContext *wctx, int accel_flags)
{
    if (!refs)
        return AVERROR(ENOMEM);

    create_cuda_devices(refs);

    return 0;
}

static int show_amf_info(HwDeviceRefs *refs, const WriterContext *wctx, int accel_flags)
{
    return 0;
}

int show_accel_device_info(const WriterContext *wctx, enum HWInfoAccelType accel_type, int accel_flags)
{
    HwDeviceRefs refs[MAX_HW_DEVICE_NUM] = {0};

    if (!wctx)
        return AVERROR(EINVAL);

    switch (accel_type) {
    case HWINFO_ACCEL_TYPE_VAAPI:
        return show_vaapi_info(refs, wctx, accel_flags);
    case HWINFO_ACCEL_TYPE_QSV:
        return show_qsv_info(refs, wctx, accel_flags);
    case HWINFO_ACCEL_TYPE_NV:
        return show_nv_info(refs, wctx, accel_flags);
    case HWINFO_ACCEL_TYPE_AMF:
        return show_amf_info(refs, wctx, accel_flags);
    case HWINFO_ACCEL_TYPE_NONE:
    default:
        return AVERROR(EINVAL);
    }

    return 0;
}
