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

#if (CONFIG_VAAPI && CONFIG_LIBDRM && HAVE_VAAPI_DRM)
#   include <va/va_drm.h>
#   include <va/va_drmcommon.h>
#   include <xf86drm.h>
#   include "libavutil/hwcontext_drm.h"
#   include "libavutil/hwcontext_vaapi.h"
#endif

// see also https://github.com/oneapi-src/oneVPL/blob/master/tools/cli/system_analyzer/system_analyzer.cpp

/* DRM */
int create_drm_devices(HwDeviceRefs *refs)
{
#if CONFIG_LIBDRM
    int i, j, n = 0, ret = 0;
    drmDevice *drm_all[HWINFO_MAX_DEV_NUM];

    n = drmGetDevices(drm_all, FF_ARRAY_ELEMS(drm_all));
    if (n <= 0)
        return AVERROR(ENOSYS);

    for (i = n - 1, j = 0; i >= 0 && refs; --i) {
        drmDevice *drm = drm_all[i];

        if (!(drm->available_nodes & (1 << DRM_NODE_RENDER)))
            continue;

        ret = av_hwdevice_ctx_create(&refs[j].drm_ref, AV_HWDEVICE_TYPE_DRM,
                                     drm->nodes[DRM_NODE_RENDER], NULL, 0);
        if (ret < 0)
            continue;

        refs[j++].device_path_drm = drm->nodes[DRM_NODE_RENDER];
    }

    drmFreeDevices(drm_all, n);
    return 0;
#else
    return AVERROR(ENOSYS);
#endif
}

/* DRM -> VAAPI */
void create_derive_vaapi_devices_from_drm(HwDeviceRefs *refs)
{
    for (int i = 0; i < HWINFO_MAX_DEV_NUM && refs && refs[i].drm_ref; i++) {
        av_hwdevice_ctx_create_derived(&refs[i].vaapi_ref, AV_HWDEVICE_TYPE_VAAPI,
                                       refs[i].drm_ref, 0);
    }
}

/* DRM -> VULKAN */
void create_derive_vulkan_devices_from_drm(HwDeviceRefs *refs)
{
    for (int i = 0; i < HWINFO_MAX_DEV_NUM && refs && refs[i].drm_ref; i++) {
        av_hwdevice_ctx_create_derived(&refs[i].vulkan_ref, AV_HWDEVICE_TYPE_VULKAN,
                                       refs[i].drm_ref, 0);
    }
}

/* VAAPI -> QSV */
void create_derive_qsv_devices_from_vaapi(HwDeviceRefs *refs)
{
    for (int i = 0; i < HWINFO_MAX_DEV_NUM && refs && refs[i].vaapi_ref; i++) {
        if (refs[i].device_vendor_id != HWINFO_VENDOR_ID_INTEL)
            continue;
        av_hwdevice_ctx_create_derived(&refs[i].qsv_ref, AV_HWDEVICE_TYPE_QSV,
                                       refs[i].vaapi_ref, 0);
    }
}

/* VAAPI -> OPENCL */
void create_derive_opencl_devices_from_vaapi(HwDeviceRefs *refs)
{
    for (int i = 0; i < HWINFO_MAX_DEV_NUM && refs && refs[i].vaapi_ref; i++) {
        if (refs[i].device_vendor_id != HWINFO_VENDOR_ID_INTEL)
            continue;
        av_hwdevice_ctx_create_derived(&refs[i].opencl_ref, AV_HWDEVICE_TYPE_OPENCL,
                                       refs[i].vaapi_ref, 0);
    }
}