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

/* DRM */
int create_drm_devices(HwDeviceRefs *refs)
{
#if CONFIG_LIBDRM
    int i, j, n = 0, ret = 0;
    drmDevice *drm_all[MAX_HW_DEVICE_NUM];

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
    for (int i = 0; i < MAX_HW_DEVICE_NUM && refs && refs[i].drm_ref; i++) {
        av_hwdevice_ctx_create_derived(&refs[i].vaapi_ref, AV_HWDEVICE_TYPE_VAAPI,
                                       refs[i].drm_ref, 0);
    }
}

/* DRM -> VULKAN */
void create_derive_vulkan_devices_from_drm(HwDeviceRefs *refs)
{
    for (int i = 0; i < MAX_HW_DEVICE_NUM && refs && refs[i].drm_ref; i++) {
        av_hwdevice_ctx_create_derived(&refs[i].vulkan_ref, AV_HWDEVICE_TYPE_VULKAN,
                                       refs[i].drm_ref, 0);
    }
}

/* VAAPI -> QSV */
void create_derive_qsv_devices_from_vaapi(HwDeviceRefs *refs)
{
    for (int i = 0; i < MAX_HW_DEVICE_NUM && refs && refs[i].vaapi_ref; i++) {
        if (refs[i].device_vendor_id != HWINFO_VENDOR_ID_INTEL)
            continue;
        av_hwdevice_ctx_create_derived(&refs[i].qsv_ref, AV_HWDEVICE_TYPE_QSV,
                                       refs[i].vaapi_ref, 0);
    }
}

/* VAAPI -> OPENCL */
void create_derive_opencl_devices_from_vaapi(HwDeviceRefs *refs)
{
    for (int i = 0; i < MAX_HW_DEVICE_NUM && refs && refs[i].vaapi_ref; i++) {
        if (refs[i].device_vendor_id != HWINFO_VENDOR_ID_INTEL)
            continue;
        av_hwdevice_ctx_create_derived(&refs[i].opencl_ref, AV_HWDEVICE_TYPE_OPENCL,
                                       refs[i].vaapi_ref, 0);
    }
}

int print_drm_based_all(WriterContext *wctx, HwDeviceRefs *refs, int accel_flags)
{
    unsigned i, j;

    if (!refs || !wctx)
        return AVERROR(EINVAL);

    for (j = 0; j < MAX_HW_DEVICE_NUM && refs[j].drm_ref; j++);
    if (j == 0)
        return 0;

    mark_section_show_entries(SECTION_ID_ROOT, 1, NULL);
    mark_section_show_entries(SECTION_ID_DEVICES, 1, NULL);
    mark_section_show_entries(SECTION_ID_DEVICE, 1, NULL);
    writer_print_section_header(wctx, SECTION_ID_ROOT);
    writer_print_section_header(wctx, SECTION_ID_DEVICES);

    for (i = 0; i < j; i++) {
        writer_print_section_header(wctx, SECTION_ID_DEVICE);

        /* DRM based device path */
        print_str("DevicePathDRM", refs[i].device_path_drm);

        /* DRM device info */
        // if (accel_flags & HWINFO_FLAG_PRINT_DEV)
        //     print_drm_device_info(wctx, refs[i].drm_ref);

        /* VAAPI device info */
        // if ((accel_flags & HWINFO_FLAG_PRINT_DEV) &&
        //     (accel_flags & HWINFO_FLAG_PRINT_OS_VA))
        //     print_vaapi_device_info(wctx, refs[i].vaapi_ref);

        /* VAAPI decoder info */
        // if ((accel_flags & HWINFO_FLAG_PRINT_DEC) &&
        //     (accel_flags & HWINFO_FLAG_PRINT_OS_VA))
        //     print_vaapi_decoder_info(wctx, refs[i].vaapi_ref);

        /* VAAPI encoder info */
        // if ((accel_flags & HWINFO_FLAG_PRINT_ENC) &&
        //     (accel_flags & HWINFO_FLAG_PRINT_OS_VA))
        //     print_vaapi_encoder_info(wctx, refs[i].vaapi_ref);

        /* VAAPI vpp info */
        // if ((accel_flags & HWINFO_FLAG_PRINT_VPP) &&
        //     (accel_flags & HWINFO_FLAG_PRINT_OS_VA))
        //     print_vaapi_filter_info(wctx, refs[i].vaapi_ref);

        /* QSV device info */
        // if (accel_flags & HWINFO_FLAG_PRINT_DEV)
        //     print_qsv_device_info(wctx, refs[i].qsv_ref);

        /* QSV decoder info */
        // if (accel_flags & HWINFO_FLAG_PRINT_DEC)
        //     print_qsv_decoder_info(wctx, refs[i].qsv_ref);

        /* QSV encoder info */
        // if (accel_flags & HWINFO_FLAG_PRINT_ENC)
        //     print_qsv_encoder_info(wctx, refs[i].qsv_ref);

        /* QSV vpp info */
        // if (accel_flags & HWINFO_FLAG_PRINT_VPP)
        //     print_qsv_filter_info(wctx, refs[i].qsv_ref);

        /* OPENCL device info */
        // if ((accel_flags & HWINFO_FLAG_PRINT_COMPUTE_OPENCL) &&
        //     (accel_flags & HWINFO_FLAG_PRINT_DEV))
        //     print_opencl_device_info(wctx, refs[i].opencl_ref);

        /* VULKAN device info */
        // if ((accel_flags & HWINFO_FLAG_PRINT_COMPUTE_VULKAN) &&
        //     (accel_flags & HWINFO_FLAG_PRINT_DEV))
        //     print_vulkan_device_info(wctx, refs[i].vulkan_ref);
#if 0
        /* CUDA based device path */
        // print_int("DeviceIndexCUDA", refs[i].device_index_cuda);

        /* CUDA device info */
        // if (accel_flags & HWINFO_FLAG_PRINT_DEV)
        //     print_cuda_device_info(wctx, refs[i].cuda_ref);
#endif
        writer_print_section_footer(wctx);
    }

    writer_print_section_footer(wctx);
    writer_print_section_footer(wctx);

    return 0;
}
