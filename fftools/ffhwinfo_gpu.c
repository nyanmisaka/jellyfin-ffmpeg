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

static int show_vaapi_info(WriterContext *wctx, HwDeviceRefs *refs, int accel_flags)
{
    int ret = 0;
    if (!wctx || !refs)
        return AVERROR(EINVAL);
#if CONFIG_LIBDRM
    ret = create_drm_devices(refs);
    if (ret < 0)
        return ret;

    create_derive_vaapi_devices_from_drm(refs);

    if (accel_flags & HWINFO_FLAG_PRINT_COMPUTE_OPENCL)
        create_derive_opencl_devices_from_vaapi(refs);
    if (accel_flags & HWINFO_FLAG_PRINT_COMPUTE_VULKAN)
        create_derive_vulkan_devices_from_drm(refs);

    print_drm_based_all(wctx, refs, accel_flags | HWINFO_FLAG_PRINT_OS_VA);
#endif
    return ret;
}

static int show_qsv_info(WriterContext *wctx, HwDeviceRefs *refs, int accel_flags)
{
    int ret = 0;
    if (!wctx || !refs)
        return AVERROR(EINVAL);
#if CONFIG_D3D11VA
    ret = create_d3d11va_devices_with_filter(refs, HWINFO_VENDOR_ID_INTEL, -1, NULL);
    if (ret < 0)
        return ret;

    create_derive_qsv_devices_from_d3d11va(refs);

    if (accel_flags & HWINFO_FLAG_PRINT_COMPUTE_OPENCL)
        create_derive_opencl_devices_from_d3d11va(refs);

    print_dxgi_based_all(wctx, refs, accel_flags);
#elif CONFIG_LIBDRM
    ret = create_drm_devices(refs);
    if (ret < 0)
        return ret;

    create_derive_vaapi_devices_from_drm(refs);
    create_derive_qsv_devices_from_vaapi(refs);

    if (accel_flags & HWINFO_FLAG_PRINT_COMPUTE_OPENCL)
        create_derive_opencl_devices_from_vaapi(refs);
    if (accel_flags & HWINFO_FLAG_PRINT_COMPUTE_VULKAN)
        create_derive_vulkan_devices_from_drm(refs);

    print_drm_based_all(wctx, refs, accel_flags);
#endif
    return ret;
}

static int show_cuda_info(WriterContext *wctx, HwDeviceRefs *refs, int accel_flags)
{
    int ret = 0;
    if (!wctx || !refs)
        return AVERROR(EINVAL);
#if CONFIG_CUDA
    ret = create_cuda_devices(refs);
    if (ret < 0)
        goto exit;

    print_cuda_based_all(wctx, refs, accel_flags);
exit:
    uninit_cuda_functions();
    uninit_nvml_functions();
#endif
    return ret;
}

static int show_amf_info(WriterContext *wctx, HwDeviceRefs *refs, int accel_flags)
{
    int ret = 0;
    if (!wctx || !refs)
        return AVERROR(EINVAL);
#if CONFIG_D3D11VA
    ret = create_d3d11va_devices_with_filter(refs, HWINFO_VENDOR_ID_AMD, -1, NULL);
    if (ret < 0)
        return ret;

    if (accel_flags & HWINFO_FLAG_PRINT_COMPUTE_OPENCL)
        create_derive_opencl_devices_from_d3d11va(refs);

    print_dxgi_based_all(wctx, refs, accel_flags | HWINFO_FLAG_PRINT_OS_VA);
#endif
    return ret;
}

int show_accel_device_info(WriterContext *wctx, enum HWInfoAccelType accel_type, int accel_flags)
{
    HwDeviceRefs refs[MAX_HW_DEVICE_NUM] = {0};

    if (!wctx)
        return AVERROR(EINVAL);

    switch (accel_type) {
    case HWINFO_ACCEL_TYPE_VAAPI:
        show_vaapi_info(wctx, refs, accel_flags);
        break;
    case HWINFO_ACCEL_TYPE_QSV:
        show_qsv_info(wctx, refs, accel_flags);
        break;
    case HWINFO_ACCEL_TYPE_CUDA:
        show_cuda_info(wctx, refs, accel_flags);
        break;
    case HWINFO_ACCEL_TYPE_AMF:
        show_amf_info(wctx, refs, accel_flags);
        break;
    case HWINFO_ACCEL_TYPE_NONE:
    default:
        return AVERROR(EINVAL);
    }

    for (unsigned i = 0; i < FF_ARRAY_ELEMS(refs); i++) {
        if (refs[i].drm_ref)
            av_buffer_unref(&refs[i].drm_ref);
        if (refs[i].vaapi_ref)
            av_buffer_unref(&refs[i].vaapi_ref);
        if (refs[i].d3d11va_ref)
            av_buffer_unref(&refs[i].d3d11va_ref);
        if (refs[i].qsv_ref)
            av_buffer_unref(&refs[i].qsv_ref);
        if (refs[i].opencl_ref)
            av_buffer_unref(&refs[i].opencl_ref);
        if (refs[i].vulkan_ref)
            av_buffer_unref(&refs[i].vulkan_ref);
        if (refs[i].cuda_ref)
            av_buffer_unref(&refs[i].cuda_ref);
    }
    return 0;
}
