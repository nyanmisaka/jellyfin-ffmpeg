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

static int print_drm_based_all(WriterContext *wctx, HwDeviceRefs *refs, int accel_flags)
{
    unsigned i, j;
    if (!refs || !wctx)
        return AVERROR(EINVAL);

    for (j = 0; j < HWINFO_MAX_DEV_NUM && refs[j].drm_ref; j++);
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
        //     print_vaapi_vpp_info(wctx, refs[i].vaapi_ref);

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
        //     print_qsv_vpp_info(wctx, refs[i].qsv_ref);

        /* OPENCL device info */
        // if ((accel_flags & HWINFO_FLAG_PRINT_DEV) &&
        //     (accel_flags & HWINFO_FLAG_PRINT_OPT_OPENCL))
        //     print_opencl_device_info(wctx, refs[i].opencl_ref);

        /* VULKAN device info */
        // if ((accel_flags & HWINFO_FLAG_PRINT_DEV) &&
        //     (accel_flags & HWINFO_FLAG_PRINT_OPT_VULKAN))
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

static int print_dxgi_based_all(WriterContext *wctx, HwDeviceRefs *refs, int accel_flags)
{
    int amf_used = 0;
    unsigned i, j;
    if (!wctx || !refs)
        return AVERROR(EINVAL);

    for (j = 0; j < HWINFO_MAX_DEV_NUM && refs[j].d3d11va_ref; j++);
    if (j == 0)
        return 0;

    mark_section_show_entries(SECTION_ID_ROOT, 1, NULL);
    mark_section_show_entries(SECTION_ID_DEVICES, 1, NULL);
    mark_section_show_entries(SECTION_ID_DEVICE, 1, NULL);
    writer_print_section_header(wctx, SECTION_ID_ROOT);
    writer_print_section_header(wctx, SECTION_ID_DEVICES);

    for (i = 0; i < j; i++) {
        writer_print_section_header(wctx, SECTION_ID_DEVICE);

        /* DXGI/D3D11VA based device index */
        print_int("DeviceIndexD3D11VA", refs[i].device_index_dxgi);

        /* D3D11VA device info */
        if ((accel_flags & HWINFO_FLAG_PRINT_DEV) &&
            (accel_flags & HWINFO_FLAG_PRINT_OS_VA))
            print_d3d11va_device_info(wctx, refs[i].d3d11va_ref);

        /* D3D11VA decoder info */
        if ((accel_flags & HWINFO_FLAG_PRINT_DEC) &&
            (accel_flags & HWINFO_FLAG_PRINT_OS_VA))
            print_d3d11va_decoder_info(wctx, refs[i].d3d11va_ref);

        /* QSV device info */
        if (accel_flags & HWINFO_FLAG_PRINT_DEV)
            print_qsv_device_info(wctx, refs[i].qsv_ref);

        /* QSV decoder info */
        if (accel_flags & HWINFO_FLAG_PRINT_DEC)
            print_qsv_decoder_info(wctx, refs[i].qsv_ref);

        /* QSV encoder info */
        if (accel_flags & HWINFO_FLAG_PRINT_ENC)
            print_qsv_encoder_info(wctx, refs[i].qsv_ref);

        /* QSV vpp info */
        if (accel_flags & HWINFO_FLAG_PRINT_VPP)
            print_qsv_vpp_info(wctx, refs[i].qsv_ref);

        if (refs[i].device_vendor_id == HWINFO_VENDOR_ID_AMD &&
            ((accel_flags & HWINFO_FLAG_PRINT_DEV) ||
             (accel_flags & HWINFO_FLAG_PRINT_ENC))) {

            /* Create internal AMF device from D3D11VA */
            create_derive_amf_device_from_d3d11va(refs[i].d3d11va_ref);
            amf_used = 1;

            /* AMF device info */
            if (accel_flags & HWINFO_FLAG_PRINT_DEV)
                print_amf_device_info_from_d3d11va(wctx);

            /* AMF encoder info */
            if (accel_flags & HWINFO_FLAG_PRINT_ENC)
                print_amf_encoder_info_from_d3d11va(wctx);
        }

        /* OPENCL device info */
        // if ((accel_flags & HWINFO_FLAG_PRINT_DEV) &&
        //     (accel_flags & HWINFO_FLAG_PRINT_OPT_OPENCL))
        //     print_opencl_device_info(wctx, refs[i].opencl_ref);

        writer_print_section_footer(wctx);
    }

    writer_print_section_footer(wctx);
    writer_print_section_footer(wctx);

    if (amf_used)
        uninit_amf_functions();
    return 0;
}

static int print_cuda_based_all(WriterContext *wctx, HwDeviceRefs *refs, int accel_flags)
{
    unsigned i, j;
    int nvml_ret = AVERROR_EXTERNAL;
    if (!refs || !wctx)
        return AVERROR(EINVAL);

    for (j = 0; j < HWINFO_MAX_DEV_NUM && refs[j].cuda_ref; j++);
    if (j == 0)
        return 0;

    /* Init NVML for the optional version info */
    nvml_ret = init_nvml_driver_version();

    mark_section_show_entries(SECTION_ID_ROOT, 1, NULL);
    mark_section_show_entries(SECTION_ID_DEVICES, 1, NULL);
    mark_section_show_entries(SECTION_ID_DEVICE, 1, NULL);
    writer_print_section_header(wctx, SECTION_ID_ROOT);
    writer_print_section_header(wctx, SECTION_ID_DEVICES);

    for (i = 0; i < j; i++) {
        writer_print_section_header(wctx, SECTION_ID_DEVICE);

        /* CUDA based device index */
        print_int("DeviceIndexCUDA", refs[i].device_index_cuda);

        /* CUDA device info */
        if (accel_flags & HWINFO_FLAG_PRINT_DEV)
            print_cuda_device_info(wctx, refs[i].cuda_ref, nvml_ret);

        /* CUDA decoder info */
        if (accel_flags & HWINFO_FLAG_PRINT_DEC)
            print_cuda_decoder_info(wctx, refs[i].cuda_ref);

        /* CUDA encoder info */
        if (accel_flags & HWINFO_FLAG_PRINT_ENC)
            print_cuda_encoder_info(wctx, refs[i].cuda_ref);
#if 1
        /* VULKAN device info */
        // if ((accel_flags & HWINFO_FLAG_PRINT_DEV) &&
        //     (accel_flags & HWINFO_FLAG_PRINT_OPT_VULKAN))
        //     print_vulkan_device_info(wctx, refs[i].vulkan_ref);

        /* DXGI/D3D11VA based device index */
        if ((accel_flags & HWINFO_FLAG_PRINT_OPT_D3D11VA) && refs[i].d3d11va_ref)
            print_int("DeviceIndexD3D11VA", refs[i].device_index_dxgi);

        /* D3D11VA device info */
        if ((accel_flags & HWINFO_FLAG_PRINT_DEV) &&
            (accel_flags & HWINFO_FLAG_PRINT_OPT_D3D11VA))
            print_d3d11va_device_info(wctx, refs[i].d3d11va_ref);

        if ((accel_flags & HWINFO_FLAG_PRINT_DEC) &&
            (accel_flags & HWINFO_FLAG_PRINT_OPT_D3D11VA))
            print_d3d11va_decoder_info(wctx, refs[i].d3d11va_ref);
#endif
        writer_print_section_footer(wctx);
    }

    writer_print_section_footer(wctx);
    writer_print_section_footer(wctx);

    return 0;
}

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

    if (accel_flags & HWINFO_FLAG_PRINT_OPT_OPENCL)
        create_derive_opencl_devices_from_vaapi(refs);
    if (accel_flags & HWINFO_FLAG_PRINT_OPT_VULKAN)
        create_derive_vulkan_devices_from_drm(refs);

    print_drm_based_all(wctx, refs, (accel_flags | HWINFO_FLAG_PRINT_OS_VA));
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

    if (accel_flags & HWINFO_FLAG_PRINT_OPT_OPENCL)
        create_derive_opencl_devices_from_d3d11va(refs);

    print_dxgi_based_all(wctx, refs, accel_flags);
#elif CONFIG_LIBDRM
    ret = create_drm_devices(refs);
    if (ret < 0)
        return ret;

    create_derive_vaapi_devices_from_drm(refs);
    create_derive_qsv_devices_from_vaapi(refs);

    if (accel_flags & HWINFO_FLAG_PRINT_OPT_OPENCL)
        create_derive_opencl_devices_from_vaapi(refs);
    if (accel_flags & HWINFO_FLAG_PRINT_OPT_VULKAN)
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

#if CONFIG_D3D11VA
    if (accel_flags & HWINFO_FLAG_PRINT_OPT_D3D11VA)
        create_derive_d3d11va_devices_from_cuda(refs);
#endif

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

    if (accel_flags & HWINFO_FLAG_PRINT_OPT_OPENCL)
        create_derive_opencl_devices_from_d3d11va(refs);

    print_dxgi_based_all(wctx, refs, (accel_flags | HWINFO_FLAG_PRINT_OS_VA));
#endif
    return ret;
}

int show_accel_device_info(WriterContext *wctx, enum HWInfoAccelType accel_type, int accel_flags)
{
    HwDeviceRefs refs[HWINFO_MAX_DEV_NUM] = {0};

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
