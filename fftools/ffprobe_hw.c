/*
 * Copyright (C) 2026 NyanMisaka
 *
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

#include <errno.h>
#include <stdlib.h>

#include "ffprobe_hw_internal.h"

av_unused static void get_hwaccel_device_index(int *device_idx, const char *hwaccel_device)
{
    if (!device_idx)
        return;

    *device_idx = -1;

    if (hwaccel_device) {
        char *end_ptr = NULL;
        long val = strtol(hwaccel_device, &end_ptr, 10);

        if (hwaccel_device == end_ptr ||
            *end_ptr != '\0'          ||
            errno == ERANGE           ||
            val < 0                   ||
            val > FFPROBE_HW_MAX_DEV_NUM) {
            *device_idx = -1;
        } else
            *device_idx = (int)val;
    }
}

#if CONFIG_CUDA
static int print_cuda_based_all(AVTextFormatContext *tfc,
                                HwDeviceRefs *refs, int hwaccel_flags)
{
    unsigned i;
    int nvml_ret = AVERROR_EXTERNAL;
    if (!refs || !tfc)
        return AVERROR(EINVAL);

    for (i = 0; i < FFPROBE_HW_MAX_DEV_NUM && refs[i].cuda_ref; i++);
    if (!i)
        return 0;

    /* Init NVML for the optional version info */
    if ((hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEV) ||
        (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_UTIL))
        nvml_ret = init_nvml_driver_version();

    mark_section_show_entries(SECTION_ID_HWACCEL_DEVICE, 1, NULL);

    for (i = 0; i < FFPROBE_HW_MAX_DEV_NUM && refs[i].cuda_ref; i++) {
        avtext_print_section_header(tfc, NULL, SECTION_ID_HWACCEL_DEVICE);

        /* CUDA based device index */
        print_int("device_index_cuda", refs[i].device_index_cuda);

        /* CUDA device info */
        if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEV)
            print_cuda_device_info(tfc, refs[i].cuda_ref, nvml_ret);

        /* CUDA device util */
        if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_UTIL)
            print_cuda_device_util(tfc, refs[i].cuda_ref, nvml_ret);

        /* CUDA decoder info */
        if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEC)
            print_cuda_decoder_info(tfc, refs[i].cuda_ref);

        /* CUDA encoder info */
        if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_ENC)
            print_cuda_encoder_info(tfc, refs[i].cuda_ref);

#   if CONFIG_D3D11VA
        /* DXGI/D3D11VA based device index */
        if ((hwaccel_flags & FFPROBE_HW_FLAG_PRINT_OPT_DX11) && refs[i].d3d11va_ref)
            print_int("device_index_d3d11va", refs[i].device_index_dxgi);

        /* D3D11VA device info */
        if ((hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEV) &&
            (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_OPT_DX11))
            print_d3d11va_device_info(tfc, refs[i].d3d11va_ref);

        /* D3D11VA device util */
        if ((hwaccel_flags & FFPROBE_HW_FLAG_PRINT_UTIL) &&
            (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_OPT_DX11))
            print_d3d11va_device_util(tfc, refs[i].d3d11va_ref);

        /* D3D11VA decoder info */
        if ((hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEC) &&
            (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_OPT_DX11))
            print_d3d11va_decoder_info(tfc, refs[i].d3d11va_ref);
#   endif

        avtext_print_section_footer(tfc); // SECTION_ID_HWACCEL_DEVICE
    }

    return 0;
}
#endif

static int show_cuda_info(AVTextFormatContext *tfc, HwDeviceRefs *refs,
                          int hwaccel_flags, const char *hwaccel_device)
{
    av_unused int device_idx = -1;
    int ret = 0;
    if (!tfc || !refs)
        return AVERROR(EINVAL);

#if CONFIG_CUDA
    get_hwaccel_device_index(&device_idx, hwaccel_device);

    ret = create_cuda_devices(refs, device_idx);
    if (ret < 0)
        goto exit;

#   if CONFIG_D3D11VA
    if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_OPT_DX11)
        create_derive_d3d11va_devices_from_cuda(refs);
#   endif

    print_cuda_based_all(tfc, refs, hwaccel_flags);
exit:
    uninit_cuvid_functions();
    uninit_nvenc_functions();
    uninit_cuda_functions();
    uninit_nvml_functions();
#endif
    return ret;
}

#if CONFIG_D3D11VA
static int print_dxgi_based_all(AVTextFormatContext *tfc,
                                HwDeviceRefs *refs,
                                int hwaccel_flags, int mf_only)
{
    unsigned i;
    if (!tfc || !refs)
        return AVERROR(EINVAL);

    for (i = 0; i < FFPROBE_HW_MAX_DEV_NUM && refs[i].d3d11va_ref; i++);
    if (!i)
        return 0;

    mark_section_show_entries(SECTION_ID_HWACCEL_DEVICE, 1, NULL);

    for (i = 0; i < FFPROBE_HW_MAX_DEV_NUM && refs[i].d3d11va_ref; i++) {
        avtext_print_section_header(tfc, NULL, SECTION_ID_HWACCEL_DEVICE);

        /* DXGI/D3D11VA based device index */
        print_int("device_index_d3d11va", refs[i].device_index_dxgi);

        /* D3D11VA device info */
        if ((hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEV) &&
            (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_SUB_DEV))
            print_d3d11va_device_info(tfc, refs[i].d3d11va_ref);

        /* D3D11VA device util */
        if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_UTIL)
            print_d3d11va_device_util(tfc, refs[i].d3d11va_ref);

        /* D3D11VA decoder info */
        if ((hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEC) &&
            (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_SUB_DEV))
            print_d3d11va_decoder_info(tfc, refs[i].d3d11va_ref);

        if (!mf_only && refs[i].device_vendor_id == FFPROBE_HW_VENDOR_ID_INTEL) {
            /* QSV device info */
            if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEV)
                print_qsv_device_info(tfc, refs[i].qsv_ref);

            /* QSV decoder info */
            if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEC)
                print_qsv_decoder_info(tfc, refs[i].qsv_ref);

            /* QSV encoder info */
            if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_ENC)
                print_qsv_encoder_info(tfc, refs[i].qsv_ref);

            /* QSV vpp info */
            if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_VPP)
                print_qsv_vpp_info(tfc, refs[i].qsv_ref);
        }

        if (!mf_only && refs[i].device_vendor_id == FFPROBE_HW_VENDOR_ID_AMD &&
            ((hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEV) ||
             (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_ENC))) {
            /* Create internal AMF device from D3D11VA */
            create_derive_amf_device_from_d3d11va(refs[i].d3d11va_ref,
                                                  refs[i].amf_dll_path);

            /* AMF device info */
            if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEV)
                print_amf_device_info_from_d3d11va(tfc);

            /* AMF encoder info */
            if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_ENC)
                print_amf_encoder_info_from_d3d11va(tfc);

            uninit_amf_functions();
        }

        if (mf_only && (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_ENC)) {
            /* Create internal MF device from D3D11VA */
            create_derive_mf_device_from_d3d11va(refs[i].d3d11va_ref);

            /* MF encoder info */
            print_mf_encoder_info_from_d3d11va(tfc, refs[i].device_luid_dxgi);

            uninit_mf_functions();
        }

        /* OPENCL device info */
        if ((hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEV) &&
            (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_OPT_OCL))
            print_opencl_device_info(tfc, refs[i].opencl_ref);

        avtext_print_section_footer(tfc); // SECTION_ID_HWACCEL_DEVICE
    }

    return 0;
}
#endif

#if CONFIG_LIBDRM
static int print_drm_based_all(AVTextFormatContext *tfc,
                               HwDeviceRefs *refs, int hwaccel_flags)
{
    unsigned i;
    if (!tfc || !refs)
        return AVERROR(EINVAL);

    for (i = 0; i < FFPROBE_HW_MAX_DEV_NUM && refs[i].drm_ref; i++);
    if (!i)
        return 0;

    mark_section_show_entries(SECTION_ID_HWACCEL_DEVICE, 1, NULL);

    for (i = 0; i < FFPROBE_HW_MAX_DEV_NUM && refs[i].drm_ref; i++) {
        avtext_print_section_header(tfc, NULL, SECTION_ID_HWACCEL_DEVICE);

        /* DRM based device path */
        print_str("device_path_drm", refs[i].device_path_drm);

        /* DRM device info */
        if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEV)
            print_drm_device_info(tfc, refs[i].drm_ref);
#   if 0
        /* DRM device util */
        if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_UTIL)
            print_drm_device_util(tfc, refs[i].drm_ref);
#   endif
        /* VAAPI device info */
        if ((hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEV) &&
            (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_SUB_DEV))
            print_vaapi_device_info(tfc, refs[i].vaapi_ref,
                                         refs[i].drm_ref);

        /* VAAPI decoder info */
        if ((hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEC) &&
            (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_SUB_DEV))
            print_vaapi_decoder_info(tfc, refs[i].vaapi_ref);

        /* VAAPI encoder info */
        if ((hwaccel_flags & FFPROBE_HW_FLAG_PRINT_ENC) &&
            (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_SUB_DEV))
            print_vaapi_encoder_info(tfc, refs[i].vaapi_ref);

        /* VAAPI vpp info */
        if ((hwaccel_flags & FFPROBE_HW_FLAG_PRINT_VPP) &&
            (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_SUB_DEV))
            print_vaapi_vpp_info(tfc, refs[i].vaapi_ref);

        if (refs[i].device_vendor_id == FFPROBE_HW_VENDOR_ID_INTEL) {
            /* QSV device info */
            if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEV)
                print_qsv_device_info(tfc, refs[i].qsv_ref);

            /* QSV decoder info */
            if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEC)
                print_qsv_decoder_info(tfc, refs[i].qsv_ref);

            /* QSV encoder info */
            if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_ENC)
                print_qsv_encoder_info(tfc, refs[i].qsv_ref);

            /* QSV vpp info */
            if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_VPP)
                print_qsv_vpp_info(tfc, refs[i].qsv_ref);
        }

        /* OPENCL device info */
        if ((hwaccel_flags & FFPROBE_HW_FLAG_PRINT_DEV) &&
            (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_OPT_OCL))
            print_opencl_device_info(tfc, refs[i].opencl_ref);

        avtext_print_section_footer(tfc); // SECTION_ID_HWACCEL_DEVICE
    }

    return 0;
}
#endif

static int show_vaapi_info(AVTextFormatContext *tfc, HwDeviceRefs *refs,
                           int hwaccel_flags, const char *hwaccel_device)
{
    int ret = 0;
    if (!tfc || !refs)
        return AVERROR(EINVAL);

#if (CONFIG_LIBDRM && CONFIG_VAAPI && HAVE_VAAPI_DRM)
    ret = create_drm_devices(refs, hwaccel_device);
    if (ret < 0)
        return ret;

    create_derive_vaapi_devices_from_drm(refs);

    if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_OPT_OCL)
        create_derive_opencl_devices_from_vaapi(refs);

    print_drm_based_all(tfc, refs, (hwaccel_flags | FFPROBE_HW_FLAG_PRINT_SUB_DEV));

    uninit_udev_functions();
#endif
    return ret;
}

static int show_d3d11va_info(AVTextFormatContext *tfc, HwDeviceRefs *refs,
                             int hwaccel_flags, const char *hwaccel_device)
{
    av_unused int device_idx = -1;
    int ret = 0;
    if (!tfc || !refs)
        return AVERROR(EINVAL);

#if CONFIG_D3D11VA
    get_hwaccel_device_index(&device_idx, hwaccel_device);

    ret = create_d3d11va_devices(refs, device_idx);
    if (ret < 0)
        return ret;

    print_dxgi_based_all(tfc, refs, (hwaccel_flags | FFPROBE_HW_FLAG_PRINT_SUB_DEV), 1);
#endif
    return ret;
}

static int show_qsv_info(AVTextFormatContext *tfc, HwDeviceRefs *refs,
                         int hwaccel_flags, const char *hwaccel_device)
{
    av_unused int device_idx = -1;
    int ret = 0;
    if (!tfc || !refs)
        return AVERROR(EINVAL);

#if CONFIG_D3D11VA
    get_hwaccel_device_index(&device_idx, hwaccel_device);

    ret = create_d3d11va_devices_with_filter(refs, FFPROBE_HW_VENDOR_ID_INTEL, -1, NULL, device_idx);
    if (ret < 0)
        return ret;

    create_derive_qsv_devices_from_d3d11va(refs);

    if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_OPT_OCL)
        create_derive_opencl_devices_from_d3d11va(refs);

    print_dxgi_based_all(tfc, refs, hwaccel_flags, 0);
#elif (CONFIG_LIBDRM && CONFIG_VAAPI && HAVE_VAAPI_DRM)
    ret = create_drm_devices_with_filter(refs, FFPROBE_HW_VENDOR_ID_INTEL, hwaccel_device);
    if (ret < 0)
        return ret;

    create_derive_vaapi_devices_from_drm(refs);
    create_derive_qsv_devices_from_vaapi(refs);

    if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_OPT_OCL)
        create_derive_opencl_devices_from_vaapi(refs);

    print_drm_based_all(tfc, refs, hwaccel_flags);

    uninit_udev_functions();
#endif
    return ret;
}

static int show_amf_info(AVTextFormatContext *tfc, HwDeviceRefs *refs,
                         int hwaccel_flags, const char *hwaccel_device)
{
    av_unused int device_idx = -1;
    int ret = 0;
    if (!tfc || !refs)
        return AVERROR(EINVAL);

#if CONFIG_D3D11VA
    get_hwaccel_device_index(&device_idx, hwaccel_device);

    ret = create_d3d11va_devices_with_filter(refs, FFPROBE_HW_VENDOR_ID_AMD, -1, NULL, device_idx);
    if (ret < 0)
        return ret;

    if (hwaccel_flags & FFPROBE_HW_FLAG_PRINT_OPT_OCL)
        create_derive_opencl_devices_from_d3d11va(refs);

    print_dxgi_based_all(tfc, refs, (hwaccel_flags | FFPROBE_HW_FLAG_PRINT_SUB_DEV), 0);
#endif
    return ret;
}

void ffprobe_show_hwaccel_internal(AVTextFormatContext *tfc,
                                   enum AVHWDeviceType hwaccel_type,
                                   int hwaccel_flags,
                                   const char *hwaccel_device)
{
    HwDeviceRefs refs[FFPROBE_HW_MAX_DEV_NUM] = { 0 };

    if (!tfc)
        return;

    switch (hwaccel_type) {
    case AV_HWDEVICE_TYPE_VAAPI:
        show_vaapi_info(tfc, refs, hwaccel_flags, hwaccel_device);
        break;
    case AV_HWDEVICE_TYPE_D3D11VA:
        show_d3d11va_info(tfc, refs, hwaccel_flags, hwaccel_device);
        break;
    case AV_HWDEVICE_TYPE_CUDA:
        show_cuda_info(tfc, refs, hwaccel_flags, hwaccel_device);
        break;
    case AV_HWDEVICE_TYPE_QSV:
        show_qsv_info(tfc, refs, hwaccel_flags, hwaccel_device);
        break;
    case AV_HWDEVICE_TYPE_AMF:
        show_amf_info(tfc, refs, hwaccel_flags, hwaccel_device);
        break;
    default:
        return;
    }

    for (unsigned i = 0; i < FF_ARRAY_ELEMS(refs); i++) {
        if (refs[i].drm_ref)
            av_buffer_unref(&refs[i].drm_ref);
        if (refs[i].vaapi_ref)
            av_buffer_unref(&refs[i].vaapi_ref);
        if (refs[i].d3d11va_ref)
            av_buffer_unref(&refs[i].d3d11va_ref);
        if (refs[i].cuda_ref)
            av_buffer_unref(&refs[i].cuda_ref);
        if (refs[i].qsv_ref)
            av_buffer_unref(&refs[i].qsv_ref);
        if (refs[i].opencl_ref)
            av_buffer_unref(&refs[i].opencl_ref);
        if (refs[i].vulkan_ref)
            av_buffer_unref(&refs[i].vulkan_ref);
    }
}
