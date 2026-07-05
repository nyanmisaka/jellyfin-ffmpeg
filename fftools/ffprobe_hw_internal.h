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

#ifndef FFTOOLS_FFPROBE_HW_INTERNAL_H
#define FFTOOLS_FFPROBE_HW_INTERNAL_H

#include "config.h"
#include "config_components.h"

#include "textformat/avtextformat.h"
#include "libavcodec/avcodec.h"
#include "libavutil/hwcontext.h"
#include "libavutil/pixdesc.h"

#include "ffprobe.h"
#include "ffprobe_hw.h"

#define print_int(k, v) avtext_print_integer(tfc, k, v, 0)
#define print_str(k, v) avtext_print_string(tfc, k, v, 0)

#define FFPROBE_HW_MAX_DEV_NUM 16

#define FFPROBE_HW_VENDOR_ID_AMD        0x1002
#define FFPROBE_HW_VENDOR_ID_INTEL      0x8086
#define FFPROBE_HW_VENDOR_ID_NVIDIA     0x10de
#define FFPROBE_HW_VENDOR_ID_MICROSOFT  0x1414

typedef struct HwRes {
    const char    *name;
    const unsigned width;
    const unsigned height;
} HwRes;

static const HwRes hw_res_ascend[] = {
    { "16x16",     16,   16   },
    { "64x64",     64,   64   },
    { "128x128",   128,  128  },
    { "144x144",   144,  144  },
    { "256x256",   256,  256  },
    { "720x480",   720,  480  },
    { "1280x720",  1280, 720  },
    { "2048x1024", 2048, 1024 },
    { "1920x1080", 1920, 1080 },
    { "1920x1088", 1920, 1088 },
    { "2560x1440", 2560, 1440 },
    { "2048x2048", 2048, 2048 },
    { "3840x2160", 3840, 2160 },
    { "4096x2160", 4096, 2160 },
    { "4096x2304", 4096, 2304 },
    { "4096x2318", 4096, 2318 },
    { "3840x3840", 3840, 3840 },
    { "4080x4080", 4080, 4080 },
    { "4096x4096", 4096, 4096 },
    { "7680x4320", 7680, 4320 },
    { "8192x4320", 8192, 4320 },
    { "8192x4352", 8192, 4352 },
    { "8192x8192", 8192, 8192 },
    { NULL,        0,    0    },
};

typedef struct HwDeviceRefs {
    AVBufferRef *drm_ref;
    char         device_path_drm[4096]; /* PATH_MAX */

    AVBufferRef *vaapi_ref;

    AVBufferRef *d3d11va_ref;
    int          device_index_dxgi;
    uint64_t     device_luid_dxgi;

    AVBufferRef *cuda_ref;
    int          device_index_cuda;

    AVBufferRef *qsv_ref;
    AVBufferRef *opencl_ref;
    AVBufferRef *vulkan_ref;

    int          device_vendor_id;
    wchar_t      amf_dll_path[260*2]; /* MAX_PATH*2 */
} HwDeviceRefs;

/* DRM */
int init_udev_functions(void);
void uninit_udev_functions(void);
int create_drm_devices(HwDeviceRefs *refs, const char *device_path);
int create_drm_devices_with_filter(HwDeviceRefs *refs, int vendor_id, const char *device_path);
void create_derive_vaapi_devices_from_drm(HwDeviceRefs *refs);
void create_derive_vulkan_devices_from_drm(HwDeviceRefs *refs);
int print_drm_device_info(AVTextFormatContext *tfc, AVBufferRef *drm_ref);

/* VAAPI */
void create_derive_qsv_devices_from_vaapi(HwDeviceRefs *refs);
void create_derive_opencl_devices_from_vaapi(HwDeviceRefs *refs);
int print_vaapi_device_info(AVTextFormatContext *tfc, AVBufferRef *vaapi_ref, AVBufferRef *drm_ref);
int print_vaapi_decoder_info(AVTextFormatContext *tfc, AVBufferRef *vaapi_ref);
int print_vaapi_encoder_info(AVTextFormatContext *tfc, AVBufferRef *vaapi_ref);
int print_vaapi_vpp_info(AVTextFormatContext *tfc, AVBufferRef *vaapi_ref);

/* DXGI/D3D11VA */
int create_d3d11va_devices(HwDeviceRefs *refs, int device_idx);
int create_d3d11va_devices_with_filter(HwDeviceRefs *refs, int vendor_id, int idx_luid, const char *luid, int device_idx);
void create_derive_qsv_devices_from_d3d11va(HwDeviceRefs *refs);
void create_derive_opencl_devices_from_d3d11va(HwDeviceRefs *refs);
int print_d3d11va_device_info(AVTextFormatContext *tfc, AVBufferRef *d3d11va_ref);
int print_d3d11va_device_util(AVTextFormatContext *tfc, AVBufferRef *d3d11va_ref);
int print_d3d11va_decoder_info(AVTextFormatContext *tfc, AVBufferRef *d3d11va_ref);

/* CUDA */
int init_cuda_functions(void);
void uninit_cuda_functions(void);
int init_nvml_functions(void);
void uninit_nvml_functions(void);
int init_cuvid_functions(void);
void uninit_cuvid_functions(void);
int init_nvenc_functions(void);
void uninit_nvenc_functions(void);
int init_nvml_driver_version(void);
int create_cuda_devices(HwDeviceRefs *refs, int device_idx);
void create_derive_d3d11va_devices_from_cuda(HwDeviceRefs *refs);
int print_cuda_device_info(AVTextFormatContext *tfc, AVBufferRef *cuda_ref, int nvml_ret);
int print_cuda_device_util(AVTextFormatContext *tfc, AVBufferRef *cuda_ref, int nvml_ret);
int print_cuda_decoder_info(AVTextFormatContext *tfc, AVBufferRef *cuda_ref);
int print_cuda_encoder_info(AVTextFormatContext *tfc, AVBufferRef *cuda_ref);

/* QSV */
int print_qsv_device_info(AVTextFormatContext *tfc, AVBufferRef *qsv_ref);
int print_qsv_decoder_info(AVTextFormatContext *tfc, AVBufferRef *qsv_ref);
int print_qsv_encoder_info(AVTextFormatContext *tfc, AVBufferRef *qsv_ref);
int print_qsv_vpp_info(AVTextFormatContext *tfc, AVBufferRef *qsv_ref);

/* AMF */
int init_amf_functions(const wchar_t *amf_dll_path);
void uninit_amf_functions(void);
int create_derive_amf_device_from_d3d11va(AVBufferRef *d3d11va_ref, const wchar_t *amf_dll_path);
int print_amf_device_info_from_d3d11va(AVTextFormatContext *tfc);
int print_amf_encoder_info_from_d3d11va(AVTextFormatContext *tfc);

/* MF/MFT */
int init_mf_functions(void);
void uninit_mf_functions(void);
int create_derive_mf_device_from_d3d11va(AVBufferRef *d3d11va_ref);
int print_mf_encoder_info_from_d3d11va(AVTextFormatContext *tfc, uint64_t dxgi_luid);

/* OPENCL */
int print_opencl_device_info(AVTextFormatContext *tfc, AVBufferRef *opencl_ref);

#endif /* FFTOOLS_FFPROBE_HW_INTERNAL_H */
