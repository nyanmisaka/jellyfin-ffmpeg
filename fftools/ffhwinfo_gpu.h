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

#ifndef FFTOOLS_FFHWINFO_GPU_H
#define FFTOOLS_FFHWINFO_GPU_H

#include "ffhwinfo_utils.h"

#define print_int(k, v)  writer_print_integer(wctx, k, v)
#define print_str(k, v)  writer_print_string(wctx, k, v, 0)

#define HWINFO_MAX_DEV_NUM 16

#define HWINFO_VENDOR_ID_AMD     0x1002
#define HWINFO_VENDOR_ID_INTEL   0x8086
#define HWINFO_VENDOR_ID_NVIDIA  0x10de

/**
 * Print the hardware info of the devices.
 */
#define HWINFO_FLAG_PRINT_DEV                  (1 << 0)

/**
 * Print the hardware decoder info of the devices.
 */
#define HWINFO_FLAG_PRINT_DEC                  (1 << 1)

/**
 * Print the hardware encoder info of the devices.
 */
#define HWINFO_FLAG_PRINT_ENC                  (1 << 2)

/**
 * Print the hardware vpp info of the devices.
 */
#define HWINFO_FLAG_PRINT_VPP                  (1 << 3)

/**
 * Print the optional OPENCL info of the devices.
 */
#define HWINFO_FLAG_PRINT_OPT_OPENCL           (1 << 4)

/**
 * Print the optional VULKAN info of the devices.
 */
#define HWINFO_FLAG_PRINT_OPT_VULKAN           (1 << 5)

/**
 * Print the optional D3D11VA info of the devices.
 * This works only in CUDA to print the corresponding D3D11VA devices.
 */
#define HWINFO_FLAG_PRINT_OPT_D3D11VA          (1 << 6)

/**
 * Print the OS native hardware dev/enc/vpp info of the devices.
 * This works only in QSV to print its VAAPI and D3D11VA sub-devices,
 * and this is force enabled internally for both VAAPI and D3D11VA/AMF.
 */
#define HWINFO_FLAG_PRINT_OS_VA                (1 << 7)

#define HWINFO_DEFAULT_PRINT_FLAGS (HWINFO_FLAG_PRINT_DEV | \
                                    HWINFO_FLAG_PRINT_DEC | \
                                    HWINFO_FLAG_PRINT_ENC | \
                                    HWINFO_FLAG_PRINT_VPP | \
                                    HWINFO_FLAG_PRINT_OPT_OPENCL | \
                                    HWINFO_FLAG_PRINT_OPT_VULKAN)

enum HWInfoAccelType {
    HWINFO_ACCEL_TYPE_NONE,
    HWINFO_ACCEL_TYPE_VAAPI,
    HWINFO_ACCEL_TYPE_QSV,
    HWINFO_ACCEL_TYPE_CUDA,
    HWINFO_ACCEL_TYPE_AMF,
};

typedef struct HwDeviceRefs {
    AVBufferRef *drm_ref;
    char        *device_path_drm;

    AVBufferRef *vaapi_ref;

    AVBufferRef *d3d11va_ref;
    int          device_index_dxgi;

    AVBufferRef *qsv_ref;
    AVBufferRef *opencl_ref;
    AVBufferRef *vulkan_ref;

    AVBufferRef *cuda_ref;
    int          device_index_cuda;

    int          device_vendor_id;
} HwDeviceRefs;

int show_accel_device_info(WriterContext *wctx, enum HWInfoAccelType accel_type, int accel_flags);

int create_drm_devices(HwDeviceRefs *refs);
void create_derive_vaapi_devices_from_drm(HwDeviceRefs *refs);
void create_derive_vulkan_devices_from_drm(HwDeviceRefs *refs);
void create_derive_qsv_devices_from_vaapi(HwDeviceRefs *refs);
void create_derive_opencl_devices_from_vaapi(HwDeviceRefs *refs);

int create_d3d11va_devices(HwDeviceRefs *refs);
int create_d3d11va_devices_with_filter(HwDeviceRefs *refs, int vendor_id, int idx_refs, char *luid);
void create_derive_qsv_devices_from_d3d11va(HwDeviceRefs *refs);
void create_derive_opencl_devices_from_d3d11va(HwDeviceRefs *refs);
void create_derive_cuda_devices_from_d3d11va(HwDeviceRefs *refs);
int print_d3d11va_device_info(WriterContext *wctx, AVBufferRef *d3d11va_ref);
int print_d3d11va_decoder_info(WriterContext *wctx, AVBufferRef *d3d11va_ref);

int init_cuda_functions(void);
void uninit_cuda_functions(void);
int init_nvml_functions(void);
void uninit_nvml_functions(void);
int init_nvml_driver_version(void);
int create_cuda_devices(HwDeviceRefs *refs);
void create_derive_d3d11va_devices_from_cuda(HwDeviceRefs *refs);
int print_cuda_device_info(WriterContext *wctx, AVBufferRef *cuda_ref, int nvml_ret);
int print_cuda_decoder_info(WriterContext *wctx, AVBufferRef *cuda_ref);
int print_cuda_encoder_info(WriterContext *wctx, AVBufferRef *cuda_ref);

int print_qsv_device_info(WriterContext *wctx, AVBufferRef *qsv_ref);
int print_qsv_decoder_info(WriterContext *wctx, AVBufferRef *qsv_ref);
int print_qsv_encoder_info(WriterContext *wctx, AVBufferRef *qsv_ref);
int print_qsv_vpp_info(WriterContext *wctx, AVBufferRef *qsv_ref);

int init_amf_functions(void);
void uninit_amf_functions(void);
int create_derive_amf_device_from_d3d11va(AVBufferRef *d3d11va_ref);
int print_amf_device_info_from_d3d11va(WriterContext *wctx);
int print_amf_encoder_info_from_d3d11va(WriterContext *wctx);

#endif /* FFTOOLS_FFHWINFO_GPU_H */
