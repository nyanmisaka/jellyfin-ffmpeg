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

enum HWInfoAccelType {
    HWINFO_ACCEL_TYPE_NONE,
    HWINFO_ACCEL_TYPE_VAAPI,
    HWINFO_ACCEL_TYPE_QSV,
    HWINFO_ACCEL_TYPE_CUDA,
    HWINFO_ACCEL_TYPE_AMF,
};

/**
 * Print the hardware device info of the base or derived devices.
 */
#define HWINFO_FLAG_PRINT_DEV                  (1 << 0)

/**
 * Print the hardware decoder info of the base or derived devices.
 */
#define HWINFO_FLAG_PRINT_DEC                  (1 << 1)

/**
 * Print the hardware encoder info of the base or derived devices.
 */
#define HWINFO_FLAG_PRINT_ENC                  (1 << 2)

/**
 * Print the hardware vpp info of the base or derived devices.
 */
#define HWINFO_FLAG_PRINT_VPP                  (1 << 3)

/**
 * Print the OPENCL compute device info of the base or derived devices.
 */
#define HWINFO_FLAG_PRINT_COMPUTE_OPENCL       (1 << 4)

/**
 * Print the VULKAN compute device info of the base or derived devices.
 */
#define HWINFO_FLAG_PRINT_COMPUTE_VULKAN       (1 << 5)

/**
 * Print the OS native hardware dev/enc/vpp info of the base or derived devices.
 */
#define HWINFO_FLAG_PRINT_OS_VA                (1 << 6)

#define HWINFO_DEFAULT_PRINT_FLAGS (HWINFO_FLAG_PRINT_DEV | \
                                    HWINFO_FLAG_PRINT_DEC | \
                                    HWINFO_FLAG_PRINT_ENC | \
                                    HWINFO_FLAG_PRINT_VPP | \
                                    HWINFO_FLAG_PRINT_COMPUTE_OPENCL | \
                                    HWINFO_FLAG_PRINT_COMPUTE_VULKAN)

int show_accel_device_info(WriterContext *wctx, enum HWInfoAccelType accel_type, int accel_flags);

#define MAX_HW_DEVICE_NUM 16

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

int create_drm_devices(HwDeviceRefs *refs);
void create_derive_vaapi_devices_from_drm(HwDeviceRefs *refs);
void create_derive_vulkan_devices_from_drm(HwDeviceRefs *refs);
void create_derive_qsv_devices_from_vaapi(HwDeviceRefs *refs);
void create_derive_opencl_devices_from_vaapi(HwDeviceRefs *refs);
int print_drm_based_all(WriterContext *wctx, HwDeviceRefs *refs, int accel_flags);

int create_d3d11va_devices(HwDeviceRefs *refs);
int create_d3d11va_devices_with_filter(HwDeviceRefs *refs, int vendor_id, int idx_refs, char *luid);
void create_derive_qsv_devices_from_d3d11va(HwDeviceRefs *refs);
void create_derive_opencl_devices_from_d3d11va(HwDeviceRefs *refs);
void create_derive_cuda_devices_from_d3d11va(HwDeviceRefs *refs);
int print_dxgi_based_all(WriterContext *wctx, HwDeviceRefs *refs, int accel_flags);

int init_cuda_functions(void);
void uninit_cuda_functions(void);
int init_nvml_functions(void);
void uninit_nvml_functions(void);
int create_cuda_devices(HwDeviceRefs *refs);
void create_derive_d3d11va_devices_from_cuda(HwDeviceRefs *refs);
int print_cuda_based_all(WriterContext *wctx, HwDeviceRefs *refs, int accel_flags);

#define print_int(k, v)  writer_print_integer(wctx, k, v)
#define print_str(k, v)  writer_print_string(wctx, k, v, 0)

#define HWINFO_VENDOR_ID_AMD     0x1002
#define HWINFO_VENDOR_ID_INTEL   0x8086
#define HWINFO_VENDOR_ID_NVIDIA  0x10de

#endif /* FFTOOLS_FFHWINFO_GPU_H */
