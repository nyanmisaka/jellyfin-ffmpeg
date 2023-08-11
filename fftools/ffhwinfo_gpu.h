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
    HWINFO_ACCEL_TYPE_NV,
    HWINFO_ACCEL_TYPE_AMF,
};

/**
 * Print the hardware decoder infos of the base or derived devices.
 */
#define HWINFO_FLAG_PRINT_DECODER (1 << 0)

/**
 * Print the hardware encoder infos of the base or derived devices.
 */
#define HWINFO_FLAG_PRINT_ENCODER (1 << 1)

/**
 * Print the hardware filter infos of the base or derived devices.
 */
#define HWINFO_FLAG_PRINT_FILTER  (1 << 2)

#define HWINFO_DEFAULT_PRINT_FLAGS (HWINFO_FLAG_PRINT_DECODER | \
                                    HWINFO_FLAG_PRINT_ENCODER | \
                                    HWINFO_FLAG_PRINT_FILTER)

int show_accel_device_info(const WriterContext *wctx, enum HWInfoAccelType accel_type, int accel_flags);

#define MAX_HW_DEVICE_NUM 16

typedef struct HwDeviceRefs {
    AVBufferRef *drm_ref;
    AVBufferRef *vaapi_ref;

    AVBufferRef *d3d11va_ref;

    AVBufferRef *qsv_ref;
    AVBufferRef *opencl_ref;
    AVBufferRef *vulkan_ref;

    AVBufferRef *cuda_ref;

    int          device_index;
    char        *device_path;
} HwDeviceRefs;

int create_drm_devices(HwDeviceRefs *refs);
void create_derive_vaapi_devices_from_drm(HwDeviceRefs *refs);
void create_derive_vulkan_devices_from_drm(HwDeviceRefs *refs);
void create_derive_qsv_devices_from_vaapi(HwDeviceRefs *refs);
void create_derive_opencl_devices_from_vaapi(HwDeviceRefs *refs);

int create_d3d11va_devices(HwDeviceRefs *refs);
void create_derive_qsv_devices_from_d3d11va(HwDeviceRefs *refs);
void create_derive_opencl_devices_from_d3d11va(HwDeviceRefs *refs);

int create_cuda_devices(HwDeviceRefs *refs);

#endif /* FFTOOLS_FFHWINFO_GPU_H */
