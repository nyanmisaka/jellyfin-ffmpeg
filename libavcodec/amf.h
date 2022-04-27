/*
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

#ifndef AVCODEC_AMF_H
#define AVCODEC_AMF_H

#include <AMF/core/Factory.h>
#include <AMF/core/Surface.h>
#include <AMF/components/ColorSpace.h>

#include "config.h"
#include "avcodec.h"

#include "libavutil/pixdesc.h"

#if CONFIG_D3D11VA
#include "libavutil/hwcontext_d3d11va.h"
#endif

#if CONFIG_DXVA2
#define COBJMACROS
#include "libavutil/hwcontext_dxva2.h"
#endif

#if CONFIG_OPENCL
#include "libavutil/hwcontext_opencl.h"
#endif

#ifdef _WIN32
#include "compat/w32dlfcn.h"
#else
#include <dlfcn.h>
#endif

/**
* Error handling helper
*/
#define AMF_RETURN_IF_FALSE(avctx, exp, ret_value, /*message,*/ ...) \
    if (!(exp)) { \
        av_log(avctx, AV_LOG_ERROR, __VA_ARGS__); \
        return ret_value; \
    }

#define AMF_GOTO_FAIL_IF_FALSE(avctx, exp, ret_value, /*message,*/ ...) \
    if (!(exp)) { \
        av_log(avctx, AV_LOG_ERROR, __VA_ARGS__); \
        ret = ret_value; \
        goto fail; \
    }

/**
* AMF trace writer callback class
* Used to capture all AMF logging
*/
typedef struct AVAMFLogger {
    AMFTraceWriterVtbl *vtbl;
    void               *avcl;
} AVAMFLogger;

typedef struct AVAMFContext {
    void               *avclass;
    int                 log_to_dbg;

    // access to AMF runtime
    amf_handle          library; ///< handle to DLL library
    AMFFactory         *factory; ///< pointer to AMF factory
    AMFDebug           *debug;   ///< pointer to AMF debug interface
    AMFTrace           *trace;   ///< pointer to AMF trace interface

    amf_uint64          version; ///< version of AMF runtime
    AVAMFLogger         logger;  ///< AMF writer registered with AMF
    AMFContext         *context; ///< AMF context
} AVAMFContext;

/**
* Surface/Pixel format
*/
typedef struct FormatMap {
    enum AVPixelFormat      av_format;
    enum AMF_SURFACE_FORMAT amf_format;
} FormatMap;

extern const FormatMap format_map[];
enum AMF_SURFACE_FORMAT amf_av_to_amf_format(enum AVPixelFormat fmt);
enum AVPixelFormat amf_to_av_format(enum AMF_SURFACE_FORMAT fmt);

/**
* Color Transfer
*/
typedef struct ColorTransferMap {
    enum AVColorTransferCharacteristic          av_color_trc;
    enum AMF_COLOR_TRANSFER_CHARACTERISTIC_ENUM amf_color_trc;
} ColorTransferMap;

extern const ColorTransferMap color_trc_map[];
enum AMF_COLOR_TRANSFER_CHARACTERISTIC_ENUM amf_av_to_amf_color_trc(enum AVColorTransferCharacteristic trc);
enum AVColorTransferCharacteristic amf_to_av_color_trc(enum AMF_COLOR_TRANSFER_CHARACTERISTIC_ENUM trc);

/**
* Color Primaries
*/
typedef struct ColorPrimariesMap {
    enum AVColorPrimaries         av_color_prm;
    enum AMF_COLOR_PRIMARIES_ENUM amf_color_prm;
} ColorPrimariesMap;

extern const ColorPrimariesMap color_prm_map[];
enum AMF_COLOR_PRIMARIES_ENUM amf_av_to_amf_color_prm(enum AVColorPrimaries prm);
enum AVColorPrimaries amf_to_av_color_prm(enum AMF_COLOR_PRIMARIES_ENUM prm);

/**
* Load AMFContext
*/
int amf_load_library(AVAMFContext *ctx);
int amf_create_context(AVAMFContext *ctx);
void amf_unload_library(AVAMFContext *ctx);

/**
* Init AMFContext standalone
*/
int amf_context_init_dx11(AVAMFContext *ctx);
int amf_context_init_dx9(AVAMFContext *ctx);
int amf_context_init_vulkan(AVAMFContext *ctx);
int amf_context_init_opencl(AVAMFContext *ctx);

/**
* Derive AMFContext from builtin hwcontext
*/
#if CONFIG_D3D11VA
int amf_context_derive_dx11(AVAMFContext *ctx, AVD3D11VADeviceContext *hwctx);
#endif

#if CONFIG_DXVA2
int amf_context_derive_dx9(AVAMFContext *ctx, AVDXVA2DeviceContext *hwctx);
#endif

#if CONFIG_OPENCL
int amf_context_derive_opencl(AVAMFContext *ctx, AVOpenCLDeviceContext *hwctx);
#endif

#endif /* AVCODEC_AMF_H */
