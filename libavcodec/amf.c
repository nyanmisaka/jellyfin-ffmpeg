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

#include "amf.h"

#define FFMPEG_AMF_WRITER_ID L"ffmpeg_amf"

const FormatMap format_map[] =
{
    { AV_PIX_FMT_NONE,       AMF_SURFACE_UNKNOWN },
    { AV_PIX_FMT_NV12,       AMF_SURFACE_NV12    },
    { AV_PIX_FMT_P010,       AMF_SURFACE_P010    },
    { AV_PIX_FMT_BGR0,       AMF_SURFACE_BGRA    },
    { AV_PIX_FMT_RGB0,       AMF_SURFACE_RGBA    },
    { AV_PIX_FMT_GRAY8,      AMF_SURFACE_GRAY8   },
    { AV_PIX_FMT_YUV420P,    AMF_SURFACE_YUV420P },
    { AV_PIX_FMT_YUYV422,    AMF_SURFACE_YUY2    },
};

enum AMF_SURFACE_FORMAT amf_av_to_amf_format(enum AVPixelFormat fmt)
{
    int i;
    for (i = 0; i < amf_countof(format_map); i++) {
        if (format_map[i].av_format == fmt) {
            return format_map[i].amf_format;
        }
    }
    return AMF_SURFACE_UNKNOWN;
}

enum AVPixelFormat amf_to_av_format(enum AMF_SURFACE_FORMAT fmt)
{
    int i;
    for (i = 0; i < amf_countof(format_map); i++) {
        if (format_map[i].amf_format == fmt) {
            return format_map[i].av_format;
        }
    }
    return AMF_SURFACE_UNKNOWN;
}

const ColorTransferMap color_trc_map[] =
{
    { AVCOL_TRC_RESERVED0,       AMF_COLOR_TRANSFER_CHARACTERISTIC_UNDEFINED    },
    { AVCOL_TRC_BT709,           AMF_COLOR_TRANSFER_CHARACTERISTIC_BT709        },
    { AVCOL_TRC_UNSPECIFIED,     AMF_COLOR_TRANSFER_CHARACTERISTIC_UNSPECIFIED  },
    { AVCOL_TRC_RESERVED,        AMF_COLOR_TRANSFER_CHARACTERISTIC_RESERVED     },
    { AVCOL_TRC_GAMMA22,         AMF_COLOR_TRANSFER_CHARACTERISTIC_GAMMA22      },
    { AVCOL_TRC_GAMMA28,         AMF_COLOR_TRANSFER_CHARACTERISTIC_GAMMA28      },
    { AVCOL_TRC_SMPTE170M,       AMF_COLOR_TRANSFER_CHARACTERISTIC_SMPTE170M    },
    { AVCOL_TRC_SMPTE240M,       AMF_COLOR_TRANSFER_CHARACTERISTIC_SMPTE240M    },
    { AVCOL_TRC_LINEAR,          AMF_COLOR_TRANSFER_CHARACTERISTIC_LINEAR       },
    { AVCOL_TRC_LOG,             AMF_COLOR_TRANSFER_CHARACTERISTIC_LOG          },
    { AVCOL_TRC_LOG_SQRT,        AMF_COLOR_TRANSFER_CHARACTERISTIC_LOG_SQRT     },
    { AVCOL_TRC_IEC61966_2_4,    AMF_COLOR_TRANSFER_CHARACTERISTIC_IEC61966_2_4 },
    { AVCOL_TRC_BT1361_ECG,      AMF_COLOR_TRANSFER_CHARACTERISTIC_BT1361_ECG   },
    { AVCOL_TRC_IEC61966_2_1,    AMF_COLOR_TRANSFER_CHARACTERISTIC_IEC61966_2_1 },
    { AVCOL_TRC_BT2020_10,       AMF_COLOR_TRANSFER_CHARACTERISTIC_BT2020_10    },
    { AVCOL_TRC_BT2020_12,       AMF_COLOR_TRANSFER_CHARACTERISTIC_BT2020_12    },
    { AVCOL_TRC_SMPTE2084,       AMF_COLOR_TRANSFER_CHARACTERISTIC_SMPTE2084    },
    { AVCOL_TRC_SMPTE428,        AMF_COLOR_TRANSFER_CHARACTERISTIC_SMPTE428     },
    { AVCOL_TRC_ARIB_STD_B67,    AMF_COLOR_TRANSFER_CHARACTERISTIC_ARIB_STD_B67 },
};

enum AMF_COLOR_TRANSFER_CHARACTERISTIC_ENUM amf_av_to_amf_color_trc(enum AVColorTransferCharacteristic trc)
{
    int i;
    for (i = 0; i < amf_countof(color_trc_map); i++) {
        if (color_trc_map[i].av_color_trc == trc) {
            return color_trc_map[i].amf_color_trc;
        }
    }
    return AMF_COLOR_TRANSFER_CHARACTERISTIC_UNDEFINED;
}

const ColorPrimariesMap color_prm_map[] =
{
    { AVCOL_PRI_RESERVED0,      AMF_COLOR_PRIMARIES_UNDEFINED   },
    { AVCOL_PRI_BT709,          AMF_COLOR_PRIMARIES_BT709       },
    { AVCOL_PRI_UNSPECIFIED,    AMF_COLOR_PRIMARIES_UNSPECIFIED },
    { AVCOL_PRI_RESERVED,       AMF_COLOR_PRIMARIES_RESERVED    },
    { AVCOL_PRI_BT470M,         AMF_COLOR_PRIMARIES_BT470M      },
    { AVCOL_PRI_BT470BG,        AMF_COLOR_PRIMARIES_BT470BG     },
    { AVCOL_PRI_SMPTE170M,      AMF_COLOR_PRIMARIES_SMPTE170M   },
    { AVCOL_PRI_SMPTE240M,      AMF_COLOR_PRIMARIES_SMPTE240M   },
    { AVCOL_PRI_FILM,           AMF_COLOR_PRIMARIES_FILM        },
    { AVCOL_PRI_BT2020,         AMF_COLOR_PRIMARIES_BT2020      },
    { AVCOL_PRI_SMPTE428,       AMF_COLOR_PRIMARIES_SMPTE428    },
    { AVCOL_PRI_SMPTE431,       AMF_COLOR_PRIMARIES_SMPTE431    },
    { AVCOL_PRI_SMPTE432,       AMF_COLOR_PRIMARIES_SMPTE432    },
    { AVCOL_PRI_JEDEC_P22,      AMF_COLOR_PRIMARIES_JEDEC_P22   },
};

enum AMF_COLOR_PRIMARIES_ENUM amf_av_to_amf_color_prm(enum AVColorPrimaries prm)
{
    int i;
    for (i = 0; i < amf_countof(color_prm_map); i++) {
        if (color_prm_map[i].av_color_prm == prm) {
            return color_prm_map[i].amf_color_prm;
        }
    }
    return AMF_COLOR_PRIMARIES_UNDEFINED;
}

static void AMF_CDECL_CALL AMFTraceWriter_Write(AMFTraceWriter *pThis,
                                                const wchar_t *scope, const wchar_t *message)
{
    AVAMFLogger *logger = (AVAMFLogger*)pThis;
    av_log(logger->avcl, AV_LOG_DEBUG, "%ls: %ls", scope, message);
}

static void AMF_CDECL_CALL AMFTraceWriter_Flush(AMFTraceWriter *pThis) {}

static AMFTraceWriterVtbl tracer_vtbl =
{
    .Write = AMFTraceWriter_Write,
    .Flush = AMFTraceWriter_Flush,
};

int amf_load_library(AVAMFContext *ctx)
{
    AMFInit_Fn         init_fun;
    AMFQueryVersion_Fn version_fun;
    AMF_RESULT         res;

    ctx->library = dlopen(AMF_DLL_NAMEA, RTLD_NOW | RTLD_LOCAL);
    AMF_RETURN_IF_FALSE(ctx->avclass, ctx->library != NULL,
        AVERROR_UNKNOWN, "DLL %s failed to open\n", AMF_DLL_NAMEA);

    init_fun = (AMFInit_Fn)dlsym(ctx->library, AMF_INIT_FUNCTION_NAME);
    AMF_RETURN_IF_FALSE(ctx->avclass, init_fun != NULL,
        AVERROR_UNKNOWN, "DLL %s failed to find function %s\n", AMF_DLL_NAMEA, AMF_INIT_FUNCTION_NAME);

    version_fun = (AMFQueryVersion_Fn)dlsym(ctx->library, AMF_QUERY_VERSION_FUNCTION_NAME);
    AMF_RETURN_IF_FALSE(ctx->avclass, version_fun != NULL,
        AVERROR_UNKNOWN, "DLL %s failed to find function %s\n", AMF_DLL_NAMEA, AMF_QUERY_VERSION_FUNCTION_NAME);

    res = version_fun(&ctx->version);
    AMF_RETURN_IF_FALSE(ctx->avclass, res == AMF_OK,
        AVERROR_UNKNOWN, "%s failed with error %d\n", AMF_QUERY_VERSION_FUNCTION_NAME, res);

    res = init_fun(AMF_FULL_VERSION, &ctx->factory);
    AMF_RETURN_IF_FALSE(ctx->avclass, res == AMF_OK,
        AVERROR_UNKNOWN, "%s failed with error %d\n", AMF_INIT_FUNCTION_NAME, res);

    res = ctx->factory->pVtbl->GetTrace(ctx->factory, &ctx->trace);
    AMF_RETURN_IF_FALSE(ctx->avclass, res == AMF_OK,
        AVERROR_UNKNOWN, "GetTrace() failed with error %d\n", res);

    res = ctx->factory->pVtbl->GetDebug(ctx->factory, &ctx->debug);
    AMF_RETURN_IF_FALSE(ctx->avclass, res == AMF_OK,
        AVERROR_UNKNOWN, "GetDebug() failed with error %d\n", res);

    return 0;
}

int amf_create_context(AVAMFContext *ctx)
{
    AMF_RESULT res;

    // configure AMF logger
    ctx->trace->pVtbl->EnableWriter(ctx->trace, AMF_TRACE_WRITER_DEBUG_OUTPUT, !!ctx->log_to_dbg);
    if (ctx->log_to_dbg)
        ctx->trace->pVtbl->SetWriterLevel(ctx->trace, AMF_TRACE_WRITER_DEBUG_OUTPUT, AMF_TRACE_TRACE);
    ctx->trace->pVtbl->EnableWriter(ctx->trace, AMF_TRACE_WRITER_CONSOLE, 0);
    ctx->trace->pVtbl->SetGlobalLevel(ctx->trace, AMF_TRACE_TRACE);

    // connect AMF logger to av_log
    ctx->logger.vtbl = &tracer_vtbl;
    ctx->logger.avcl = ctx->avclass;
    ctx->trace->pVtbl->RegisterWriter(ctx->trace, FFMPEG_AMF_WRITER_ID, (AMFTraceWriter*)&ctx->logger, 1);
    ctx->trace->pVtbl->SetWriterLevel(ctx->trace, FFMPEG_AMF_WRITER_ID, AMF_TRACE_TRACE);

    res = ctx->factory->pVtbl->CreateContext(ctx->factory, &ctx->context);
    AMF_RETURN_IF_FALSE(ctx->avclass, res == AMF_OK,
        AVERROR_UNKNOWN, "CreateContext() failed with error %d\n", res);

    return 0;
}

void amf_unload_library(AVAMFContext *ctx)
{
    if (ctx->context) {
        ctx->context->pVtbl->Terminate(ctx->context);
        ctx->context->pVtbl->Release(ctx->context);
        ctx->context = NULL;
    }
    if (ctx->trace) {
        ctx->trace->pVtbl->UnregisterWriter(ctx->trace, FFMPEG_AMF_WRITER_ID);
    }
    if (ctx->library) {
        dlclose(ctx->library);
        ctx->library = NULL;
    }
    ctx->trace = NULL;
    ctx->debug = NULL;
    ctx->factory = NULL;
    ctx->version = 0;
}

int amf_context_init_dx11(AVAMFContext *ctx)
{
    AMF_RESULT res;

    res = ctx->context->pVtbl->InitDX11(ctx->context, NULL, AMF_DX11_1);
    if (res != AMF_OK) {
        res = ctx->context->pVtbl->InitDX11(ctx->context, NULL, AMF_DX11_0);
    }

    if (res == AMF_OK) {
        av_log(ctx->avclass, AV_LOG_VERBOSE, "AMF initialization succeeded via DX11\n");
    } else {
        if (res == AMF_NOT_SUPPORTED)
            av_log(ctx->avclass, AV_LOG_ERROR, "AMF via DX11 is not supported on the given device\n");
        else
            av_log(ctx->avclass, AV_LOG_ERROR, "AMF failed to initialize on the default DX11 device: %d\n", res);
    }
    return res;
}

int amf_context_init_dx9(AVAMFContext *ctx)
{
    AMF_RESULT res;

    res = ctx->context->pVtbl->InitDX9(ctx->context, NULL);
    if (res == AMF_OK) {
        av_log(ctx->avclass, AV_LOG_VERBOSE, "AMF initialization succeeded via DX9\n");
    } else {
        if (res == AMF_NOT_SUPPORTED)
            av_log(ctx->avclass, AV_LOG_ERROR, "AMF via DX9 is not supported on the given device\n");
        else
            av_log(ctx->avclass, AV_LOG_ERROR, "AMF failed to initialize on the default DX9 device: %d\n", res);
    }
    return res;
}

int amf_context_init_vulkan(AVAMFContext *ctx)
{
    AMF_RESULT res;
    AMFContext1* context1 = NULL;
    AMFGuid guid = IID_AMFContext1();

    res = ctx->context->pVtbl->QueryInterface(ctx->context, &guid, (void**)&context1);
    AMF_RETURN_IF_FALSE(ctx->avclass, res == AMF_OK, AVERROR_UNKNOWN, "CreateContext1() failed with error %d\n", res);

    res = context1->pVtbl->InitVulkan(context1, NULL);
    context1->pVtbl->Release(context1);
    if (res == AMF_OK) {
        av_log(ctx->avclass, AV_LOG_VERBOSE, "AMF initialization succeeded via Vulkan\n");
    } else {
        if (res == AMF_NOT_SUPPORTED)
            av_log(ctx->avclass, AV_LOG_ERROR, "AMF via Vulkan is not supported on the given device\n");
        else
            av_log(ctx->avclass, AV_LOG_ERROR, "AMF failed to initialize on the default Vulkan device: %d\n", res);
    }
    return res;
}

int amf_context_init_opencl(AVAMFContext *ctx)
{
    AMF_RESULT res;

    res = ctx->context->pVtbl->InitOpenCL(ctx->context, NULL);
    if (res == AMF_OK) {
        av_log(ctx->avclass, AV_LOG_VERBOSE, "AMF initialization succeeded via OpenCL\n");
    } else {
        if (res == AMF_NOT_SUPPORTED)
            av_log(ctx->avclass, AV_LOG_ERROR, "AMF via OpenCL is not supported on the given device\n");
        else
            av_log(ctx->avclass, AV_LOG_ERROR, "AMF failed to initialize on the default OpenCL device: %d\n", res);
    }
    return res;
}

#if CONFIG_D3D11VA
int amf_context_derive_dx11(AVAMFContext *ctx, AVD3D11VADeviceContext *hwctx)
{
    AMF_RESULT res;

    res = ctx->context->pVtbl->InitDX11(ctx->context, hwctx->device, AMF_DX11_1);
    if (res != AMF_OK) {
        res = ctx->context->pVtbl->InitDX11(ctx->context, hwctx->device, AMF_DX11_0);
    }

    if (res == AMF_OK) {
        av_log(ctx->avclass, AV_LOG_VERBOSE, "AMF derived succeeded via DX11\n");
    } else {
        if (res == AMF_NOT_SUPPORTED)
            av_log(ctx->avclass, AV_LOG_ERROR, "AMF via DX11 is not supported on the given device\n");
        else
            av_log(ctx->avclass, AV_LOG_ERROR, "AMF failed to derive from the given DX11 device: %d\n", res);
        return AVERROR(ENODEV);
    }
    return res;
}
#endif

#if CONFIG_DXVA2
int amf_context_derive_dx9(AVAMFContext *ctx, AVDXVA2DeviceContext *hwctx)
{
    AMF_RESULT res;
    HRESULT hr;
    HANDLE device_handle;
    IDirect3DDevice9* device;

    hr = IDirect3DDeviceManager9_OpenDeviceHandle(hwctx->devmgr, &device_handle);
    if (FAILED(hr)) {
        av_log(ctx->avclass, AV_LOG_ERROR, "Failed to open device handle for DX9 device: %lx\n", (unsigned long)hr);
        return AVERROR_EXTERNAL;
    }

    hr = IDirect3DDeviceManager9_LockDevice(hwctx->devmgr, device_handle, &device, FALSE);
    if (SUCCEEDED(hr)) {
        IDirect3DDeviceManager9_UnlockDevice(hwctx->devmgr, device_handle, FALSE);
    } else {
        av_log(ctx->avclass, AV_LOG_ERROR, "Failed to lock device handle for DX9 device: %lx\n", (unsigned long)hr);
        return AVERROR_EXTERNAL;
    }

    IDirect3DDeviceManager9_CloseDeviceHandle(hwctx->devmgr, device_handle);

    res = ctx->context->pVtbl->InitDX9(ctx->context, device);

    IDirect3DDevice9_Release(device);

    if (res == AMF_OK) {
        av_log(ctx->avclass, AV_LOG_VERBOSE, "AMF derived succeeded via DX9\n");
    } else {
        if (res == AMF_NOT_SUPPORTED)
            av_log(ctx->avclass, AV_LOG_ERROR, "AMF via DX9 is not supported on the given device\n");
        else
            av_log(ctx->avclass, AV_LOG_ERROR, "AMF failed to derive from the given DX9 device: %d\n", res);
        return AVERROR(ENODEV);
    }
    return res;
}
#endif

#if CONFIG_OPENCL
int amf_context_derive_opencl(AVAMFContext *ctx, AVOpenCLDeviceContext *hwctx)
{
    AMF_RESULT res;

    res = ctx->context->pVtbl->InitOpenCL(ctx->context, hwctx->command_queue);
    if (res == AMF_OK) {
        av_log(ctx->avclass, AV_LOG_VERBOSE, "AMF derived succeeded via OpenCL\n");
    } else {
        if (res == AMF_NOT_SUPPORTED)
            av_log(ctx->avclass, AV_LOG_ERROR, "AMF via OpenCL is not supported on the given device\n");
        else
            av_log(ctx->avclass, AV_LOG_ERROR, "AMF failed to derive from the given OpenCL device: %d\n", res);
        return AVERROR(ENODEV);
    }
    return res;
}
#endif
