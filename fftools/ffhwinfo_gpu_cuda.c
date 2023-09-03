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

#if CONFIG_CUDA
#   define CHECK_CU(x) FF_CUDA_CHECK_DL(NULL, cu, x)
#   define CHECK_ML(x) FF_NVML_CHECK_DL(NULL, nvml_ext, x)
#   include "libavutil/cuda_check.h"
#   include "libavutil/hwcontext_cuda_internal.h"
#endif

#if (CONFIG_CUDA && CONFIG_NVENC)
#define NVENCAPI_CHECK_VERSION(major, minor) \
    ((major) < NVENCAPI_MAJOR_VERSION || ((major) == NVENCAPI_MAJOR_VERSION && (minor) <= NVENCAPI_MINOR_VERSION))
// SDK 8.1 compile time feature checks
#if NVENCAPI_CHECK_VERSION(8, 1)
#define NVENC_HAVE_BFRAME_REF_MODE
#define NVENC_HAVE_QP_MAP_MODE
#endif

// SDK 9.0 compile time feature checks
#if NVENCAPI_CHECK_VERSION(9, 0)
#define NVENC_HAVE_HEVC_BFRAME_REF_MODE
#endif

// SDK 9.1 compile time feature checks
#if NVENCAPI_CHECK_VERSION(9, 1)
#define NVENC_HAVE_MULTIPLE_REF_FRAMES
#define NVENC_HAVE_CUSTREAM_PTR
#define NVENC_HAVE_GETLASTERRORSTRING
#endif

// SDK 10.0 compile time feature checks
#if NVENCAPI_CHECK_VERSION(10, 0)
#define NVENC_HAVE_NEW_PRESETS
#define NVENC_HAVE_MULTIPASS
#define NVENC_HAVE_LDKFS
#define NVENC_HAVE_H264_LVL6
#define NVENC_HAVE_HEVC_CONSTRAINED_ENCODING
#endif

// SDK 11.1 compile time feature checks
#if NVENCAPI_CHECK_VERSION(11, 1)
#define NVENC_HAVE_QP_CHROMA_OFFSETS
#define NVENC_HAVE_SINGLE_SLICE_INTRA_REFRESH
#endif

// SDK 12.1 compile time feature checks
#if NVENCAPI_CHECK_VERSION(12, 1)
#define NVENC_NO_DEPRECATED_RC
#endif

static NvencFunctions *nvenc = NULL;
static NV_ENCODE_API_FUNCTION_LIST nvenc_fns;
typedef struct NvencMode {
    const char    *name;
    enum AVCodecID codec;
    const int     *profiles;
    const int     *formats;
} NvencMode;

static const int enc_profiles_h264[] = { FF_PROFILE_H264_BASELINE,
                                         FF_PROFILE_H264_MAIN,
                                         FF_PROFILE_H264_HIGH,
                                         FF_PROFILE_H264_HIGH_444,
                                         FF_PROFILE_UNKNOWN };
static const int enc_profiles_hevc[] = { FF_PROFILE_HEVC_MAIN,
                                         FF_PROFILE_HEVC_MAIN_10,
                                         FF_PROFILE_HEVC_REXT,
                                         FF_PROFILE_UNKNOWN };
static const int enc_profiles_av1[]  = { FF_PROFILE_AV1_MAIN,
                                         FF_PROFILE_UNKNOWN };
#define ENC_FMTS_8_YUV_RGB \
    AV_PIX_FMT_NV12, AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV444P, \
    AV_PIX_FMT_RGB32, AV_PIX_FMT_0RGB32, AV_PIX_FMT_BGR32, AV_PIX_FMT_0BGR32, AV_PIX_FMT_GBRP

#define ENC_FMTS_10_YUV_RGB \
    AV_PIX_FMT_P010, AV_PIX_FMT_P016, AV_PIX_FMT_YUV444P16, \
    AV_PIX_FMT_X2RGB10, AV_PIX_FMT_X2BGR10, AV_PIX_FMT_GBRP16
static const int enc_formats_8_yuv_rgb[]    = { ENC_FMTS_8_YUV_RGB, AV_PIX_FMT_NONE };
static const int enc_formats_8_10_yuv_rgb[] = { ENC_FMTS_8_YUV_RGB, ENC_FMTS_10_YUV_RGB, AV_PIX_FMT_NONE };
#undef ENC_FMTS_8_YUV_RGB
#undef ENC_FMTS_10_YUV_RGB
static const NvencMode nvenc_modes[] = {
    { "NVENC H.264 encoder", AV_CODEC_ID_H264, enc_profiles_h264, enc_formats_8_yuv_rgb },
    { "NVENC HEVC encoder",  AV_CODEC_ID_HEVC, enc_profiles_hevc, enc_formats_8_10_yuv_rgb },
    { "NVENC AV1 encoder",   AV_CODEC_ID_AV1,  enc_profiles_av1,  enc_formats_8_10_yuv_rgb },
    { NULL, 0, NULL },
};
static const struct {
    int         cap_val;
    const char *cap_str;
} nvenc_codec_caps[] = {
    { NV_ENC_CAPS_NUM_MAX_BFRAMES,                "MaxBFrames"                     },
    { NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES,    "RateControlModesMask"           },
    { NV_ENC_CAPS_SUPPORT_FIELD_ENCODING,         "SupportFieldEncoding"           },
    { NV_ENC_CAPS_SUPPORT_MONOCHROME,             "SupportMonochrome"              },
    { NV_ENC_CAPS_SUPPORT_FMO,                    "SupportFMO"                     },
    { NV_ENC_CAPS_SUPPORT_QPELMV,                 "SupportQPMotionEstimation"      },
    { NV_ENC_CAPS_SUPPORT_BDIRECT_MODE,           "SupportBiDirect"                },
    { NV_ENC_CAPS_SUPPORT_CABAC,                  "SupportCABAC"                   },
    { NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM,     "SupportAdaptiveTransform"       },
    { NV_ENC_CAPS_SUPPORT_STEREO_MVC,             "SupportStereoMVC"               },
    { NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS,        "SupportMaxTemporalLayers"       },
    { NV_ENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES,   "SupportHierarchicalPFrames"     },
    { NV_ENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES,   "SupportHierarchicalBFrames"     },
    { NV_ENC_CAPS_LEVEL_MAX,                      "MaxLevel"                       },
    { NV_ENC_CAPS_LEVEL_MIN,                      "MinLevel"                       },
    { NV_ENC_CAPS_SEPARATE_COLOUR_PLANE,          "SupportSeparateColourPlane"     },
    { NV_ENC_CAPS_WIDTH_MAX,                      "MaxWidth"                       },
    { NV_ENC_CAPS_HEIGHT_MAX,                     "MaxHeight"                      },
    { NV_ENC_CAPS_SUPPORT_TEMPORAL_SVC,           "SupportTemporalSVC"             },
    { NV_ENC_CAPS_SUPPORT_DYN_RES_CHANGE,         "SupportDynResChange"            },
    { NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE,     "SupportDynBitrateChange"        },
    { NV_ENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP,      "SupportDynForceConstQP"         },
    { NV_ENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE,      "SupportDynRcModeChange"         },
    { NV_ENC_CAPS_SUPPORT_SUBFRAME_READBACK,      "SupportSubFrameReadback"        },
    { NV_ENC_CAPS_SUPPORT_CONSTRAINED_ENCODING,   "SupportConstrainedEncoding"     },
    { NV_ENC_CAPS_SUPPORT_INTRA_REFRESH,          "SupportIntraRefresh"            },
    { NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE,    "SupportCustomVBVBufSize"        },
    { NV_ENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE,     "SupportDynSliceMode"            },
    { NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION,   "SupportRefPicInvalidation"      },
    { NV_ENC_CAPS_PREPROC_SUPPORT,                "PreProcMask"                    },
    { NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT,           "SupportAsyncEncode"             },
    { NV_ENC_CAPS_MB_NUM_MAX,                     "MaxMBPerFrame"                  },
    { NV_ENC_CAPS_MB_PER_SEC_MAX,                 "MaxMBPerSec"                    },
    { NV_ENC_CAPS_SUPPORT_YUV444_ENCODE,          "SupportYuv444Encode"            },
    { NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE,        "SupportLosslessEncode"          },
    { NV_ENC_CAPS_SUPPORT_SAO,                    "SupportSAO"                     },
    { NV_ENC_CAPS_SUPPORT_MEONLY_MODE,            "SupportMEOnlyMode"              },
    { NV_ENC_CAPS_SUPPORT_LOOKAHEAD,              "SupportLookahead"               },
    { NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ,            "SupportIntraTemporalAQ"         },
    { NV_ENC_CAPS_SUPPORT_10BIT_ENCODE,           "Support10bitEncode"             },
    { NV_ENC_CAPS_NUM_MAX_LTR_FRAMES,             "MaxLtrFrames"                   },
    { NV_ENC_CAPS_SUPPORT_WEIGHTED_PREDICTION,    "SupportWeightPrediction"        },
    { NV_ENC_CAPS_DYNAMIC_QUERY_ENCODER_CAPACITY, "SupportDynQueryEncoderCapacity" },
    { NV_ENC_CAPS_SUPPORT_BFRAME_REF_MODE,        "SupportBframeRefMode"           },
    { NV_ENC_CAPS_SUPPORT_EMPHASIS_LEVEL_MAP,     "SupportEmphasisLevelMap"        },
    { NV_ENC_CAPS_WIDTH_MIN,                      "MinWidth"                       },
    { NV_ENC_CAPS_HEIGHT_MIN,                     "MinHeight"                      },
    { NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES,    "SupportMultiRefFrames"          },
    { NV_ENC_CAPS_SUPPORT_ALPHA_LAYER_ENCODING,   "SupportAlphaLayerEncoding"      },
    { NV_ENC_CAPS_NUM_ENCODER_ENGINES,            "EncoderEngines"                 },
    { NV_ENC_CAPS_SINGLE_SLICE_INTRA_REFRESH,     "SupportSingleSliceIntraRefresh" },
};

static const struct {
    const GUID *preset_val;
    const char *preset_str;
} nvenc_codec_presets[] = {
#ifdef NVENC_HAVE_NEW_PRESETS
    { &NV_ENC_PRESET_P1_GUID,                  "p1"         },
    { &NV_ENC_PRESET_P2_GUID,                  "p2"         },
    { &NV_ENC_PRESET_P3_GUID,                  "p3"         },
    { &NV_ENC_PRESET_P4_GUID,                  "p4"         },
    { &NV_ENC_PRESET_P5_GUID,                  "p5"         },
    { &NV_ENC_PRESET_P6_GUID,                  "p6"         },
    { &NV_ENC_PRESET_P7_GUID,                  "p7"         },
#else
    { &NV_ENC_PRESET_DEFAULT_GUID,             "default"    },
    { &NV_ENC_PRESET_HP_GUID,                  "hp"         },
    { &NV_ENC_PRESET_HQ_GUID,                  "hq"         },
    { &NV_ENC_PRESET_BD_GUID,                  "bd"         },
    { &NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID, "ll"         },
    { &NV_ENC_PRESET_LOW_LATENCY_HQ_GUID,      "llhq"       },
    { &NV_ENC_PRESET_LOW_LATENCY_HP_GUID,      "llhp"       },
    { &NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID,    "lossless"   },
    { &NV_ENC_PRESET_LOSSLESS_HP_GUID,         "losslesshp" },
#endif
};
#endif

#if (CONFIG_CUDA && (CONFIG_CUVID || CONFIG_NVDEC))
static CuvidFunctions *cuvid = NULL;
typedef struct CuvidMode {
    const char    *name;
    enum AVCodecID codec;
    const int     *formats;
} CuvidMode;

static const int dec_formats_8_420[]        = { AV_PIX_FMT_NV12, AV_PIX_FMT_NONE };
static const int dec_formats_8_10_420[]     = { AV_PIX_FMT_NV12, AV_PIX_FMT_P010, AV_PIX_FMT_NONE };
static const int dec_formats_8_12_420[]     = { AV_PIX_FMT_NV12, AV_PIX_FMT_P010, AV_PIX_FMT_P016, AV_PIX_FMT_NONE };
static const int dec_formats_8_12_420_444[] = { AV_PIX_FMT_NV12, AV_PIX_FMT_P010, AV_PIX_FMT_P016,
                                                AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV444P16, AV_PIX_FMT_NONE };
static const CuvidMode cuvid_modes[] = {
    { "NVDEC / CUVID MPEG1 decoder", AV_CODEC_ID_MPEG1VIDEO, dec_formats_8_420 },
    { "NVDEC / CUVID MPEG2 decoder", AV_CODEC_ID_MPEG2VIDEO, dec_formats_8_420 },
    { "NVDEC / CUVID MPEG4 decoder", AV_CODEC_ID_MPEG4,      dec_formats_8_420 },
    { "NVDEC / CUVID VC1 decoder",   AV_CODEC_ID_VC1,        dec_formats_8_420 },
    { "NVDEC / CUVID VC1 decoder",   AV_CODEC_ID_WMV3,       dec_formats_8_420 },
    { "NVDEC / CUVID H.264 decoder", AV_CODEC_ID_H264,       dec_formats_8_420 },
    { "NVDEC / CUVID JPEG decoder",  AV_CODEC_ID_MJPEG,      dec_formats_8_420 },
    { "NVDEC / CUVID HEVC decoder",  AV_CODEC_ID_HEVC,       dec_formats_8_12_420_444 },
    { "NVDEC / CUVID VP8 decoder",   AV_CODEC_ID_VP8,        dec_formats_8_420 },
    { "NVDEC / CUVID VP9 decoder",   AV_CODEC_ID_VP9,        dec_formats_8_12_420 },
    { "NVDEC / CUVID AV1 decoder",   AV_CODEC_ID_AV1,        dec_formats_8_10_420 },
    { NULL, 0, NULL },
};
#endif

#if CONFIG_CUDA
static CudaFunctions *cu = NULL;
static CudaFunctionsExt *cu_ext = NULL;
static NvmlFunctionsExt *nvml_ext = NULL;
static char drv_ver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE+1] = {0};
static char nvml_ver[NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE+1] = {0};
static const struct {
    int         attr_val;
    const char *attr_str;
} cuda_device_attrs[] = {
    { CU_DEVICE_ATTRIBUTE_CLOCK_RATE,               "ClockRate"              },
    { CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,        "TextureAlignment"       },
    { CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,     "MultiprocessorCount"    },
    { CU_DEVICE_ATTRIBUTE_INTEGRATED,               "Integrated"             },
    { CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,      "CanMapHostMemory"       },
    { CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,             "ComputeMode"            },
    { CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,       "ConcurrentKernels"      },
    { CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,               "PciBusId"               },
    { CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,            "PciDeviceId"            },
    { CU_DEVICE_ATTRIBUTE_TCC_DRIVER,               "TccDriver"              },
    { CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,        "MemoryClockRate"        },
    { CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,  "GlobalMemoryBusWidth"   },
    { CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,       "AsyncEngineCount"       },
    { CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,       "UnifiedAddressing"      },
    { CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,            "PciDomainId"            },
    { CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,  "TexturePitchAlignment"  },
    { CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "ComputeCapabilityMajor" },
    { CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, "ComputeCapabilityMinor" },
    { CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,           "ManagedMemory"          },
    { CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,          "MultiGpuBoard"          },
    { CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, "MultiGpuBoardGroupId"   },
};
#endif

int init_cuda_functions(void)
{
#if CONFIG_CUDA
    int ret = 0;
    if (!cu) {
        ret = cuda_load_functions(&cu, NULL);
        if (ret < 0)
            goto exit;

        ret = CHECK_CU(cu->cuInit(0));
        if (ret < 0)
            goto exit;
    }
    if (!cu_ext) {
        ret = cuda_ext_load_functions(&cu_ext, NULL);
        if (ret < 0)
            goto exit;
    }
    return 0;
exit:
    if (cu)
        cuda_free_functions(&cu);
    if (cu_ext)
        cuda_ext_free_functions(&cu_ext);
    return ret;
#else
    return AVERROR(ENOSYS);
#endif
}

void uninit_cuda_functions(void)
{
#if CONFIG_CUDA
    if (cu)
        cuda_free_functions(&cu);
    if (cu_ext)
        cuda_ext_free_functions(&cu_ext);
#endif
}

int init_nvml_functions(void)
{
#if CONFIG_CUDA
    int ret = 0;
    if (!nvml_ext) {
        ret = nvml_ext_load_functions(&nvml_ext, NULL);
        if (ret < 0)
            goto exit;

        ret = CHECK_ML(nvml_ext->nvmlInit());
        if (ret < 0)
            goto exit;
    }
    return 0;
exit:
    if (nvml_ext) {
        CHECK_ML(nvml_ext->nvmlShutdown());
        nvml_ext_free_functions(&nvml_ext);
    }
    return ret;
#else
    return AVERROR(ENOSYS);
#endif
}

void uninit_nvml_functions(void)
{
#if CONFIG_CUDA
    if (nvml_ext) {
        CHECK_ML(nvml_ext->nvmlShutdown());
        nvml_ext_free_functions(&nvml_ext);
    }
#endif
}

int init_cuvid_functions(void)
{
#if (CONFIG_CUDA && (CONFIG_CUVID || CONFIG_NVDEC))
    int ret = 0;
    if (!cuvid) {
        ret = cuvid_load_functions(&cuvid, NULL);
        if (ret < 0)
            goto exit;
    }
    return 0;
exit:
    if (cuvid)
        cuvid_free_functions(&cuvid);
    return ret;
#else
    return AVERROR(ENOSYS);
#endif
}

void uninit_cuvid_functions(void)
{
#if (CONFIG_CUDA && (CONFIG_CUVID || CONFIG_NVDEC))
    if (cuvid)
        cuvid_free_functions(&cuvid);
#endif
}

int init_nvenc_functions(void)
{
#if (CONFIG_CUDA && CONFIG_NVENC)
    int ret = 0;
    uint32_t max_ver = 0;
    NVENCSTATUS err = 0;
    if (!nvenc) {
        ret = nvenc_load_functions(&nvenc, NULL);
        if (ret < 0)
            goto exit;

        err = nvenc->NvEncodeAPIGetMaxSupportedVersion(&max_ver);
        if (err != NV_ENC_SUCCESS) {
            ret = AVERROR(ENOSYS);
            goto exit;
        }

        av_log(NULL, AV_LOG_DEBUG, "Loaded Nvenc version %d.%d\n",
               max_ver >> 4, max_ver & 0xf);

        if ((NVENCAPI_MAJOR_VERSION << 4 | NVENCAPI_MINOR_VERSION) > max_ver) {
            av_log(NULL, AV_LOG_ERROR, "Driver does not support the required Nvenc API version. "
                   "Required: %d.%d Found: %d.%d\n",
                   NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION,
                   max_ver >> 4, max_ver & 0xf);
            ret = AVERROR(ENOSYS);
            goto exit;
        }

        nvenc_fns.version = NV_ENCODE_API_FUNCTION_LIST_VER;
        err = nvenc->NvEncodeAPICreateInstance(&nvenc_fns);
        if (err != NV_ENC_SUCCESS) {
            av_log(NULL, AV_LOG_ERROR, "Failed to create Nvenc instance\n");
            ret = AVERROR(ENOSYS);
            goto exit;
        }
    }
    return 0;
exit:
    if (nvenc)
        nvenc_free_functions(&nvenc);
    return ret;
#else
    return AVERROR(ENOSYS);
#endif
}

void uninit_nvenc_functions(void)
{
#if (CONFIG_CUDA && CONFIG_NVENC)
    if (nvenc)
        nvenc_free_functions(&nvenc);
#endif
}

/* CUDA */
int create_cuda_devices(HwDeviceRefs *refs)
{
#if CONFIG_CUDA
    unsigned i, j;
    int n = 0, ret = 0;
    char ibuf[4];

    if ((ret = init_cuda_functions()) < 0)
        goto exit;

    ret = CHECK_CU(cu->cuDeviceGetCount(&n));
    if (ret < 0)
        goto exit;

    if (n <= 0) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    n = FFMIN(n, HWINFO_MAX_DEV_NUM);
    for (i = 0, j = 0; i < n && refs; i++) {
        snprintf(ibuf, sizeof(ibuf), "%d", i);
        ret = av_hwdevice_ctx_create(&refs[j].cuda_ref, AV_HWDEVICE_TYPE_CUDA,
                                     ibuf, NULL, 0);
        if (ret < 0)
            continue;

        refs[j].device_index_cuda = i;
        refs[j].device_vendor_id  = HWINFO_VENDOR_ID_NVIDIA;
        ++j;
    }

    ret = 0;

exit:
    return ret;
#else
    return AVERROR(ENOSYS);
#endif
}

// win32 cudaDeviceProp::luid -> DXGI_ADAPTER_DESC::AdapterLuid
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#matching-device-luids

/* CUDA -> D3D11VA */
void create_derive_d3d11va_devices_from_cuda(HwDeviceRefs *refs)
{
#if CONFIG_CUDA
    int ret = 0;

    if (!refs)
        return;
    if ((ret = init_cuda_functions()) < 0)
        return;
    if (!cu_ext->cuDeviceGetLuid)
        return;

    for (unsigned i = 0; i < HWINFO_MAX_DEV_NUM && refs[i].cuda_ref; i++) {
        char cuda_luid[8];
        unsigned int node_mask;
        AVHWDeviceContext *dev_ctx = (AVHWDeviceContext*)refs[i].cuda_ref->data;
        AVCUDADeviceContext *hwctx = dev_ctx->hwctx;

        /* Values are undefined on TCC and non-Windows platforms */
        ret = CHECK_CU(cu_ext->cuDeviceGetLuid(cuda_luid, &node_mask,
                                               hwctx->internal->cuda_device));
        if (ret < 0)
            continue;

        create_d3d11va_devices_with_filter(refs, -1, i, cuda_luid);
    }
#endif
}

int init_nvml_driver_version(void)
{
#if CONFIG_CUDA
    int ret = 0;

    if ((ret = init_nvml_functions()) < 0)
        return ret;

    ret = CHECK_ML(nvml_ext->nvmlSystemGetDriverVersion(drv_ver,
                                                        NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE));
    if (ret < 0)
        return ret;

    ret = CHECK_ML(nvml_ext->nvmlSystemGetNVMLVersion(nvml_ver,
                                                      NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE));
    if (ret < 0)
        return ret;

    return 0;
#else
    return AVERROR(ENOSYS);
#endif
}

int print_cuda_device_info(WriterContext *wctx, AVBufferRef *cuda_ref, int nvml_ret)
{
#if CONFIG_CUDA
    AVHWDeviceContext *dev_ctx = NULL;
    AVCUDADeviceContext *hwctx = NULL;
    CUdevice dev;
    int val, cuda_ver = 0, ret = 0;
    char device_name[256] = {0};

    if (!wctx || !cuda_ref)
        return AVERROR(EINVAL);

    if ((ret = init_cuda_functions()) < 0)
        return AVERROR(ENOSYS);

    dev_ctx = (AVHWDeviceContext*)cuda_ref->data;
    hwctx = dev_ctx->hwctx;
    dev = hwctx->internal->cuda_device;

    ret = CHECK_CU(cu->cuDeviceGetName(device_name, sizeof(device_name), dev));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu_ext->cuDriverGetVersion(&cuda_ver));
    if (ret < 0)
        return ret;

    mark_section_show_entries(SECTION_ID_DEVICE_INFO_CUDA, 1, NULL);
    writer_print_section_header(wctx, SECTION_ID_DEVICE_INFO_CUDA);

    print_str("DeviceName", device_name);
    if (nvml_ret == 0) {
        print_str("DriverVersion", drv_ver);
        print_str("NvmlVersion", nvml_ver);
    }
    print_int("CudaVersion", cuda_ver);

    for (unsigned i = 0; i < FF_ARRAY_ELEMS(cuda_device_attrs); i++) {
        val = 0;
        ret = CHECK_CU(cu->cuDeviceGetAttribute(&val, cuda_device_attrs[i].attr_val, dev));
        if (ret == 0)
            print_int(cuda_device_attrs[i].attr_str, val);
    }

    writer_print_section_footer(wctx);

    return ret;
#else
    return 0;
#endif
}

#if (CONFIG_CUDA && (CONFIG_CUVID || CONFIG_NVDEC))
static int cuda_map_av_to_cuvid_codec(enum AVCodecID codec)
{
    switch (codec) {
    case AV_CODEC_ID_MPEG1VIDEO: return cudaVideoCodec_MPEG1;
    case AV_CODEC_ID_MPEG2VIDEO: return cudaVideoCodec_MPEG2;
    case AV_CODEC_ID_MPEG4:      return cudaVideoCodec_MPEG4;
    case AV_CODEC_ID_WMV3:       /* fallthrough */
    case AV_CODEC_ID_VC1:        return cudaVideoCodec_VC1;
    case AV_CODEC_ID_H264:       return cudaVideoCodec_H264;
    case AV_CODEC_ID_MJPEG:      return cudaVideoCodec_JPEG;
    case AV_CODEC_ID_HEVC:       return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_VP8:        return cudaVideoCodec_VP8;
    case AV_CODEC_ID_VP9:        return cudaVideoCodec_VP9;
    case AV_CODEC_ID_AV1:        return cudaVideoCodec_AV1;
    default:                     return -1;
    }
}

static int cuda_map_av_to_cuvid_chroma(enum AVPixelFormat pix_fmt)
{
    switch (pix_fmt) {
    case AV_PIX_FMT_NV12:      /* fallthrough */
    case AV_PIX_FMT_P010:      /* fallthrough */
    case AV_PIX_FMT_P016:      return cudaVideoChromaFormat_420;
    case AV_PIX_FMT_YUV444P:   /* fallthrough */
    case AV_PIX_FMT_YUV444P16: return cudaVideoChromaFormat_444;
    default:                   return -1;
    }
}

static int cuda_map_av_to_cuvid_surface(enum AVPixelFormat pix_fmt)
{
    switch (pix_fmt) {
    case AV_PIX_FMT_NV12:      return cudaVideoSurfaceFormat_NV12;
    case AV_PIX_FMT_P010:      /* fallthrough */
    case AV_PIX_FMT_P016:      return cudaVideoSurfaceFormat_P016;
    case AV_PIX_FMT_YUV444P:   return cudaVideoSurfaceFormat_YUV444;
    case AV_PIX_FMT_YUV444P16: return cudaVideoSurfaceFormat_YUV444_16Bit;
    default:                   return -1;
    }
}
#endif

int print_cuda_decoder_info(WriterContext *wctx, AVBufferRef *cuda_ref)
{
#if (CONFIG_CUDA && (CONFIG_CUVID || CONFIG_NVDEC))
    AVHWDeviceContext *dev_ctx = NULL;
    AVCUDADeviceContext *hwctx = NULL;
    CUVIDDECODECAPS caps = {0};
    CUcontext dummy;
    int header_printed = 0;
    int ret = 0;
    unsigned i, j;

    if (!wctx || !cuda_ref)
        return AVERROR(EINVAL);

    if ((ret = init_cuda_functions()) < 0)
        return AVERROR(ENOSYS);

    if ((ret = init_cuvid_functions()) < 0)
        return AVERROR(ENOSYS);

    if (!cuvid->cuvidGetDecoderCaps)
        return AVERROR(ENOSYS);

    dev_ctx = (AVHWDeviceContext*)cuda_ref->data;
    hwctx = dev_ctx->hwctx;

    ret = CHECK_CU(cu->cuCtxPushCurrent(hwctx->cuda_ctx));
    if (ret < 0)
        return ret;

    for (i = 0; cuvid_modes[i].name; i++) {
        int header2_printed = 0;
        const CuvidMode *mode = &cuvid_modes[i];
        if (!mode->formats)
            continue;

        caps.eCodecType = cuda_map_av_to_cuvid_codec(mode->codec);
        if (caps.eCodecType < 0)
            continue;

        for (j = 0; mode->formats[j] != AV_PIX_FMT_NONE; j++) {
            int surface = -1;
            const int format = mode->formats[j];
            const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(format);

            caps.nBitDepthMinus8 = FFMIN(desc->comp[0].depth, 12) - 8;
            caps.eChromaFormat = cuda_map_av_to_cuvid_chroma(format);
            if (caps.eChromaFormat < 0)
                continue;

            ret = CHECK_CU(cuvid->cuvidGetDecoderCaps(&caps));
            if (ret < 0)
                continue;

            if (!caps.bIsSupported)
                continue;

            surface = cuda_map_av_to_cuvid_surface(format);
            if (surface < 0 || !(caps.nOutputFormatMask & (1 << surface)))
                continue;

            if (!header_printed) {
                mark_section_show_entries(SECTION_ID_DECODERS_CUDA, 1, NULL);
                writer_print_section_header(wctx, SECTION_ID_DECODERS_CUDA);
                header_printed = 1;
            }

            if (!header2_printed) {
                mark_section_show_entries(SECTION_ID_DECODER, 1, NULL);
                writer_print_section_header(wctx, SECTION_ID_DECODER);
                print_str("CodecName", avcodec_get_name(mode->codec));
                print_int("CodecId", mode->codec);
                print_str("CodecDesc", mode->name);
                print_int("MinWidth", caps.nMinWidth);
                print_int("MinHeight", caps.nMinHeight);
                print_int("MaxWidth", caps.nMaxWidth);
                print_int("MaxHeight", caps.nMaxHeight);
                print_int("MaxMBCount", caps.nMaxMBCount);
                mark_section_show_entries(SECTION_ID_PIXEL_FORMATS, 1, NULL);
                writer_print_section_header(wctx, SECTION_ID_PIXEL_FORMATS);
                header2_printed = 1;
            }

            mark_section_show_entries(SECTION_ID_PIXEL_FORMAT, 1, NULL);
            writer_print_section_header(wctx, SECTION_ID_PIXEL_FORMAT);
            print_str("FormatName", av_get_pix_fmt_name(format));
            print_int("FormatId", format);
            writer_print_section_footer(wctx);
        }

        if (header2_printed) {
            writer_print_section_footer(wctx);
            writer_print_section_footer(wctx);
        }
    }

    if (header_printed)
        writer_print_section_footer(wctx);

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return 0;
#else
    return 0;
#endif
}

#if (CONFIG_CUDA && CONFIG_NVENC)
static GUID cuda_map_av_to_nvenc_codec_guid(enum AVCodecID codec)
{
    switch (codec) {
    case AV_CODEC_ID_H264: return NV_ENC_CODEC_H264_GUID;
    case AV_CODEC_ID_HEVC: return NV_ENC_CODEC_HEVC_GUID;
    case AV_CODEC_ID_AV1:  return NV_ENC_CODEC_AV1_GUID;
    default:
        {
            GUID g;
            memset(&g, 0, sizeof(g));
            return g;
        }
    }
}

static GUID cuda_map_av_to_nvenc_profile_guid(enum AVCodecID codec, int profile)
{
    GUID g;
    if (codec == AV_CODEC_ID_H264) {
        switch (profile) {
        case FF_PROFILE_H264_BASELINE: return NV_ENC_H264_PROFILE_BASELINE_GUID;
        case FF_PROFILE_H264_MAIN:     return NV_ENC_H264_PROFILE_MAIN_GUID;
        case FF_PROFILE_H264_HIGH:     return NV_ENC_H264_PROFILE_HIGH_GUID;
        case FF_PROFILE_H264_HIGH_444: return NV_ENC_H264_PROFILE_HIGH_444_GUID;
        }
    } else if (codec == AV_CODEC_ID_HEVC) {
        switch (profile) {
        case FF_PROFILE_HEVC_MAIN:     return NV_ENC_HEVC_PROFILE_MAIN_GUID;
        case FF_PROFILE_HEVC_MAIN_10:  return NV_ENC_HEVC_PROFILE_MAIN10_GUID;
        case FF_PROFILE_HEVC_REXT:     return NV_ENC_HEVC_PROFILE_FREXT_GUID;
        }
    } else if (codec == AV_CODEC_ID_AV1) {
        switch (profile) {
        case FF_PROFILE_AV1_MAIN:      return NV_ENC_AV1_PROFILE_MAIN_GUID;
        }
    }

    memset(&g, 0, sizeof(g));
    return g;
}

static int cuda_map_av_to_nvenc_buffer_format(enum AVPixelFormat pix_fmt)
{
    switch (pix_fmt) {
    case AV_PIX_FMT_NV12:      return NV_ENC_BUFFER_FORMAT_NV12_PL;
    case AV_PIX_FMT_YUV420P:   return NV_ENC_BUFFER_FORMAT_YV12_PL;
    case AV_PIX_FMT_P010:      /* fallthrough */
    case AV_PIX_FMT_P016:      return NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
    case AV_PIX_FMT_GBRP:      /* fallthrough */
    case AV_PIX_FMT_YUV444P:   return NV_ENC_BUFFER_FORMAT_YUV444_PL;
    case AV_PIX_FMT_GBRP16:    /* fallthrough */
    case AV_PIX_FMT_YUV444P16: return NV_ENC_BUFFER_FORMAT_YUV444_10BIT;
    case AV_PIX_FMT_0RGB32:    /* fallthrough */
    case AV_PIX_FMT_RGB32:     return NV_ENC_BUFFER_FORMAT_ARGB;
    case AV_PIX_FMT_0BGR32:    /* fallthrough */
    case AV_PIX_FMT_BGR32:     return NV_ENC_BUFFER_FORMAT_ABGR;
    case AV_PIX_FMT_X2RGB10:   return NV_ENC_BUFFER_FORMAT_ARGB10;
    case AV_PIX_FMT_X2BGR10:   return NV_ENC_BUFFER_FORMAT_ABGR10;
    default:                   return NV_ENC_BUFFER_FORMAT_UNDEFINED;
    }
}
#endif

int print_cuda_encoder_info(WriterContext *wctx, AVBufferRef *cuda_ref)
{
#if (CONFIG_CUDA && CONFIG_NVENC)
    AVHWDeviceContext *dev_ctx = NULL;
    AVCUDADeviceContext *hwctx = NULL;
    CUdevice dev;
    CUcontext dummy;
    unsigned i, j, k;
    unsigned major, minor;
    int val, ret = 0;
    int header_printed = 0;
    void *nvenc_hdl = NULL;
    NVENCSTATUS err = 0;
    unsigned codec_cnt = 0;
    GUID *codec_list = NULL;

    if (!wctx || !cuda_ref)
        return AVERROR(EINVAL);

    if ((ret = init_cuda_functions()) < 0)
        return AVERROR(ENOSYS);

    if ((ret = init_nvenc_functions()) < 0)
        return AVERROR(ENOSYS);

    dev_ctx = (AVHWDeviceContext*)cuda_ref->data;
    hwctx = dev_ctx->hwctx;
    dev = hwctx->internal->cuda_device;

    ret = CHECK_CU(cu->cuDeviceComputeCapability(&major, &minor, dev));
    if (ret < 0)
        return ret;

    if (((major << 4) | minor) < 0x30)
        return AVERROR(EINVAL);

    ret = CHECK_CU(cu->cuCtxPushCurrent(hwctx->cuda_ctx));
    if (ret < 0)
        return ret;

    {
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS params = {
            .version    = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
            .apiVersion = NVENCAPI_VERSION,
            .deviceType = NV_ENC_DEVICE_TYPE_CUDA,
            .device     = hwctx->cuda_ctx,
        };

        err = nvenc_fns.nvEncOpenEncodeSessionEx(&params, &nvenc_hdl);
        if (err != NV_ENC_SUCCESS) {
            nvenc_hdl = NULL;
            av_log(NULL, AV_LOG_ERROR, "Nvenc OpenEncodeSessionEx failed\n");
            ret = AVERROR(ENOSYS);
            goto exit;
        }
    }

    err = nvenc_fns.nvEncGetEncodeGUIDCount(nvenc_hdl, &codec_cnt);
    if (err != NV_ENC_SUCCESS || !codec_cnt) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    codec_list = av_malloc_array(codec_cnt, sizeof(*codec_list));
    if (!codec_list) {
        av_free(codec_list);
        ret = AVERROR(EINVAL);
        goto exit;
    }

    err = nvenc_fns.nvEncGetEncodeGUIDs(nvenc_hdl, codec_list, codec_cnt, &codec_cnt);
    if (err != NV_ENC_SUCCESS) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    for (i = 0; nvenc_modes[i].name; i++) {
        int supported = 0;
        int header2_printed = 0;
        int header3_printed = 0;
        int header4_printed = 0;
        unsigned profile_cnt, fmt_cnt, preset_cnt;
        GUID *profile_list = NULL, *preset_list = NULL;
        NV_ENC_BUFFER_FORMAT *fmt_list = NULL;
        const NvencMode *mode = &nvenc_modes[i];
        const GUID codec_guid = cuda_map_av_to_nvenc_codec_guid(mode->codec);

        for (const GUID *g = &codec_list[0]; !supported && g < &codec_list[codec_cnt]; g++) {
            supported = !memcmp(g, &codec_guid, sizeof(*g));
        }
        if (!supported)
            continue;

        if (!mode->formats)
            continue;

        if (!header_printed) {
            mark_section_show_entries(SECTION_ID_ENCODERS_CUDA, 1, NULL);
            writer_print_section_header(wctx, SECTION_ID_ENCODERS_CUDA);
            header_printed = 1;
        }

        mark_section_show_entries(SECTION_ID_DECODER, 1, NULL);
        writer_print_section_header(wctx, SECTION_ID_DECODER);
        print_str("CodecName", avcodec_get_name(mode->codec));
        print_int("CodecId", mode->codec);
        print_str("CodecDesc", mode->name);

        for (j = 0; j < FF_ARRAY_ELEMS(nvenc_codec_caps); j++) {
            NV_ENC_CAPS_PARAM params = {
                .version     = NV_ENC_CAPS_PARAM_VER,
                .capsToQuery = nvenc_codec_caps[j].cap_val,
            };

            val = 0;
            err = nvenc_fns.nvEncGetEncodeCaps(nvenc_hdl, codec_guid, &params, &val);
            if (err == NV_ENC_SUCCESS)
                print_int(nvenc_codec_caps[j].cap_str, val);
        }

        /* profiles */
        err = nvenc_fns.nvEncGetEncodeProfileGUIDCount(nvenc_hdl, codec_guid, &profile_cnt);
        if (err == NV_ENC_SUCCESS && profile_cnt) {
            profile_list = av_malloc_array(profile_cnt, sizeof(*profile_list));
            if (profile_list)
                err = nvenc_fns.nvEncGetEncodeProfileGUIDs(nvenc_hdl, codec_guid,
                                                           profile_list, profile_cnt, &profile_cnt);
        }
        for (j = 0; mode->profiles[j] != FF_PROFILE_UNKNOWN; j++) {
            GUID profile_guid;
            if (!profile_cnt || !profile_list || err != NV_ENC_SUCCESS)
                break;

            profile_guid = cuda_map_av_to_nvenc_profile_guid(mode->codec, mode->profiles[j]);
            for (const GUID *g = &profile_list[0]; g < &profile_list[profile_cnt]; g++) {
                if (!memcmp(g, &profile_guid, sizeof(*g))) {
                    if (!header2_printed) {
                        mark_section_show_entries(SECTION_ID_PROFILES, 1, NULL);
                        writer_print_section_header(wctx, SECTION_ID_PROFILES);
                        header2_printed = 1;
                    }
                    mark_section_show_entries(SECTION_ID_PROFILE, 1, NULL);
                    writer_print_section_header(wctx, SECTION_ID_PROFILE);
                    print_str("ProfileName", avcodec_profile_name(mode->codec, mode->profiles[j]));
                    print_int("ProfileId", mode->profiles[j]);
                    writer_print_section_footer(wctx);
                    break;
                }
            }
        }
        if (header2_printed)
            writer_print_section_footer(wctx);

        /* formats */
        err = nvenc_fns.nvEncGetInputFormatCount(nvenc_hdl, codec_guid, &fmt_cnt);
        if (err == NV_ENC_SUCCESS && fmt_cnt) {
            fmt_list = av_malloc_array(fmt_cnt, sizeof(*fmt_list));
            if (fmt_list)
                err = nvenc_fns.nvEncGetInputFormats(nvenc_hdl, codec_guid,
                                                     fmt_list, fmt_cnt, &fmt_cnt);
        }
        for (j = 0; mode->formats[j] != AV_PIX_FMT_NONE; j++) {
            int buffer_format;
            if (!fmt_cnt || !fmt_list || err != NV_ENC_SUCCESS)
                break;

            buffer_format = cuda_map_av_to_nvenc_buffer_format(mode->formats[j]);
            if (buffer_format == NV_ENC_BUFFER_FORMAT_UNDEFINED)
                continue;

            for (k = 0; k < fmt_cnt; k++) {
                if (buffer_format == fmt_list[k]) {
                    if (!header3_printed) {
                        mark_section_show_entries(SECTION_ID_PIXEL_FORMATS, 1, NULL);
                        writer_print_section_header(wctx, SECTION_ID_PIXEL_FORMATS);
                        header3_printed = 1;
                    }
                    mark_section_show_entries(SECTION_ID_PIXEL_FORMAT, 1, NULL);
                    writer_print_section_header(wctx, SECTION_ID_PIXEL_FORMAT);
                    print_str("FormatName", av_get_pix_fmt_name(mode->formats[j]));
                    print_int("FormatId", mode->formats[j]);
                    writer_print_section_footer(wctx);
                    break;
                }
            }
        }
        if (header3_printed)
            writer_print_section_footer(wctx);

        /* presets */
        err = nvenc_fns.nvEncGetEncodePresetCount(nvenc_hdl, codec_guid, &preset_cnt);
        if (err == NV_ENC_SUCCESS && preset_cnt) {
            preset_list = av_malloc_array(preset_cnt, sizeof(*preset_list));
            if (preset_list)
                err = nvenc_fns.nvEncGetEncodePresetGUIDs(nvenc_hdl, codec_guid,
                                                          preset_list, preset_cnt, &preset_cnt);
        }
        for (j = 0; j < FF_ARRAY_ELEMS(nvenc_codec_presets); j++) {
            if (!preset_cnt || !preset_list || err != NV_ENC_SUCCESS)
                break;

            for (const GUID *g = &preset_list[0]; g < &preset_list[preset_cnt]; g++) {
                if (!memcmp(g, nvenc_codec_presets[j].preset_val, sizeof(*g))) {
                    if (!header4_printed) {
                        mark_section_show_entries(SECTION_ID_PRESETS, 1, NULL);
                        writer_print_section_header(wctx, SECTION_ID_PRESETS);
                        header4_printed = 1;
                    }
                    mark_section_show_entries(SECTION_ID_PRESET, 1, NULL);
                    writer_print_section_header(wctx, SECTION_ID_PRESET);
                    print_str("PresetName", nvenc_codec_presets[j].preset_str);
                    writer_print_section_footer(wctx);
                    break;
                }
            }
        }
        if (header4_printed)
            writer_print_section_footer(wctx);

        writer_print_section_footer(wctx);

        if (profile_list)
            av_free(profile_list);
        if (fmt_list)
            av_free(fmt_list);
        if (preset_list)
            av_free(preset_list);
    }

    if (header_printed)
        writer_print_section_footer(wctx);

exit:
    if (codec_list)
        av_free(codec_list);
    if (nvenc_hdl)
        nvenc_fns.nvEncDestroyEncoder(nvenc_hdl);

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return ret;
#else
    return 0;
#endif
}
