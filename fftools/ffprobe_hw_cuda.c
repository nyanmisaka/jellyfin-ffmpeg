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

#include "ffprobe_hw_internal.h"

#if CONFIG_CUDA
#   define CHECK_CU(x) FF_CUDA_CHECK_DL(NULL, cu, x)
#   define CHECK_CU_EXT(x) FF_CUDA_CHECK_DL(NULL, cu_ext, x)
#   define CHECK_ML(x) FF_NVML_CHECK_DL(NULL, nvml_ext, x)
#   include "libavutil/cuda_check.h"
#   include "libavutil/hwcontext_cuda_internal.h"
#endif

#if (CONFIG_CUDA && CONFIG_NVENC)
#   define NVENCAPI_CHECK_VERSION(major, minor) \
        ((major) < NVENCAPI_MAJOR_VERSION || \
         ((major) == NVENCAPI_MAJOR_VERSION && \
          (minor) <= NVENCAPI_MINOR_VERSION))

static NvencFunctions *nvenc = NULL;
static NV_ENCODE_API_FUNCTION_LIST nvenc_fns;

typedef struct NvencMode {
    const char               *name;
    const enum AVCodecID      codec;
    const int                *profiles;
    const enum AVPixelFormat *formats;
} NvencMode;

static const int enc_profiles_h264[] = {
    AV_PROFILE_H264_BASELINE,
    AV_PROFILE_H264_MAIN,
    AV_PROFILE_H264_HIGH,
    AV_PROFILE_H264_HIGH_10,
    AV_PROFILE_H264_HIGH_422,
    AV_PROFILE_H264_HIGH_444_PREDICTIVE,
    AV_PROFILE_UNKNOWN
};
static const int enc_profiles_hevc[] = {
    AV_PROFILE_HEVC_MAIN,
    AV_PROFILE_HEVC_MAIN_10,
    AV_PROFILE_HEVC_REXT,
    AV_PROFILE_UNKNOWN
};
static const int enc_profiles_av1[]  = {
    AV_PROFILE_AV1_MAIN,
    AV_PROFILE_UNKNOWN
};

static const enum AVPixelFormat enc_formats_8_10_yuv_rgb[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NV16,
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_RGB32,
    AV_PIX_FMT_0RGB32,
    AV_PIX_FMT_BGR32,
    AV_PIX_FMT_0BGR32,
    AV_PIX_FMT_GBRP,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P016,
    AV_PIX_FMT_P210,
    AV_PIX_FMT_P216,
    AV_PIX_FMT_YUV444P10MSB,
    AV_PIX_FMT_YUV444P16,
    AV_PIX_FMT_X2RGB10,
    AV_PIX_FMT_X2BGR10,
    AV_PIX_FMT_GBRP10MSB,
    AV_PIX_FMT_GBRP16,
    AV_PIX_FMT_NONE
};

static const NvencMode nvenc_modes[] = {
    { "NVENC H.264 Encoder", AV_CODEC_ID_H264, enc_profiles_h264, enc_formats_8_10_yuv_rgb },
    { "NVENC HEVC Encoder",  AV_CODEC_ID_HEVC, enc_profiles_hevc, enc_formats_8_10_yuv_rgb },
    { "NVENC AV1 Encoder",   AV_CODEC_ID_AV1,  enc_profiles_av1,  enc_formats_8_10_yuv_rgb },
    { NULL, 0, NULL },
};

static const struct {
    const int   cap_val;
    const char *cap_str;
} nvenc_codec_caps[] = {
    { NV_ENC_CAPS_NUM_MAX_BFRAMES,                "max_b_frames"                       },
    { NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES,    "rate_control_modes_mask"            },
    { NV_ENC_CAPS_SUPPORT_FIELD_ENCODING,         "support_field_encoding"             },
    { NV_ENC_CAPS_SUPPORT_MONOCHROME,             "support_monochrome"                 },
    { NV_ENC_CAPS_SUPPORT_FMO,                    "support_fmo"                        },
    { NV_ENC_CAPS_SUPPORT_QPELMV,                 "support_qp_motion_estimation"       },
    { NV_ENC_CAPS_SUPPORT_BDIRECT_MODE,           "support_bi_direct"                  },
    { NV_ENC_CAPS_SUPPORT_CABAC,                  "support_cabac"                      },
    { NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM,     "support_adaptive_transform"         },
    { NV_ENC_CAPS_SUPPORT_STEREO_MVC,             "support_stereo_mvc"                 },
    { NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS,        "support_max_temporal_layers"        },
    { NV_ENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES,   "support_hierarchical_p_frames"      },
    { NV_ENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES,   "support_hierarchical_b_frames"      },
    { NV_ENC_CAPS_LEVEL_MAX,                      "max_level"                          },
    { NV_ENC_CAPS_LEVEL_MIN,                      "min_level"                          },
    { NV_ENC_CAPS_SEPARATE_COLOUR_PLANE,          "support_separate_colour_plane"      },
    { NV_ENC_CAPS_WIDTH_MAX,                      "max_width"                          },
    { NV_ENC_CAPS_HEIGHT_MAX,                     "max_height"                         },
    { NV_ENC_CAPS_SUPPORT_TEMPORAL_SVC,           "support_temporal_svc"               },
    { NV_ENC_CAPS_SUPPORT_DYN_RES_CHANGE,         "support_dyn_res_change"             },
    { NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE,     "support_dyn_bitrate_change"         },
    { NV_ENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP,      "support_dyn_force_const_qp"         },
    { NV_ENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE,      "support_dyn_rc_mode_change"         },
    { NV_ENC_CAPS_SUPPORT_SUBFRAME_READBACK,      "support_sub_frame_readback"         },
    { NV_ENC_CAPS_SUPPORT_CONSTRAINED_ENCODING,   "support_constrained_encoding"       },
    { NV_ENC_CAPS_SUPPORT_INTRA_REFRESH,          "support_intra_refresh"              },
    { NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE,    "support_custom_vbv_buf_size"        },
    { NV_ENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE,     "support_dyn_slice_mode"             },
    { NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION,   "support_ref_pic_invalidation"       },
    { NV_ENC_CAPS_PREPROC_SUPPORT,                "pre_proc_mask"                      },
    { NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT,           "support_async_encode"               },
    { NV_ENC_CAPS_MB_NUM_MAX,                     "max_mb_per_frame"                   },
    { NV_ENC_CAPS_MB_PER_SEC_MAX,                 "max_mb_per_sec"                     },
    { NV_ENC_CAPS_SUPPORT_YUV444_ENCODE,          "support_yuv444_encode"              },
    { NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE,        "support_lossless_encode"            },
    { NV_ENC_CAPS_SUPPORT_SAO,                    "support_sao"                        },
    { NV_ENC_CAPS_SUPPORT_MEONLY_MODE,            "support_me_only_mode"               },
    { NV_ENC_CAPS_SUPPORT_LOOKAHEAD,              "support_lookahead"                  },
    { NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ,            "support_intra_temporal_aq"          },
    { NV_ENC_CAPS_SUPPORT_10BIT_ENCODE,           "support_10bit_encode"               },
    { NV_ENC_CAPS_NUM_MAX_LTR_FRAMES,             "max_ltr_frames"                     },
    { NV_ENC_CAPS_SUPPORT_WEIGHTED_PREDICTION,    "support_weight_prediction"          },
#   if NVENCAPI_CHECK_VERSION(8, 1)
    { NV_ENC_CAPS_DYNAMIC_QUERY_ENCODER_CAPACITY, "support_dyn_query_encoder_capacity" },
    { NV_ENC_CAPS_SUPPORT_BFRAME_REF_MODE,        "support_bframe_ref_mode"            },
    { NV_ENC_CAPS_SUPPORT_EMPHASIS_LEVEL_MAP,     "support_emphasis_level_map"         },
#   endif
#   if NVENCAPI_CHECK_VERSION(9, 1)
    { NV_ENC_CAPS_WIDTH_MIN,                      "min_width"                          },
    { NV_ENC_CAPS_HEIGHT_MIN,                     "min_height"                         },
    { NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES,    "support_multi_ref_frames"           },
#   endif
#   if NVENCAPI_CHECK_VERSION(11, 0)
    { NV_ENC_CAPS_SUPPORT_ALPHA_LAYER_ENCODING,   "support_alpha_layer_encoding"       },
    { NV_ENC_CAPS_NUM_ENCODER_ENGINES,            "encoder_engines"                    },
#   endif
#   if NVENCAPI_CHECK_VERSION(11, 1)
    { NV_ENC_CAPS_SINGLE_SLICE_INTRA_REFRESH,     "support_single_slice_intra_refresh" },
#   endif
#   if NVENCAPI_CHECK_VERSION(12, 1)
    { NV_ENC_CAPS_DISABLE_ENC_STATE_ADVANCE,      "disable_enc_state_advance"          },
    { NV_ENC_CAPS_OUTPUT_RECON_SURFACE,           "output_recon_surface"               },
    { NV_ENC_CAPS_OUTPUT_BLOCK_STATS,             "output_block_stats"                 },
    { NV_ENC_CAPS_OUTPUT_ROW_STATS,               "output_row_stats"                   },
#   endif
#   if NVENCAPI_CHECK_VERSION(12, 2)
    { NV_ENC_CAPS_SUPPORT_TEMPORAL_FILTER,        "support_temporal_filter"            },
    { NV_ENC_CAPS_SUPPORT_LOOKAHEAD_LEVEL,        "support_lookahead_level"            },
    { NV_ENC_CAPS_SUPPORT_UNIDIRECTIONAL_B,       "support_unidirectional_b"           },
#   endif
#   if NVENCAPI_CHECK_VERSION(13, 0)
    { NV_ENC_CAPS_SUPPORT_MVHEVC_ENCODE,          "support_mvhevc_encode"              },
    { NV_ENC_CAPS_SUPPORT_YUV422_ENCODE,          "support_yuv422_encode"              },
#   endif
};

static const struct {
    const GUID *preset_val;
    const char *preset_str;
} nvenc_codec_presets[] = {
#   if NVENCAPI_CHECK_VERSION(10, 0)
    { &NV_ENC_PRESET_P1_GUID,                  "p1"         },
    { &NV_ENC_PRESET_P2_GUID,                  "p2"         },
    { &NV_ENC_PRESET_P3_GUID,                  "p3"         },
    { &NV_ENC_PRESET_P4_GUID,                  "p4"         },
    { &NV_ENC_PRESET_P5_GUID,                  "p5"         },
    { &NV_ENC_PRESET_P6_GUID,                  "p6"         },
    { &NV_ENC_PRESET_P7_GUID,                  "p7"         },
#   else
    { &NV_ENC_PRESET_DEFAULT_GUID,             "default"    },
    { &NV_ENC_PRESET_HP_GUID,                  "hp"         },
    { &NV_ENC_PRESET_HQ_GUID,                  "hq"         },
    { &NV_ENC_PRESET_BD_GUID,                  "bd"         },
    { &NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID, "ll"         },
    { &NV_ENC_PRESET_LOW_LATENCY_HQ_GUID,      "llhq"       },
    { &NV_ENC_PRESET_LOW_LATENCY_HP_GUID,      "llhp"       },
    { &NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID,    "lossless"   },
    { &NV_ENC_PRESET_LOSSLESS_HP_GUID,         "losslesshp" },
#   endif
};
#endif

#if (CONFIG_CUDA && (CONFIG_CUVID || CONFIG_NVDEC))
#   if defined(NVDECAPI_MAJOR_VERSION) && defined(NVDECAPI_MINOR_VERSION)
#   define NVDECAPI_CHECK_VERSION(major, minor) \
        ((major) < NVDECAPI_MAJOR_VERSION || \
         ((major) == NVDECAPI_MAJOR_VERSION && \
          (minor) <= NVDECAPI_MINOR_VERSION))
#   else
/* version macros were added in SDK 8.1 ffnvcodec */
#   define NVDECAPI_CHECK_VERSION(major, minor) \
        ((major) < 8 || ((major) == 8 && (minor) <= 0))
#   endif

static CuvidFunctions *cuvid = NULL;

typedef struct CuvidMode {
    const char               *name;
    const enum AVCodecID      codec;
    const enum AVPixelFormat *formats;
} CuvidMode;

static const enum AVPixelFormat dec_formats_8_420[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NONE
};
static const enum AVPixelFormat dec_formats_8_10_420[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_NONE
};
static const enum AVPixelFormat dec_formats_8_10_420_422[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NV16,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P210,
    AV_PIX_FMT_NONE
};
static const enum AVPixelFormat dec_formats_8_12_420[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
#   if !FF_API_NVDEC_OLD_PIX_FMTS
    AV_PIX_FMT_P012,
#   else
    AV_PIX_FMT_P016,
#   endif
    AV_PIX_FMT_NONE
};
static const enum AVPixelFormat dec_formats_8_12_420_422_444[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NV16,
    AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P210,
#   if !FF_API_NVDEC_OLD_PIX_FMTS
    AV_PIX_FMT_P012,
    AV_PIX_FMT_P212,
    AV_PIX_FMT_YUV444P10MSB,
    AV_PIX_FMT_YUV444P12MSB,
#   else
    AV_PIX_FMT_P016,
    AV_PIX_FMT_P216,
    AV_PIX_FMT_YUV444P16,
#   endif
    AV_PIX_FMT_NONE
};

static const CuvidMode cuvid_modes[] = {
    { "NVDEC/CUVID MPEG-1 Decoder", AV_CODEC_ID_MPEG1VIDEO, dec_formats_8_420            },
    { "NVDEC/CUVID MPEG-2 Decoder", AV_CODEC_ID_MPEG2VIDEO, dec_formats_8_420            },
    { "NVDEC/CUVID MPEG-4 Decoder", AV_CODEC_ID_MPEG4,      dec_formats_8_420            },
    { "NVDEC/CUVID VC-1 Decoder",   AV_CODEC_ID_VC1,        dec_formats_8_420            },
    { "NVDEC/CUVID VC-1 Decoder",   AV_CODEC_ID_WMV3,       dec_formats_8_420            },
    { "NVDEC/CUVID H.264 Decoder",  AV_CODEC_ID_H264,       dec_formats_8_10_420_422     },
    { "NVDEC/CUVID MJPEG Decoder",  AV_CODEC_ID_MJPEG,      dec_formats_8_420            },
    { "NVDEC/CUVID HEVC Decoder",   AV_CODEC_ID_HEVC,       dec_formats_8_12_420_422_444 },
    { "NVDEC/CUVID VP8 Decoder",    AV_CODEC_ID_VP8,        dec_formats_8_420            },
    { "NVDEC/CUVID VP9 Decoder",    AV_CODEC_ID_VP9,        dec_formats_8_12_420         },
    { "NVDEC/CUVID AV1 Decoder",    AV_CODEC_ID_AV1,        dec_formats_8_10_420         },
    { NULL, 0, NULL },
};
#endif

#if CONFIG_CUDA
static CudaFunctionsExt *cu_ext = NULL;
static NvmlFunctionsExt *nvml_ext = NULL;
static char drv_ver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE+1] = { 0 };
static char nvml_ver[NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE+1] = { 0 };

static const struct {
    const int   attr_val;
    const char *attr_str;
} cuda_device_attrs[] = {
    { 13 /* CU_DEVICE_ATTRIBUTE_CLOCK_RATE */,               "device_clock_rate"               },
    { 14 /* CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT */,        "device_texture_alignment"        },
    { 16 /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */,     "device_multiprocessor_count"     },
    { 18 /* CU_DEVICE_ATTRIBUTE_INTEGRATED */,               "device_integrated"               },
    { 19 /* CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY */,      "device_can_map_host_memory"      },
    { 20 /* CU_DEVICE_ATTRIBUTE_COMPUTE_MODE */,             "device_compute_mode"             },
    { 31 /* CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS */,       "device_concurrent_kernels"       },
    { 33 /* CU_DEVICE_ATTRIBUTE_PCI_BUS_ID */,               "device_pci_bus_id"               },
    { 34 /* CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID */,            "device_pci_device_id"            },
    { 35 /* CU_DEVICE_ATTRIBUTE_TCC_DRIVER */,               "device_tcc_driver"               },
    { 36 /* CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE */,        "device_memory_clock_rate"        },
    { 37 /* CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH */,  "device_global_memory_bus_width"  },
    { 40 /* CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT */,       "device_async_engine_count"       },
    { 41 /* CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING */,       "device_unified_addressing"       },
    { 50 /* CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID */,            "device_pci_domain_id"            },
    { 51 /* CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT */,  "device_texture_pitch_alignment"  },
    { 75 /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */, "device_compute_capability_major" },
    { 76 /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR */, "device_compute_capability_minor" },
    { 83 /* CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY */,           "device_managed_memory"           },
    { 84 /* CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD */,          "device_multi_gpu_board"          },
    { 85 /* CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID */, "device_multi_gpu_board_group_id" },
};
#endif

int init_cuda_functions(void)
{
#if CONFIG_CUDA
    int ret = 0;

    if (!cu_ext) {
        ret = cuda_ext_load_functions(&cu_ext, NULL);
        if (ret < 0)
            goto exit;

        ret = CHECK_CU_EXT(cu_ext->cuInit(0));
        if (ret < 0)
            goto exit;
    }
    return 0;
exit:
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
               max_ver >> 4, max_ver & 0xF);

        if ((NVENCAPI_MAJOR_VERSION << 4 | NVENCAPI_MINOR_VERSION) > max_ver) {
            av_log(NULL, AV_LOG_DEBUG, "Driver does not support the required Nvenc API version. "
                   "Required: %d.%d Found: %d.%d\n",
                   NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION,
                   max_ver >> 4, max_ver & 0xF);
            ret = AVERROR(ENOSYS);
            goto exit;
        }

        nvenc_fns.version = NV_ENCODE_API_FUNCTION_LIST_VER;
        err = nvenc->NvEncodeAPICreateInstance(&nvenc_fns);
        if (err != NV_ENC_SUCCESS) {
            av_log(NULL, AV_LOG_DEBUG, "Failed to create Nvenc instance\n");
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
int create_cuda_devices(HwDeviceRefs *refs, int device_idx)
{
#if CONFIG_CUDA
    unsigned i, j;
    const int start_idx = device_idx < 0 ? 0 : device_idx;
    int n = 0, ret = 0;
    char ibuf[4];

    if ((ret = init_cuda_functions()) < 0)
        goto exit;

    ret = CHECK_CU_EXT(cu_ext->cuDeviceGetCount(&n));
    if (ret < 0)
        goto exit;

    if (n <= 0) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    n = FFMIN(n, FFPROBE_HW_MAX_DEV_NUM);
    for (i = start_idx, j = 0; i < n && refs; i++) {
        snprintf(ibuf, sizeof(ibuf), "%d", i);
        ret = av_hwdevice_ctx_create(&refs[j].cuda_ref, AV_HWDEVICE_TYPE_CUDA,
                                     ibuf, NULL, 0);
        if (ret < 0) {
            if (device_idx < 0) continue;
            else break;
        }

        refs[j].device_index_cuda = i;
        refs[j].device_vendor_id  = FFPROBE_HW_VENDOR_ID_NVIDIA;
        j++;

        /* Filter by the requested device index */
        if (device_idx >= 0)
            break;
    }

    ret = 0;

exit:
    return ret;
#else
    return AVERROR(ENOSYS);
#endif
}

/* CUDA -> D3D11VA */
void create_derive_d3d11va_devices_from_cuda(HwDeviceRefs *refs)
{
#if (CONFIG_CUDA && CONFIG_D3D11VA)
    int ret = 0;

    if (!refs)
        return;
    if ((ret = init_cuda_functions()) < 0)
        return;
    if (!cu_ext->cuDeviceGetLuid)
        return;

    for (unsigned i = 0; i < FFPROBE_HW_MAX_DEV_NUM && refs[i].cuda_ref; i++) {
        AVHWDeviceContext *dev_ctx = NULL;
        AVCUDADeviceContext *hwctx = NULL;
        CudaFunctions *cu = NULL;
        char cuda_luid[8] = { 0 };
        int is_tcc_drv = 0;
        unsigned node_mask = 0;

        dev_ctx = (AVHWDeviceContext*)refs[i].cuda_ref->data;
        if (!dev_ctx)
            continue;

        hwctx = dev_ctx->hwctx;
        cu = hwctx->internal->cuda_dl;

        /* Values are undefined on TCC and non-Windows platforms */
        ret = CHECK_CU(cu->cuDeviceGetAttribute(&is_tcc_drv,
                                                35 /* CU_DEVICE_ATTRIBUTE_TCC_DRIVER */,
                                                hwctx->internal->cuda_device));
        if (ret < 0 || is_tcc_drv)
            continue;

        ret = CHECK_CU_EXT(cu_ext->cuDeviceGetLuid(cuda_luid, &node_mask,
                                                   hwctx->internal->cuda_device));
        if (ret < 0)
            continue;

        create_d3d11va_devices_with_filter(refs, FFPROBE_HW_VENDOR_ID_NVIDIA, i, cuda_luid, -1);
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

int print_cuda_device_info(AVTextFormatContext *tfc, AVBufferRef *cuda_ref, int nvml_ret)
{
#if CONFIG_CUDA
    AVHWDeviceContext *dev_ctx = NULL;
    AVCUDADeviceContext *hwctx = NULL;
    CudaFunctions *cu = NULL;
    CUdevice dev;
    CUuuid cuda_uuid = { 0 };
    int has_uuid = 0;
    int val, cuda_ver = 0, ret = 0;
    char device_name[256] = { 0 };
    char uuid_buf[32+4+1] = { 0 };
    size_t device_memory = 0;

    if (!tfc || !cuda_ref)
        return AVERROR(EINVAL);

    if ((ret = init_cuda_functions()) < 0)
        return AVERROR(ENOSYS);

    dev_ctx = (AVHWDeviceContext*)cuda_ref->data;
    if (!dev_ctx)
        return AVERROR(ENOSYS);

    hwctx = dev_ctx->hwctx;
    cu = hwctx->internal->cuda_dl;
    dev = hwctx->internal->cuda_device;

    ret = CHECK_CU(cu->cuDeviceGetName(device_name, sizeof(device_name), dev));
    if (ret < 0)
        return ret;

    ret = CHECK_CU_EXT(cu_ext->cuDriverGetVersion(&cuda_ver));
    if (ret < 0)
        return ret;

    ret = CHECK_CU_EXT(cu_ext->cuDeviceTotalMem(&device_memory, dev));
    if (ret < 0)
        return ret;

    if (cu_ext->cuDeviceGetUuid_v2) {
        ret = CHECK_CU_EXT(cu_ext->cuDeviceGetUuid_v2(&cuda_uuid, dev));
        if (!ret)
            has_uuid = 1;
    } else if (cu_ext->cuDeviceGetUuid) {
        ret = CHECK_CU_EXT(cu_ext->cuDeviceGetUuid(&cuda_uuid, dev));
        if (!ret)
            has_uuid = 1;
    }

    mark_section_show_entries(SECTION_ID_DEVICE_INFO_CUDA, 1, NULL);
    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_INFO_CUDA);

    print_str("device_name", device_name);
    if (!nvml_ret) {
        print_str("device_driver_version", drv_ver);
        print_str("device_nvml_version", nvml_ver);
    }
    print_int("device_cuda_version", cuda_ver);
    if (has_uuid) {
        const uint8_t *u = (const uint8_t*)cuda_uuid.bytes;

        snprintf(
            uuid_buf, sizeof(uuid_buf),
            "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
            u[0], u[1], u[2],  u[3],  u[4],  u[5],  u[6],  u[7],
            u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]
        );
        print_str("device_uuid", uuid_buf);
    }
    print_int("device_memory", device_memory);

    for (unsigned i = 0; i < FF_ARRAY_ELEMS(cuda_device_attrs); i++) {
        val = 0;
        if (cuda_device_attrs[i].attr_val <= 0)
            break;
        ret = CHECK_CU(cu->cuDeviceGetAttribute(&val, cuda_device_attrs[i].attr_val, dev));
        if (!ret)
            print_int(cuda_device_attrs[i].attr_str, val);
    }

    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_INFO_CUDA

    return ret;
#else
    return 0;
#endif
}

int print_cuda_device_util(AVTextFormatContext *tfc, AVBufferRef *cuda_ref, int nvml_ret)
{
#if CONFIG_CUDA
    AVHWDeviceContext *dev_ctx = NULL;
    AVCUDADeviceContext *hwctx = NULL;
    CudaFunctions *cu = NULL;
    CUdevice dev;
    CUuuid cuda_uuid = { 0 };
    nvmlDevice_t nvml_dev = NULL;
    int ret = 0;
    char device_name[256] = { 0 };
    char uuid_buf[4+32+4+1] = { 0 };

    if (!tfc || !cuda_ref || nvml_ret < 0)
        return AVERROR(EINVAL);

    if ((ret = init_cuda_functions()) < 0)
        return AVERROR(ENOSYS);

    dev_ctx = (AVHWDeviceContext*)cuda_ref->data;
    if (!dev_ctx)
        return AVERROR(ENOSYS);

    hwctx = dev_ctx->hwctx;
    cu = hwctx->internal->cuda_dl;
    dev = hwctx->internal->cuda_device;

    ret = CHECK_CU(cu->cuDeviceGetName(device_name, sizeof(device_name), dev));
    if (ret < 0)
        return ret;

    if (cu_ext->cuDeviceGetUuid_v2) {
        ret = CHECK_CU_EXT(cu_ext->cuDeviceGetUuid_v2(&cuda_uuid, dev));
        if (ret < 0)
            return ret;
    } else if (cu_ext->cuDeviceGetUuid) {
        ret = CHECK_CU_EXT(cu_ext->cuDeviceGetUuid(&cuda_uuid, dev));
        if (ret < 0)
            return ret;
    }

    for (unsigned i = 0; i < 2; i++) {
        const uint8_t *u = (const uint8_t*)cuda_uuid.bytes;

        snprintf(
            uuid_buf, sizeof(uuid_buf),
            "%s%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
            (i ? "MIG-" : "GPU-"),
            u[0], u[1], u[2],  u[3],  u[4],  u[5],  u[6],  u[7],
            u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]
        );
        ret = CHECK_ML(nvml_ext->nvmlDeviceGetHandleByUUID(uuid_buf, &nvml_dev));
        if (!ret)
            break;
    }
    if (!nvml_dev)
        return AVERROR(ENOSYS);

    mark_section_show_entries(SECTION_ID_DEVICE_UTIL_CUDA, 1, NULL);
    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_UTIL_CUDA);

    print_str("device_name", device_name);
    print_str("device_uuid", uuid_buf);

    if (nvml_ext->nvmlDeviceGetMemoryInfo) {
        nvmlMemory_t memory_info = { 0 };

        ret = CHECK_ML(nvml_ext->nvmlDeviceGetMemoryInfo(nvml_dev, &memory_info));
        if (!ret) {
            print_int("device_memory_total", memory_info.total);
            print_int("device_memory_used", memory_info.used);
            print_int("device_memory_free", memory_info.free);
        }
    }
    if (nvml_ext->nvmlDeviceGetUtilizationRates) {
        nvmlUtilization_t util = { 0 };

        ret = CHECK_ML(nvml_ext->nvmlDeviceGetUtilizationRates(nvml_dev, &util));
        if (!ret) {
            print_int("device_gpu_utilization", util.gpu);
            print_int("device_memory_utilization", util.memory);
        }
    }
    if (nvml_ext->nvmlDeviceGetDecoderUtilization) {
        unsigned dec_util = 0, sample_us = 0;

        ret = CHECK_ML(nvml_ext->nvmlDeviceGetDecoderUtilization(nvml_dev, &dec_util, &sample_us));
        if (!ret)
            print_int("device_decoder_utilization", dec_util);
    }
    if (nvml_ext->nvmlDeviceGetEncoderUtilization) {
        unsigned enc_util = 0, sample_us = 0;

        ret = CHECK_ML(nvml_ext->nvmlDeviceGetEncoderUtilization(nvml_dev, &enc_util, &sample_us));
        if (!ret)
            print_int("device_encoder_utilization", enc_util);
    }
    if (nvml_ext->nvmlDeviceGetEncoderStats) {
        unsigned sessions = 0, avg_fps = 0, avg_latency = 0;

        ret = CHECK_ML(nvml_ext->nvmlDeviceGetEncoderStats(nvml_dev, &sessions, &avg_fps, &avg_latency));
        if (!ret) {
            print_int("device_encoder_sessions", sessions);
            print_int("device_encoder_avg_fps", avg_fps);
            print_int("device_encoder_avg_latency", avg_latency);
        }
    }
    if (nvml_ext->nvmlDeviceGetPcieThroughput) {
        unsigned util = 0;
        static const struct {
            nvmlPcieUtilCounter_t pcie_util_val;
            const char           *pcie_util_str;
        } params[] = {
            { 0 /* NVML_PCIE_UTIL_TX_BYTES */, "device_pcie_tx_throughput" },
            { 1 /* NVML_PCIE_UTIL_RX_BYTES */, "device_pcie_rx_throughput" },
        };

        for (unsigned i = 0; i < FF_ARRAY_ELEMS(params); i++) {
            ret = CHECK_ML(nvml_ext->nvmlDeviceGetPcieThroughput(nvml_dev, params[i].pcie_util_val, &util));
            if (!ret)
                print_int(params[i].pcie_util_str, util);
        }
    }

    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_UTIL_CUDA

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
#   if NVDECAPI_CHECK_VERSION(11, 0)
    case AV_CODEC_ID_AV1:        return cudaVideoCodec_AV1;
#   endif
    default:                     return -1;
    }
}

static int cuda_map_av_to_cuvid_chroma(enum AVPixelFormat pix_fmt)
{
    switch (pix_fmt) {
    case AV_PIX_FMT_NV12:         /* fallthrough */
    case AV_PIX_FMT_P010:         /* fallthrough */
    case AV_PIX_FMT_P012:         /* fallthrough */
    case AV_PIX_FMT_P016:      return cudaVideoChromaFormat_420;
    case AV_PIX_FMT_NV16:         /* fallthrough */
    case AV_PIX_FMT_P210:         /* fallthrough */
    case AV_PIX_FMT_P212:         /* fallthrough */
    case AV_PIX_FMT_P216:      return cudaVideoChromaFormat_422;
    case AV_PIX_FMT_YUV444P:      /* fallthrough */
    case AV_PIX_FMT_YUV444P10MSB: /* fallthrough */
    case AV_PIX_FMT_YUV444P12MSB: /* fallthrough */
    case AV_PIX_FMT_YUV444P16: return cudaVideoChromaFormat_444;
    default:                   return -1;
    }
}

static int cuda_map_av_to_cuvid_surface(enum AVPixelFormat pix_fmt)
{
    switch (pix_fmt) {
    case AV_PIX_FMT_NV12:      return cudaVideoSurfaceFormat_NV12;
    case AV_PIX_FMT_P010:         /* fallthrough */
    case AV_PIX_FMT_P012:         /* fallthrough */
    case AV_PIX_FMT_P016:      return cudaVideoSurfaceFormat_P016;
#   if NVDECAPI_CHECK_VERSION(13, 0)
    case AV_PIX_FMT_NV16:      return cudaVideoSurfaceFormat_NV16;
    case AV_PIX_FMT_P210:         /* fallthrough */
    case AV_PIX_FMT_P212:         /* fallthrough */
    case AV_PIX_FMT_P216:      return cudaVideoSurfaceFormat_P216;
#   endif
#   if NVDECAPI_CHECK_VERSION(9, 0)
    case AV_PIX_FMT_YUV444P:   return cudaVideoSurfaceFormat_YUV444;
    case AV_PIX_FMT_YUV444P10MSB: /* fallthrough */
    case AV_PIX_FMT_YUV444P12MSB: /* fallthrough */
    case AV_PIX_FMT_YUV444P16: return cudaVideoSurfaceFormat_YUV444_16Bit;
#   endif
    default:                   return -1;
    }
}
#endif

int print_cuda_decoder_info(AVTextFormatContext *tfc, AVBufferRef *cuda_ref)
{
#if (CONFIG_CUDA && (CONFIG_CUVID || CONFIG_NVDEC))
    AVHWDeviceContext *dev_ctx = NULL;
    AVCUDADeviceContext *hwctx = NULL;
    CUVIDDECODECAPS caps = { 0 };
    CudaFunctions *cu = NULL;
    CUcontext dummy;
    int header_printed = 0;
    int ret = 0;
    unsigned i, j;

    if (!tfc || !cuda_ref)
        return AVERROR(EINVAL);

    if ((ret = init_cuvid_functions()) < 0)
        return AVERROR(ENOSYS);

    if (!cuvid->cuvidGetDecoderCaps)
        return AVERROR(ENOSYS);

    dev_ctx = (AVHWDeviceContext*)cuda_ref->data;
    if (!dev_ctx)
        return AVERROR(ENOSYS);

    hwctx = dev_ctx->hwctx;
    cu = hwctx->internal->cuda_dl;

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
            const AVPixFmtDescriptor *desc;

            surface = cuda_map_av_to_cuvid_surface(format);
            if (surface < 0)
                continue;

            desc = av_pix_fmt_desc_get(format);
            caps.nBitDepthMinus8 = FFMIN(desc->comp[0].depth, 12) - 8;
            caps.eChromaFormat = cuda_map_av_to_cuvid_chroma(format);
            if (caps.eChromaFormat < 0)
                continue;

            ret = CHECK_CU(cuvid->cuvidGetDecoderCaps(&caps));
            if (ret < 0)
                continue;

            if (!caps.bIsSupported || !(caps.nOutputFormatMask & (1 << surface)))
                continue;

            if (!header_printed) {
                mark_section_show_entries(SECTION_ID_DEVICE_DECODERS_CUDA, 1, NULL);
                avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_DECODERS_CUDA);
                header_printed = 1;
            }

            if (!header2_printed) {
                mark_section_show_entries(SECTION_ID_DEVICE_DECODER, 1, NULL);
                avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_DECODER);
                print_str("codec_name", avcodec_get_name(mode->codec));
                print_int("codec_id", mode->codec);
                print_str("codec_desc", mode->name);
                print_int("min_width", caps.nMinWidth);
                print_int("min_height", caps.nMinHeight);
                print_int("max_width", caps.nMaxWidth);
                print_int("max_height", caps.nMaxHeight);
                print_int("max_mb_count", caps.nMaxMBCount);
                mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMTS, 1, NULL);
                avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMTS);
                header2_printed = 1;
            }

            mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMT, 1, NULL);
            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMT);
            print_str("format_name", av_get_pix_fmt_name(format));
            print_int("format_id", format);
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMT
        }

        if (header2_printed) {
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMTS
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_DECODER
        }
    }
    if (header_printed)
        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_DECODERS_CUDA

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
#endif
    return 0;
}

#if (CONFIG_CUDA && CONFIG_NVENC)
static GUID cuda_map_av_to_nvenc_codec_guid(enum AVCodecID codec)
{
    static const GUID g = { 0 };

    switch (codec) {
    case AV_CODEC_ID_H264: return NV_ENC_CODEC_H264_GUID;
    case AV_CODEC_ID_HEVC: return NV_ENC_CODEC_HEVC_GUID;
#   if NVENCAPI_CHECK_VERSION(12, 0)
    case AV_CODEC_ID_AV1:  return NV_ENC_CODEC_AV1_GUID;
#   endif
    default:               return g;
    }
}

static GUID cuda_map_av_to_nvenc_profile_guid(enum AVCodecID codec, int profile)
{
    static const GUID g = { 0 };

    if (codec == AV_CODEC_ID_H264) {
        switch (profile) {
        case AV_PROFILE_H264_BASELINE:            return NV_ENC_H264_PROFILE_BASELINE_GUID;
        case AV_PROFILE_H264_MAIN:                return NV_ENC_H264_PROFILE_MAIN_GUID;
        case AV_PROFILE_H264_HIGH:                return NV_ENC_H264_PROFILE_HIGH_GUID;
#   if NVENCAPI_CHECK_VERSION(13, 0)
        case AV_PROFILE_H264_HIGH_10:             return NV_ENC_H264_PROFILE_HIGH_10_GUID;
        case AV_PROFILE_H264_HIGH_422:            return NV_ENC_H264_PROFILE_HIGH_422_GUID;
#   endif
        case AV_PROFILE_H264_HIGH_444_PREDICTIVE: return NV_ENC_H264_PROFILE_HIGH_444_GUID;
        }
    } else if (codec == AV_CODEC_ID_HEVC) {
        switch (profile) {
        case AV_PROFILE_HEVC_MAIN:    return NV_ENC_HEVC_PROFILE_MAIN_GUID;
        case AV_PROFILE_HEVC_MAIN_10: return NV_ENC_HEVC_PROFILE_MAIN10_GUID;
        case AV_PROFILE_HEVC_REXT:    return NV_ENC_HEVC_PROFILE_FREXT_GUID;
        }
    } else if (codec == AV_CODEC_ID_AV1) {
        switch (profile) {
#   if NVENCAPI_CHECK_VERSION(12, 0)
        case AV_PROFILE_AV1_MAIN:      return NV_ENC_AV1_PROFILE_MAIN_GUID;
#   endif
        }
    }

    return g;
}

static NV_ENC_BUFFER_FORMAT cuda_map_av_to_nvenc_buffer_format(enum AVPixelFormat pix_fmt)
{
    switch (pix_fmt) {
    case AV_PIX_FMT_NV12:      return NV_ENC_BUFFER_FORMAT_NV12;
#   if NVENCAPI_CHECK_VERSION(13, 0)
    case AV_PIX_FMT_NV16:      return NV_ENC_BUFFER_FORMAT_NV16;
#   endif
    case AV_PIX_FMT_YUV420P:   return NV_ENC_BUFFER_FORMAT_YV12;
    case AV_PIX_FMT_P010:      /* fallthrough */
    case AV_PIX_FMT_P016:      return NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
#   if NVENCAPI_CHECK_VERSION(13, 0)
    case AV_PIX_FMT_P210:      /* fallthrough */
    case AV_PIX_FMT_P216:      return NV_ENC_BUFFER_FORMAT_P210;
#   endif
    case AV_PIX_FMT_GBRP:      /* fallthrough */
    case AV_PIX_FMT_YUV444P:   return NV_ENC_BUFFER_FORMAT_YUV444;
    case AV_PIX_FMT_GBRP10MSB:
    case AV_PIX_FMT_GBRP16:    /* fallthrough */
    case AV_PIX_FMT_YUV444P10MSB:
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

int print_cuda_encoder_info(AVTextFormatContext *tfc, AVBufferRef *cuda_ref)
{
#if (CONFIG_CUDA && CONFIG_NVENC)
    AVHWDeviceContext *dev_ctx = NULL;
    AVCUDADeviceContext *hwctx = NULL;
    CudaFunctions *cu = NULL;
    CUdevice dev;
    CUcontext dummy;
    unsigned i, j, k;
    unsigned major = 0, minor = 0;
    int val, ret = 0;
    int header_printed = 0;
    void *nvenc_hdl = NULL;
    NVENCSTATUS err = 0;
    unsigned codec_cnt = 0;
    GUID *codec_list = NULL;

    if (!tfc || !cuda_ref)
        return AVERROR(EINVAL);

    if ((ret = init_cuda_functions()) < 0)
        return AVERROR(ENOSYS);

    if ((ret = init_nvenc_functions()) < 0)
        return AVERROR(ENOSYS);

    dev_ctx = (AVHWDeviceContext*)cuda_ref->data;
    if (!dev_ctx)
        return AVERROR(ENOSYS);

    hwctx = dev_ctx->hwctx;
    cu = hwctx->internal->cuda_dl;
    dev = hwctx->internal->cuda_device;

    ret = CHECK_CU(cu->cuDeviceComputeCapability(&major, &minor, dev));
    if (ret < 0)
        return ret;

    if (((major << 4) | minor) < 0x30 /* NVENC_CAP */)
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
            av_log(NULL, AV_LOG_DEBUG, "Nvenc OpenEncodeSessionEx failed\n");
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
            mark_section_show_entries(SECTION_ID_DEVICE_ENCODERS_CUDA, 1, NULL);
            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_ENCODERS_CUDA);
            header_printed = 1;
        }

        mark_section_show_entries(SECTION_ID_DEVICE_ENCODER, 1, NULL);
        avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_ENCODER);
        print_str("codec_name", avcodec_get_name(mode->codec));
        print_int("codec_id", mode->codec);
        print_str("codec_desc", mode->name);

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

        /* Profiles */
        err = nvenc_fns.nvEncGetEncodeProfileGUIDCount(nvenc_hdl, codec_guid, &profile_cnt);
        if (err == NV_ENC_SUCCESS && profile_cnt) {
            profile_list = av_malloc_array(profile_cnt, sizeof(*profile_list));
            if (profile_list)
                err = nvenc_fns.nvEncGetEncodeProfileGUIDs(nvenc_hdl, codec_guid,
                                                           profile_list, profile_cnt, &profile_cnt);
        }
        for (j = 0; mode->profiles[j] != AV_PROFILE_UNKNOWN; j++) {
            GUID profile_guid = { 0 };

            if (!profile_cnt || !profile_list || err != NV_ENC_SUCCESS)
                break;

            profile_guid = cuda_map_av_to_nvenc_profile_guid(mode->codec, mode->profiles[j]);
            for (const GUID *g = &profile_list[0]; g < &profile_list[profile_cnt]; g++) {
                if (!memcmp(g, &profile_guid, sizeof(*g))) {
                    if (!header2_printed) {
                        mark_section_show_entries(SECTION_ID_DEVICE_PROFILES, 1, NULL);
                        avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PROFILES);
                        header2_printed = 1;
                    }
                    mark_section_show_entries(SECTION_ID_DEVICE_PROFILE, 1, NULL);
                    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PROFILE);
                    print_str("profile_name", avcodec_profile_name(mode->codec, mode->profiles[j]));
                    print_int("profile_id", mode->profiles[j]);
                    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PROFILE
                    break;
                }
            }
        }
        if (header2_printed)
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PROFILES

        /* Formats */
        err = nvenc_fns.nvEncGetInputFormatCount(nvenc_hdl, codec_guid, &fmt_cnt);
        if (err == NV_ENC_SUCCESS && fmt_cnt) {
            fmt_list = av_malloc_array(fmt_cnt, sizeof(*fmt_list));
            if (fmt_list)
                err = nvenc_fns.nvEncGetInputFormats(nvenc_hdl, codec_guid,
                                                     fmt_list, fmt_cnt, &fmt_cnt);
        }
        for (j = 0; mode->formats[j] != AV_PIX_FMT_NONE; j++) {
            int buffer_format = NV_ENC_BUFFER_FORMAT_UNDEFINED;

            if (!fmt_cnt || !fmt_list || err != NV_ENC_SUCCESS)
                break;

            buffer_format = cuda_map_av_to_nvenc_buffer_format(mode->formats[j]);
            if (buffer_format == NV_ENC_BUFFER_FORMAT_UNDEFINED)
                continue;

            for (k = 0; k < fmt_cnt; k++) {
                if (buffer_format == fmt_list[k]) {
                    if (!header3_printed) {
                        mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMTS, 1, NULL);
                        avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMTS);
                        header3_printed = 1;
                    }
                    mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMT, 1, NULL);
                    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMT);
                    print_str("format_name", av_get_pix_fmt_name(mode->formats[j]));
                    print_int("format_id", mode->formats[j]);
                    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMT
                    break;
                }
            }
        }
        if (header3_printed)
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMTS

        /* Presets */
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
                        mark_section_show_entries(SECTION_ID_DEVICE_PRESETS, 1, NULL);
                        avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PRESETS);
                        header4_printed = 1;
                    }
                    mark_section_show_entries(SECTION_ID_DEVICE_PRESET, 1, NULL);
                    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PRESET);
                    print_str("preset_name", nvenc_codec_presets[j].preset_str);
                    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PRESET
                    break;
                }
            }
        }
        if (header4_printed)
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PRESETS

        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_ENCODER

        if (profile_list)
            av_free(profile_list);
        if (fmt_list)
            av_free(fmt_list);
        if (preset_list)
            av_free(preset_list);
    }
    if (header_printed)
        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_ENCODERS_CUDA

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
