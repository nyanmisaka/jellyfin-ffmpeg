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
#include "libavutil/wchar_filename.h"

#if (CONFIG_AMF && CONFIG_D3D11VA)
#   include <AMF/core/Factory.h>
#   include <AMF/components/VideoEncoderVCE.h>
#   include <AMF/components/VideoEncoderHEVC.h>
#   include <AMF/components/VideoEncoderAV1.h>

#   define COBJMACROS
#   include <windows.h>
#   include <initguid.h>
#   include <d3d11.h>
#   include <dxgi1_2.h>
#   include "libavutil/hwcontext_d3d11va.h"
#   include "compat/w32dlfcn.h"
#endif

#if (CONFIG_AMF && CONFIG_D3D11VA)
amf_handle         amf_lib = NULL;
AMFInit_Fn         amf_init_fn = NULL;
AMFQueryVersion_Fn amf_ver_fn = NULL;
amf_uint64         amf_ver = 0;
AMFFactory        *amf_factory = NULL;
AMFContext        *amf_ctx = NULL;

typedef struct AmfEncCap {
    const wchar_t *cap_val;
    const char    *cap_str;
} AmfEncCap;

typedef struct AmfEncMode {
    const char               *name;
    const enum AVCodecID      codec;
    const wchar_t            *comp_name;
    const int                *profiles;
    const enum AVPixelFormat *formats;
    const AmfEncCap          *caps;
    const size_t              nb_caps;
} AmfEncMode;

static const int enc_profiles_h264[] = {
    AV_PROFILE_H264_BASELINE,
    AV_PROFILE_H264_CONSTRAINED_BASELINE,
    AV_PROFILE_H264_MAIN,
    AV_PROFILE_H264_HIGH,
    AV_PROFILE_UNKNOWN
};
static const int enc_profiles_hevc[] = {
    AV_PROFILE_HEVC_MAIN,
    AV_PROFILE_HEVC_MAIN_10,
    AV_PROFILE_UNKNOWN
};
static const int enc_profiles_av1[] = {
    AV_PROFILE_AV1_MAIN,
    AV_PROFILE_UNKNOWN
};

static const enum AVPixelFormat enc_formats_8_10_yuv_rgb[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_BGR0,
    AV_PIX_FMT_RGB0,
    AV_PIX_FMT_BGRA,
    AV_PIX_FMT_ARGB,
    AV_PIX_FMT_RGBA,
    AV_PIX_FMT_X2BGR10,
    AV_PIX_FMT_NONE
};

static const AmfEncCap enc_caps_h264[] = {
    { AMF_VIDEO_ENCODER_CAP_MAX_BITRATE,                     "max_bitrate"                },
    { AMF_VIDEO_ENCODER_CAP_NUM_OF_STREAMS,                  "num_of_streams"             },
    { AMF_VIDEO_ENCODER_CAP_MAX_PROFILE,                     "max_profile"                },
    { AMF_VIDEO_ENCODER_CAP_MAX_LEVEL,                       "max_level"                  },
    { AMF_VIDEO_ENCODER_CAP_BFRAMES,                         "bframes"                    },
    { AMF_VIDEO_ENCODER_CAP_MIN_REFERENCE_FRAMES,            "min_reference_frames"       },
    { AMF_VIDEO_ENCODER_CAP_MAX_REFERENCE_FRAMES,            "max_reference_frames"       },
    { AMF_VIDEO_ENCODER_CAP_MAX_TEMPORAL_LAYERS,             "max_temporal_layers"        },
    { AMF_VIDEO_ENCODER_CAP_FIXED_SLICE_MODE,                "fixed_slice_mode"           },
    { AMF_VIDEO_ENCODER_CAP_NUM_OF_HW_INSTANCES,             "num_of_hw_instances"        },
    { AMF_VIDEO_ENCODER_CAP_COLOR_CONVERSION,                "color_conversion"           },
    { AMF_VIDEO_ENCODER_CAP_PRE_ANALYSIS,                    "pre_analysis"               },
    { AMF_VIDEO_ENCODER_CAP_ROI,                             "roi"                        },
    { AMF_VIDEO_ENCODER_CAP_MAX_THROUGHPUT,                  "max_throughput"             },
    { AMF_VIDEO_ENCODER_CAP_REQUESTED_THROUGHPUT,            "requested_throughput"       },
    { AMF_VIDEO_ENCODER_CAP_QUERY_TIMEOUT_SUPPORT,           "query_timeout_support"      },
    { AMF_VIDEO_ENCODER_CAP_SUPPORT_SLICE_OUTPUT,            "support_slice_output"       },
    { AMF_VIDEO_ENCODER_CAP_SUPPORT_SMART_ACCESS_VIDEO,      "support_smart_access_video" },
};
static const AmfEncCap enc_caps_hevc[] = {
    { AMF_VIDEO_ENCODER_HEVC_CAP_MAX_BITRATE,                "max_bitrate"                },
    { AMF_VIDEO_ENCODER_HEVC_CAP_NUM_OF_STREAMS,             "num_of_streams"             },
    { AMF_VIDEO_ENCODER_HEVC_CAP_MAX_PROFILE,                "max_profile"                },
    { AMF_VIDEO_ENCODER_HEVC_CAP_MAX_TIER,                   "max_tier"                   },
    { AMF_VIDEO_ENCODER_HEVC_CAP_MAX_LEVEL,                  "max_level"                  },
    { AMF_VIDEO_ENCODER_HEVC_CAP_MIN_REFERENCE_FRAMES,       "min_reference_frames"       },
    { AMF_VIDEO_ENCODER_HEVC_CAP_MAX_REFERENCE_FRAMES,       "max_reference_frames"       },
    { AMF_VIDEO_ENCODER_HEVC_CAP_MAX_TEMPORAL_LAYERS,        "max_temporal_layers"        },
    { AMF_VIDEO_ENCODER_HEVC_CAP_NUM_OF_HW_INSTANCES,        "num_of_hw_instances"        },
    { AMF_VIDEO_ENCODER_HEVC_CAP_COLOR_CONVERSION,           "color_conversion"           },
    { AMF_VIDEO_ENCODER_HEVC_CAP_PRE_ANALYSIS,               "pre_analysis"               },
    { AMF_VIDEO_ENCODER_HEVC_CAP_ROI,                        "roi"                        },
    { AMF_VIDEO_ENCODER_HEVC_CAP_MAX_THROUGHPUT,             "max_throughput"             },
    { AMF_VIDEO_ENCODER_HEVC_CAP_REQUESTED_THROUGHPUT,       "requested_throughput"       },
    { AMF_VIDEO_ENCODER_HEVC_CAP_QUERY_TIMEOUT_SUPPORT,      "query_timeout_support"      },
    { AMF_VIDEO_ENCODER_HEVC_CAP_SUPPORT_SLICE_OUTPUT,       "support_slice_output"       },
    { AMF_VIDEO_ENCODER_HEVC_CAP_SUPPORT_SMART_ACCESS_VIDEO, "support_smart_access_video" },
};
static const AmfEncCap enc_caps_av1[] = {
    { AMF_VIDEO_ENCODER_AV1_CAP_NUM_OF_HW_INSTANCES,         "num_of_hw_instances"        },
    { AMF_VIDEO_ENCODER_AV1_CAP_MAX_THROUGHPUT,              "max_throughput"             },
    { AMF_VIDEO_ENCODER_AV1_CAP_REQUESTED_THROUGHPUT,        "requested_throughput"       },
    { AMF_VIDEO_ENCODER_AV1_CAP_COLOR_CONVERSION,            "color_conversion"           },
    { AMF_VIDEO_ENCODER_AV1_CAP_PRE_ANALYSIS,                "pre_analysis"               },
    { AMF_VIDEO_ENCODER_AV1_CAP_MAX_BITRATE,                 "max_bitrate"                },
    { AMF_VIDEO_ENCODER_AV1_CAP_MAX_PROFILE,                 "max_profile"                },
    { AMF_VIDEO_ENCODER_AV1_CAP_MAX_LEVEL,                   "max_level"                  },
    { AMF_VIDEO_ENCODER_AV1_CAP_MAX_NUM_TEMPORAL_LAYERS,     "max_num_temporal_layers"    },
    { AMF_VIDEO_ENCODER_AV1_CAP_MAX_NUM_LTR_FRAMES,          "max_num_ltr_frames"         },
    { AMF_VIDEO_ENCODER_AV1_CAP_SUPPORT_TILE_OUTPUT,         "support_tile_output"        },
    { AMF_VIDEO_ENCODER_AV1_CAP_BFRAMES,                     "bframes"                    },
    { AMF_VIDEO_ENCODER_AV1_CAP_SUPPORT_SMART_ACCESS_VIDEO,  "support_smart_access_video" },
    { AMF_VIDEO_ENCODER_AV1_CAP_WIDTH_ALIGNMENT_FACTOR,      "width_alignment_factor"     },
    { AMF_VIDEO_ENCODER_AV1_CAP_HEIGHT_ALIGNMENT_FACTOR,     "height_alignment_factor"    },
};

static const AmfEncMode amfenc_modes[] = {
    { "AMF H.264 Encoder", AV_CODEC_ID_H264, AMFVideoEncoderVCE_AVC,
        enc_profiles_h264, enc_formats_8_10_yuv_rgb, enc_caps_h264, FF_ARRAY_ELEMS(enc_caps_h264) },
    { "AMF HEVC Encoder",  AV_CODEC_ID_HEVC, AMFVideoEncoder_HEVC,
        enc_profiles_hevc, enc_formats_8_10_yuv_rgb, enc_caps_hevc, FF_ARRAY_ELEMS(enc_caps_hevc) },
    { "AMF AV1 Encoder",   AV_CODEC_ID_AV1,  AMFVideoEncoder_AV1,
        enc_profiles_av1,  enc_formats_8_10_yuv_rgb, enc_caps_av1,  FF_ARRAY_ELEMS(enc_caps_av1)  },
    { NULL, 0, NULL, NULL, NULL, NULL, 0 },
};
#endif

int init_amf_functions(const wchar_t *amf_dll_path)
{
    int ret = AVERROR(ENOSYS);
#if (CONFIG_AMF && CONFIG_D3D11VA)
    AMF_RESULT res = AMF_OK;

    if (amf_dll_path) {
        char *amf_dll_path_utf8 = NULL;

        if (!wchartoutf8(amf_dll_path, &amf_dll_path_utf8)) {
            amf_lib = dlopen(amf_dll_path_utf8, RTLD_NOW | RTLD_LOCAL);
            av_free(amf_dll_path_utf8);
        }
    }
    if (!amf_lib) {
        amf_lib = dlopen(AMF_DLL_NAMEA, RTLD_NOW | RTLD_LOCAL);
        if (!amf_lib)
            return AVERROR(ENOSYS);
    }

    if (!amf_ver_fn) {
        amf_ver_fn = (AMFQueryVersion_Fn)dlsym(amf_lib, AMF_QUERY_VERSION_FUNCTION_NAME);
        if (!amf_ver_fn) {
            ret = AVERROR(ENOSYS);
            goto exit;
        }
        res = amf_ver_fn(&amf_ver);
        if (res != AMF_OK) {
            ret = AVERROR(ENOSYS);
            goto exit;
        }
    }

    if (!amf_init_fn) {
        amf_init_fn = (AMFInit_Fn)dlsym(amf_lib, AMF_INIT_FUNCTION_NAME);
        if (!amf_init_fn) {
            ret = AVERROR(ENOSYS);
            goto exit;
        }
    }

    if (!amf_factory) {
        res = amf_init_fn(AMF_FULL_VERSION, &amf_factory);
        if (res != AMF_OK) {
            ret = AVERROR(ENOSYS);
            goto exit;
        }
    }

    if (amf_ctx) {
        amf_ctx->pVtbl->Terminate(amf_ctx);
        amf_ctx->pVtbl->Release(amf_ctx);
        amf_ctx = NULL;
    }

    res = amf_factory->pVtbl->CreateContext(amf_factory, &amf_ctx);
    if (res != AMF_OK) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    return 0;
exit:
    uninit_amf_functions();
#endif
    return ret;
}

void uninit_amf_functions(void)
{
#if (CONFIG_AMF && CONFIG_D3D11VA)
    if (amf_ctx) {
        amf_ctx->pVtbl->Terminate(amf_ctx);
        amf_ctx->pVtbl->Release(amf_ctx);
        amf_ctx = NULL;
    }
    if (amf_lib) {
        dlclose(amf_lib);
        amf_lib = NULL;
        amf_init_fn = NULL;
        amf_ver_fn = NULL;
    }
    amf_ver = 0;
    amf_factory = NULL;
#endif
}

int create_derive_amf_device_from_d3d11va(AVBufferRef *d3d11va_ref, const wchar_t *amf_dll_path)
{
    int ret = AVERROR(ENOSYS);
#if (CONFIG_AMF && CONFIG_D3D11VA)
    AVHWDeviceContext    *dev_ctx = NULL;
    AVD3D11VADeviceContext *hwctx = NULL;
    AMF_RESULT res = AMF_OK;

    if (!d3d11va_ref)
        return AVERROR(EINVAL);
    if ((ret = init_amf_functions(amf_dll_path)) < 0)
        goto exit;

    dev_ctx = (AVHWDeviceContext*)d3d11va_ref->data;
    hwctx = dev_ctx->hwctx;

    res = amf_ctx->pVtbl->InitDX11(amf_ctx, hwctx->device, AMF_DX11_1);
    if (res != AMF_OK && res != AMF_ALREADY_INITIALIZED) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }
    return 0;
exit:
#endif
    return ret;
}

int print_amf_device_info_from_d3d11va(AVTextFormatContext *tfc)
{
#if (CONFIG_AMF && CONFIG_D3D11VA)
    static const amf_uint64 api_ver = AMF_FULL_VERSION;

    if (!tfc || !amf_ctx)
        return AVERROR(EINVAL);

    mark_section_show_entries(SECTION_ID_DEVICE_INFO_AMF, 1, NULL);
    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_INFO_AMF);

    print_int("device_amf_impl_version_major", AMF_GET_MAJOR_VERSION(amf_ver));
    print_int("device_amf_impl_version_minor", AMF_GET_MINOR_VERSION(amf_ver));
    print_int("device_amf_impl_version_sub_minor", AMF_GET_SUBMINOR_VERSION(amf_ver));
    print_int("device_amf_impl_version_build", AMF_GET_BUILD_VERSION(amf_ver));
    print_int("device_amf_api_version_major", AMF_GET_MAJOR_VERSION(api_ver));
    print_int("device_amf_api_version_minor", AMF_GET_MINOR_VERSION(api_ver));
    print_int("device_amf_api_version_sub_minor", AMF_GET_SUBMINOR_VERSION(api_ver));
    print_int("device_amf_api_version_build", AMF_GET_BUILD_VERSION(api_ver));

    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_INFO_AMF
#endif
    return 0;
}

#if (CONFIG_AMF && CONFIG_D3D11VA)
static int amf_map_av_to_amfenc_profile(enum AVCodecID codec, int profile)
{
    if (codec == AV_CODEC_ID_H264) {
        switch (profile) {
        case AV_PROFILE_H264_BASELINE:             return AMF_VIDEO_ENCODER_PROFILE_BASELINE;
        case AV_PROFILE_H264_CONSTRAINED_BASELINE: return AMF_VIDEO_ENCODER_PROFILE_CONSTRAINED_BASELINE;
        case AV_PROFILE_H264_MAIN:                 return AMF_VIDEO_ENCODER_PROFILE_MAIN;
        case AV_PROFILE_H264_HIGH:                 return AMF_VIDEO_ENCODER_PROFILE_HIGH;
        }
    } else if (codec == AV_CODEC_ID_HEVC) {
        switch (profile) {
        case AV_PROFILE_HEVC_MAIN:    return AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN;
        case AV_PROFILE_HEVC_MAIN_10: return AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN_10;
        }
    } else if (codec == AV_CODEC_ID_AV1) {
        switch (profile) {
        case AV_PROFILE_AV1_MAIN:     return AMF_VIDEO_ENCODER_AV1_PROFILE_MAIN;
        }
    }

    return -1;
}

static AMF_SURFACE_FORMAT amf_map_av_to_amf_surface_format(enum AVPixelFormat pix_fmt)
{
    switch (pix_fmt) {
    case AV_PIX_FMT_YUV420P: return AMF_SURFACE_YUV420P;
    case AV_PIX_FMT_NV12:    return AMF_SURFACE_NV12;
    case AV_PIX_FMT_P010:    return AMF_SURFACE_P010;
    case AV_PIX_FMT_YUYV422: return AMF_SURFACE_YUY2;
    case AV_PIX_FMT_GRAY8:   return AMF_SURFACE_GRAY8;
    case AV_PIX_FMT_BGR0:    return AMF_SURFACE_BGRA;
    case AV_PIX_FMT_RGB0:    return AMF_SURFACE_RGBA;
    case AV_PIX_FMT_BGRA:    return AMF_SURFACE_BGRA;
    case AV_PIX_FMT_ARGB:    return AMF_SURFACE_ARGB;
    case AV_PIX_FMT_RGBA:    return AMF_SURFACE_RGBA;
    case AV_PIX_FMT_X2BGR10: return AMF_SURFACE_R10G10B10A2;
    default:                 return AMF_SURFACE_UNKNOWN;
    }
}
#endif

int print_amf_encoder_info_from_d3d11va(AVTextFormatContext *tfc)
{
#if (CONFIG_AMF && CONFIG_D3D11VA)
    AMF_RESULT res = AMF_OK;
    int header_printed = 0;
    unsigned i, j, k;

    if (!tfc || !amf_ctx || !amf_factory)
        return AVERROR(EINVAL);

    for (i = 0; amfenc_modes[i].name; i++) {
        int header2_printed = 0;
        int header3_printed = 0;
        AMFComponent *encoder = NULL;
        AMFCaps *encoder_caps = NULL;
        AMFIOCaps *input_caps = NULL;
        const AmfEncMode *mode = &amfenc_modes[i];
        int max_depth = 8;
        amf_int64 max_profile = -1;
        amf_int32 num_formats = 0;

        if (!mode->comp_name || !mode->formats || !mode->caps)
            continue;

        res = amf_factory->pVtbl->CreateComponent(amf_factory, amf_ctx, mode->comp_name, &encoder);
        if (res != AMF_OK || !encoder)
            continue;

        if (!header_printed) {
            mark_section_show_entries(SECTION_ID_DEVICE_ENCODERS_AMF, 1, NULL);
            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_ENCODERS_AMF);
            header_printed = 1;
        }

        mark_section_show_entries(SECTION_ID_DEVICE_ENCODER, 1, NULL);
        avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_ENCODER);
        print_str("codec_name", avcodec_get_name(mode->codec));
        print_int("codec_id", mode->codec);
        print_str("codec_desc", mode->name);

        res = encoder->pVtbl->GetCaps(encoder, &encoder_caps);
        if (res == AMF_OK && encoder_caps) {
            res = encoder_caps->pVtbl->GetInputCaps(encoder_caps, &input_caps);
            if (res == AMF_OK && input_caps) {
                amf_int32 min_w = 0, max_w = 0;
                amf_int32 min_h = 0, max_h = 0;
                amf_int32 vert_align = 0;
                amf_bool interlaced_support = 0;

                input_caps->pVtbl->GetWidthRange(input_caps, &min_w, &max_w);
                input_caps->pVtbl->GetHeightRange(input_caps, &min_h, &max_h);
                print_int("min_width",  min_w);
                print_int("min_height", min_h);
                print_int("max_width",  max_w);
                print_int("max_height", max_h);
                vert_align = input_caps->pVtbl->GetVertAlign(input_caps);
                print_int("vertical_align", vert_align);
                interlaced_support = input_caps->pVtbl->IsInterlacedSupported(input_caps);
                print_int("interlaced_support", interlaced_support);

                num_formats = input_caps->pVtbl->GetNumOfFormats(input_caps);
            }

            for (j = 0; j < mode->nb_caps; j++) {
                AMFVariantStruct val = { 0 };

                res = encoder_caps->pVtbl->GetProperty(encoder_caps, mode->caps[j].cap_val, &val);
                if (res == AMF_OK) {
                    if (val.type == AMF_VARIANT_BOOL)
                        print_int(mode->caps[j].cap_str, val.boolValue);
                    else if (val.type == AMF_VARIANT_INT64) {
                        print_int(mode->caps[j].cap_str, val.int64Value);

                        if (!wcscmp(mode->caps[j].cap_val, AMF_VIDEO_ENCODER_CAP_MAX_PROFILE) ||
                            !wcscmp(mode->caps[j].cap_val, AMF_VIDEO_ENCODER_HEVC_CAP_MAX_PROFILE) ||
                            !wcscmp(mode->caps[j].cap_val, AMF_VIDEO_ENCODER_AV1_CAP_MAX_PROFILE)) {
                            max_profile = val.int64Value;
                        }
                    }
                    AMFVariantClear(&val);
                }
            }

            /* Formats */
            for (j = 0; mode->formats[j] != AV_PIX_FMT_NONE; j++) {
                AMF_SURFACE_FORMAT surface_format = AMF_SURFACE_UNKNOWN;

                if (!num_formats)
                    break;

                surface_format = amf_map_av_to_amf_surface_format(mode->formats[j]);
                if (surface_format == AMF_SURFACE_UNKNOWN)
                    continue;

                for (k = 0; k < num_formats; k++) {
                    AMF_SURFACE_FORMAT input_format = AMF_SURFACE_UNKNOWN;
                    amf_bool is_native = 0;

                    res = input_caps->pVtbl->GetFormatAt(input_caps, k, &input_format, &is_native);
                    if (res != AMF_OK)
                        continue;

                    if (surface_format == input_format) {
                        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(mode->formats[j]);

                        if (desc)
                            max_depth = FFMAX(FFMIN(desc->comp[0].depth, 12), max_depth);

                        if (!header2_printed) {
                            mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMTS, 1, NULL);
                            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMTS);
                            header2_printed = 1;
                        }
                        mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMT, 1, NULL);
                        avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMT);
                        print_str("format_name", av_get_pix_fmt_name(mode->formats[j]));
                        print_int("format_id", mode->formats[j]);
                        print_int("is_native", is_native);
                        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMT
                        break;
                    }
                }
            }
            if (header2_printed)
                avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMTS

            /* Profiles */
            for (j = 0; mode->profiles[j] != AV_PROFILE_UNKNOWN; j++) {
                const int amf_profile = amf_map_av_to_amfenc_profile(mode->codec, mode->profiles[j]);

                /* Fixup legacy drivers */
                if (mode->codec == AV_CODEC_ID_HEVC && max_depth >= 10)
                    max_profile = AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN_10;

                if (amf_profile >= 0 && max_profile >= 0 && amf_profile <= max_profile) {
                    if (!header3_printed) {
                        mark_section_show_entries(SECTION_ID_DEVICE_PROFILES, 1, NULL);
                        avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PROFILES);
                        header3_printed = 1;
                    }
                    mark_section_show_entries(SECTION_ID_DEVICE_PROFILE, 1, NULL);
                    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PROFILE);
                    print_str("profile_name", avcodec_profile_name(mode->codec, mode->profiles[j]));
                    print_int("profile_id", mode->profiles[j]);
                    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PROFILE
                }
            }
            if (header3_printed)
                avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PROFILES

            if (input_caps)
                input_caps->pVtbl->Release(input_caps);
        }
        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_ENCODER

        if (encoder_caps)
            encoder_caps->pVtbl->Release(encoder_caps);
        if (encoder) {
            encoder->pVtbl->Terminate(encoder);
            encoder->pVtbl->Release(encoder);
        }
    }
    if (header_printed)
        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_ENCODERS_AMF
#endif
    return 0;
}
