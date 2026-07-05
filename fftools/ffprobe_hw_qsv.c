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

#if CONFIG_QSV
#   include <mfxvideo.h>
#   include <mfxjpeg.h>
#   include <mfxvp8.h>
#   include "libavutil/hwcontext_qsv.h"
#endif

#if CONFIG_QSV
#   define QSV_VERSION_ATLEAST(MAJOR, MINOR) \
        (MFX_VERSION_MAJOR > (MAJOR) || \
         MFX_VERSION_MAJOR == (MAJOR) && \
         MFX_VERSION_MINOR >= (MINOR))

#   define QSV_RUNTIME_VERSION_ATLEAST(MFX_VERSION, MAJOR, MINOR) \
        ((MFX_VERSION).Major > (MAJOR) || \
         ((MFX_VERSION).Major == (MAJOR) && \
          (MFX_VERSION).Minor >= (MINOR)))

#   define QSV_MFX_PROFILE_H264_CONSTRAINTS \
        (MFX_PROFILE_AVC_CONSTRAINT_SET0 | \
         MFX_PROFILE_AVC_CONSTRAINT_SET1 | \
         MFX_PROFILE_AVC_CONSTRAINT_SET2 | \
         MFX_PROFILE_AVC_CONSTRAINT_SET3 | \
         MFX_PROFILE_AVC_CONSTRAINT_SET4 | \
         MFX_PROFILE_AVC_CONSTRAINT_SET5)

typedef struct QsvDecMode {
    const char               *name;
    const enum AVCodecID      codec;
    const int                *profiles;
    const enum AVPixelFormat *formats;
    const unsigned            legacy;
} QsvDecMode;

static const int profiles_mpeg2[] = {
    AV_PROFILE_MPEG2_SIMPLE,
    AV_PROFILE_MPEG2_MAIN,
    AV_PROFILE_UNKNOWN
};
static const int profiles_vc1[] = {
    AV_PROFILE_VC1_SIMPLE,
    AV_PROFILE_VC1_MAIN,
    AV_PROFILE_VC1_ADVANCED,
    AV_PROFILE_UNKNOWN
};
static const int profiles_h264[] = {
    AV_PROFILE_H264_BASELINE,
    AV_PROFILE_H264_CONSTRAINED_BASELINE,
    AV_PROFILE_H264_MAIN,
    AV_PROFILE_H264_EXTENDED,
    AV_PROFILE_H264_HIGH,
    AV_PROFILE_H264_HIGH_10,
    AV_PROFILE_UNKNOWN
};
static const int profiles_hevc[] = {
    AV_PROFILE_HEVC_MAIN,
    AV_PROFILE_HEVC_MAIN_10,
    AV_PROFILE_HEVC_REXT,
    AV_PROFILE_UNKNOWN
};
static const int profiles_vp9[] = {
    AV_PROFILE_VP9_0,
    AV_PROFILE_VP9_1,
    AV_PROFILE_VP9_2,
    AV_PROFILE_VP9_3,
    AV_PROFILE_UNKNOWN
};
static const int profiles_av1[] = {
    AV_PROFILE_AV1_MAIN,
    AV_PROFILE_UNKNOWN
};
static const int profiles_vvc[] = {
    AV_PROFILE_VVC_MAIN_10,
    AV_PROFILE_UNKNOWN
};
static const int profiles_mjpeg[] = {
    AV_PROFILE_MJPEG_HUFFMAN_BASELINE_DCT,
    AV_PROFILE_UNKNOWN
};

static const enum AVPixelFormat dec_formats_8_420[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NONE
};
static const enum AVPixelFormat dec_formats_8_420_422[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUYV422,
    AV_PIX_FMT_NONE
};
static const enum AVPixelFormat dec_formats_8_10_420[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_NONE
};
static const enum AVPixelFormat dec_formats_8_12_420_444[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P012,
    AV_PIX_FMT_VUYX,
    AV_PIX_FMT_XV30,
    AV_PIX_FMT_XV36,
    AV_PIX_FMT_NONE
};
static const enum AVPixelFormat dec_formats_8_12_420_422_444[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P012,
    AV_PIX_FMT_YUYV422,
    AV_PIX_FMT_Y210,
    AV_PIX_FMT_Y212,
    AV_PIX_FMT_VUYX,
    AV_PIX_FMT_XV30,
    AV_PIX_FMT_XV36,
    AV_PIX_FMT_NONE
};

static const QsvDecMode qsvdec_modes[] = {
    { "QSV MPEG-2 Decoder", AV_CODEC_ID_MPEG2VIDEO, profiles_mpeg2, dec_formats_8_420,            1 },
    { "QSV VC-1 Decoder",   AV_CODEC_ID_VC1,        profiles_vc1,   dec_formats_8_420,            1 },
    { "QSV H.264 Decoder",  AV_CODEC_ID_H264,       profiles_h264,  dec_formats_8_10_420,         1 },
    { "QSV HEVC Decoder",   AV_CODEC_ID_HEVC,       profiles_hevc,  dec_formats_8_12_420_422_444, 0 },
#   ifndef _WIN32
    { "QSV VP8 Decoder",    AV_CODEC_ID_VP8,        NULL,           dec_formats_8_420,            1 },
#   endif
    { "QSV VP9 Decoder",    AV_CODEC_ID_VP9,        profiles_vp9,   dec_formats_8_12_420_444,     0 },
    { "QSV AV1 Decoder",    AV_CODEC_ID_AV1,        profiles_av1,   dec_formats_8_10_420,         0 },
    { "QSV VVC Decoder",    AV_CODEC_ID_VVC,        profiles_vvc,   dec_formats_8_10_420,         0 },
    { "QSV MJPEG Decoder",  AV_CODEC_ID_MJPEG,      profiles_mjpeg, dec_formats_8_420_422,        0 },
    { NULL, 0, NULL, NULL, 0 },
};

typedef struct QsvEncMode {
    const char                *name;
    const enum AVCodecID       codec;
    const int                 *profiles;
    const enum AVPixelFormat  *formats;
    const unsigned short      *rc_modes;
    const unsigned             low_power;
    const unsigned             legacy;
} QsvEncMode;

static const enum AVPixelFormat enc_formats_8_420_rgb[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_BGRA,
    AV_PIX_FMT_NONE
};
static const enum AVPixelFormat enc_formats_8_420_422_rgb[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUYV422,
    AV_PIX_FMT_BGRA,
    AV_PIX_FMT_NONE
};
static const enum AVPixelFormat enc_formats_8_10_420_rgb[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_BGRA,
    AV_PIX_FMT_X2RGB10,
    AV_PIX_FMT_NONE
};
static const enum AVPixelFormat enc_formats_8_10_420_444[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_VUYX,
    AV_PIX_FMT_XV30,
    AV_PIX_FMT_NONE
};
static const enum AVPixelFormat enc_formats_8_10_420_422_444_rgb[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_YUYV422,
    AV_PIX_FMT_Y210,
    AV_PIX_FMT_VUYX,
    AV_PIX_FMT_XV30,
    AV_PIX_FMT_BGRA,
    AV_PIX_FMT_X2RGB10,
    AV_PIX_FMT_NONE
};
static const unsigned short enc_rc_modes_h26x[] = {
    MFX_RATECONTROL_CQP,
    MFX_RATECONTROL_CBR,
    MFX_RATECONTROL_VBR,
    MFX_RATECONTROL_QVBR,
    MFX_RATECONTROL_ICQ,
    0
};
static const unsigned short enc_rc_modes_av1[] = {
    MFX_RATECONTROL_CQP,
    MFX_RATECONTROL_CBR,
    MFX_RATECONTROL_VBR,
    MFX_RATECONTROL_ICQ,
    0
};
static const unsigned short enc_rc_modes_vp9[] = {
    MFX_RATECONTROL_CQP,
    MFX_RATECONTROL_CBR,
    MFX_RATECONTROL_VBR,
    0
};

static const QsvEncMode qsvenc_modes[] = {
    { "QSV H.264 Encoder",             AV_CODEC_ID_H264,  profiles_h264,  enc_formats_8_420_rgb,            enc_rc_modes_h26x, 0, 1 },
    { "QSV H.264 Encoder (Low-Power)", AV_CODEC_ID_H264,  profiles_h264,  enc_formats_8_420_rgb,            enc_rc_modes_h26x, 1, 1 },
    { "QSV HEVC Encoder",              AV_CODEC_ID_HEVC,  profiles_hevc,  enc_formats_8_10_420_422_444_rgb, enc_rc_modes_h26x, 0, 0 },
    { "QSV HEVC Encoder (Low-Power)",  AV_CODEC_ID_HEVC,  profiles_hevc,  enc_formats_8_10_420_422_444_rgb, enc_rc_modes_h26x, 1, 0 },
    { "QSV AV1 Encoder (Low-Power)",   AV_CODEC_ID_AV1,   profiles_av1,   enc_formats_8_10_420_rgb,         enc_rc_modes_av1,  1, 0 },
    { "QSV VP9 Encoder (Low-Power)",   AV_CODEC_ID_VP9,   profiles_vp9,   enc_formats_8_10_420_444,         enc_rc_modes_vp9,  1, 0 },
    { "QSV MJPEG Encoder",             AV_CODEC_ID_MJPEG, profiles_mjpeg, enc_formats_8_420_422_rgb,        NULL,              0, 0 },
    { NULL, 0, NULL, NULL, NULL, 0, 0 },
};

enum QsvVppType {
    QSV_VPP_SCALE,
    QSV_VPP_DEINT,
    QSV_VPP_OVERLAY,
    QSV_VPP_ROTATE,
    QSV_VPP_FLIP,
    QSV_VPP_DENOISE,
    QSV_VPP_DETAIL,
    QSV_VPP_FRAMERATE,
    QSV_VPP_PROCAMP,
    QSV_VPP_TONEMAP,
    QSV_VPP_NONE,
};

typedef struct QsvVppMode {
    const char               *name;
    const enum QsvVppType     vpp;
    const enum AVPixelFormat *formats;
} QsvVppMode;

static const enum AVPixelFormat vpp_formats_10_12_420_422_444[] = {
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P012,
    AV_PIX_FMT_Y210,
    AV_PIX_FMT_Y212,
    AV_PIX_FMT_XV30,
    AV_PIX_FMT_XV36,
    AV_PIX_FMT_NONE
};
static const enum AVPixelFormat vpp_formats_8_10_12_420_422_444_rgb[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P012,
    AV_PIX_FMT_YUYV422,
    AV_PIX_FMT_Y210,
    AV_PIX_FMT_Y212,
    AV_PIX_FMT_VUYX,
    AV_PIX_FMT_XV30,
    AV_PIX_FMT_XV36,
    AV_PIX_FMT_BGRA,
    AV_PIX_FMT_NONE
};

static const QsvVppMode qsvvpp_modes[] = {
    { "QSV VPP Scale Filter",       QSV_VPP_SCALE,     vpp_formats_8_10_12_420_422_444_rgb },
    { "QSV VPP Deinterlace Filter", QSV_VPP_DEINT,     vpp_formats_8_10_12_420_422_444_rgb },
    { "QSV VPP Overlay Filter",     QSV_VPP_OVERLAY,   vpp_formats_8_10_12_420_422_444_rgb },
    { "QSV VPP Rotate Filter",      QSV_VPP_ROTATE,    vpp_formats_8_10_12_420_422_444_rgb },
    { "QSV VPP Flip Filter",        QSV_VPP_FLIP,      vpp_formats_8_10_12_420_422_444_rgb },
    { "QSV VPP Denoise Filter",     QSV_VPP_DENOISE,   vpp_formats_8_10_12_420_422_444_rgb },
    { "QSV VPP Detail Filter",      QSV_VPP_DETAIL,    vpp_formats_8_10_12_420_422_444_rgb },
    { "QSV VPP Framerate Filter",   QSV_VPP_FRAMERATE, vpp_formats_8_10_12_420_422_444_rgb },
    { "QSV VPP Procamp Filter",     QSV_VPP_PROCAMP,   vpp_formats_8_10_12_420_422_444_rgb },
    { "QSV VPP Tonemap Filter",     QSV_VPP_TONEMAP,   vpp_formats_10_12_420_422_444       },
    { NULL, QSV_VPP_NONE, NULL },
};

enum QsvTestType {
    QSV_TEST_BREAK    = -2,
    QSV_TEST_CONTINUE = -1,
    QSV_TEST_SUCCESS  =  0,
};
#endif

int print_qsv_device_info(AVTextFormatContext *tfc, AVBufferRef *qsv_ref)
{
#if CONFIG_QSV
    AVHWDeviceContext *dev_ctx = NULL;
    AVQSVDeviceContext *hwctx = NULL;
    mfxStatus sts;
    mfxIMPL impl;
    mfxVersion ver = { 0 };
    mfxPlatform platform = { 0 };

    if (!tfc || !qsv_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)qsv_ref->data;
    if (!dev_ctx)
        return AVERROR(ENOSYS);

    hwctx = dev_ctx->hwctx;

    sts = MFXQueryIMPL(hwctx->session, &impl);
    if (sts != MFX_ERR_NONE)
        return AVERROR(ENOSYS);

    sts = MFXQueryVersion(hwctx->session, &ver);
    if (sts != MFX_ERR_NONE)
        return AVERROR(ENOSYS);

    mark_section_show_entries(SECTION_ID_DEVICE_INFO_QSV, 1, NULL);
    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_INFO_QSV);

    print_int("device_mfx_impl", impl);
    print_int("device_mfx_impl_version_major", ver.Major);
    print_int("device_mfx_impl_version_minor", ver.Minor);
    print_int("device_mfx_api_version_major", MFX_VERSION_MAJOR);
    print_int("device_mfx_api_version_minor", MFX_VERSION_MINOR);

    sts = MFXVideoCORE_QueryPlatform(hwctx->session, &platform);
    if (sts == MFX_ERR_NONE) {
        print_int("device_mfx_platfrom_code_name", platform.CodeName);
        print_int("device_mfx_platfrom_device_id", platform.DeviceId);
        print_int("device_mfx_platfrom_media_adapter_type", platform.MediaAdapterType);
    }

    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_INFO_QSV
#endif
    return 0;
}

#if CONFIG_QSV
static int qsv_map_av_to_mfx_codec(enum AVCodecID codec)
{
    switch (codec) {
    case AV_CODEC_ID_H264:       return MFX_CODEC_AVC;
    case AV_CODEC_ID_HEVC:       return MFX_CODEC_HEVC;
    case AV_CODEC_ID_MPEG1VIDEO:
    case AV_CODEC_ID_MPEG2VIDEO: return MFX_CODEC_MPEG2;
    case AV_CODEC_ID_VC1:        return MFX_CODEC_VC1;
    case AV_CODEC_ID_VP8:        return MFX_CODEC_VP8;
    case AV_CODEC_ID_MJPEG:      return MFX_CODEC_JPEG;
    case AV_CODEC_ID_VP9:        return MFX_CODEC_VP9;
#   if QSV_VERSION_ATLEAST(1, 34)
    case AV_CODEC_ID_AV1:        return MFX_CODEC_AV1;
#   endif
#   if QSV_VERSION_ATLEAST(2, 11)
    case AV_CODEC_ID_VVC:        return MFX_CODEC_VVC;
#   endif
    default:                     return -1;
    }
}

static uint32_t qsv_map_av_to_mfx_fourcc(enum AVPixelFormat pix_fmt)
{
    switch (pix_fmt) {
    case AV_PIX_FMT_NV12:    return MFX_FOURCC_NV12;
    case AV_PIX_FMT_YUYV422: return MFX_FOURCC_YUY2;
    case AV_PIX_FMT_VUYX:    return MFX_FOURCC_AYUV;
    case AV_PIX_FMT_P010:    return MFX_FOURCC_P010;
    case AV_PIX_FMT_Y210:    return MFX_FOURCC_Y210;
    case AV_PIX_FMT_XV30:    return MFX_FOURCC_Y410;
#   if QSV_VERSION_ATLEAST(1, 31)
    case AV_PIX_FMT_P012:    return MFX_FOURCC_P016;
    case AV_PIX_FMT_Y212:    return MFX_FOURCC_Y216;
    case AV_PIX_FMT_XV36:    return MFX_FOURCC_Y416;
#   endif
    case AV_PIX_FMT_BGRA:    return MFX_FOURCC_RGB4;
    case AV_PIX_FMT_X2RGB10: return MFX_FOURCC_A2RGB10;
    default:                 return 0;
    }
}

static int qsv_map_av_to_mfx_chroma(enum AVPixelFormat pix_fmt)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(pix_fmt);
    return MFX_CHROMAFORMAT_YUV420 +
        !desc->log2_chroma_w + !desc->log2_chroma_h;
}

static unsigned short qsv_map_av_to_mfx_profile(enum AVCodecID codec, int profile)
{
    if (codec == AV_CODEC_ID_MPEG2VIDEO) {
        switch (profile) {
        case AV_PROFILE_MPEG2_SIMPLE: return MFX_PROFILE_MPEG2_SIMPLE;
        case AV_PROFILE_MPEG2_MAIN:   return MFX_PROFILE_MPEG2_MAIN;
        case AV_PROFILE_MPEG2_HIGH:   return MFX_PROFILE_MPEG2_HIGH;
        }
    } else if (codec == AV_CODEC_ID_VC1) {
        switch (profile) {
        case AV_PROFILE_VC1_SIMPLE:   return MFX_PROFILE_VC1_SIMPLE;
        case AV_PROFILE_VC1_MAIN:     return MFX_PROFILE_VC1_MAIN;
        case AV_PROFILE_VC1_ADVANCED: return MFX_PROFILE_VC1_ADVANCED;
        }
    } else if (codec == AV_CODEC_ID_H264) {
        switch (profile) {
        case AV_PROFILE_H264_BASELINE:             return MFX_PROFILE_AVC_BASELINE;
        case AV_PROFILE_H264_CONSTRAINED_BASELINE: return MFX_PROFILE_AVC_CONSTRAINED_BASELINE;
        case AV_PROFILE_H264_MAIN:                 return MFX_PROFILE_AVC_MAIN;
        case AV_PROFILE_H264_EXTENDED:             return MFX_PROFILE_AVC_EXTENDED;
        case AV_PROFILE_H264_HIGH:                 return MFX_PROFILE_AVC_HIGH;
        case AV_PROFILE_H264_HIGH_10:              return MFX_PROFILE_AVC_HIGH10;
        case AV_PROFILE_H264_HIGH_422:             return MFX_PROFILE_AVC_HIGH_422;
        }
    } else if (codec == AV_CODEC_ID_HEVC) {
        switch (profile) {
        case AV_PROFILE_HEVC_MAIN:    return MFX_PROFILE_HEVC_MAIN;
        case AV_PROFILE_HEVC_MAIN_10: return MFX_PROFILE_HEVC_MAIN10;
        case AV_PROFILE_HEVC_REXT:    return MFX_PROFILE_HEVC_REXT;
        }
    } else if (codec == AV_CODEC_ID_VP9) {
        switch (profile) {
        case AV_PROFILE_VP9_0:        return MFX_PROFILE_VP9_0;
        case AV_PROFILE_VP9_1:        return MFX_PROFILE_VP9_1;
        case AV_PROFILE_VP9_2:        return MFX_PROFILE_VP9_2;
        case AV_PROFILE_VP9_3:        return MFX_PROFILE_VP9_3;
        }
    } else if (codec == AV_CODEC_ID_AV1) {
#   if QSV_VERSION_ATLEAST(1, 34)
        switch (profile) {
        case AV_PROFILE_AV1_MAIN:         return MFX_PROFILE_AV1_MAIN;
        case AV_PROFILE_AV1_HIGH:         return MFX_PROFILE_AV1_HIGH;
        case AV_PROFILE_AV1_PROFESSIONAL: return MFX_PROFILE_AV1_PRO;
        }
#   endif
    } else if (codec == AV_CODEC_ID_VVC) {
#   if QSV_VERSION_ATLEAST(2, 11)
        switch (profile) {
        case AV_PROFILE_VVC_MAIN_10: return MFX_PROFILE_VVC_MAIN10;
        }
#   endif
    } else if (codec == AV_CODEC_ID_MJPEG) {
        switch (profile) {
        case AV_PROFILE_MJPEG_HUFFMAN_BASELINE_DCT: return MFX_PROFILE_JPEG_BASELINE;
        }
    }

    return MFX_PROFILE_UNKNOWN;
}

static void qsv_set_default_mfx_profile_level(unsigned short *mfx_profile,
                                              unsigned short *mfx_level,
                                              enum AVCodecID codec,
                                              unsigned short bit_depth,
                                              unsigned short mfx_chroma)
{
    if (!mfx_profile || !mfx_level)
        return;

    switch (codec) {
    case AV_CODEC_ID_MPEG2VIDEO:
        *mfx_profile = MFX_PROFILE_MPEG2_MAIN;
        *mfx_level = MFX_LEVEL_MPEG2_HIGH;
        break;
    case AV_CODEC_ID_VC1:
        *mfx_profile = MFX_PROFILE_VC1_ADVANCED;
        *mfx_level = MFX_LEVEL_VC1_4;
        break;
    case AV_CODEC_ID_H264:
        *mfx_profile = (bit_depth <= 10 && mfx_chroma == MFX_CHROMAFORMAT_YUV422) ?
            MFX_PROFILE_AVC_HIGH_422 :
            (bit_depth == 10 ? MFX_PROFILE_AVC_HIGH10 : MFX_PROFILE_AVC_HIGH);
        *mfx_level = MFX_LEVEL_AVC_52;
        break;
    case AV_CODEC_ID_HEVC:
        *mfx_profile = (bit_depth <= 10 && mfx_chroma == MFX_CHROMAFORMAT_YUV420) ?
            (bit_depth == 10 ? MFX_PROFILE_HEVC_MAIN10 : MFX_PROFILE_HEVC_MAIN) :
            MFX_PROFILE_HEVC_REXT;
        *mfx_level = MFX_LEVEL_HEVC_62;
        break;
    case AV_CODEC_ID_VP8:
        *mfx_profile = MFX_PROFILE_VP8_3;
        *mfx_level = MFX_LEVEL_UNKNOWN;
        break;
    case AV_CODEC_ID_VP9:
        *mfx_profile = (mfx_chroma == MFX_CHROMAFORMAT_YUV420) ?
            (bit_depth < 10 ? MFX_PROFILE_VP9_0 : MFX_PROFILE_VP9_2) :
            (bit_depth < 10 ? MFX_PROFILE_VP9_1 : MFX_PROFILE_VP9_3);
        *mfx_level = MFX_LEVEL_UNKNOWN;
        break;
#   if QSV_VERSION_ATLEAST(1, 34)
    case AV_CODEC_ID_AV1:
        *mfx_profile = MFX_PROFILE_AV1_MAIN;
        *mfx_level = MFX_LEVEL_AV1_63;
        break;
#   endif
#   if QSV_VERSION_ATLEAST(2, 11)
    case AV_CODEC_ID_VVC:
        *mfx_profile = MFX_PROFILE_VVC_MAIN10;
        *mfx_level = MFX_LEVEL_VVC_63;
        break;
#   endif
    case AV_CODEC_ID_MJPEG:
        *mfx_profile = MFX_PROFILE_JPEG_BASELINE;
        *mfx_level = MFX_LEVEL_UNKNOWN;
        break;
    default:
        *mfx_profile = MFX_PROFILE_UNKNOWN;
        *mfx_level = MFX_LEVEL_UNKNOWN;
        break;
    }
}

static int qsv_check_comp_shift(enum AVPixelFormat pix_fmt)
{
    int first = -1;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(pix_fmt);

    for (int i = 0; desc && i < desc->nb_components; i++) {
        if (desc->comp[i].step > 0) {
            if (desc->comp[i].shift <= 0 ||
                (first >= 0 && desc->comp[i].shift != first))
                return 0;
            first = desc->comp[i].shift;
        }
    }
    return first > 0;
}

static enum QsvTestType qsv_test_mfx_dec_params(mfxSession mfx_session,
                                                mfxVideoParam mfx_params,
                                                mfxVideoParam *mfx_params_out,
                                                int do_init_test)
{
    mfxFrameAllocRequest mfx_alloc_req = { 0 };
    mfxStatus sts;

    if (!mfx_params_out)
        return QSV_TEST_CONTINUE;

    mfx_params_out->mfx.CodecId = mfx_params.mfx.CodecId;

    sts = MFXVideoDECODE_Query(mfx_session, &mfx_params, mfx_params_out);
    if (sts != MFX_ERR_NONE && sts != MFX_WRN_INCOMPATIBLE_VIDEO_PARAM)
        return QSV_TEST_CONTINUE;

    sts = MFXVideoDECODE_QueryIOSurf(mfx_session, &mfx_params, &mfx_alloc_req);
    if (sts != MFX_ERR_NONE && sts != MFX_WRN_INCOMPATIBLE_VIDEO_PARAM)
        return QSV_TEST_CONTINUE;

    if (!do_init_test)
        return QSV_TEST_SUCCESS;

    sts = MFXVideoDECODE_Init(mfx_session, &mfx_params);
    MFXVideoDECODE_Close(mfx_session);
    if (sts != MFX_ERR_NONE && sts != MFX_WRN_INCOMPATIBLE_VIDEO_PARAM)
        return QSV_TEST_CONTINUE;

    return QSV_TEST_SUCCESS;
}
#endif

int print_qsv_decoder_info(AVTextFormatContext *tfc, AVBufferRef *qsv_ref)
{
#if CONFIG_QSV
    AVHWDeviceContext *dev_ctx = NULL;
    AVQSVDeviceContext *hwctx = NULL;
    mfxVersion ver = { 0 };
    mfxPlatform platform = { 0 };
    mfxStatus sts;
    int header_printed = 0;
    int do_init_test = 0;
    unsigned i, j;

    if (!tfc || !qsv_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)qsv_ref->data;
    if (!dev_ctx)
        return AVERROR(ENOSYS);

    hwctx = dev_ctx->hwctx;

    sts = MFXQueryVersion(hwctx->session, &ver);
    if (sts != MFX_ERR_NONE)
        return AVERROR(ENOSYS);

    do_init_test = !QSV_RUNTIME_VERSION_ATLEAST(ver, 1, 255);

    MFXVideoCORE_QueryPlatform(hwctx->session, &platform);

    for (i = 0; qsvdec_modes[i].name; i++) {
        mfxVideoParam mfx_params = { 0 };
        uint32_t mfx_fourcc = 0;
        int mfx_codec = -1;
        const QsvDecMode *mode = &qsvdec_modes[i];
        const AVPixFmtDescriptor *desc = NULL;
        enum AVPixelFormat format = AV_PIX_FMT_NONE;
        unsigned min_width = 0, min_height = 0;
        unsigned max_width = 0, max_height = 0;
        unsigned short max_profile = MFX_PROFILE_UNKNOWN;
        unsigned short max_chroma = MFX_CHROMAFORMAT_YUV420;
        int header2_printed = 0;
        int header3_printed = 0;
        enum QsvTestType test_ret = QSV_TEST_SUCCESS;
        int do_init_test_dec = do_init_test | (mode->codec == AV_CODEC_ID_MPEG2VIDEO);

        if (platform.CodeName > 0) {
            switch (mode->codec) {
            case AV_CODEC_ID_HEVC: { if (platform.CodeName <  6) continue; } break; /* CHERRYTRAIL */
            case AV_CODEC_ID_VP9:  { if (platform.CodeName <  8) continue; } break; /* APOLLOLAKE  */
            case AV_CODEC_ID_AV1:  { if (platform.CodeName < 40) continue; } break; /* TIGERLAKE   */
            case AV_CODEC_ID_VVC:  { if (platform.CodeName < 53) continue; } break; /* LUNARLAKE   */
            }
        }

        if (!mode->formats)
            continue;

        mfx_codec = qsv_map_av_to_mfx_codec(mode->codec);
        if (mfx_codec < 0)
            continue;

        /* Use the most basic format for this codec */
        format = mode->formats[0];
        desc = av_pix_fmt_desc_get(format);
        if (!desc)
            continue;
        mfx_fourcc = qsv_map_av_to_mfx_fourcc(format);
        if (!mfx_fourcc)
            continue;

        mfx_params.mfx.CodecId = mfx_codec;
        mfx_params.mfx.FrameInfo.BitDepthLuma = FFMIN(desc->comp[0].depth, 12);
        mfx_params.mfx.FrameInfo.BitDepthChroma = mfx_params.mfx.FrameInfo.BitDepthLuma;
        mfx_params.mfx.FrameInfo.Shift = qsv_check_comp_shift(format);
        mfx_params.mfx.FrameInfo.FourCC = mfx_fourcc;
        mfx_params.mfx.FrameInfo.FrameRateExtN = 25;
        mfx_params.mfx.FrameInfo.FrameRateExtD = 1;
        mfx_params.mfx.FrameInfo.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
        mfx_params.mfx.FrameInfo.ChromaFormat = qsv_map_av_to_mfx_chroma(format);
        mfx_params.AsyncDepth = 1;
        mfx_params.IOPattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

        qsv_set_default_mfx_profile_level(&mfx_params.mfx.CodecProfile,
                                          &mfx_params.mfx.CodecLevel,
                                          mode->codec,
                                          mfx_params.mfx.FrameInfo.BitDepthLuma,
                                          mfx_params.mfx.FrameInfo.ChromaFormat);

        if (!QSV_RUNTIME_VERSION_ATLEAST(ver, 1, 9)) {
            mfx_params.mfx.FrameInfo.BitDepthLuma   = 0;
            mfx_params.mfx.FrameInfo.BitDepthChroma = 0;
            mfx_params.mfx.FrameInfo.Shift          = 0;
        }

        /* Check min res first */
        for (const HwRes *r = &hw_res_ascend[0]; r->name; r++) {
            mfxVideoParam mfx_params_out = { 0 };

            if (mode->legacy && (r->width > 4096 || r->height > 4096))
                break;

            mfx_params.mfx.FrameInfo.Width  = r->width;
            mfx_params.mfx.FrameInfo.Height = r->height;

            test_ret = qsv_test_mfx_dec_params(hwctx->session, mfx_params, &mfx_params_out, do_init_test_dec);
            if (test_ret != QSV_TEST_SUCCESS) {
                if (test_ret == QSV_TEST_BREAK) break;
                else continue;
            }
            min_width  = r->width;
            min_height = r->height;
            break;
        }
        if (!min_width || !min_height)
            continue;

        /* Check max res */
        for (const HwRes *r = &hw_res_ascend[FF_ARRAY_ELEMS(hw_res_ascend) - 1]; r >= &hw_res_ascend[0]; r--) {
            mfxVideoParam mfx_params_out = { 0 };

            if (!r->name)
                continue;
            if (r->width <= min_width && r->height <= min_height) {
                max_width  = r->width;
                max_height = r->height;
                break;
            }
            if (mode->legacy && (r->width > 4096 || r->height > 4096))
                continue;

            mfx_params.mfx.FrameInfo.Width  = r->width;
            mfx_params.mfx.FrameInfo.Height = r->height;

            test_ret = qsv_test_mfx_dec_params(hwctx->session, mfx_params, &mfx_params_out, do_init_test_dec);
            if (test_ret != QSV_TEST_SUCCESS) {
                if (test_ret == QSV_TEST_BREAK) break;
                else continue;
            }
            max_width  = r->width;
            max_height = r->height;
            break;
        }
        if (!max_width || !max_height)
            continue;

        if (!header_printed) {
            mark_section_show_entries(SECTION_ID_DEVICE_DECODERS_QSV, 1, NULL);
            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_DECODERS_QSV);
            header_printed = 1;
        }

        mark_section_show_entries(SECTION_ID_DEVICE_DECODER, 1, NULL);
        avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_DECODER);
        print_str("codec_name", avcodec_get_name(mode->codec));
        print_int("codec_id", mode->codec);
        print_str("codec_desc", mode->name);
        print_int("min_width", min_width);
        print_int("min_height", min_height);
        print_int("max_width", max_width);
        print_int("max_height", max_height);

        /* Formats */
        for (j = 0; mode->formats[j] != AV_PIX_FMT_NONE; j++) {
            mfxVideoParam mfx_params_out = { 0 };

            format = mode->formats[j];
            desc = av_pix_fmt_desc_get(format);
            if (!desc)
                continue;
            mfx_fourcc = qsv_map_av_to_mfx_fourcc(format);
            if (!mfx_fourcc)
                continue;
            if (!j)
                goto skip;

            mfx_params.mfx.FrameInfo.Width  = min_width;
            mfx_params.mfx.FrameInfo.Height = min_height;
            mfx_params.mfx.FrameInfo.BitDepthLuma = FFMIN(desc->comp[0].depth, 12);
            mfx_params.mfx.FrameInfo.BitDepthChroma = mfx_params.mfx.FrameInfo.BitDepthLuma;
            mfx_params.mfx.FrameInfo.Shift = qsv_check_comp_shift(format);
            mfx_params.mfx.FrameInfo.FourCC = mfx_fourcc;
            mfx_params.mfx.FrameInfo.ChromaFormat = qsv_map_av_to_mfx_chroma(format);

            qsv_set_default_mfx_profile_level(&mfx_params.mfx.CodecProfile,
                                              &mfx_params.mfx.CodecLevel,
                                              mode->codec,
                                              mfx_params.mfx.FrameInfo.BitDepthLuma,
                                              mfx_params.mfx.FrameInfo.ChromaFormat);

            if (!QSV_RUNTIME_VERSION_ATLEAST(ver, 1, 9)) {
                mfx_params.mfx.FrameInfo.BitDepthLuma   = 0;
                mfx_params.mfx.FrameInfo.BitDepthChroma = 0;
                mfx_params.mfx.FrameInfo.Shift          = 0;
            }

            test_ret = qsv_test_mfx_dec_params(hwctx->session, mfx_params, &mfx_params_out, do_init_test_dec);
            if (test_ret != QSV_TEST_SUCCESS) {
                if (test_ret == QSV_TEST_BREAK) break;
                else continue;
            }
skip:
            if (!header2_printed) {
                mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMTS, 1, NULL);
                avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMTS);
                header2_printed = 1;
            }

            max_profile = FFMAX(mfx_params.mfx.CodecProfile, max_profile);
            max_chroma  = FFMAX(mfx_params.mfx.FrameInfo.ChromaFormat, max_chroma);

            mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMT, 1, NULL);
            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMT);
            print_str("format_name", av_get_pix_fmt_name(format));
            print_int("format_id", format);
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMT
        }
        if (header2_printed)
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMTS

        /* Profiles */
        for (j = 0; mode->profiles && mode->profiles[j] != AV_PROFILE_UNKNOWN; j++) {
            unsigned short mfx_profile = qsv_map_av_to_mfx_profile(mode->codec, mode->profiles[j]);

            if (mode->codec == AV_CODEC_ID_H264)
                mfx_profile &= ~QSV_MFX_PROFILE_H264_CONSTRAINTS;
            if (mode->codec == AV_CODEC_ID_VP9 &&
                max_chroma == MFX_CHROMAFORMAT_YUV420 && !(mfx_profile & 1))
                continue;

            if (mfx_profile > 0 && max_profile > 0 && mfx_profile <= max_profile) {
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

        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_DECODER
    }

    if (header_printed)
        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_DECODERS_QSV
#endif
    return 0;
}

#if CONFIG_QSV
static void qsv_set_default_mfx_enc_rc_params(mfxVideoParam* mfx_params,
                                              unsigned short mfx_rc_mode)
{
    if (!mfx_params)
        return;

    mfx_params->mfx.RateControlMethod  = 0;
    mfx_params->mfx.QPI                = 0;
    mfx_params->mfx.QPP                = 0;
    mfx_params->mfx.QPB                = 0;
    mfx_params->mfx.ICQQuality         = 0;
    mfx_params->mfx.BRCParamMultiplier = 0;
    mfx_params->mfx.TargetKbps         = 0;
    mfx_params->mfx.MaxKbps            = 0;
    mfx_params->mfx.BufferSizeInKB     = 0;
    mfx_params->mfx.InitialDelayInKB   = 0;

    switch (mfx_rc_mode) {
    case MFX_RATECONTROL_CBR:
    case MFX_RATECONTROL_VBR:
    case MFX_RATECONTROL_QVBR:
        {
            mfxU32 target_kbps = 4000;
            mfxU32 max_kbps    = target_kbps << (mfx_rc_mode != MFX_RATECONTROL_CBR);
            mfxU32 buf_size    = (max_kbps * 2) / 8;
            mfxU32 init_delay  = buf_size / 2;
            mfxU32 max_val     = FFMAX(FFMAX3(target_kbps, max_kbps, buf_size), init_delay);
            mfxU16 multiplier  = (max_val + 0x10000) / 0x10000;

            multiplier = FFMAX(multiplier, 1);
            mfx_params->mfx.RateControlMethod  = mfx_rc_mode;
            mfx_params->mfx.BRCParamMultiplier = multiplier;
            mfx_params->mfx.TargetKbps         = (mfxU16)(target_kbps / multiplier);
            mfx_params->mfx.MaxKbps            = (mfxU16)(max_kbps / multiplier);
            mfx_params->mfx.BufferSizeInKB     = (mfxU16)(buf_size / multiplier);
            mfx_params->mfx.InitialDelayInKB   = (mfxU16)(init_delay / multiplier);
        }
        break;
    case MFX_RATECONTROL_CQP:
        mfx_params->mfx.RateControlMethod = mfx_rc_mode;
        mfx_params->mfx.QPI =
        mfx_params->mfx.QPP =
        mfx_params->mfx.QPB = 26;
        break;
    case MFX_RATECONTROL_ICQ:
        mfx_params->mfx.RateControlMethod = mfx_rc_mode;
        mfx_params->mfx.ICQQuality = 23;
        break;
    }
}

static enum QsvTestType qsv_test_mfx_enc_params(mfxSession mfx_session,
                                                mfxVideoParam mfx_params,
                                                mfxVideoParam *mfx_params_out,
                                                int do_init_test)
{
    mfxFrameAllocRequest mfx_alloc_req = { 0 };
    mfxStatus sts;

    if (!mfx_params_out)
        return QSV_TEST_CONTINUE;

    mfx_params_out->mfx.CodecId = mfx_params.mfx.CodecId;

    sts = MFXVideoENCODE_Query(mfx_session, &mfx_params, mfx_params_out);
    if (sts != MFX_ERR_NONE && sts != MFX_WRN_INCOMPATIBLE_VIDEO_PARAM)
        return QSV_TEST_CONTINUE;
    else if (mfx_params.mfx.LowPower != mfx_params_out->mfx.LowPower)
        return QSV_TEST_BREAK;

    sts = MFXVideoENCODE_QueryIOSurf(mfx_session, &mfx_params, &mfx_alloc_req);
    if (sts != MFX_ERR_NONE && sts != MFX_WRN_INCOMPATIBLE_VIDEO_PARAM)
        return QSV_TEST_CONTINUE;

    if (!do_init_test)
        return QSV_TEST_SUCCESS;

    sts = MFXVideoENCODE_Init(mfx_session, &mfx_params);
    MFXVideoENCODE_Close(mfx_session);
    if (sts != MFX_ERR_NONE && sts != MFX_WRN_INCOMPATIBLE_VIDEO_PARAM)
        return QSV_TEST_CONTINUE;

    return QSV_TEST_SUCCESS;
}

static const char *qsv_map_mfx_rc_mode_to_str(unsigned short mfx_rc_mode, int suffix)
{
    switch (mfx_rc_mode) {
    case MFX_RATECONTROL_CBR:  return suffix ? "cbr_ratecontol"  : "cbr";
    case MFX_RATECONTROL_VBR:  return suffix ? "vbr_ratecontol"  : "vbr";
    case MFX_RATECONTROL_CQP:  return suffix ? "cqp_ratecontol"  : "cqp";
    case MFX_RATECONTROL_ICQ:  return suffix ? "icq_ratecontol"  : "icq";
    case MFX_RATECONTROL_QVBR: return suffix ? "qvbr_ratecontol" : "qvbr";
    default:                   return "";
    }
}
#endif

int print_qsv_encoder_info(AVTextFormatContext *tfc, AVBufferRef *qsv_ref)
{
#if CONFIG_QSV
    AVHWDeviceContext *dev_ctx = NULL;
    AVQSVDeviceContext *hwctx = NULL;
    mfxVersion ver = { 0 };
    mfxPlatform platform = { 0 };
    mfxStatus sts;
    int header_printed = 0;
    int do_init_test = 0;
    unsigned i, j, k;

    if (!tfc || !qsv_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)qsv_ref->data;
    if (!dev_ctx)
        return AVERROR(ENOSYS);

    hwctx = dev_ctx->hwctx;

    sts = MFXQueryVersion(hwctx->session, &ver);
    if (sts != MFX_ERR_NONE)
        return AVERROR(ENOSYS);

    do_init_test = !QSV_RUNTIME_VERSION_ATLEAST(ver, 1, 255);

    MFXVideoCORE_QueryPlatform(hwctx->session, &platform);

    for (i = 0; qsvenc_modes[i].name; i++) {
        mfxVideoParam mfx_params = { 0 };
        uint32_t mfx_fourcc = 0;
        int mfx_codec = -1;
        const QsvEncMode *mode = &qsvenc_modes[i];
        const AVPixFmtDescriptor *desc = NULL;
        enum AVPixelFormat format = AV_PIX_FMT_NONE;
        unsigned min_width = 0, min_height = 0;
        unsigned max_width = 0, max_height = 0;
        unsigned short max_profile = MFX_PROFILE_UNKNOWN;
        unsigned short max_chroma = MFX_CHROMAFORMAT_YUV420;
        int header2_printed = 0;
        int header3_printed = 0;
        enum QsvTestType test_ret = QSV_TEST_SUCCESS;

        if (!mode->low_power && mode->codec != AV_CODEC_ID_MJPEG &&
            (platform.CodeName == 32 || platform.CodeName == 33 || /* JASPERLAKE/ELKHARTLAKE */
             platform.CodeName == 45 || platform.CodeName == 46 || /* DG2/ATS_M  */
             platform.CodeName >= 51))                             /* METEORLAKE */
            continue;

        if (platform.CodeName > 0) {
            switch (mode->codec) {
            case AV_CODEC_ID_HEVC: { if (platform.CodeName <   7) continue; } break; /* SKYLAKE       */
            case AV_CODEC_ID_VP9:  { if (platform.CodeName <  30) continue; } break; /* ICELAKE       */
            case AV_CODEC_ID_AV1:  { if (platform.CodeName <  45) continue;          /* ARCTICSOUND_P */
                                     if (platform.CodeName == 50) continue;          /* ALDERLAKE_N   */
                                     if (platform.CodeName == 55) continue; } break; /* KEEMBAY       */
            }
        }

        if (!mode->formats)
            continue;

        mfx_codec = qsv_map_av_to_mfx_codec(mode->codec);
        if (mfx_codec < 0)
            continue;

        /* Use the most basic format for this codec */
        format = mode->formats[0];
        desc = av_pix_fmt_desc_get(format);
        if (!desc)
            continue;
        mfx_fourcc = qsv_map_av_to_mfx_fourcc(format);
        if (!mfx_fourcc)
            continue;

        mfx_params.mfx.CodecId = mfx_codec;
        mfx_params.mfx.FrameInfo.BitDepthLuma = FFMIN(desc->comp[0].depth, 10);
        mfx_params.mfx.FrameInfo.BitDepthChroma = mfx_params.mfx.FrameInfo.BitDepthLuma;
        mfx_params.mfx.FrameInfo.Shift = qsv_check_comp_shift(format);
        mfx_params.mfx.FrameInfo.FourCC = mfx_fourcc;
        mfx_params.mfx.FrameInfo.FrameRateExtN = 25;
        mfx_params.mfx.FrameInfo.FrameRateExtD = 1;
        mfx_params.mfx.FrameInfo.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
        mfx_params.mfx.FrameInfo.ChromaFormat = qsv_map_av_to_mfx_chroma(format);
        mfx_params.AsyncDepth = 1;
        mfx_params.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY;

        if (mode->codec == AV_CODEC_ID_MJPEG) {
            mfx_params.mfx.Interleaved = MFX_SCANTYPE_INTERLEAVED;
            mfx_params.mfx.Quality = 100;
        } else {
            mfx_params.mfx.TargetUsage = MFX_TARGETUSAGE_BALANCED;
            mfx_params.mfx.LowPower = mode->low_power ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF;
            if (!QSV_RUNTIME_VERSION_ATLEAST(ver, 1, 15)) {
                mfx_params.mfx.LowPower = MFX_CODINGOPTION_UNKNOWN;
                if (mode->low_power)
                    continue;
            }
            qsv_set_default_mfx_enc_rc_params(&mfx_params, MFX_RATECONTROL_CQP);
        }

        qsv_set_default_mfx_profile_level(&mfx_params.mfx.CodecProfile,
                                          &mfx_params.mfx.CodecLevel,
                                          mode->codec,
                                          mfx_params.mfx.FrameInfo.BitDepthLuma,
                                          mfx_params.mfx.FrameInfo.ChromaFormat);

        if (!QSV_RUNTIME_VERSION_ATLEAST(ver, 1, 9)) {
            mfx_params.mfx.FrameInfo.BitDepthLuma   = 0;
            mfx_params.mfx.FrameInfo.BitDepthChroma = 0;
            mfx_params.mfx.FrameInfo.Shift          = 0;
        }

        /* Check min res first */
        for (const HwRes *r = &hw_res_ascend[0]; r->name; r++) {
            mfxVideoParam mfx_params_out = { 0 };

            if (mode->legacy && (r->width > 4096 || r->height > 4096))
                break;

            mfx_params.mfx.FrameInfo.CropW = mfx_params.mfx.FrameInfo.Width  = r->width;
            mfx_params.mfx.FrameInfo.CropH = mfx_params.mfx.FrameInfo.Height = r->height;

            test_ret = qsv_test_mfx_enc_params(hwctx->session, mfx_params, &mfx_params_out, do_init_test);
            if (test_ret != QSV_TEST_SUCCESS) {
                if (test_ret == QSV_TEST_BREAK) break;
                else continue;
            }
            min_width  = r->width;
            min_height = r->height;
            break;
        }
        if (!min_width || !min_height)
            continue;

        /* Check max res */
        for (const HwRes *r = &hw_res_ascend[FF_ARRAY_ELEMS(hw_res_ascend) - 1]; r >= &hw_res_ascend[0]; r--) {
            mfxVideoParam mfx_params_out = { 0 };

            if (!r->name)
                continue;
            if (r->width <= min_width && r->height <= min_height) {
                max_width  = r->width;
                max_height = r->height;
                break;
            }
            if (mode->legacy && (r->width > 4096 || r->height > 4096))
                continue;

            mfx_params.mfx.FrameInfo.CropW = mfx_params.mfx.FrameInfo.Width  = r->width;
            mfx_params.mfx.FrameInfo.CropH = mfx_params.mfx.FrameInfo.Height = r->height;

            test_ret = qsv_test_mfx_enc_params(hwctx->session, mfx_params, &mfx_params_out, do_init_test);
            if (test_ret != QSV_TEST_SUCCESS) {
                if (test_ret == QSV_TEST_BREAK) break;
                else continue;
            }
            max_width  = r->width;
            max_height = r->height;
            break;
        }
        if (!max_width || !max_height)
            continue;

        if (!header_printed) {
            mark_section_show_entries(SECTION_ID_DEVICE_ENCODERS_QSV, 1, NULL);
            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_ENCODERS_QSV);
            header_printed = 1;
        }

        mark_section_show_entries(SECTION_ID_DEVICE_ENCODER, 1, NULL);
        avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_ENCODER);
        print_str("codec_name", avcodec_get_name(mode->codec));
        print_int("codec_id", mode->codec);
        print_str("codec_desc", mode->name);
        print_int("min_width", min_width);
        print_int("min_height", min_height);
        print_int("max_width", max_width);
        print_int("max_height", max_height);

        mfx_params.mfx.FrameInfo.CropW = mfx_params.mfx.FrameInfo.Width  = min_width;
        mfx_params.mfx.FrameInfo.CropH = mfx_params.mfx.FrameInfo.Height = min_height;

        /* Check RC and CO */
        if (mode->codec != AV_CODEC_ID_MJPEG) {
            static const struct {
                const char    *attr_str;
                const size_t   attr_offset;
                const unsigned need_brc;
                const unsigned co_version;
                const unsigned mfx_ver_major;
                const unsigned mfx_ver_minor;
            } mfx_co_attrs[] = {
                { "enctools_la", offsetof(mfxExtCodingOption2, ExtBRC), 1, 2, 1, 6 },
                { "mbbrc",       offsetof(mfxExtCodingOption2, MBBRC),  1, 2, 1, 6 },
            };

            print_int("low_power", mfx_params.mfx.LowPower == MFX_CODINGOPTION_ON);

            for (j = 0; mode->rc_modes && mode->rc_modes[j] != 0; j++) {
                mfxVideoParam mfx_params_rc = mfx_params;
                mfxVideoParam mfx_params_rc_out = { 0 };

                test_ret = QSV_TEST_SUCCESS;

                if ((mode->rc_modes[j] == MFX_RATECONTROL_ICQ &&
                     !QSV_RUNTIME_VERSION_ATLEAST(ver, 1, 8)) ||
                    (mode->rc_modes[j] == MFX_RATECONTROL_QVBR &&
                     !QSV_RUNTIME_VERSION_ATLEAST(ver, 1, 11)))
                    continue;

                if (mode->rc_modes[j] != MFX_RATECONTROL_CQP) {
                    qsv_set_default_mfx_enc_rc_params(&mfx_params_rc, mode->rc_modes[j]);
                    test_ret = qsv_test_mfx_enc_params(hwctx->session, mfx_params_rc, &mfx_params_rc_out, do_init_test);
                }

                print_int(qsv_map_mfx_rc_mode_to_str(mode->rc_modes[j], 1), test_ret == QSV_TEST_SUCCESS);
                if (test_ret != QSV_TEST_SUCCESS)
                    continue;

                /* mfxExtCodingOption2 */
                for (k = 0; k < FF_ARRAY_ELEMS(mfx_co_attrs) && mfx_co_attrs[k].attr_str; k++) {
                    mfxExtCodingOption2 extco2_in  = { .Header.BufferId = MFX_EXTBUFF_CODING_OPTION2,
                                                       .Header.BufferSz = sizeof(mfxExtCodingOption2) };
                    mfxExtCodingOption2 extco2_out = { .Header.BufferId = MFX_EXTBUFF_CODING_OPTION2,
                                                       .Header.BufferSz = sizeof(mfxExtCodingOption2) };
                    mfxExtBuffer* mfx_ext_bufs_in[1]  = { (mfxExtBuffer*)&extco2_in  };
                    mfxExtBuffer* mfx_ext_bufs_out[1] = { (mfxExtBuffer*)&extco2_out };
                    mfxVideoParam mfx_params_in  = mfx_params_rc;
                    mfxVideoParam mfx_params_out = { .ExtParam = mfx_ext_bufs_out, .NumExtParam = 1 };
                    const int is_extbrc = mfx_co_attrs[k].attr_offset == offsetof(mfxExtCodingOption2, ExtBRC);
                    char full_str[128] = { 0 };

                    if (mfx_co_attrs[k].co_version != 2)
                        continue;
                    if (!QSV_RUNTIME_VERSION_ATLEAST(ver, mfx_co_attrs[k].mfx_ver_major,
                                                          mfx_co_attrs[k].mfx_ver_minor))
                        continue;
                    if (mfx_co_attrs[k].need_brc && mode->rc_modes[j] == MFX_RATECONTROL_CQP)
                        continue;
                    if (is_extbrc && ((platform.CodeName > 0 && platform.CodeName < 40 /* TIGERLAKE */) ||
                                       mode->codec == AV_CODEC_ID_VP9 ||
                                       (mode->rc_modes[j] != MFX_RATECONTROL_CBR &&
                                        mode->rc_modes[j] != MFX_RATECONTROL_VBR)))
                        continue;

                    *(mfxU16*)((char*)&extco2_in + mfx_co_attrs[k].attr_offset) = MFX_CODINGOPTION_ON;
                    if (is_extbrc)
                        extco2_in.LookAheadDepth = 8;

                    mfx_params_in.ExtParam = mfx_ext_bufs_in;
                    mfx_params_in.NumExtParam = 1;
                    test_ret = qsv_test_mfx_enc_params(hwctx->session, mfx_params_in, &mfx_params_out, 1);
                    if (test_ret == QSV_TEST_SUCCESS) {
                        mfxU16 attr_out = *(mfxU16*)((char*)&extco2_out + mfx_co_attrs[k].attr_offset);

                        test_ret = (attr_out == MFX_CODINGOPTION_ON) ? QSV_TEST_SUCCESS : QSV_TEST_CONTINUE;
                        if (is_extbrc && extco2_in.LookAheadDepth != extco2_out.LookAheadDepth)
                            test_ret = QSV_TEST_CONTINUE;
                    }

                    snprintf(full_str, sizeof(full_str), "%s_%s",
                             qsv_map_mfx_rc_mode_to_str(mode->rc_modes[j], 0), mfx_co_attrs[k].attr_str);

                    print_int(full_str, test_ret == QSV_TEST_SUCCESS);
                }
            }
        }

        /* Formats */
        for (j = 0; mode->formats[j] != AV_PIX_FMT_NONE; j++) {
            mfxVideoParam mfx_params_out = { 0 };

            format = mode->formats[j];
            desc = av_pix_fmt_desc_get(format);
            if (!desc)
                continue;
            mfx_fourcc = qsv_map_av_to_mfx_fourcc(format);
            if (!mfx_fourcc)
                continue;
            if (!j)
                goto skip;

            mfx_params.mfx.FrameInfo.BitDepthLuma = FFMIN(desc->comp[0].depth, 10);
            mfx_params.mfx.FrameInfo.BitDepthChroma = mfx_params.mfx.FrameInfo.BitDepthLuma;
            mfx_params.mfx.FrameInfo.Shift = qsv_check_comp_shift(format);
            mfx_params.mfx.FrameInfo.FourCC = mfx_fourcc;
            mfx_params.mfx.FrameInfo.ChromaFormat = qsv_map_av_to_mfx_chroma(format);

            qsv_set_default_mfx_profile_level(&mfx_params.mfx.CodecProfile,
                                              &mfx_params.mfx.CodecLevel,
                                              mode->codec,
                                              mfx_params.mfx.FrameInfo.BitDepthLuma,
                                              mfx_params.mfx.FrameInfo.ChromaFormat);

            if (!QSV_RUNTIME_VERSION_ATLEAST(ver, 1, 9)) {
                mfx_params.mfx.FrameInfo.BitDepthLuma   = 0;
                mfx_params.mfx.FrameInfo.BitDepthChroma = 0;
                mfx_params.mfx.FrameInfo.Shift          = 0;
            }

            test_ret = qsv_test_mfx_enc_params(hwctx->session, mfx_params, &mfx_params_out, do_init_test);
            if (test_ret != QSV_TEST_SUCCESS) {
                if (test_ret == QSV_TEST_BREAK) break;
                else continue;
            }
skip:
            if (!header2_printed) {
                mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMTS, 1, NULL);
                avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMTS);
                header2_printed = 1;
            }

            max_profile = FFMAX(mfx_params.mfx.CodecProfile, max_profile);
            max_chroma  = FFMAX(mfx_params.mfx.FrameInfo.ChromaFormat, max_chroma);

            mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMT, 1, NULL);
            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMT);
            print_str("format_name", av_get_pix_fmt_name(format));
            print_int("format_id", format);
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMT
        }
        if (header2_printed)
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMTS

        /* Profiles */
        for (j = 0; mode->profiles && mode->profiles[j] != AV_PROFILE_UNKNOWN; j++) {
            unsigned short mfx_profile = qsv_map_av_to_mfx_profile(mode->codec, mode->profiles[j]);

            if (mode->codec == AV_CODEC_ID_H264)
                mfx_profile &= ~QSV_MFX_PROFILE_H264_CONSTRAINTS;
            if (mode->codec == AV_CODEC_ID_VP9 &&
                max_chroma == MFX_CHROMAFORMAT_YUV420 && !(mfx_profile & 1))
                continue;

            if (mfx_profile > 0 && max_profile > 0 && mfx_profile <= max_profile) {
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

        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_ENCODER
    }

    if (header_printed)
        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_ENCODERS_QSV
#endif
    return 0;
}

#if CONFIG_QSV
static enum QsvTestType qsv_test_mfx_vpp_params(mfxSession mfx_session,
                                                mfxVideoParam mfx_params,
                                                mfxVideoParam *mfx_params_out,
                                                int do_init_test)
{
    mfxFrameAllocRequest mfx_alloc_req[2] = { 0 };
    mfxStatus sts;

    if (!mfx_params_out)
        return QSV_TEST_CONTINUE;

    *mfx_params_out = mfx_params;

    sts = MFXVideoVPP_Query(mfx_session, &mfx_params, mfx_params_out);
    if (sts != MFX_ERR_NONE && sts != MFX_WRN_INCOMPATIBLE_VIDEO_PARAM)
        return QSV_TEST_CONTINUE;

    sts = MFXVideoVPP_QueryIOSurf(mfx_session, &mfx_params, mfx_alloc_req);
    if (sts != MFX_ERR_NONE && sts != MFX_WRN_INCOMPATIBLE_VIDEO_PARAM)
        return QSV_TEST_CONTINUE;

    if (!do_init_test)
        return QSV_TEST_SUCCESS;

    sts = MFXVideoVPP_Init(mfx_session, &mfx_params);
    MFXVideoVPP_Close(mfx_session);
    if (sts != MFX_ERR_NONE && sts != MFX_WRN_INCOMPATIBLE_VIDEO_PARAM)
        return QSV_TEST_CONTINUE;

    return QSV_TEST_SUCCESS;
}

static const char *qsv_map_vpp_type_to_str(enum QsvVppType vpp)
{
    switch (vpp) {
    case QSV_VPP_SCALE:     return "scale";
    case QSV_VPP_DEINT:     return "deint";
    case QSV_VPP_OVERLAY:   return "overlay";
    case QSV_VPP_ROTATE:    return "rotate";
    case QSV_VPP_FLIP:      return "flip";
    case QSV_VPP_DENOISE:   return "denoise";
    case QSV_VPP_DETAIL:    return "detail";
    case QSV_VPP_FRAMERATE: return "framerate";
    case QSV_VPP_PROCAMP:   return "procamp";
    case QSV_VPP_TONEMAP:   return "tonemap";
    default:                return "";
    }
}
#endif

int print_qsv_vpp_info(AVTextFormatContext *tfc, AVBufferRef *qsv_ref)
{
#if CONFIG_QSV
    AVHWDeviceContext *dev_ctx = NULL;
    AVQSVDeviceContext *hwctx = NULL;
    mfxVersion ver = { 0 };
    mfxStatus sts;
    int header_printed = 0;
    int do_init_test = 0;
    unsigned i, j;

    mfxExtVPPScaling scale_conf = { 0 };
    mfxExtVPPDeinterlacing deint_conf = {
        .Mode = MFX_DEINTERLACING_BOB
    };
    mfxVPPCompInputStream overlay_inputs[2] = {
        { 0 },
        { .GlobalAlpha = 255, .PixelAlphaEnable = 1 }
    };
    mfxExtVPPComposite overlay_conf = {
        .InputStream = overlay_inputs,
        .NumInputStream = 2
    };
    mfxExtVPPRotation rotate_conf = {
        .Angle = MFX_ANGLE_180
    };
    mfxExtVPPMirroring flip_conf = {
        .Type = MFX_MIRRORING_HORIZONTAL
    };
    mfxExtVPPDenoise denoise_conf = {
        .DenoiseFactor = 1
    };
    mfxExtVPPDetail detail_conf = {
        .DetailFactor = 1
    };
    mfxExtVPPFrameRateConversion frc_conf = {
        .Algorithm = MFX_FRCALGM_FRAME_INTERPOLATION
    };
    mfxExtVPPProcAmp procamp_conf = {
        .Brightness = 1.f,
        .Contrast = 1.f,
        .Hue = 1.f,
        .Saturation = 1.f
    };
    mfxExtVideoSignalInfo invsi_conf = {
        .ColourPrimaries = AVCOL_PRI_BT2020,
        .TransferCharacteristics = AVCOL_TRC_SMPTE2084,
        .MatrixCoefficients = AVCOL_SPC_BT2020_NCL,
        .ColourDescriptionPresent = 1
    };
    mfxExtVideoSignalInfo outvsi_conf = {
        .ColourPrimaries = AVCOL_PRI_BT709,
        .TransferCharacteristics = AVCOL_TRC_BT709,
        .MatrixCoefficients = AVCOL_SPC_BT709,
        .ColourDescriptionPresent = 1
    };
    mfxExtMasteringDisplayColourVolume mdcv_conf = {
        .DisplayPrimariesX = { 13250, 7500, 34000 },
        .DisplayPrimariesY = { 34500, 3000, 16000 },
        .WhitePointX = 15635, .WhitePointY = 16450,
        .MaxDisplayMasteringLuminance = 10000000,
        .MinDisplayMasteringLuminance = 50
    };
    mfxExtContentLightLevelInfo clli_conf = {
        .MaxContentLightLevel = 1000,
        .MaxPicAverageLightLevel = 400
    };

    if (!tfc || !qsv_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)qsv_ref->data;
    if (!dev_ctx)
        return AVERROR(ENOSYS);

    hwctx = dev_ctx->hwctx;

    sts = MFXQueryVersion(hwctx->session, &ver);
    if (sts != MFX_ERR_NONE)
        return AVERROR(ENOSYS);

    do_init_test = !QSV_RUNTIME_VERSION_ATLEAST(ver, 1, 255);

    for (i = 0; qsvvpp_modes[i].name; i++) {
        mfxVideoParam mfx_params = { 0 };
        mfxExtBuffer* mfx_ext_bufs[4] = { NULL };
        uint32_t mfx_fourcc = 0;
        const QsvVppMode *mode = &qsvvpp_modes[i];
        const AVPixFmtDescriptor *desc = NULL;
        enum AVPixelFormat format = AV_PIX_FMT_NONE;
        unsigned min_width = 0, min_height = 0;
        unsigned max_width = 0, max_height = 0;
        int header2_printed = 0;
        enum QsvTestType test_ret = QSV_TEST_SUCCESS;

        if (!mode->formats)
            continue;

        /* Use the most basic format for this vpp */
        format = mode->formats[0];
        desc = av_pix_fmt_desc_get(format);
        if (!desc)
            continue;
        mfx_fourcc = qsv_map_av_to_mfx_fourcc(format);
        if (!mfx_fourcc)
            continue;

#   define VPP_EXTBUF(id, conf, optional, ver_major, ver_minor) \
        {                                                                    \
            if (!QSV_RUNTIME_VERSION_ATLEAST(ver, (ver_major), (ver_minor))) \
                if (!(optional))                                             \
                    continue;                                                \
            if (mfx_params.NumExtParam >= FF_ARRAY_ELEMS(mfx_ext_bufs))      \
                continue;                                                    \
            (conf).Header.BufferId                 = (id);                   \
            (conf).Header.BufferSz                 = sizeof(conf);           \
            mfx_ext_bufs[mfx_params.NumExtParam++] = (mfxExtBuffer*)&(conf); \
            mfx_params.ExtParam                    = mfx_ext_bufs;           \
        }

        switch (mode->vpp) {
        case QSV_VPP_SCALE:
            VPP_EXTBUF(MFX_EXTBUFF_VPP_SCALING,                        scale_conf,   1, 1,  19) break;
        case QSV_VPP_DEINT:
            VPP_EXTBUF(MFX_EXTBUFF_VPP_DEINTERLACING,                  deint_conf,   0, 1,   8) break;
        case QSV_VPP_OVERLAY:
            VPP_EXTBUF(MFX_EXTBUFF_VPP_COMPOSITE,                      overlay_conf, 0, 1,   9) break;
        case QSV_VPP_ROTATE:
            VPP_EXTBUF(MFX_EXTBUFF_VPP_ROTATION,                       rotate_conf,  0, 1,  17) break;
        case QSV_VPP_FLIP:
            VPP_EXTBUF(MFX_EXTBUFF_VPP_MIRRORING,                      flip_conf,    0, 1,  17) break;
        case QSV_VPP_DENOISE:
            VPP_EXTBUF(MFX_EXTBUFF_VPP_DENOISE,                        denoise_conf, 0, 1,   1) break;
        case QSV_VPP_DETAIL:
            VPP_EXTBUF(MFX_EXTBUFF_VPP_DETAIL,                         detail_conf,  0, 1,   1) break;
        case QSV_VPP_FRAMERATE:
            VPP_EXTBUF(MFX_EXTBUFF_VPP_FRAME_RATE_CONVERSION,          frc_conf,     0, 1,   3) break;
        case QSV_VPP_PROCAMP:
            VPP_EXTBUF(MFX_EXTBUFF_VPP_PROCAMP,                        procamp_conf, 0, 1,   1) break;
        case QSV_VPP_TONEMAP:
            VPP_EXTBUF(MFX_EXTBUFF_VIDEO_SIGNAL_INFO_IN,               invsi_conf,   0, 1, 255)
            VPP_EXTBUF(MFX_EXTBUFF_VIDEO_SIGNAL_INFO_OUT,              outvsi_conf,  0, 1, 255)
            VPP_EXTBUF(MFX_EXTBUFF_MASTERING_DISPLAY_COLOUR_VOLUME_IN, mdcv_conf,    0, 1, 255)
            VPP_EXTBUF(MFX_EXTBUFF_CONTENT_LIGHT_LEVEL_INFO,           clli_conf,    0, 1, 255) break;
        default: continue;
        }
#   undef VPP_EXTBUF

        mfx_params.vpp.Out.BitDepthLuma   = mfx_params.vpp.In.BitDepthLuma = FFMIN(desc->comp[0].depth, 12);
        mfx_params.vpp.Out.BitDepthChroma = mfx_params.vpp.In.BitDepthChroma = mfx_params.mfx.FrameInfo.BitDepthLuma;
        mfx_params.vpp.Out.Shift          = mfx_params.vpp.In.Shift = qsv_check_comp_shift(format);
        mfx_params.vpp.Out.FourCC         = mfx_params.vpp.In.FourCC = mfx_fourcc;
        mfx_params.vpp.Out.FrameRateExtN  = mfx_params.vpp.In.FrameRateExtN = 25;
        mfx_params.vpp.Out.FrameRateExtD  = mfx_params.vpp.In.FrameRateExtD = 1;
        mfx_params.vpp.Out.PicStruct      = mfx_params.vpp.In.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
        mfx_params.vpp.Out.ChromaFormat   = mfx_params.vpp.In.ChromaFormat = qsv_map_av_to_mfx_chroma(format);

        if (!QSV_RUNTIME_VERSION_ATLEAST(ver, 1, 9)) {
            mfx_params.vpp.Out.BitDepthLuma   = mfx_params.vpp.In.BitDepthLuma   = 0;
            mfx_params.vpp.Out.BitDepthChroma = mfx_params.vpp.In.BitDepthChroma = 0;
            mfx_params.vpp.Out.Shift          = mfx_params.vpp.In.Shift          = 0;
        }
        if (mode->vpp == QSV_VPP_DEINT)
            mfx_params.vpp.In.PicStruct = MFX_PICSTRUCT_FIELD_TFF;
        if (mode->vpp == QSV_VPP_FRAMERATE)
            mfx_params.vpp.Out.FrameRateExtD <<= 1;

        mfx_params.AsyncDepth = 1;
        mfx_params.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

        /* Check min res first */
        for (const HwRes *r = &hw_res_ascend[0]; r->name; r++) {
            mfxVideoParam mfx_params_out = { 0 };

            overlay_inputs[0].DstW   = overlay_inputs[1].DstW    = r->width;
            overlay_inputs[0].DstH   = overlay_inputs[1].DstH    = r->height;
            mfx_params.vpp.In.CropW  = mfx_params.vpp.In.Width   = r->width;
            mfx_params.vpp.In.CropH  = mfx_params.vpp.In.Height  = r->height;
            mfx_params.vpp.Out.CropW = mfx_params.vpp.Out.Width  = r->width;
            mfx_params.vpp.Out.CropH = mfx_params.vpp.Out.Height = r->height;

            test_ret = qsv_test_mfx_vpp_params(hwctx->session, mfx_params, &mfx_params_out, do_init_test);
            if (test_ret != QSV_TEST_SUCCESS) {
                if (test_ret == QSV_TEST_BREAK) break;
                else continue;
            }
            min_width  = r->width;
            min_height = r->height;
            break;
        }
        if (!min_width || !min_height)
            continue;

        /* Check max res */
        for (const HwRes *r = &hw_res_ascend[FF_ARRAY_ELEMS(hw_res_ascend) - 1]; r >= &hw_res_ascend[0]; r--) {
            mfxVideoParam mfx_params_out = { 0 };

            if (!r->name)
                continue;
            if (r->width <= min_width && r->height <= min_height) {
                max_width  = r->width;
                max_height = r->height;
                break;
            }

            overlay_inputs[0].DstW   = overlay_inputs[1].DstW    = r->width;
            overlay_inputs[0].DstH   = overlay_inputs[1].DstH    = r->height;
            mfx_params.vpp.In.CropW  = mfx_params.vpp.In.Width   = r->width;
            mfx_params.vpp.In.CropH  = mfx_params.vpp.In.Height  = r->height;
            mfx_params.vpp.Out.CropW = mfx_params.vpp.Out.Width  = r->width;
            mfx_params.vpp.Out.CropH = mfx_params.vpp.Out.Height = r->height;

            test_ret = qsv_test_mfx_vpp_params(hwctx->session, mfx_params, &mfx_params_out, do_init_test);
            if (test_ret != QSV_TEST_SUCCESS) {
                if (test_ret == QSV_TEST_BREAK) break;
                else continue;
            }
            max_width  = r->width;
            max_height = r->height;
            break;
        }
        if (!max_width || !max_height)
            continue;

        if (!header_printed) {
            mark_section_show_entries(SECTION_ID_DEVICE_FILTERS_QSV, 1, NULL);
            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_FILTERS_QSV);
            header_printed = 1;
        }

        mark_section_show_entries(SECTION_ID_DEVICE_FILTER, 1, NULL);
        avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_FILTER);
        print_str("filter_name", qsv_map_vpp_type_to_str(mode->vpp));
        print_str("filter_desc", mode->name);
        print_int("min_width", min_width);
        print_int("min_height", min_height);
        print_int("max_width", max_width);
        print_int("max_height", max_height);

        overlay_inputs[0].DstW   = overlay_inputs[1].DstW    = min_width;
        overlay_inputs[0].DstH   = overlay_inputs[1].DstH    = min_height;
        mfx_params.vpp.In.CropW  = mfx_params.vpp.In.Width   = min_width;
        mfx_params.vpp.In.CropH  = mfx_params.vpp.In.Height  = min_height;
        mfx_params.vpp.Out.CropW = mfx_params.vpp.Out.Width  = min_width;
        mfx_params.vpp.Out.CropH = mfx_params.vpp.Out.Height = min_height;

        if (mfx_params.NumExtParam > 0) {
            mfxVideoParam mfx_params_out = { 0 };

            switch (mode->vpp) {
            case QSV_VPP_SCALE:
                scale_conf.ScalingMode = MFX_SCALING_MODE_LOWPOWER;
                test_ret = qsv_test_mfx_vpp_params(hwctx->session, mfx_params, &mfx_params_out, do_init_test);
                print_int("scale_mode_low_power", test_ret == QSV_TEST_SUCCESS);

                scale_conf.ScalingMode = MFX_SCALING_MODE_QUALITY;
                test_ret = qsv_test_mfx_vpp_params(hwctx->session, mfx_params, &mfx_params_out, do_init_test);
                print_int("scale_mode_quality", test_ret == QSV_TEST_SUCCESS);

                scale_conf.ScalingMode = 1001; /* MFX_SCALING_MODE_INTEL_GEN_COMPUTE */
                test_ret = qsv_test_mfx_vpp_params(hwctx->session, mfx_params, &mfx_params_out, do_init_test);
                print_int("scale_mode_compute", test_ret == QSV_TEST_SUCCESS);

                scale_conf.ScalingMode = MFX_SCALING_MODE_DEFAULT;
                break;
            case QSV_VPP_DEINT:
                deint_conf.Mode = MFX_DEINTERLACING_ADVANCED;
                test_ret = qsv_test_mfx_vpp_params(hwctx->session, mfx_params, &mfx_params_out, do_init_test);
                print_int("deint_mode_bob", 1);
                print_int("deint_mode_advanced", test_ret == QSV_TEST_SUCCESS);

                deint_conf.Mode = MFX_DEINTERLACING_BOB;
                break;
            }
        }

        /* Formats */
        for (j = 0; mode->formats[j] != AV_PIX_FMT_NONE; j++) {
            mfxVideoParam mfx_params_out = { 0 };

            format = mode->formats[j];
            desc = av_pix_fmt_desc_get(format);
            if (!desc)
                continue;
            mfx_fourcc = qsv_map_av_to_mfx_fourcc(format);
            if (!mfx_fourcc)
                continue;
            if (!j)
                goto skip;

            mfx_params.vpp.Out.BitDepthLuma   = mfx_params.vpp.In.BitDepthLuma = FFMIN(desc->comp[0].depth, 12);
            mfx_params.vpp.Out.BitDepthChroma = mfx_params.vpp.In.BitDepthChroma = mfx_params.mfx.FrameInfo.BitDepthLuma;
            mfx_params.vpp.Out.Shift          = mfx_params.vpp.In.Shift = qsv_check_comp_shift(format);
            mfx_params.vpp.Out.FourCC         = mfx_params.vpp.In.FourCC = mfx_fourcc;
            mfx_params.vpp.Out.ChromaFormat   = mfx_params.vpp.In.ChromaFormat = qsv_map_av_to_mfx_chroma(format);

            if (!QSV_RUNTIME_VERSION_ATLEAST(ver, 1, 9)) {
                mfx_params.vpp.Out.BitDepthLuma   = mfx_params.vpp.In.BitDepthLuma   = 0;
                mfx_params.vpp.Out.BitDepthChroma = mfx_params.vpp.In.BitDepthChroma = 0;
                mfx_params.vpp.Out.Shift          = mfx_params.vpp.In.Shift          = 0;
            }

            test_ret = qsv_test_mfx_vpp_params(hwctx->session, mfx_params, &mfx_params_out, do_init_test);
            if (test_ret != QSV_TEST_SUCCESS) {
                if (test_ret == QSV_TEST_BREAK) break;
                else continue;
            }
skip:
            if (!header2_printed) {
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
        if (header2_printed)
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMTS

        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_FILTER
    }

    if (header_printed)
        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_FILTERS_QSV
#endif
    return 0;
}
