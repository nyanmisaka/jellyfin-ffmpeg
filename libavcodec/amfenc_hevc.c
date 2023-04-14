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

#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "amfenc.h"
#include "codec_internal.h"
#include "internal.h"

#define OFFSET(x) offsetof(AMFEncContext, x)
#define VE AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM
#define ENUM(a, b, c, d) { a, b, 0, AV_OPT_TYPE_CONST, { .i64 = c }, 0, 0, VE, d }

static const enum AVPixelFormat ff_amfenc_hevc_pix_fmts[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_BGRA,
    AV_PIX_FMT_BGR0,
#if CONFIG_D3D11VA
    AV_PIX_FMT_D3D11,
#endif
#if CONFIG_DXVA2
    AV_PIX_FMT_DXVA2_VLD,
#endif
    AV_PIX_FMT_NONE
};

static const AVOption options[] = {
    { "usage",                 "Encoder Usage",                                        OFFSET(usage),                 AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_HEVC_USAGE_TRANSCODING }, AMF_VIDEO_ENCODER_HEVC_USAGE_TRANSCODING, AMF_VIDEO_ENCODER_HEVC_USAGE_LOW_LATENCY_HIGH_QUALITY, VE, "usage" },
        ENUM("transcoding",     "Transcoding, video editing",           AMF_VIDEO_ENCODER_HEVC_USAGE_TRANSCODING,              "usage"),
        ENUM("ultralowlatency", "Video game streaming",                 AMF_VIDEO_ENCODER_HEVC_USAGE_ULTRA_LOW_LATENCY,        "usage"),
        ENUM("lowlatency",      "Video collaboration, RDP",             AMF_VIDEO_ENCODER_HEVC_USAGE_LOW_LATENCY,              "usage"),
        ENUM("webcam",          "Video conferencing",                   AMF_VIDEO_ENCODER_HEVC_USAGE_WEBCAM,                   "usage"),
        ENUM("highquality",     "High-quality encoding",                AMF_VIDEO_ENCODER_HEVC_USAGE_HIGH_QUALITY,             "usage"),
        ENUM("llhighquality",   "High-quality encoding (low latency)",  AMF_VIDEO_ENCODER_HEVC_USAGE_LOW_LATENCY_HIGH_QUALITY, "usage"),

    { "profile",               "Profile",                                              OFFSET(profile),               AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN }, AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN, AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN_10, VE, "profile" },
        ENUM("main",            "",                                     AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN,    "profile"),
        ENUM("main10",          "",                                     AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN_10, "profile"),

    { "profile_tier",          "Profile Tier",                                         OFFSET(tier),                  AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_HEVC_TIER_MAIN }, AMF_VIDEO_ENCODER_HEVC_TIER_MAIN, AMF_VIDEO_ENCODER_HEVC_TIER_HIGH, VE, "tier" },
        ENUM("main",            "",                                     AMF_VIDEO_ENCODER_HEVC_TIER_MAIN, "tier"),
        ENUM("high",            "",                                     AMF_VIDEO_ENCODER_HEVC_TIER_HIGH, "tier"),

    { "level",                 "Profile Level",                                        OFFSET(level),                 AV_OPT_TYPE_INT,  { .i64 = 0 }, 0, AMF_LEVEL_6_2, VE, "level" },
        ENUM("auto",            "",                                     0,             "level"),
        ENUM("1.0",             "",                                     AMF_LEVEL_1,   "level"),
        ENUM("2.0",             "",                                     AMF_LEVEL_2,   "level"),
        ENUM("2.1",             "",                                     AMF_LEVEL_2_1, "level"),
        ENUM("3.0",             "",                                     AMF_LEVEL_3,   "level"),
        ENUM("3.1",             "",                                     AMF_LEVEL_3_1, "level"),
        ENUM("4.0",             "",                                     AMF_LEVEL_4,   "level"),
        ENUM("4.1",             "",                                     AMF_LEVEL_4_1, "level"),
        ENUM("5.0",             "",                                     AMF_LEVEL_5,   "level"),
        ENUM("5.1",             "",                                     AMF_LEVEL_5_1, "level"),
        ENUM("5.2",             "",                                     AMF_LEVEL_5_2, "level"),
        ENUM("6.0",             "",                                     AMF_LEVEL_6,   "level"),
        ENUM("6.1",             "",                                     AMF_LEVEL_6_1, "level"),
        ENUM("6.2",             "",                                     AMF_LEVEL_6_2, "level"),

    { "quality",               "Quality Preset",                                       OFFSET(quality),               AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_SPEED }, AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_QUALITY, AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_SPEED, VE, "quality" },
        ENUM("speed",           "Prefer Speed",                         AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_SPEED,    "quality"),
        ENUM("balanced",        "Balanced",                             AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_BALANCED, "quality"),
        ENUM("quality",         "Prefer Quality",                       AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_QUALITY,  "quality"),

    { "rc",                    "Rate Control Method",                                  OFFSET(rate_control_mode),     AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_UNKNOWN }, AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_UNKNOWN, AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CBR, VE, "rc" },
        ENUM("cqp",             "Constant Quantization Parameter",      AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CONSTANT_QP,             "rc"),
        ENUM("cbr",             "Constant Bitrate",                     AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CBR,                     "rc"),
        ENUM("vbr_peak",        "Peak Contrained Variable Bitrate",     AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR,    "rc"),
        ENUM("vbr_latency",     "Latency Constrained Variable Bitrate", AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_LATENCY_CONSTRAINED_VBR, "rc"),

    { "header_insertion_mode", "Set header insertion mode",                            OFFSET(header_insertion_mode), AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_HEVC_HEADER_INSERTION_MODE_NONE }, AMF_VIDEO_ENCODER_HEVC_HEADER_INSERTION_MODE_NONE, AMF_VIDEO_ENCODER_HEVC_HEADER_INSERTION_MODE_IDR_ALIGNED, VE, "hdrmode" },
        ENUM("none",            "",                                     AMF_VIDEO_ENCODER_HEVC_HEADER_INSERTION_MODE_NONE,        "hdrmode"),
        ENUM("gop",             "",                                     AMF_VIDEO_ENCODER_HEVC_HEADER_INSERTION_MODE_GOP_ALIGNED, "hdrmode"),
        ENUM("idr",             "",                                     AMF_VIDEO_ENCODER_HEVC_HEADER_INSERTION_MODE_IDR_ALIGNED, "hdrmode"),

    { "gops_per_idr",          "GOPs per IDR 0-no IDR will be inserted",               OFFSET(gops_per_idr),          AV_OPT_TYPE_INT,  { .i64 = 1  },  0, INT_MAX, VE },
    { "preanalysis",           "Enable Pre-Encode/Analysis for rate rontrol (2-Pass)", OFFSET(pre_encode),            AV_OPT_TYPE_BOOL, { .i64 = 0  },  0, 1, VE },
    { "vbaq",                  "Enable VBAQ",                                          OFFSET(enable_vbaq),           AV_OPT_TYPE_BOOL, { .i64 = 0  },  0, 1, VE },
    { "hmqb",                  "Enable High Motion Quality Boost",                     OFFSET(enable_hmqb),           AV_OPT_TYPE_BOOL, { .i64 = 0  },  0, 1, VE },
    { "enforce_hrd",           "Enforce HRD",                                          OFFSET(enforce_hrd),           AV_OPT_TYPE_BOOL, { .i64 = 0  },  0, 1, VE },
    { "filler_data",           "Filler Data Enable",                                   OFFSET(filler_data),           AV_OPT_TYPE_BOOL, { .i64 = 0  },  0, 1, VE },
    { "max_au_size",           "Maximum Access Unit Size for rate control (in bits)",  OFFSET(max_au_size),           AV_OPT_TYPE_INT,  { .i64 = 0  },  0, INT_MAX, VE},
    { "min_qp_i",              "Min Quantization Parameter for I-frame",               OFFSET(min_qp_i),              AV_OPT_TYPE_INT,  { .i64 = -1 }, -1, 51, VE },
    { "max_qp_i",              "Max Quantization Parameter for I-frame",               OFFSET(max_qp_i),              AV_OPT_TYPE_INT,  { .i64 = -1 }, -1, 51, VE },
    { "min_qp_p",              "Min Quantization Parameter for P-frame",               OFFSET(min_qp_p),              AV_OPT_TYPE_INT,  { .i64 = -1 }, -1, 51, VE },
    { "max_qp_p",              "Max Quantization Parameter for P-frame",               OFFSET(max_qp_p),              AV_OPT_TYPE_INT,  { .i64 = -1 }, -1, 51, VE },
    { "qp_p",                  "Quantization Parameter for P-frame",                   OFFSET(qp_p),                  AV_OPT_TYPE_INT,  { .i64 = -1 }, -1, 51, VE },
    { "qp_i",                  "Quantization Parameter for I-frame",                   OFFSET(qp_i),                  AV_OPT_TYPE_INT,  { .i64 = -1 }, -1, 51, VE },
    { "skip_frame",            "Rate Control Based Frame Skip",                        OFFSET(skip_frame),            AV_OPT_TYPE_BOOL, { .i64 = 0  },  0, 1, VE },
    { "me_half_pel",           "Enable ME Half Pixel",                                 OFFSET(me_half_pel),           AV_OPT_TYPE_BOOL, { .i64 = 1  },  0, 1, VE },
    { "me_quarter_pel",        "Enable ME Quarter Pixel",                              OFFSET(me_quarter_pel),        AV_OPT_TYPE_BOOL, { .i64 = 1  },  0, 1, VE },
    { "aud",                   "Inserts AU Delimiter NAL unit",                        OFFSET(aud),                   AV_OPT_TYPE_BOOL, { .i64 = 0  },  0, 1, VE },
    { "log_to_dbg",            "Enable AMF logging to debug output",                   OFFSET(log_to_dbg),            AV_OPT_TYPE_BOOL, { .i64 = 0  },  0, 1, VE },
    { NULL }
};

static av_cold int amf_encode_init_hevc(AVCodecContext *avctx)
{
    int                 ret = 0;
    AMF_RESULT          res = AMF_OK;
    AMFEncContext      *ctx = avctx->priv_data;
    AMFVariantStruct    var = { 0 };
    amf_int64           profile = 0;
    amf_int64           profile_level = 0;
    AMFBuffer          *buffer;
    AMFGuid             guid;
    AMFRate             framerate;
    AMFSize             framesize = AMFConstructSize(avctx->width, avctx->height);
    int                 deblocking_filter = (avctx->flags & AV_CODEC_FLAG_LOOP_FILTER) ? 1 : 0;

    if (avctx->framerate.num > 0 && avctx->framerate.den > 0)
        framerate = AMFConstructRate(avctx->framerate.num, avctx->framerate.den);
    else
        framerate = AMFConstructRate(avctx->time_base.den, avctx->time_base.num * avctx->ticks_per_frame);

    if ((ret = ff_amf_encode_init(avctx)) < 0)
        return ret;

    // Static parameters
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_USAGE, ctx->usage);

    AMF_ASSIGN_PROPERTY_SIZE(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_FRAMESIZE, framesize);

    AMF_ASSIGN_PROPERTY_RATE(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_FRAMERATE, framerate);

    switch (avctx->profile) {
    case FF_PROFILE_HEVC_MAIN:
        profile = AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN;
        break;
    case FF_PROFILE_HEVC_MAIN_10:
        profile = AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN_10;
        break;
    default:
        break;
    }
    if (profile == 0)
        profile = ctx->profile;
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_PROFILE, profile);

    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_TIER, ctx->tier);

    profile_level = avctx->level;
    if (profile_level == FF_LEVEL_UNKNOWN)
        profile_level = ctx->level;
    if (profile_level != 0)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_PROFILE_LEVEL, profile_level);

    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET, ctx->quality);
    // Maximum Reference Frames
    if (avctx->refs != -1)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MAX_NUM_REFRAMES, avctx->refs);
    // Aspect Ratio
    if (avctx->sample_aspect_ratio.den && avctx->sample_aspect_ratio.num) {
        AMFRatio ratio = AMFConstructRatio(avctx->sample_aspect_ratio.num, avctx->sample_aspect_ratio.den);
        AMF_ASSIGN_PROPERTY_RATIO(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_ASPECT_RATIO, ratio);
    }

    // Picture control properties
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_NUM_GOPS_PER_IDR, ctx->gops_per_idr);
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_GOP_SIZE, avctx->gop_size);
    if (avctx->slices > 1)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_SLICES_PER_FRAME, avctx->slices);
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_DE_BLOCKING_FILTER_DISABLE, deblocking_filter);
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_HEADER_INSERTION_MODE, ctx->header_insertion_mode);

    // Rate control properties
    // Auto detect rate control method
    if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_UNKNOWN) {
        if (ctx->min_qp_i != -1 || ctx->max_qp_i != -1 ||
            ctx->min_qp_p != -1 || ctx->max_qp_p != -1 ||
            ctx->qp_i !=-1 || ctx->qp_p != -1) {
            ctx->rate_control_mode = AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CONSTANT_QP;
            av_log(ctx, AV_LOG_DEBUG, "Rate control method turned to CQP\n");
        } else if (avctx->rc_max_rate > 0) {
            ctx->rate_control_mode = AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR;
            av_log(ctx, AV_LOG_DEBUG, "Rate control method turned to Peak VBR\n");
        } else {
            ctx->rate_control_mode = AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CBR;
            av_log(ctx, AV_LOG_DEBUG, "Rate control method turned to CBR\n");
        }
    }

    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD, ctx->rate_control_mode);
    if (avctx->rc_buffer_size) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_VBV_BUFFER_SIZE, avctx->rc_buffer_size);

        if (avctx->rc_initial_buffer_occupancy != 0) {
            int amf_buffer_fullness = avctx->rc_initial_buffer_occupancy * 64 / avctx->rc_buffer_size;
            if (amf_buffer_fullness > 64)
                amf_buffer_fullness = 64;
            AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_INITIAL_VBV_BUFFER_FULLNESS, amf_buffer_fullness);
        }
    }

    // Pre-Encode/Two-Pass(pre-encode assisted rate control)
    if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CONSTANT_QP) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_PREENCODE_ENABLE, 0);
        if (ctx->pre_encode) {
            ctx->pre_encode = 0;
            av_log(ctx, AV_LOG_WARNING, "Pre-Encode is not supported by CQP rate control method, automatically disabled\n");
        }
    } else {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_PREENCODE_ENABLE, ctx->pre_encode);
    }

    // VBAQ
    if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CONSTANT_QP) {
        AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_ENABLE_VBAQ, 0);
        if (ctx->enable_vbaq) {
            ctx->enable_vbaq = 0;
            av_log(ctx, AV_LOG_WARNING, "VBAQ is not supported by CQP rate control method, automatically disabled\n");
        }
    } else {
        AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_ENABLE_VBAQ, !!ctx->enable_vbaq);
    }

    // High Motion Quality Boost mode
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_HIGH_MOTION_QUALITY_BOOST_ENABLE, !!ctx->enable_hmqb);

    // Motion estimation
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MOTION_HALF_PIXEL, ctx->me_half_pel);
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MOTION_QUARTERPIXEL, ctx->me_quarter_pel);

    // Dynamic rate control params
    if (ctx->max_au_size)
        ctx->enforce_hrd = 1;
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_ENFORCE_HRD, ctx->enforce_hrd);
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_FILLER_DATA_ENABLE, ctx->filler_data);

    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_TARGET_BITRATE, avctx->bit_rate);

    if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CBR)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_PEAK_BITRATE, avctx->bit_rate);
    if (avctx->rc_max_rate) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_PEAK_BITRATE, avctx->rc_max_rate);
    } else if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR) {
        av_log(ctx, AV_LOG_WARNING, "Rate control method is PEAK_CONSTRAINED_VBR but rc_max_rate is not set\n");
    }

    // Color Range (Studio/Partial/TV/MPEG or Full/PC/JPEG)
    if (avctx->color_range == AVCOL_RANGE_JPEG) {
        AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_NOMINAL_RANGE, AMF_VIDEO_ENCODER_HEVC_NOMINAL_RANGE_FULL);
    } else {
        AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_NOMINAL_RANGE, AMF_VIDEO_ENCODER_HEVC_NOMINAL_RANGE_STUDIO);
    }

    // Output color profile, transfer and primaries
    if (ctx->out_color_profile > AMF_VIDEO_CONVERTER_COLOR_PROFILE_UNKNOWN)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_OUTPUT_COLOR_PROFILE, ctx->out_color_profile);
    if (ctx->out_color_trc > AMF_COLOR_TRANSFER_CHARACTERISTIC_UNDEFINED)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_OUTPUT_TRANSFER_CHARACTERISTIC, ctx->out_color_trc);
    if (ctx->out_color_prm > AMF_COLOR_PRIMARIES_UNDEFINED)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_OUTPUT_COLOR_PRIMARIES, ctx->out_color_prm);

    // Set 10-bit encoding if possible
    if (ctx->bit_depth == 10)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_COLOR_BIT_DEPTH, AMF_COLOR_BIT_DEPTH_10);

    // Init encoder
    res = ctx->encoder->pVtbl->Init(ctx->encoder, ctx->format, avctx->width, avctx->height);
    AMF_RETURN_IF_FALSE(ctx, res == AMF_OK, AVERROR_BUG, "encoder->Init() failed with error %d\n", res);

    // Dynamic picture control params
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MAX_AU_SIZE, ctx->max_au_size);

    // QP Minimum / Maximum
    if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CONSTANT_QP) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MIN_QP_I, 0);
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MAX_QP_I, 51);
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MIN_QP_P, 0);
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MAX_QP_P, 51);
    } else {
        if (ctx->min_qp_i != -1) {
            AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MIN_QP_I, ctx->min_qp_i);
        } else if (avctx->qmin != -1) {
            int qval = avctx->qmin > 51 ? 51 : avctx->qmin;
            AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MIN_QP_I, qval);
        }
        if (ctx->max_qp_i != -1) {
            AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MAX_QP_I, ctx->max_qp_i);
        } else if (avctx->qmax != -1) {
            int qval = avctx->qmax > 51 ? 51 : avctx->qmax;
            AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MAX_QP_I, qval);
        }
        if (ctx->min_qp_p != -1) {
            AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MIN_QP_P, ctx->min_qp_p);
        } else if (avctx->qmin != -1) {
            int qval = avctx->qmin > 51 ? 51 : avctx->qmin;
            AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MIN_QP_P, qval);
        }
        if (ctx->max_qp_p != -1) {
            AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MAX_QP_P, ctx->max_qp_p);
        } else if (avctx->qmax != -1) {
            int qval = avctx->qmax > 51 ? 51 : avctx->qmax;
            AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MAX_QP_P, qval);
        }
        if (ctx->min_qp_i == -1 && ctx->max_qp_i == -1 && ctx->min_qp_p == -1 && ctx->max_qp_p == -1 &&
            avctx->qmin == -1 && avctx->qmax == -1) {
            switch (ctx->usage) {
            case AMF_VIDEO_ENCODER_HEVC_USAGE_TRANSCONDING:
                AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MIN_QP_I, 18);
                AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MAX_QP_I, 46);
                AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MIN_QP_P, 18);
                AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MAX_QP_P, 46);
                break;
            case AMF_VIDEO_ENCODER_HEVC_USAGE_ULTRA_LOW_LATENCY:
            case AMF_VIDEO_ENCODER_HEVC_USAGE_LOW_LATENCY:
            case AMF_VIDEO_ENCODER_HEVC_USAGE_WEBCAM:
                AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MIN_QP_I, 22);
                AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MAX_QP_I, 48);
                AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MIN_QP_P, 22);
                AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_MAX_QP_P, 48);
                break;
            }
        }
    }

    if (ctx->qp_p != -1)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_QP_P, ctx->qp_p);
    if (ctx->qp_i != -1)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_QP_I, ctx->qp_i);
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_SKIP_FRAME_ENABLE, ctx->skip_frame);

    // Fill extradata
    res = AMFVariantInit(&var);
    AMF_RETURN_IF_FALSE(ctx, res == AMF_OK, AVERROR_BUG, "AMFVariantInit() failed with error %d\n", res);

    res = ctx->encoder->pVtbl->GetProperty(ctx->encoder, AMF_VIDEO_ENCODER_HEVC_EXTRADATA, &var);
    AMF_RETURN_IF_FALSE(ctx, res == AMF_OK, AVERROR_BUG, "GetProperty(AMF_VIDEO_ENCODER_EXTRADATA) failed with error %d\n", res);
    AMF_RETURN_IF_FALSE(ctx, var.pInterface != NULL, AVERROR_BUG, "GetProperty(AMF_VIDEO_ENCODER_EXTRADATA) returned NULL\n");

    guid = IID_AMFBuffer();

    res = var.pInterface->pVtbl->QueryInterface(var.pInterface, &guid, (void**)&buffer); // query for buffer interface
    if (res != AMF_OK)
        var.pInterface->pVtbl->Release(var.pInterface);
    AMF_RETURN_IF_FALSE(ctx, res == AMF_OK, AVERROR_BUG, "QueryInterface(IID_AMFBuffer) failed with error %d\n", res);

    avctx->extradata_size = (int)buffer->pVtbl->GetSize(buffer);
    avctx->extradata = av_mallocz(avctx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!avctx->extradata) {
        buffer->pVtbl->Release(buffer);
        var.pInterface->pVtbl->Release(var.pInterface);
        return AVERROR(ENOMEM);
    }
    memcpy(avctx->extradata, buffer->pVtbl->GetNative(buffer), avctx->extradata_size);

    buffer->pVtbl->Release(buffer);
    var.pInterface->pVtbl->Release(var.pInterface);

    return 0;
}

static const FFCodecDefault defaults[] = {
    { "refs",       "-1"  },
    { "aspect",     "0"   },
    { "b",          "2M"  },
    { "g",          "250" },
    { "slices",     "1"   },
    { "qmin",       "-1"  },
    { "qmax",       "-1"  },
    { NULL                },
};

static const AVClass hevc_amf_class = {
    .class_name = "hevc_amf",
    .item_name = av_default_item_name,
    .option = options,
    .version = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_hevc_amf_encoder = {
    .p.name         = "hevc_amf",
    CODEC_LONG_NAME("AMD AMF HEVC encoder"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_HEVC,
    .init           = amf_encode_init_hevc,
    FF_CODEC_RECEIVE_PACKET_CB(ff_amf_receive_packet),
    .close          = ff_amf_encode_close,
    .priv_data_size = sizeof(AMFEncContext),
    .p.priv_class   = &hevc_amf_class,
    .defaults       = defaults,
    .p.capabilities = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_HARDWARE |
                      AV_CODEC_CAP_DR1,
    .caps_internal  = FF_CODEC_CAP_NOT_INIT_THREADSAFE |
                      FF_CODEC_CAP_INIT_CLEANUP,
    .p.pix_fmts     = ff_amfenc_hevc_pix_fmts,
    .p.wrapper_name = "amf",
    .hw_configs     = ff_amfenc_hw_configs,
};
