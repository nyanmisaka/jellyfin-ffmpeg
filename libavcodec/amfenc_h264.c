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

static const enum AVPixelFormat ff_amfenc_h264_pix_fmts[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV420P,
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
    { "usage",            "Encoder Usage",                                        OFFSET(usage),                AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_USAGE_TRANSCODING }, AMF_VIDEO_ENCODER_USAGE_TRANSCODING, AMF_VIDEO_ENCODER_USAGE_LOW_LATENCY_HIGH_QUALITY, VE, "usage" },
        ENUM("transcoding",          "Transcoding, video editing",                AMF_VIDEO_ENCODER_USAGE_TRANSCODING,              "usage"),
        ENUM("ultralowlatency",      "Video game streaming",                      AMF_VIDEO_ENCODER_USAGE_ULTRA_LOW_LATENCY,        "usage"),
        ENUM("lowlatency",           "Video collaboration, RDP",                  AMF_VIDEO_ENCODER_USAGE_LOW_LATENCY,              "usage"),
        ENUM("webcam",               "Video conferencing",                        AMF_VIDEO_ENCODER_USAGE_WEBCAM,                   "usage"),
        ENUM("highquality",          "High-quality encoding",                     AMF_VIDEO_ENCODER_USAGE_HIGH_QUALITY,             "usage"),
        ENUM("llhighquality",        "High-quality encoding (low latency)",       AMF_VIDEO_ENCODER_USAGE_LOW_LATENCY_HIGH_QUALITY, "usage"),

    { "profile",          "Profile",                                              OFFSET(profile),              AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_PROFILE_MAIN }, AMF_VIDEO_ENCODER_PROFILE_BASELINE, AMF_VIDEO_ENCODER_PROFILE_CONSTRAINED_HIGH, VE, "profile" },
        ENUM("main",                 "",                                          AMF_VIDEO_ENCODER_PROFILE_MAIN,                 "profile"),
        ENUM("high",                 "",                                          AMF_VIDEO_ENCODER_PROFILE_HIGH,                 "profile"),
        ENUM("constrained_baseline", "",                                          AMF_VIDEO_ENCODER_PROFILE_CONSTRAINED_BASELINE, "profile"),
        ENUM("constrained_high",     "",                                          AMF_VIDEO_ENCODER_PROFILE_CONSTRAINED_HIGH,     "profile"),

    { "level",            "Profile Level",                                        OFFSET(level),                AV_OPT_TYPE_INT,  { .i64 = 0 }, 0, 62, VE, "level" },
        ENUM("auto",                 "",                                          0,  "level"),
        ENUM("1.0",                  "",                                          10, "level"),
        ENUM("1.1",                  "",                                          11, "level"),
        ENUM("1.2",                  "",                                          12, "level"),
        ENUM("1.3",                  "",                                          13, "level"),
        ENUM("2.0",                  "",                                          20, "level"),
        ENUM("2.1",                  "",                                          21, "level"),
        ENUM("2.2",                  "",                                          22, "level"),
        ENUM("3.0",                  "",                                          30, "level"),
        ENUM("3.1",                  "",                                          31, "level"),
        ENUM("3.2",                  "",                                          32, "level"),
        ENUM("4.0",                  "",                                          40, "level"),
        ENUM("4.1",                  "",                                          41, "level"),
        ENUM("4.2",                  "",                                          42, "level"),
        ENUM("5.0",                  "",                                          50, "level"),
        ENUM("5.1",                  "",                                          51, "level"),
        ENUM("5.2",                  "",                                          52, "level"),
        ENUM("6.0",                  "",                                          60, "level"),
        ENUM("6.1",                  "",                                          61, "level"),
        ENUM("6.2",                  "",                                          62, "level"),

    { "quality",          "Quality Preset",                                       OFFSET(quality),              AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_QUALITY_PRESET_SPEED }, AMF_VIDEO_ENCODER_QUALITY_PRESET_BALANCED, AMF_VIDEO_ENCODER_QUALITY_PRESET_QUALITY, VE, "quality" },
        ENUM("speed",                "Prefer Speed",                              AMF_VIDEO_ENCODER_QUALITY_PRESET_SPEED,    "quality"),
        ENUM("balanced",             "Balanced",                                  AMF_VIDEO_ENCODER_QUALITY_PRESET_BALANCED, "quality"),
        ENUM("quality",              "Prefer Quality",                            AMF_VIDEO_ENCODER_QUALITY_PRESET_QUALITY,  "quality"),

    { "rc",               "Rate Control Method",                                  OFFSET(rate_control_mode),    AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_UNKNOWN }, AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_UNKNOWN, AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_QUALITY_VBR, VE, "rc" },
        ENUM("cqp",                  "Constant Quantization Parameter",           AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CONSTANT_QP,             "rc"),
        ENUM("cbr",                  "Constant Bitrate",                          AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CBR,                     "rc"),
        ENUM("vbr_peak",             "Peak Constrained Variable Bitrate",         AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR,    "rc"),
        ENUM("vbr_latency",          "Latency Constrained Variable Bitrate",      AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_LATENCY_CONSTRAINED_VBR, "rc"),
        ENUM("qvbr",                 "Quality-defined Variable Bitrate",          AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_QUALITY_VBR,             "rc"),

    { "preanalysis",      "Enable Pre-Encode/Analysis for Rate Control (2-Pass)", OFFSET(pre_encode),           AV_OPT_TYPE_BOOL, { .i64 = 0  },  0,   1, VE },
    { "vbaq",             "Enable VBAQ",                                          OFFSET(enable_vbaq),          AV_OPT_TYPE_BOOL, { .i64 = 0  },  0,   1, VE },
    { "hmqb",             "Enable High Motion Quality Boost",                     OFFSET(enable_hmqb),          AV_OPT_TYPE_BOOL, { .i64 = 0  },  0,   1, VE },
    { "enforce_hrd",      "Enforce HRD",                                          OFFSET(enforce_hrd),          AV_OPT_TYPE_BOOL, { .i64 = 0  },  0,   1, VE },
    { "filler_data",      "Filler Data Enable",                                   OFFSET(filler_data),          AV_OPT_TYPE_BOOL, { .i64 = 0  },  0,   1, VE },
    { "frame_skipping",   "Rate Control Based Frame Skip",                        OFFSET(skip_frame),           AV_OPT_TYPE_BOOL, { .i64 = 0  },  0,   1, VE },
    { "qvbr_level",       "Quality level for QVBR rate control",                  OFFSET(qvbr_level),           AV_OPT_TYPE_INT,  { .i64 = 23 },  1,   51, VE },
    { "qp_i",             "Quantization Parameter for I-Frame",                   OFFSET(qp_i),                 AV_OPT_TYPE_INT,  { .i64 = -1 }, -1,   51, VE },
    { "qp_p",             "Quantization Parameter for P-Frame",                   OFFSET(qp_p),                 AV_OPT_TYPE_INT,  { .i64 = -1 }, -1,   51, VE },
    { "qp_b",             "Quantization Parameter for B-Frame",                   OFFSET(qp_b),                 AV_OPT_TYPE_INT,  { .i64 = -1 }, -1,   51, VE },
    { "max_au_size",      "Maximum Access Unit Size for rate control (in bits)",  OFFSET(max_au_size),          AV_OPT_TYPE_INT,  { .i64 = 0  },  0,   INT_MAX, VE },
    { "header_spacing",   "Header Insertion Spacing",                             OFFSET(header_spacing),       AV_OPT_TYPE_INT,  { .i64 = -1 }, -1,   1000, VE },
    { "bf_delta_qp",      "B-Picture Delta QP",                                   OFFSET(b_frame_delta_qp),     AV_OPT_TYPE_INT,  { .i64 = 4  }, -10,  10, VE },
    { "bf_ref",           "Enable Reference to B-Frames",                         OFFSET(b_frame_ref),          AV_OPT_TYPE_BOOL, { .i64 = 1  },  0,   1, VE },
    { "bf_ref_delta_qp",  "Reference B-Picture Delta QP",                         OFFSET(ref_b_frame_delta_qp), AV_OPT_TYPE_INT,  { .i64 = 4  }, -10,  10, VE },
    { "intra_refresh_mb", "Intra Refresh MBs Number Per Slot in Macroblocks",     OFFSET(intra_refresh_mb),     AV_OPT_TYPE_INT,  { .i64 = 0  },  0,   INT_MAX, VE },

    { "coder",            "Coding Type",                                          OFFSET(coding_mode),          AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_UNDEFINED }, AMF_VIDEO_ENCODER_UNDEFINED, AMF_VIDEO_ENCODER_CALV, VE, "coder" },
        ENUM("auto",                 "Automatic",                                 AMF_VIDEO_ENCODER_UNDEFINED, "coder"),
        ENUM("cavlc",                "Context Adaptive Variable-Length Coding",   AMF_VIDEO_ENCODER_CALV,      "coder"),
        ENUM("cabac",                "Context Adaptive Binary Arithmetic Coding", AMF_VIDEO_ENCODER_CABAC,     "coder"),

    { "me_half_pel",      "Enable ME Half Pixel",                                 OFFSET(me_half_pel),          AV_OPT_TYPE_BOOL, { .i64 = 1  },  0,   1, VE },
    { "me_quarter_pel",   "Enable ME Quarter Pixel",                              OFFSET(me_quarter_pel),       AV_OPT_TYPE_BOOL, { .i64 = 1  },  0,   1, VE },
    { "aud",              "Inserts AU Delimiter NAL unit",                        OFFSET(aud),                  AV_OPT_TYPE_BOOL, { .i64 = 0  },  0,   1, VE },

    { "log_to_dbg",       "Enable AMF logging to debug output",                   OFFSET(log_to_dbg),           AV_OPT_TYPE_BOOL, { .i64 = 0  },  0,   1, VE },
    { NULL }
};

static av_cold int amf_encode_init_h264(AVCodecContext *avctx)
{
    int                              ret = 0;
    AMF_RESULT                       res = AMF_OK;
    AMFEncContext                   *ctx = avctx->priv_data;
    AMFVariantStruct                 var = { 0 };
    amf_int64                        profile = 0;
    amf_int64                        profile_level = 0;
    AMFBuffer                       *buffer;
    AMFGuid                          guid;
    AMFRate                          framerate;
    AMFSize                          framesize = AMFConstructSize(avctx->width, avctx->height);
    int                              probed_rc_mode = AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_UNKNOWN;
    int                              deblocking_filter = (avctx->flags & AV_CODEC_FLAG_LOOP_FILTER) ? 1 : 0;

    if (avctx->framerate.num > 0 && avctx->framerate.den > 0)
        framerate = AMFConstructRate(avctx->framerate.num, avctx->framerate.den);
    else
        framerate = AMFConstructRate(avctx->time_base.den, avctx->time_base.num * avctx->ticks_per_frame);

    if ((ret = ff_amf_encode_init(avctx)) != 0)
        return ret;

    // Static parameters
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_USAGE, ctx->usage);

    AMF_ASSIGN_PROPERTY_SIZE(res, ctx->encoder, AMF_VIDEO_ENCODER_FRAMESIZE, framesize);

    AMF_ASSIGN_PROPERTY_RATE(res, ctx->encoder, AMF_VIDEO_ENCODER_FRAMERATE, framerate);

    switch (avctx->profile) {
    case FF_PROFILE_H264_BASELINE:
        profile = AMF_VIDEO_ENCODER_PROFILE_BASELINE;
        break;
    case FF_PROFILE_H264_MAIN:
        profile = AMF_VIDEO_ENCODER_PROFILE_MAIN;
        break;
    case FF_PROFILE_H264_HIGH:
        profile = AMF_VIDEO_ENCODER_PROFILE_HIGH;
        break;
    case FF_PROFILE_H264_CONSTRAINED_BASELINE:
        profile = AMF_VIDEO_ENCODER_PROFILE_CONSTRAINED_BASELINE;
        break;
    case (FF_PROFILE_H264_HIGH | FF_PROFILE_H264_CONSTRAINED):
        profile = AMF_VIDEO_ENCODER_PROFILE_CONSTRAINED_HIGH;
        break;
    }
    if (profile == 0)
        profile = ctx->profile;

    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_PROFILE, profile);

    profile_level = avctx->level;
    if (profile_level == FF_LEVEL_UNKNOWN)
        profile_level = ctx->level;
    if (profile_level != 0)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_PROFILE_LEVEL, profile_level);

    // Maximum Reference Frames
    if (avctx->refs != -1)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_MAX_NUM_REFRAMES, avctx->refs);
    if (avctx->sample_aspect_ratio.den && avctx->sample_aspect_ratio.num) {
        AMFRatio ratio = AMFConstructRatio(avctx->sample_aspect_ratio.num, avctx->sample_aspect_ratio.den);
        AMF_ASSIGN_PROPERTY_RATIO(res, ctx->encoder, AMF_VIDEO_ENCODER_ASPECT_RATIO, ratio);
    }

    // Auto detect rate control method
    if (ctx->qp_i != -1 || ctx->qp_p != -1 || ctx->qp_b != -1) {
        probed_rc_mode = AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CONSTANT_QP;
    } else if (avctx->rc_max_rate > 0 ) {
        probed_rc_mode = AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR;
    } else {
        probed_rc_mode = AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CBR;
    }

    if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_UNKNOWN) {
        switch (probed_rc_mode) {
        case AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CONSTANT_QP:
            ctx->rate_control_mode = probed_rc_mode;
            av_log(ctx, AV_LOG_DEBUG, "Rate control method turned to CQP\n");
            break;
        case AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR:
            ctx->rate_control_mode = probed_rc_mode;
            av_log(ctx, AV_LOG_DEBUG, "Rate control method turned to Peak VBR\n");
            break;
        case AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CBR:
            ctx->rate_control_mode = probed_rc_mode;
            av_log(ctx, AV_LOG_DEBUG, "Rate control method turned to CBR\n");
            break;
        }
    }

    // Pre-Encode/Two-Pass(pre-encode assisted rate control)
    if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CONSTANT_QP) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_PREENCODE_ENABLE, AMF_VIDEO_ENCODER_PREENCODE_DISABLED);
        if (ctx->pre_encode) {
            ctx->pre_encode = 0;
            av_log(ctx, AV_LOG_WARNING, "Pre-Encode is not supported by CQP rate control method, automatically disabled\n");
        }
    } else {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_PREENCODE_ENABLE, ctx->pre_encode);
    }

    // Quality preset
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_QUALITY_PRESET, ctx->quality);

    // Dynamic parmaters
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD, ctx->rate_control_mode);
    if (res != AMF_OK && ctx->rate_control_mode == AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_QUALITY_VBR) {
        ctx->rate_control_mode = probed_rc_mode;
        av_log(ctx, AV_LOG_WARNING, "QVBR is not supported by this GPU, switch to auto detect rate control method\n");
    }

    // High Motion Quality Boost mode
    if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_QUALITY_VBR) {
        AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_HIGH_MOTION_QUALITY_BOOST_ENABLE, 0);
        if (ctx->enable_hmqb) {
            ctx->enable_hmqb = 0;
            av_log(ctx, AV_LOG_WARNING, "VBAQ is not supported by QVBR rate control method, automatically disabled\n");
        }
    } else {
        AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_HIGH_MOTION_QUALITY_BOOST_ENABLE, !!ctx->enable_hmqb);
    }

    // VBV Buffer
    if (avctx->rc_buffer_size != 0) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_VBV_BUFFER_SIZE, avctx->rc_buffer_size);
        if (avctx->rc_initial_buffer_occupancy != 0) {
            int amf_buffer_fullness = avctx->rc_initial_buffer_occupancy * 64 / avctx->rc_buffer_size;
            if (amf_buffer_fullness > 64)
                amf_buffer_fullness = 64;
            AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_INITIAL_VBV_BUFFER_FULLNESS, amf_buffer_fullness);
        }
    }

    // Maximum Access Unit Size
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_MAX_AU_SIZE, ctx->max_au_size);

    if (ctx->max_au_size)
        ctx->enforce_hrd = 1;

    // QP Minimum / Maximum
    if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CONSTANT_QP) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_MIN_QP, 0);
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_MAX_QP, 51);
    } else if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_QUALITY_VBR) {
        if (ctx->qvbr_level) {
            AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_QVBR_QUALITY_LEVEL, ctx->qvbr_level);
        }
    } else {
        if (avctx->qmin == -1 && avctx->qmax == -1) {
            switch (ctx->usage) {
            case AMF_VIDEO_ENCODER_USAGE_TRANSCONDING:
                AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_MIN_QP, 18);
                AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_MAX_QP, 46);
                break;
            case AMF_VIDEO_ENCODER_USAGE_ULTRA_LOW_LATENCY:
            case AMF_VIDEO_ENCODER_USAGE_LOW_LATENCY:
            case AMF_VIDEO_ENCODER_USAGE_WEBCAM:
                AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_MIN_QP, 22);
                AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_MAX_QP, 48);
                break;
            }
        }
        if (avctx->qmin != -1) {
            int qval = avctx->qmin > 51 ? 51 : avctx->qmin;
            AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_MIN_QP, qval);
        }
        if (avctx->qmax != -1) {
            int qval = avctx->qmax > 51 ? 51 : avctx->qmax;
            AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_MAX_QP, qval);
        }
    }
    // QP Values
    if (ctx->qp_i != -1)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_QP_I, ctx->qp_i);
    if (ctx->qp_p != -1)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_QP_P, ctx->qp_p);
    if (ctx->qp_b != -1)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_QP_B, ctx->qp_b);

    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_TARGET_BITRATE, avctx->bit_rate);

    if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CBR)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_PEAK_BITRATE, avctx->bit_rate);

    if (avctx->rc_max_rate) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_PEAK_BITRATE, avctx->rc_max_rate);
    } else if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR) {
        av_log(ctx, AV_LOG_WARNING, "Rate control method is PEAK_CONSTRAINED_VBR but rc_max_rate is not set\n");
    }

    // Color Range (Partial/TV/MPEG or Full/PC/JPEG)
    if (avctx->color_range == AVCOL_RANGE_JPEG) {
        AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_FULL_RANGE_COLOR, 1);
    } else {
        AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_FULL_RANGE_COLOR, 0);
    }

    // Set output color profile, transfer and primaries
    if (ctx->out_color_profile > AMF_VIDEO_CONVERTER_COLOR_PROFILE_UNKNOWN)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_OUTPUT_COLOR_PROFILE, ctx->out_color_profile);
    if (ctx->out_color_trc > AMF_COLOR_TRANSFER_CHARACTERISTIC_UNDEFINED)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_OUTPUT_TRANSFER_CHARACTERISTIC, ctx->out_color_trc);
    if (ctx->out_color_prm > AMF_COLOR_PRIMARIES_UNDEFINED)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_OUTPUT_COLOR_PRIMARIES, ctx->out_color_prm);

    // Initialize Encoder
    res = ctx->encoder->pVtbl->Init(ctx->encoder, ctx->format, avctx->width, avctx->height);
    AMF_RETURN_IF_FALSE(ctx, res == AMF_OK, AVERROR_BUG, "encoder->Init() failed with error %d\n", res);

    // Enforce HRD, Filler Data, Frame Skipping, Deblocking Filter
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_ENFORCE_HRD, !!ctx->enforce_hrd);
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_FILLER_DATA_ENABLE, !!ctx->filler_data);
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_RATE_CONTROL_SKIP_FRAME_ENABLE, !!ctx->skip_frame);
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_DE_BLOCKING_FILTER, !!deblocking_filter);

    // VBAQ
    if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CONSTANT_QP) {
        AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_ENABLE_VBAQ, 0);
        if (ctx->enable_vbaq) {
            ctx->enable_vbaq = 0;
            av_log(ctx, AV_LOG_WARNING, "VBAQ is not supported by cqp Rate Control Method, automatically disabled\n");
        }
    } else {
        AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_ENABLE_VBAQ, !!ctx->enable_vbaq);
    }

    // B-Frames
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_B_PIC_PATTERN, avctx->max_b_frames);
    if (res != AMF_OK) {
        res = ctx->encoder->pVtbl->GetProperty(ctx->encoder, AMF_VIDEO_ENCODER_B_PIC_PATTERN, &var);
        av_log(ctx, AV_LOG_WARNING, "B-frames=%d is not supported by this GPU, switched to %d\n",
            avctx->max_b_frames, (int)var.int64Value);
        avctx->max_b_frames = (int)var.int64Value;
    }
    if (avctx->max_b_frames) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_MAX_CONSECUTIVE_BPICTURES, avctx->max_b_frames);
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_B_PIC_DELTA_QP, ctx->b_frame_delta_qp);
        AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_B_REFERENCE_ENABLE, !!ctx->b_frame_ref);
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_REF_B_PIC_DELTA_QP, ctx->ref_b_frame_delta_qp);
    }

    // Keyframe Interval
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_IDR_PERIOD, avctx->gop_size);

    // Header Insertion Spacing
    if (ctx->header_spacing >= 0)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEADER_INSERTION_SPACING, ctx->header_spacing);

    // Intra-Refresh, Slicing
    if (ctx->intra_refresh_mb > 0)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_INTRA_REFRESH_NUM_MBS_PER_SLOT, ctx->intra_refresh_mb);
    if (avctx->slices > 1)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_SLICES_PER_FRAME, avctx->slices);

    // Coding
    if (ctx->coding_mode != 0)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_CABAC_ENABLE, ctx->coding_mode);

    // Motion Estimation
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_MOTION_HALF_PIXEL, !!ctx->me_half_pel);
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_MOTION_QUARTERPIXEL, !!ctx->me_quarter_pel);

    // fill extradata
    res = AMFVariantInit(&var);
    AMF_RETURN_IF_FALSE(ctx, res == AMF_OK, AVERROR_BUG, "AMFVariantInit() failed with error %d\n", res);

    res = ctx->encoder->pVtbl->GetProperty(ctx->encoder, AMF_VIDEO_ENCODER_EXTRADATA, &var);
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
    { "refs",       "-1"    },
    { "aspect",     "0"     },
    { "qmin",       "-1"    },
    { "qmax",       "-1"    },
    { "b",          "2M"    },
    { "g",          "250"   },
    { "slices",     "1"     },
    { "flags",      "+loop" },
    { NULL                  },
};

static const AVClass h264_amf_class = {
    .class_name = "h264_amf",
    .item_name = av_default_item_name,
    .option = options,
    .version = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_h264_amf_encoder = {
    .p.name         = "h264_amf",
    CODEC_LONG_NAME("AMD AMF H.264 Encoder"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_H264,
    .init           = amf_encode_init_h264,
    FF_CODEC_RECEIVE_PACKET_CB(ff_amf_receive_packet),
    .close          = ff_amf_encode_close,
    .priv_data_size = sizeof(AMFEncContext),
    .p.priv_class   = &h264_amf_class,
    .defaults       = defaults,
    .p.capabilities = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_HARDWARE |
                      AV_CODEC_CAP_DR1,
    .caps_internal  = FF_CODEC_CAP_NOT_INIT_THREADSAFE |
                      FF_CODEC_CAP_INIT_CLEANUP,
    .p.pix_fmts     = ff_amfenc_h264_pix_fmts,
    .p.wrapper_name = "amf",
    .hw_configs     = ff_amfenc_hw_configs,
};
