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

static const enum AVPixelFormat ff_amfenc_av1_pix_fmts[] = {
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
    { "usage",                 "Encoder Usage",                                        OFFSET(usage),                 AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_AV1_USAGE_TRANSCODING }, AMF_VIDEO_ENCODER_AV1_USAGE_TRANSCODING, AMF_VIDEO_ENCODER_AV1_USAGE_LOW_LATENCY, VE, "usage" },
        ENUM("transcoding",     "Transcoding, video editing",           AMF_VIDEO_ENCODER_AV1_USAGE_TRANSCODING, "usage"),
        ENUM("lowlatency",      "Video collaboration, RDP",             AMF_VIDEO_ENCODER_AV1_USAGE_LOW_LATENCY, "usage"),

    { "profile",               "Profile",                                              OFFSET(profile),               AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_AV1_PROFILE_MAIN }, AMF_VIDEO_ENCODER_AV1_PROFILE_MAIN, AMF_VIDEO_ENCODER_AV1_PROFILE_MAIN, VE, "profile" },
        ENUM("main",            "",                                     AMF_VIDEO_ENCODER_AV1_PROFILE_MAIN, "profile"),

    { "level",                 "Profile Level",                                        OFFSET(level),                 AV_OPT_TYPE_INT,  { .i64 = 0 }, 0, AMF_VIDEO_ENCODER_AV1_LEVEL_7_3, VE, "level" },
        ENUM("auto",            "",                                     0,             "level"),
        ENUM("2.0",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_2_0, "level"),
        ENUM("2.1",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_2_1, "level"),
        ENUM("2.2",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_2_2, "level"),
        ENUM("2.3",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_2_3, "level"),
        ENUM("3.0",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_3_0, "level"),
        ENUM("3.1",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_3_1, "level"),
        ENUM("3.2",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_3_2, "level"),
        ENUM("3.3",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_3_3, "level"),
        ENUM("4.0",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_4_0, "level"),
        ENUM("4.1",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_4_1, "level"),
        ENUM("4.2",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_4_2, "level"),
        ENUM("4.3",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_4_3, "level"),
        ENUM("5.0",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_5_0, "level"),
        ENUM("5.1",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_5_1, "level"),
        ENUM("5.2",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_5_2, "level"),
        ENUM("5.3",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_5_3, "level"),
        ENUM("6.0",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_6_0, "level"),
        ENUM("6.1",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_6_1, "level"),
        ENUM("6.2",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_6_2, "level"),
        ENUM("6.3",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_6_3, "level"),
        ENUM("7.0",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_7_0, "level"),
        ENUM("7.1",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_7_1, "level"),
        ENUM("7.2",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_7_2, "level"),
        ENUM("7.3",             "",                                     AMF_VIDEO_ENCODER_AV1_LEVEL_7_3, "level"),

    { "quality",               "Quality Preset",                                       OFFSET(quality),               AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_SPEED }, AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_HIGH_QUALITY, AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_SPEED, VE, "quality" },
        ENUM("speed",           "Speed",                                AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_SPEED,        "quality"),
        ENUM("balanced",        "Balanced",                             AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_BALANCED,     "quality"),
        ENUM("quality",         "Quality",                              AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_QUALITY,      "quality"),
        ENUM("high_quality",    "High Quality",                         AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_HIGH_QUALITY, "quality"),

    { "rc",                    "Rate Control Method",                                  OFFSET(rate_control_mode),     AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_UNKNOWN }, AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_UNKNOWN, AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CBR, VE, "rc" },
        ENUM("cqp",             "Constant Quantization Parameter",      AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CONSTANT_QP,             "rc"),
        ENUM("cbr",             "Constant Bitrate",                     AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CBR,                     "rc"),
        ENUM("vbr_peak",        "Peak Contrained Variable Bitrate",     AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR,    "rc"),
        ENUM("vbr_latency",     "Latency Constrained Variable Bitrate", AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_LATENCY_CONSTRAINED_VBR, "rc"),

    { "header_insertion_mode", "Set header insertion mode",                            OFFSET(header_insertion_mode), AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_AV1_HEADER_INSERTION_MODE_NONE }, AMF_VIDEO_ENCODER_AV1_HEADER_INSERTION_MODE_NONE, AMF_VIDEO_ENCODER_AV1_HEADER_INSERTION_MODE_KEY_FRAME_ALIGNED, VE, "hdrmode" },
        ENUM("none",            "",                                     AMF_VIDEO_ENCODER_AV1_HEADER_INSERTION_MODE_NONE,              "hdrmode"),
        ENUM("gop",             "",                                     AMF_VIDEO_ENCODER_AV1_HEADER_INSERTION_MODE_GOP_ALIGNED,       "hdrmode"),
        ENUM("frame",           "",                                     AMF_VIDEO_ENCODER_AV1_HEADER_INSERTION_MODE_KEY_FRAME_ALIGNED, "hdrmode"),

    { "preanalysis",           "Enable Pre-Encode/Analysis for rate rontrol (2-Pass)", OFFSET(pre_encode),            AV_OPT_TYPE_BOOL, { .i64 = 0  },  0, 1, VE },
    { "hmqb",                  "Enable High Motion Quality Boost",                     OFFSET(enable_hmqb),           AV_OPT_TYPE_BOOL, { .i64 = 0  },  0, 1, VE },
    { "enforce_hrd",           "Enforce HRD",                                          OFFSET(enforce_hrd),           AV_OPT_TYPE_BOOL, { .i64 = 0  },  0, 1, VE },
    { "filler_data",           "Filler Data Enable",                                   OFFSET(filler_data),           AV_OPT_TYPE_BOOL, { .i64 = 0  },  0, 1, VE },

    // min_qp_i -> min_qp_intra, min_qp_p -> min_qp_inter
    { "min_qp_i",              "Min Quantization Parameter for I-frame",               OFFSET(min_qp_i),              AV_OPT_TYPE_INT,  { .i64 = -1 }, -1, 255, VE },
    { "max_qp_i",              "Max Quantization Parameter for I-frame",               OFFSET(max_qp_i),              AV_OPT_TYPE_INT,  { .i64 = -1 }, -1, 255, VE },
    { "min_qp_p",              "Min Quantization Parameter for P-frame",               OFFSET(min_qp_p),              AV_OPT_TYPE_INT,  { .i64 = -1 }, -1, 255, VE },
    { "max_qp_p",              "Max Quantization Parameter for P-frame",               OFFSET(max_qp_p),              AV_OPT_TYPE_INT,  { .i64 = -1 }, -1, 255, VE },
    { "qp_p",                  "Quantization Parameter for P-frame",                   OFFSET(qp_p),                  AV_OPT_TYPE_INT,  { .i64 = -1 }, -1, 255, VE },
    { "qp_i",                  "Quantization Parameter for I-frame",                   OFFSET(qp_i),                  AV_OPT_TYPE_INT,  { .i64 = -1 }, -1, 255, VE },
    { "skip_frame",            "Rate Control Based Frame Skip",                        OFFSET(skip_frame),            AV_OPT_TYPE_BOOL, { .i64 = 0  },  0, 1, VE },

    { "align",                 "Alignment mode",                                       OFFSET(align),                 AV_OPT_TYPE_INT,  { .i64 = AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE_NO_RESTRICTIONS }, AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE_64X16_ONLY, AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE_NO_RESTRICTIONS, VE, "align" },
        ENUM("64x16",           "",                                     AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE_64X16_ONLY,             "align"),
        ENUM("1080p",           "",                                     AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE_64X16_1080P_CODED_1082, "align"),
        ENUM("none",            "",                                     AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE_NO_RESTRICTIONS,        "align"),

    { "log_to_dbg",            "Enable AMF logging to debug output",                   OFFSET(log_to_dbg),            AV_OPT_TYPE_BOOL, { .i64 = 0  },  0, 1, VE },
    { NULL }
};

static av_cold int amf_encode_init_av1(AVCodecContext* avctx)
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

    if (avctx->framerate.num > 0 && avctx->framerate.den > 0)
        framerate = AMFConstructRate(avctx->framerate.num, avctx->framerate.den);
    else
        framerate = AMFConstructRate(avctx->time_base.den, avctx->time_base.num * avctx->ticks_per_frame);

    if ((ret = ff_amf_encode_init(avctx)) < 0)
        return ret;

    // Static parameters
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_USAGE, ctx->usage);

    AMF_ASSIGN_PROPERTY_SIZE(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_FRAMESIZE, framesize);

    AMF_ASSIGN_PROPERTY_RATE(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_FRAMERATE, framerate);

    switch (avctx->profile) {
    case FF_PROFILE_AV1_MAIN:
        profile = AMF_VIDEO_ENCODER_AV1_PROFILE_MAIN;
        break;
    default:
        break;
    }
    if (profile == 0)
        profile = ctx->profile;
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_PROFILE, profile);

    profile_level = avctx->level;
    if (profile_level == FF_LEVEL_UNKNOWN)
        profile_level = ctx->level;
    if (profile_level != 0)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_LEVEL, profile_level);

    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET, ctx->quality);

    // Maximum Reference Frames
    if (avctx->refs != -1)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_MAX_NUM_REFRAMES, avctx->refs);

    // Picture control properties
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_GOP_SIZE, avctx->gop_size);

    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_HEADER_INSERTION_MODE, ctx->header_insertion_mode);

    // Rate control properties
    // Auto detect rate control method
    if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_UNKNOWN) {
        if (ctx->min_qp_i != -1 || ctx->max_qp_i != -1 ||
            ctx->min_qp_p != -1 || ctx->max_qp_p != -1 ||
            ctx->qp_i != -1 || ctx->qp_p != -1) {
            ctx->rate_control_mode = AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CONSTANT_QP;
            av_log(ctx, AV_LOG_DEBUG, "Rate control turned to CQP\n");
        }
        else if (avctx->rc_max_rate > 0) {
            ctx->rate_control_mode = AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR;
            av_log(ctx, AV_LOG_DEBUG, "Rate control turned to Peak VBR\n");
        }
        else {
            ctx->rate_control_mode = AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CBR;
            av_log(ctx, AV_LOG_DEBUG, "Rate control turned to CBR\n");
        }
    }

    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD, ctx->rate_control_mode);
    if (avctx->rc_buffer_size) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_VBV_BUFFER_SIZE, avctx->rc_buffer_size);

        if (avctx->rc_initial_buffer_occupancy != 0) {
            int amf_buffer_fullness = avctx->rc_initial_buffer_occupancy * 64 / avctx->rc_buffer_size;
            if (amf_buffer_fullness > 64)
                amf_buffer_fullness = 64;
            AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_INITIAL_VBV_BUFFER_FULLNESS, amf_buffer_fullness);
        }
    }

    // Pre-Encode/Two-Pass(pre-encode assisted rate control)
    if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CONSTANT_QP) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_PREENCODE, 0);
        if (ctx->pre_encode) {
            ctx->pre_encode = 0;
            av_log(ctx, AV_LOG_WARNING, "Pre-Encode is not supported by CQP rate control method, automatically disabled\n");
        }
    } else {
        AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_PREENCODE, ctx->pre_encode);
    }

    // High Motion Quality Boost mode
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_HIGH_MOTION_QUALITY_BOOST, !!ctx->enable_hmqb);

    // Dynamic rate control params
    if (ctx->max_au_size)
        ctx->enforce_hrd = 1;
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_ENFORCE_HRD, ctx->enforce_hrd);
    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_FILLER_DATA, ctx->filler_data);

    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_TARGET_BITRATE, avctx->bit_rate);

    if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CBR)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_PEAK_BITRATE, avctx->bit_rate);
    if (avctx->rc_max_rate) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_PEAK_BITRATE, avctx->rc_max_rate);
    } else if (ctx->rate_control_mode == AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR)
        av_log(ctx, AV_LOG_WARNING, "rate control mode is PEAK_CONSTRAINED_VBR but rc_max_rate is not set\n");

    if (avctx->bit_rate > 0) {
        ctx->rate_control_mode = AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CBR;
        av_log(ctx, AV_LOG_DEBUG, "Rate control turned to CBR\n");
    }

    switch (ctx->align) {
    case AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE_64X16_ONLY:
        if (avctx->width / 64 * 64 != avctx->width || avctx->height / 16 * 16 != avctx->height) {
            res = AMF_NOT_SUPPORTED;
            av_log(ctx, AV_LOG_ERROR, "Resolution incorrect for alignment mode\n");
            return AVERROR_EXIT;
        }
        break;
    case AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE_64X16_1080P_CODED_1082:
        if ((avctx->width / 64 * 64 == avctx->width && avctx->height / 16 * 16 == avctx->height) || (avctx->width == 1920 && avctx->height == 1080)) {
            res = AMF_OK;
        } else {
            res = AMF_NOT_SUPPORTED;
            av_log(ctx, AV_LOG_ERROR, "Resolution incorrect for alignment mode\n");
            return AVERROR_EXIT;
        }
        break;
    case AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE_NO_RESTRICTIONS:
        res = AMF_OK;
        break;
    default:
        res = AMF_NOT_SUPPORTED;
        av_log(ctx, AV_LOG_ERROR, "Invalid alignment mode\n");
        return AVERROR_EXIT;
    }
    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE, ctx->align);

    // Output color profile, transfer and primaries
    if (ctx->out_color_profile > AMF_VIDEO_CONVERTER_COLOR_PROFILE_UNKNOWN)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_OUTPUT_COLOR_PROFILE, ctx->out_color_profile);
    if (ctx->out_color_trc > AMF_COLOR_TRANSFER_CHARACTERISTIC_UNDEFINED)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_OUTPUT_TRANSFER_CHARACTERISTIC, ctx->out_color_trc);
    if (ctx->out_color_prm > AMF_COLOR_PRIMARIES_UNDEFINED)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_OUTPUT_COLOR_PRIMARIES, ctx->out_color_prm);

    // Init encoder
    res = ctx->encoder->pVtbl->Init(ctx->encoder, ctx->format, avctx->width, avctx->height);
    AMF_RETURN_IF_FALSE(ctx, res == AMF_OK, AVERROR_BUG, "encoder->Init() failed with error %d\n", res);

    // Dynamic picture control params
    if (ctx->min_qp_i != -1) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_MIN_Q_INDEX_INTRA, ctx->min_qp_i);
    } else if (avctx->qmin != -1) {
        int qval = avctx->qmin > 255 ? 255 : avctx->qmin;
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_MIN_Q_INDEX_INTRA, qval);
    }

    if (ctx->max_qp_i != -1) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_MAX_Q_INDEX_INTRA, ctx->max_qp_i);
    } else if (avctx->qmax != -1) {
        int qval = avctx->qmax > 255 ? 255 : avctx->qmax;
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_MAX_Q_INDEX_INTRA, qval);
    }

    if (ctx->min_qp_p != -1) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_MIN_Q_INDEX_INTER, ctx->min_qp_p);
    } else if (avctx->qmin != -1) {
        int qval = avctx->qmin > 255 ? 255 : avctx->qmin;
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_MIN_Q_INDEX_INTER, qval);
    }

    if (ctx->max_qp_p != -1) {
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_MAX_Q_INDEX_INTER, ctx->max_qp_p);
    } else if (avctx->qmax != -1) {
        int qval = avctx->qmax > 255 ? 255 : avctx->qmax;
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_MAX_Q_INDEX_INTER, qval);
    }

    if (ctx->qp_p != -1)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_Q_INDEX_INTER, ctx->qp_p);
    if (ctx->qp_i != -1)
        AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_Q_INDEX_INTRA, ctx->qp_i);

    AMF_ASSIGN_PROPERTY_BOOL(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_SKIP_FRAME, ctx->skip_frame);

    // Fill extradata
    res = AMFVariantInit(&var);
    AMF_RETURN_IF_FALSE(ctx, res == AMF_OK, AVERROR_BUG, "AMFVariantInit() failed with error %d\n", res);

    res = ctx->encoder->pVtbl->GetProperty(ctx->encoder, AMF_VIDEO_ENCODER_AV1_EXTRA_DATA, &var);
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
    { "qmin",       "-1"  },
    { "qmax",       "-1"  },
    { NULL                },
};

static const AVClass av1_amf_class = {
    .class_name = "av1_amf",
    .item_name = av_default_item_name,
    .option = options,
    .version = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_av1_amf_encoder = {
    .p.name         = "av1_amf",
    CODEC_LONG_NAME("AMD AMF AV1 encoder"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_AV1,
    .init           = amf_encode_init_av1,
    FF_CODEC_RECEIVE_PACKET_CB(ff_amf_receive_packet),
    .close          = ff_amf_encode_close,
    .priv_data_size = sizeof(AMFEncContext),
    .p.priv_class   = &av1_amf_class,
    .defaults       = defaults,
    .p.capabilities = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_HARDWARE |
                      AV_CODEC_CAP_DR1,
    .caps_internal  = FF_CODEC_CAP_NOT_INIT_THREADSAFE |
                      FF_CODEC_CAP_INIT_CLEANUP,
    .p.pix_fmts     = ff_amfenc_av1_pix_fmts,
    .p.wrapper_name = "amf",
    .hw_configs     = ff_amfenc_hw_configs,
};
