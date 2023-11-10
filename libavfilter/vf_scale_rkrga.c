/*
 * Copyright (c) 2023 NyanMisaka
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

/**
 * @file
 * Rockchip RGA (2D Raster Graphic Acceleration) video converter (scale/crop/transpose)
 */

#include "libavutil/common.h"
#include "libavutil/eval.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "filters.h"
#include "scale_eval.h"
#include "transpose.h"

#include "rkrga_common.h"

typedef struct RGAScaleContext {
    RKRGAContext rga;

    enum AVPixelFormat format;
    int transpose;
    int passthrough;
    int force_original_aspect_ratio;
    int force_divisible_by;
    int scheduler_core;

    int in_rotate_mode;

    char *ow, *oh;
    char *cx, *cy, *cw, *ch;
    int crop;

    int act_x, act_y;
    int act_w, act_h;
} RGAScaleContext;

static const char *const var_names[] = {
    "iw", "in_w",
    "ih", "in_h",
    "ow", "out_w", "w",
    "oh", "out_h", "h",
    "cw",
    "ch",
    "cx",
    "cy",
    "a", "dar",
    "sar",
    NULL
};

enum var_name {
    VAR_IW, VAR_IN_W,
    VAR_IH, VAR_IN_H,
    VAR_OW, VAR_OUT_W, VAR_W,
    VAR_OH, VAR_OUT_H, VAR_H,
    VAR_CW,
    VAR_CH,
    VAR_CX,
    VAR_CY,
    VAR_A, VAR_DAR,
    VAR_SAR,
    VAR_VARS_NB
};

static av_cold int eval_expr(AVFilterContext *ctx,
                             int *ret_w, int *ret_h,
                             int *ret_cx, int *ret_cy,
                             int *ret_cw, int *ret_ch)
{
#define PASS_EXPR(e, s) {\
    if (s) {\
        ret = av_expr_parse(&e, s, var_names, NULL, NULL, NULL, NULL, 0, ctx); \
        if (ret < 0) {                                                  \
            av_log(ctx, AV_LOG_ERROR, "Error when passing '%s'.\n", s); \
            goto release;                                               \
        }                                                               \
    }\
}
#define CALC_EXPR(e, v, i, d) {\
    if (e)\
        i = v = av_expr_eval(e, var_values, NULL);      \
    else\
        i = v = d;\
}
    RGAScaleContext *r = ctx->priv;
    double  var_values[VAR_VARS_NB] = { NAN };
    AVExpr *w_expr  = NULL, *h_expr  = NULL;
    AVExpr *cw_expr = NULL, *ch_expr = NULL;
    AVExpr *cx_expr = NULL, *cy_expr = NULL;
    int     ret = 0;

    PASS_EXPR(cw_expr, r->cw);
    PASS_EXPR(ch_expr, r->ch);

    PASS_EXPR(w_expr, r->ow);
    PASS_EXPR(h_expr, r->oh);

    PASS_EXPR(cx_expr, r->cx);
    PASS_EXPR(cy_expr, r->cy);

    var_values[VAR_IW] =
    var_values[VAR_IN_W] = ctx->inputs[0]->w;

    var_values[VAR_IH] =
    var_values[VAR_IN_H] = ctx->inputs[0]->h;

    var_values[VAR_A] = (double)var_values[VAR_IN_W] / var_values[VAR_IN_H];
    var_values[VAR_SAR] = ctx->inputs[0]->sample_aspect_ratio.num ?
        (double)ctx->inputs[0]->sample_aspect_ratio.num / ctx->inputs[0]->sample_aspect_ratio.den : 1;
    var_values[VAR_DAR] = var_values[VAR_A] * var_values[VAR_SAR];

    /* crop params */
    CALC_EXPR(cw_expr, var_values[VAR_CW], *ret_cw, var_values[VAR_IW]);
    CALC_EXPR(ch_expr, var_values[VAR_CH], *ret_ch, var_values[VAR_IH]);

    /* calc again in case cw is relative to ch */
    CALC_EXPR(cw_expr, var_values[VAR_CW], *ret_cw, var_values[VAR_IW]);

    CALC_EXPR(w_expr,
              var_values[VAR_OUT_W] = var_values[VAR_OW] = var_values[VAR_W],
              *ret_w, var_values[VAR_CW]);
    CALC_EXPR(h_expr,
              var_values[VAR_OUT_H] = var_values[VAR_OH] = var_values[VAR_H],
              *ret_h, var_values[VAR_CH]);

    /* calc again in case ow is relative to oh */
    CALC_EXPR(w_expr,
              var_values[VAR_OUT_W] = var_values[VAR_OW] = var_values[VAR_W],
              *ret_w, var_values[VAR_CW]);

    CALC_EXPR(cx_expr, var_values[VAR_CX], *ret_cx, (var_values[VAR_IW] - var_values[VAR_OW]) / 2);
    CALC_EXPR(cy_expr, var_values[VAR_CY], *ret_cy, (var_values[VAR_IH] - var_values[VAR_OH]) / 2);

    /* calc again in case cx is relative to cy */
    CALC_EXPR(cx_expr, var_values[VAR_CX], *ret_cx, (var_values[VAR_IW] - var_values[VAR_OW]) / 2);

    r->crop = (*ret_cw != var_values[VAR_IW]) || (*ret_ch != var_values[VAR_IH]);

release:
    av_expr_free(w_expr);
    av_expr_free(h_expr);
    av_expr_free(cw_expr);
    av_expr_free(ch_expr);
    av_expr_free(cx_expr);
    av_expr_free(cy_expr);
#undef PASS_EXPR
#undef CALC_EXPR

    return ret;
}

static av_cold int set_size_info(AVFilterContext *ctx,
                                 AVFilterLink *inlink,
                                 AVFilterLink *outlink)
{
    RGAScaleContext *r = ctx->priv;
    int w, h, ret;

    if (inlink->w < 2 || inlink->w > 8192 ||
        inlink->h < 2 || inlink->h > 8192) {
        av_log(ctx, AV_LOG_ERROR, "Supported input size is range from 2x2 ~ 8192x8192\n");
        return AVERROR(EINVAL);
    }

    if ((ret = eval_expr(ctx, &w, &h, &r->act_x, &r->act_y, &r->act_w, &r->act_h)) < 0)
        return ret;

    r->act_x = FFMAX(FFMIN(r->act_x, inlink->w), 0);
    r->act_y = FFMAX(FFMIN(r->act_y, inlink->h), 0);
    r->act_w = FFMAX(FFMIN(r->act_w, inlink->w), 0);
    r->act_h = FFMAX(FFMIN(r->act_h, inlink->h), 0);

    r->act_x = FFMIN(r->act_x, inlink->w - r->act_w);
    r->act_y = FFMIN(r->act_y, inlink->h - r->act_h);
    r->act_w = FFMIN(r->act_w, inlink->w - r->act_x);
    r->act_h = FFMIN(r->act_h, inlink->h - r->act_y);

    ff_scale_adjust_dimensions(inlink, &w, &h,
                               r->force_original_aspect_ratio, r->force_divisible_by);

    if (((int64_t)h * inlink->w) > INT_MAX ||
        ((int64_t)w * inlink->h) > INT_MAX) {
        av_log(ctx, AV_LOG_ERROR, "Rescaled value for width or height is too big.\n");
        return AVERROR(EINVAL);
    }

    outlink->w = w;
    outlink->h = h;
    if (outlink->w < 2 || outlink->w > 8192 ||
        outlink->h < 2 || outlink->h > 8192) {
        av_log(ctx, AV_LOG_ERROR, "Supported output size is range from 2x2 ~ 8192x8192\n");
        return AVERROR(EINVAL);
    }

    if (inlink->sample_aspect_ratio.num)
        outlink->sample_aspect_ratio = av_mul_q((AVRational){outlink->h * inlink->w,
                                                             outlink->w * inlink->h},
                                                inlink->sample_aspect_ratio);
    else
        outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;

    if (r->transpose >= 0) {
        switch (r->transpose) {
        case TRANSPOSE_CCLOCK_FLIP:
            r->in_rotate_mode = 0x07 | (0x01 << 4); /* HAL_TRANSFORM_ROT_270 | (HAL_TRANSFORM_FLIP_H << 4) */
            FFSWAP(int, outlink->w, outlink->h);
            FFSWAP(int, outlink->sample_aspect_ratio.num, outlink->sample_aspect_ratio.den);
            break;
        case TRANSPOSE_CLOCK:
            r->in_rotate_mode = 0x04; /* HAL_TRANSFORM_ROT_90 */
            FFSWAP(int, outlink->w, outlink->h);
            FFSWAP(int, outlink->sample_aspect_ratio.num, outlink->sample_aspect_ratio.den);
            break;
        case TRANSPOSE_CCLOCK:
            r->in_rotate_mode = 0x07; /* HAL_TRANSFORM_ROT_270 */
            FFSWAP(int, outlink->w, outlink->h);
            FFSWAP(int, outlink->sample_aspect_ratio.num, outlink->sample_aspect_ratio.den);
            break;
        case TRANSPOSE_CLOCK_FLIP:
            r->in_rotate_mode = 0x04 | (0x01 << 4); /* HAL_TRANSFORM_ROT_90 | (HAL_TRANSFORM_FLIP_H << 4) */
            FFSWAP(int, outlink->w, outlink->h);
            FFSWAP(int, outlink->sample_aspect_ratio.num, outlink->sample_aspect_ratio.den);
            break;
        case TRANSPOSE_REVERSAL:
            r->in_rotate_mode = 0x03; /* HAL_TRANSFORM_ROT_180 */
            break;
        case TRANSPOSE_HFLIP:
            r->in_rotate_mode = 0x01; /* HAL_TRANSFORM_FLIP_H */
            break;
        case TRANSPOSE_VFLIP:
            r->in_rotate_mode = 0x02; /* HAL_TRANSFORM_FLIP_V */
            break;
        default:
            av_log(ctx, AV_LOG_ERROR, "Failed to set transpose mode to %d\n", r->transpose);
            return AVERROR(EINVAL);
        }
    }

    return 0;
}

static av_cold int rgascale_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    RGAScaleContext   *r = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    AVHWFramesContext *in_frames_ctx;
    enum AVPixelFormat in_format;
    enum AVPixelFormat out_format;
    RKRGAParam param = { NULL };
    int ret;

    if (!inlink->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames_ctx = (AVHWFramesContext *)inlink->hw_frames_ctx->data;
    in_format     = in_frames_ctx->sw_format;
    out_format    = (r->format == AV_PIX_FMT_NONE) ? in_format : r->format;

    ret = set_size_info(ctx, inlink, outlink);
    if (ret < 0)
        return ret;

    param.filter_frame   = NULL;
    param.out_sw_format  = out_format;
    param.in_rotate_mode = r->in_rotate_mode;
    param.in_crop        = r->crop;
    param.in_crop_x      = r->act_x;
    param.in_crop_y      = r->act_y;
    param.in_crop_w      = r->act_w;
    param.in_crop_h      = r->act_h;

    if (r->passthrough && r->transpose < 0 && !r->crop &&
        inlink->w == outlink->w && inlink->h == outlink->h && in_format == out_format) {
        outlink->hw_frames_ctx = av_buffer_ref(inlink->hw_frames_ctx);
        if (!outlink->hw_frames_ctx)
            return AVERROR(ENOMEM);
    } else {
        r->passthrough = 0;

        ret = ff_rkrga_init(ctx, &param);
        if (ret < 0)
            return ret;
    }

    av_log(ctx, AV_LOG_VERBOSE, "w:%d h:%d fmt:%s -> w:%d h:%d fmt:%s%s\n",
           inlink->w, inlink->h, av_get_pix_fmt_name(in_format),
           outlink->w, outlink->h, av_get_pix_fmt_name(out_format),
           r->passthrough ? " (passthrough)" : "");

    return 0;
}

static int rgascale_activate(AVFilterContext *ctx)
{
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    RGAScaleContext *r = ctx->priv;
    AVFrame *in = NULL;
    int ret, status = 0;
    int64_t pts = AV_NOPTS_VALUE;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    if (!r->rga.eof) {
        ret = ff_inlink_consume_frame(inlink, &in);
        if (ret < 0)
            return ret;

        if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
            if (status == AVERROR_EOF) {
                r->rga.eof = 1;
            }
        }
    }

    if (!r->passthrough) {
        if (in || r->rga.eof) {
            ret = ff_rkrga_filter_frame(&r->rga, inlink, in, NULL, NULL);
            av_frame_free(&in);
            if (ret < 0)
                return ret;
	    else if (!r->rga.got_frame)
                goto not_ready;

            if (r->rga.eof)
                goto eof;

            if (r->rga.got_frame) {
                r->rga.got_frame = 0;
                return 0;
            }
        }
    } else {
        /* pass-through mode */
        if (in) {
            if (in->pts != AV_NOPTS_VALUE)
                in->pts = av_rescale_q(in->pts, inlink->time_base, outlink->time_base);

            if (outlink->frame_rate.num && outlink->frame_rate.den)
                in->duration = av_rescale_q(1, av_inv_q(outlink->frame_rate), outlink->time_base);
            else
                in->duration = 0;

            ret = ff_filter_frame(outlink, in);
            if (ret < 0)
                return ret;

            if (r->rga.eof)
                goto eof;

            return 0;
        }
    }

not_ready:
    if (r->rga.eof)
        goto eof;

    FF_FILTER_FORWARD_WANTED(outlink, inlink);
    return FFERROR_NOT_READY;

eof:
    pts = av_rescale_q(pts, inlink->time_base, outlink->time_base);
    ff_outlink_set_status(outlink, status, pts);
    return 0;
}

static av_cold int rgascale_init(AVFilterContext *ctx)
{
    return 0;
}

static av_cold void rgascale_uninit(AVFilterContext *ctx)
{
    ff_rkrga_close(ctx);
}

static AVFrame *rgascale_get_video_buffer(AVFilterLink *inlink, int w, int h)
{
    RGAScaleContext *r = inlink->dst->priv;

    return r->passthrough ?
        ff_null_get_video_buffer(inlink, w, h) :
        ff_default_get_video_buffer(inlink, w, h);
}

#define OFFSET(x) offsetof(RGAScaleContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)

static const AVOption rgascale_options[] = {
    { "w",  "Output video width",                  OFFSET(ow), AV_OPT_TYPE_STRING, { .str = "cw" }, 0, 0, FLAGS },
    { "h",  "Output video height",                 OFFSET(oh), AV_OPT_TYPE_STRING, { .str = "w*ch/cw" }, 0, 0, FLAGS },
    { "cw", "Set the width crop area expression",  OFFSET(cw), AV_OPT_TYPE_STRING, { .str = "iw" }, 0, 0, FLAGS },
    { "ch", "Set the height crop area expression", OFFSET(ch), AV_OPT_TYPE_STRING, { .str = "ih" }, 0, 0, FLAGS },
    { "cx", "Set the x crop area expression",      OFFSET(cx), AV_OPT_TYPE_STRING, { .str = "(in_w-out_w)/2" }, 0, 0, FLAGS },
    { "cy", "Set the y crop area expression",      OFFSET(cy), AV_OPT_TYPE_STRING, { .str = "(in_h-out_h)/2" }, 0, 0, FLAGS },
    { "format", "Output video pixel format", OFFSET(format), AV_OPT_TYPE_PIXEL_FMT, { .i64 = AV_PIX_FMT_NONE }, INT_MIN, INT_MAX, .flags = FLAGS },
    { "transpose", "Set transpose direction", OFFSET(transpose), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 6, FLAGS, "transpose" },
        { "cclock_hflip", "Rotate counter-clockwise with horizontal flip", 0, AV_OPT_TYPE_CONST, { .i64 = TRANSPOSE_CCLOCK_FLIP }, 0, 0, FLAGS, "transpose" },
        { "clock",        "Rotate clockwise",                              0, AV_OPT_TYPE_CONST, { .i64 = TRANSPOSE_CLOCK       }, 0, 0, FLAGS, "transpose" },
        { "cclock",       "Rotate counter-clockwise",                      0, AV_OPT_TYPE_CONST, { .i64 = TRANSPOSE_CCLOCK      }, 0, 0, FLAGS, "transpose" },
        { "clock_hflip",  "Rotate clockwise with horizontal flip",         0, AV_OPT_TYPE_CONST, { .i64 = TRANSPOSE_CLOCK_FLIP  }, 0, 0, FLAGS, "transpose" },
        { "reversal",     "Rotate by half-turn",                           0, AV_OPT_TYPE_CONST, { .i64 = TRANSPOSE_REVERSAL    }, 0, 0, FLAGS, "transpose" },
        { "hflip",        "Flip horizontally",                             0, AV_OPT_TYPE_CONST, { .i64 = TRANSPOSE_HFLIP       }, 0, 0, FLAGS, "transpose" },
        { "vflip",        "Flip vertically",                               0, AV_OPT_TYPE_CONST, { .i64 = TRANSPOSE_VFLIP       }, 0, 0, FLAGS, "transpose" },
    { "passthrough", "Do not process frames at all if parameters match", OFFSET(passthrough), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { "force_original_aspect_ratio", "Decrease or increase w/h if necessary to keep the original AR", OFFSET(force_original_aspect_ratio), AV_OPT_TYPE_INT, { .i64 = 1 }, 0, 2, FLAGS, "force_oar" },
        { "disable",  NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 0 }, 0, 0, FLAGS, "force_oar" },
        { "decrease", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 1 }, 0, 0, FLAGS, "force_oar" },
        { "increase", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 2 }, 0, 0, FLAGS, "force_oar" },
    { "force_divisible_by", "Enforce that the output resolution is divisible by a defined integer when force_original_aspect_ratio is used", OFFSET(force_divisible_by), AV_OPT_TYPE_INT, { .i64 = 2 }, 1, 256, FLAGS },
    { "core", "Set multiRGA scheduler core [use with caution]", OFFSET(rga.scheduler_core), AV_OPT_TYPE_FLAGS, { .i64 = 0 }, 0, INT_MAX, FLAGS, "core" },
        { "default",    NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 0 }, 0, 0, FLAGS, "core" },
        { "rga3_core0", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 1 }, 0, 0, FLAGS, "core" }, /* RGA3_SCHEDULER_CORE0 */
        { "rga3_core1", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 2 }, 0, 0, FLAGS, "core" }, /* RGA3_SCHEDULER_CORE1 */
        { "rga2_core0", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 4 }, 0, 0, FLAGS, "core" }, /* RGA2_SCHEDULER_CORE0 */
    { "async_depth", "Set the internal parallelization depth", OFFSET(rga.async_depth), AV_OPT_TYPE_INT, { .i64 = 3 }, 0, INT_MAX, .flags = FLAGS },
    { NULL },
};

AVFILTER_DEFINE_CLASS(rgascale);

static const AVFilterPad rgascale_inputs[] = {
    {
        .name             = "default",
        .type             = AVMEDIA_TYPE_VIDEO,
        .get_buffer.video = rgascale_get_video_buffer,
    },
};

static const AVFilterPad rgascale_outputs[] = {
    {
        .name             = "default",
        .type             = AVMEDIA_TYPE_VIDEO,
        .config_props     = rgascale_config_props,
    },
};

const AVFilter ff_vf_scale_rkrga = {
    .name           = "scale_rkrga",
    .description    = NULL_IF_CONFIG_SMALL("Rockchip RGA (2D Raster Graphic Acceleration) video converter (scale/crop/transpose)"),
    .priv_size      = sizeof(RGAScaleContext),
    .priv_class     = &rgascale_class,
    .init           = rgascale_init,
    .uninit         = rgascale_uninit,
    FILTER_INPUTS(rgascale_inputs),
    FILTER_OUTPUTS(rgascale_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_DRM_PRIME),
    .activate       = rgascale_activate,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
