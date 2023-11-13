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
 * Rockchip RGA (2D Raster Graphic Acceleration) video compositor
 */

#include "libavutil/common.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "filters.h"
#include "framesync.h"

#include "rkrga_common.h"

typedef struct RGAOverlayContext {
    RKRGAContext rga;

    FFFrameSync fs;

    int overlay_x;
    int overlay_y;
    int global_alpha;

    enum AVPixelFormat format;
} RGAOverlayContext;

static av_cold int set_size_info(AVFilterContext *ctx,
                                 AVFilterLink *inlink_main,
                                 AVFilterLink *inlink_overlay,
                                 AVFilterLink *outlink)
{
    if (inlink_main->w < 2 || inlink_main->w > 8192 ||
        inlink_main->h < 2 || inlink_main->h > 8192 ||
        inlink_overlay->w < 2 || inlink_overlay->w > 8192 ||
        inlink_overlay->h < 2 || inlink_overlay->h > 8192) {
        av_log(ctx, AV_LOG_ERROR, "Supported input size is range from 2x2 ~ 8192x8192\n");
        return AVERROR(EINVAL);
    }

    outlink->w = inlink_main->w;
    outlink->h = inlink_main->h;

    if (inlink_main->sample_aspect_ratio.num)
        outlink->sample_aspect_ratio = av_mul_q((AVRational){outlink->h * inlink_main->w,
                                                             outlink->w * inlink_main->h},
                                                inlink_main->sample_aspect_ratio);
    else
        outlink->sample_aspect_ratio = inlink_main->sample_aspect_ratio;

    return 0;
}

static av_cold int rgaoverlay_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    RGAOverlayContext *r = ctx->priv;
    AVFilterLink *inlink_main    = ctx->inputs[0];
    AVFilterLink *inlink_overlay = ctx->inputs[1];
    AVHWFramesContext *frames_ctx_main;
    AVHWFramesContext *frames_ctx_overlay;
    enum AVPixelFormat in_format_main;
    enum AVPixelFormat in_format_overlay;
    enum AVPixelFormat out_format;
    int ret;

    RKRGAParam param = { NULL };

    if (!inlink_main->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on main input\n");
        return AVERROR(EINVAL);
    }
    frames_ctx_main = (AVHWFramesContext *)inlink_main->hw_frames_ctx->data;    
    in_format_main  = frames_ctx_main->sw_format;
    out_format      = (r->format == AV_PIX_FMT_NONE) ? in_format_main : r->format;

    if (!inlink_overlay->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on overlay input\n");
        return AVERROR(EINVAL);
    }
    frames_ctx_overlay = (AVHWFramesContext *)inlink_overlay->hw_frames_ctx->data;
    in_format_overlay  = frames_ctx_overlay->sw_format;

    ret = set_size_info(ctx, inlink_main, inlink_overlay, outlink);
    if (ret < 0)
        return ret;

    param.filter_frame    = NULL;
    param.out_sw_format   = out_format;
    param.in_global_alpha = r->global_alpha;
    param.overlay_x       = r->overlay_x;
    param.overlay_y       = r->overlay_y;

    ret = ff_rkrga_init(ctx, &param);
    if (ret < 0)
        return ret;

    av_log(ctx, AV_LOG_VERBOSE, "w:%d h:%d fmt:%s + w:%d h:%d fmt:%s -> w:%d h:%d fmt:%s\n",
           inlink_main->w, inlink_main->h, av_get_pix_fmt_name(in_format_main),
           inlink_overlay->w, inlink_overlay->h, av_get_pix_fmt_name(in_format_overlay),
           outlink->w, outlink->h, av_get_pix_fmt_name(out_format));

    ret = ff_framesync_init_dualinput(&r->fs, ctx);
    if (ret < 0)
        return ret;

    r->fs.time_base = outlink->time_base = inlink_main->time_base;

    ret = ff_framesync_configure(&r->fs);
    if (ret < 0)
        return ret;

    return 0;
}

static int rgaoverlay_on_event(FFFrameSync *fs)
{
    AVFilterContext *ctx         = fs->parent;
    AVFilterLink *inlink_main    = ctx->inputs[0];
    AVFilterLink *inlink_overlay = ctx->inputs[1];
    AVFrame *in_main = NULL, *in_overlay = NULL;
    int ret;

    RGAOverlayContext *r = ctx->priv;

    ret = ff_framesync_get_frame(fs, 0, &in_main, 0);
    if (ret < 0)
        return ret;
    ret = ff_framesync_get_frame(fs, 1, &in_overlay, 0);
    if (ret < 0)
        return ret;

    if (!in_main)
        return AVERROR_BUG;

    return ff_rkrga_filter_frame(&r->rga,
                                 inlink_main, in_main,
                                 inlink_overlay, in_overlay);
}

static av_cold int rgaoverlay_init(AVFilterContext *ctx)
{
    RGAOverlayContext *r = ctx->priv;

    r->fs.on_event = &rgaoverlay_on_event;

    return 0;
}

static av_cold void rgaoverlay_uninit(AVFilterContext *ctx)
{
    RGAOverlayContext *r = ctx->priv;

    ff_framesync_uninit(&r->fs);

    ff_rkrga_close(ctx);
}

static int rgaoverlay_activate(AVFilterContext *ctx)
{
    RGAOverlayContext *r = ctx->priv;
    AVFilterLink *inlink_main    = ctx->inputs[0];
    AVFilterLink *inlink_overlay = ctx->inputs[1];
    AVFilterLink *outlink        = ctx->outputs[0];
    int ret;

    int status = 0;
    int64_t pts = AV_NOPTS_VALUE;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink_main);

    if (r->rga.eof)
        goto eof;
    else if (ff_inlink_acknowledge_status(inlink_main, &status, &pts)) {
        if (status == AVERROR_EOF) {
            r->rga.eof = 1;
            goto eof;
        }
    }

    ret = ff_framesync_activate(&r->fs);
    if (ret < 0)
        return ret;

    if (r->fs.eof) {
        r->rga.eof = 1;
        goto eof;
    }

    if (!r->rga.got_frame) {
        FF_FILTER_FORWARD_WANTED(outlink, inlink_overlay);
        FF_FILTER_FORWARD_WANTED(outlink, inlink_main);
        return FFERROR_NOT_READY;
    } else
        r->rga.got_frame = 0;

    return 0;

eof:
    ff_rkrga_filter_frame(&r->rga,
                          inlink_main, NULL,
                          inlink_overlay, NULL);
    ff_outlink_set_status(outlink, AVERROR_EOF, AV_NOPTS_VALUE);
    return 0;
}

#define OFFSET(x) offsetof(RGAOverlayContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)

static const AVOption rgaoverlay_options[] = {
    { "x", "Set horizontal offset", OFFSET(overlay_x), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, .flags = FLAGS },
    { "y", "Set vertical offset",   OFFSET(overlay_y), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, .flags = FLAGS },
    { "alpha", "Overlay global alpha", OFFSET(global_alpha), AV_OPT_TYPE_INT, { .i64 = 255 }, 0, 255, .flags = FLAGS },
    { "format", "Output video pixel format", OFFSET(format), AV_OPT_TYPE_PIXEL_FMT, { .i64 = AV_PIX_FMT_NONE }, INT_MIN, INT_MAX, .flags = FLAGS },
    { "eof_action", "Action to take when encountering EOF from secondary input ",
        OFFSET(fs.opt_eof_action), AV_OPT_TYPE_INT, { .i64 = EOF_ACTION_REPEAT },
        EOF_ACTION_REPEAT, EOF_ACTION_PASS, .flags = FLAGS, "eof_action" },
        { "repeat", "Repeat the previous frame.",   0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_REPEAT }, .flags = FLAGS, "eof_action" },
        { "endall", "End both streams.",            0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_ENDALL }, .flags = FLAGS, "eof_action" },
        { "pass",   "Pass through the main input.", 0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_PASS },   .flags = FLAGS, "eof_action" },
    { "shortest", "Force termination when the shortest input terminates", OFFSET(fs.opt_shortest), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { "repeatlast", "Repeat overlay of the last overlay frame", OFFSET(fs.opt_repeatlast), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { "core", "Set multiRGA scheduler core [use with caution]", OFFSET(rga.scheduler_core), AV_OPT_TYPE_FLAGS, { .i64 = 0 }, 0, INT_MAX, FLAGS, "core" },
        { "default",    NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 0 }, 0, 0, FLAGS, "core" },
        { "rga3_core0", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 1 }, 0, 0, FLAGS, "core" }, /* RGA3_SCHEDULER_CORE0 */
        { "rga3_core1", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 2 }, 0, 0, FLAGS, "core" }, /* RGA3_SCHEDULER_CORE1 */
        { "rga2_core0", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 4 }, 0, 0, FLAGS, "core" }, /* RGA2_SCHEDULER_CORE0 */
    { "async_depth", "Set the internal parallelization depth", OFFSET(rga.async_depth), AV_OPT_TYPE_INT, { .i64 = 3 }, 0, INT_MAX, .flags = FLAGS },
    { NULL },
};

FRAMESYNC_DEFINE_CLASS(rgaoverlay, RGAOverlayContext, fs);

static const AVFilterPad rgaoverlay_inputs[] = {
    {
        .name             = "main",
        .type             = AVMEDIA_TYPE_VIDEO,
    },
    {
        .name             = "overlay",
        .type             = AVMEDIA_TYPE_VIDEO,
    },
};

static const AVFilterPad rgaoverlay_outputs[] = {
    {
        .name             = "default",
        .type             = AVMEDIA_TYPE_VIDEO,
        .config_props     = rgaoverlay_config_props,
    },
};

const AVFilter ff_vf_overlay_rkrga = {
    .name           = "overlay_rkrga",
    .description    = NULL_IF_CONFIG_SMALL("Rockchip RGA (2D Raster Graphic Acceleration) video compositor"),
    .priv_size      = sizeof(RGAOverlayContext),
    .priv_class     = &rgaoverlay_class,
    .init           = rgaoverlay_init,
    .uninit         = rgaoverlay_uninit,
    .activate       = rgaoverlay_activate,
    FILTER_INPUTS(rgaoverlay_inputs),
    FILTER_OUTPUTS(rgaoverlay_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_DRM_PRIME),
    .preinit        = rgaoverlay_framesync_preinit,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
