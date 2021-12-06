/*
 * Copyright (c) 2021 NyanMisaka
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
 * Provide a blank video input with alpha channel.
 */

#include "libavutil/avstring.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/parseutils.h"
#include "filters.h"
#include "avfilter.h"
#include "internal.h"
#include "formats.h"
#include "video.h"

typedef struct AlphaSrc {
    const AVClass *class;
    AVRational time_base, frame_rate;
    int64_t pts;
    int64_t duration, start;
    int out_w, out_h;
    int rgb, planar;
} AlphaSrc;

static av_cold int alphasrc_init(AVFilterContext *ctx)
{
    AlphaSrc *s = ctx->priv;

    s->time_base = av_inv_q(s->frame_rate);
    s->pts = 0;

    if (s->start > 0)
        s->pts += av_rescale_q(s->start, AV_TIME_BASE_Q, s->time_base);

    return 0;
}

static int alphasrc_query_formats(AVFilterContext *ctx)
{
    AVFilterLink *outlink = ctx->outputs[0];
    AVFilterFormats *formats;
    int ret;

    if ((ret = ff_formats_pixdesc_filter(&formats, AV_PIX_FMT_FLAG_ALPHA, 0)) ||
        (ret = ff_formats_ref(formats, &outlink->incfg.formats)))
        return ret;

    return 0;
}

static int alphasrc_config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AlphaSrc *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(outlink->format);

    s->rgb = desc->flags & AV_PIX_FMT_FLAG_RGB;
    s->planar = desc->flags & AV_PIX_FMT_FLAG_PLANAR;

    if (!s->rgb && !s->planar) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported output format.\n");
        return AVERROR(EINVAL);
    }

    if (s->out_w <= 0 || s->out_h <= 0) {
        av_log(ctx, AV_LOG_ERROR, "Invalid output video size.\n");
        return AVERROR(EINVAL);
    }

    outlink->w = s->out_w;
    outlink->h = s->out_h;
    outlink->frame_rate = s->frame_rate;
    outlink->time_base  = s->time_base;
    outlink->sample_aspect_ratio = (AVRational){1, 1};

    return 0;
}

static int alphasrc_request_frame(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AlphaSrc *s = ctx->priv;
    AVFrame *out;
    int i;

    if (s->duration > 0 &&
        av_rescale_q(s->pts, s->time_base, AV_TIME_BASE_Q) >= s->duration) {
        ff_outlink_set_status(outlink, AVERROR_EOF, s->pts);
        return 0;
    }

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out)
        return AVERROR(ENOMEM);

    for (i = 0; i < AV_NUM_DATA_POINTERS; i++) {
        if (out->buf[i]) {
            if (s->rgb)
                memset(out->buf[i]->data, 0, out->buf[i]->size);
            else if (s->planar)
                memset(out->buf[i]->data, (i == 1 || i == 2) ? 128 : 0, out->buf[i]->size);
        }
    }

    out->pts = s->pts++;

    return ff_filter_frame(outlink, out);
}

#define OFFSET(x) offsetof(AlphaSrc, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM
static const AVOption alphasrc_options[] = {
    {"duration", "set the duration of the video",        OFFSET(duration),   AV_OPT_TYPE_DURATION,   {.i64 = 0   }, 0, INT64_MAX, FLAGS},
    {"d",        "set the duration of the video",        OFFSET(duration),   AV_OPT_TYPE_DURATION,   {.i64 = 0   }, 0, INT64_MAX, FLAGS},
    {"start",    "set the start timestamp of the video", OFFSET(start),      AV_OPT_TYPE_DURATION,   {.i64 = 0   }, 0, INT64_MAX, FLAGS},
    {"rate",     "set the frame rate of the video",      OFFSET(frame_rate), AV_OPT_TYPE_VIDEO_RATE, {.str = "15"}, 1, INT_MAX, FLAGS},
    {"r",        "set the frame rate of the video",      OFFSET(frame_rate), AV_OPT_TYPE_VIDEO_RATE, {.str = "15"}, 1, INT_MAX, FLAGS},
    {"size",     "set the size of the video",            OFFSET(out_w),      AV_OPT_TYPE_IMAGE_SIZE, {.str = NULL}, 0, 0, FLAGS},
    {"s",        "set the size of the video",            OFFSET(out_w),      AV_OPT_TYPE_IMAGE_SIZE, {.str = NULL}, 0, 0, FLAGS},
    {NULL}
};

AVFILTER_DEFINE_CLASS(alphasrc);

static const AVFilterPad alphasrc_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = alphasrc_config_output,
        .request_frame = alphasrc_request_frame,
    },
    { NULL }
};

AVFilter ff_vsrc_alphasrc = {
    .name          = "alphasrc",
    .description   = NULL_IF_CONFIG_SMALL("Provide a blank video input with alpha channel."),
    .priv_size     = sizeof(AlphaSrc),
    .priv_class    = &alphasrc_class,
    .query_formats = alphasrc_query_formats,
    .init          = alphasrc_init,
    .uninit        = NULL,
    .inputs        = NULL,
    .outputs       = alphasrc_outputs,
};
