/*
 * Copyright (c) 2011 Baptiste Coudurier
 * Copyright (c) 2011 Stefano Sabatini
 * Copyright (c) 2012 Clément Bœsch
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
 * Libass subtitles burning filter.
 *
 * @see{http://www.matroska.org/technical/specs/subtitles/ssa.html}
 */

#include <ass/ass.h>

#include "config.h"
#include "config_components.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/avstring.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/parseutils.h"
#include "vf_inlinesubs.h"
#include "drawutils.h"
#include "avfilter.h"
#include "internal.h"
#include "formats.h"
#include "video.h"

typedef struct AssContext {
    const AVClass *class;
    ASS_Library  *library;
    ASS_Renderer *renderer;
    ASS_Track    *track;
    char *fontsdir;
    char *charenc;
    char *force_style;
    int alpha;
    uint8_t rgba_map[4];
    int original_w, original_h;
    int shaping;
    int got_header;
    FFDrawContext draw;
} AssContext;

#define OFFSET(x) offsetof(AssContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

#define ASS_TIME_BASE av_make_q(1, 1000)

/* libass supports a log level ranging from 0 to 7 */
static const int ass_libavfilter_log_level_map[] = {
    [0] = AV_LOG_FATAL,     /* MSGL_FATAL */
    [1] = AV_LOG_ERROR,     /* MSGL_ERR */
    [2] = AV_LOG_WARNING,   /* MSGL_WARN */
    [3] = AV_LOG_WARNING,   /* <undefined> */
    [4] = AV_LOG_INFO,      /* MSGL_INFO */
    [5] = AV_LOG_INFO,      /* <undefined> */
    [6] = AV_LOG_VERBOSE,   /* MSGL_V */
    [7] = AV_LOG_DEBUG,     /* MSGL_DBG2 */
};

static void ass_log(int ass_level, const char *fmt, va_list args, void *ctx)
{
    const int ass_level_clip = av_clip(ass_level, 0,
        FF_ARRAY_ELEMS(ass_libavfilter_log_level_map) - 1);
    const int level = ass_libavfilter_log_level_map[ass_level_clip];

    av_vlog(ctx, level, fmt, args);
    av_log(ctx, level, "\n");
}

static av_cold int inlinesubs_init(AVFilterContext *ctx)
{
    AssContext *ass = ctx->priv;

    ass->library = ass_library_init();
    if (!ass->library) {
        av_log(ctx, AV_LOG_ERROR, "Could not initialize libass.\n");
        return AVERROR(EINVAL);
    }
    ass_set_message_cb(ass->library, ass_log, ctx);

    ass_set_fonts_dir(ass->library, ass->fontsdir);
    ass_set_extract_fonts(ass->library, 1);

    ass->renderer = ass_renderer_init(ass->library);
    if (!ass->renderer) {
        av_log(ctx, AV_LOG_ERROR, "Could not initialize libass renderer.\n");
        return AVERROR(EINVAL);
    }

    ass->track = ass_new_track(ass->library);
    if (!ass->track) {
        av_log(ctx, AV_LOG_ERROR, "Could not create a libass track\n");
        return AVERROR(EINVAL);
    }

    return 0;
}

static av_cold void inlinesubs_uninit(AVFilterContext *ctx)
{
    AssContext *ass = ctx->priv;

    if (ass->track)
        ass_free_track(ass->track);
    if (ass->renderer)
        ass_renderer_done(ass->renderer);
    if (ass->library)
        ass_library_done(ass->library);
}

static int query_formats(AVFilterContext *ctx)
{
    return ff_set_common_formats(ctx, ff_draw_supported_pixel_formats(0));
}

static int config_input(AVFilterLink *inlink)
{
    AssContext *ass = inlink->dst->priv;

    ff_draw_init(&ass->draw, inlink->format, ass->alpha ? FF_DRAW_PROCESS_ALPHA : 0);

    ass_set_frame_size  (ass->renderer, inlink->w, inlink->h);
    if (ass->original_w && ass->original_h) {
        ass_set_pixel_aspect(ass->renderer, (double)inlink->w / inlink->h /
                             ((double)ass->original_w / ass->original_h));
        ass_set_storage_size(ass->renderer, ass->original_w, ass->original_h);
    } else {
        ass_set_pixel_aspect(ass->renderer, av_q2d(inlink->sample_aspect_ratio));
        ass_set_storage_size(ass->renderer, inlink->w, inlink->h);
    }

    if (ass->shaping != -1)
        ass_set_shaper(ass->renderer, ass->shaping);

    return 0;
}

/* libass stores an RGBA color in the format RRGGBBTT, where TT is the transparency level */
#define AR(c)  ( (c)>>24)
#define AG(c)  (((c)>>16)&0xFF)
#define AB(c)  (((c)>>8) &0xFF)
#define AA(c)  ((0xFF-(c)) &0xFF)

static void overlay_ass_image(AssContext *ass, AVFrame *picref,
                              const ASS_Image *image)
{
    for (; image; image = image->next) {
        uint8_t rgba_color[] = {AR(image->color), AG(image->color), AB(image->color), AA(image->color)};
        FFDrawColor color;
        ff_draw_color(&ass->draw, &color, rgba_color);
        ff_blend_mask(&ass->draw, &color,
                      picref->data, picref->linesize,
                      picref->width, picref->height,
                      image->bitmap, image->stride, image->w, image->h,
                      3, 0, image->dst_x, image->dst_y);
    }
}

static int filter_frame(AVFilterLink *inlink, AVFrame *picref)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    AssContext *ass = ctx->priv;
    int detect_change = 0;
    int64_t time_ms = av_rescale_q(picref->pts, inlink->time_base, ASS_TIME_BASE);
    ASS_Image *image = ass_render_frame(ass->renderer, ass->track,
                                        time_ms, &detect_change);

    if (detect_change)
        av_log(ctx, AV_LOG_DEBUG, "Change happened at time ms:%"PRId64"\n", time_ms);

    overlay_ass_image(ass, picref, image);

    return ff_filter_frame(outlink, picref);
}

static int process_header(AVFilterContext *link, AVCodecContext *dec_ctx)
{
    int ret = 0;
    AssContext *ass = link->priv;
    ASS_Track *track = ass->track;
//    enum AVCodecID codecID = dec_ctx->codec_id;

    if (!track)
        return AVERROR(EINVAL);

    if (ass->force_style) {
        char **list = NULL;
        char *temp = NULL;
        char *ptr = av_strtok(ass->force_style, ",", &temp);
        int i = 0;
        while (ptr) {
            av_dynarray_add(&list, &i, ptr);
            if (!list) {
                return AVERROR(ENOMEM);
            }
            ptr = av_strtok(NULL, ",", &temp);
        }
        av_dynarray_add(&list, &i, NULL);
        if (!list) {
            return AVERROR(ENOMEM);
        }
        ass_set_style_overrides(ass->library, list);
        av_free(list);
    }


    /* Decode subtitles and push them into the renderer (libass) */

    if (dec_ctx->subtitle_header)
        ass_process_codec_private(ass->track,
                                  dec_ctx->subtitle_header,
                                  dec_ctx->subtitle_header_size);
/*
    if (codecID == AV_CODEC_ID_ASS) {
        ass_process_codec_private(track, dec_ctx->extradata,
                                  dec_ctx->extradata_size);
    }
*/
    ass->got_header = 1;

    return ret;
}

void avfilter_inlinesubs_append_data(AVFilterContext *link, AVCodecContext *dec_ctx,
                                     AVSubtitle *sub)
{
    AssContext *ass = link->priv;
    int i;

    if (!ass->got_header)
        process_header(link, dec_ctx);

    av_log(NULL, AV_LOG_VERBOSE, "avfilter_inlinesubs_append_data!\n");

    for (i = 0; i < sub->num_rects; i++) {
        const char *ass_line = sub->rects[i]->ass;
        int64_t duration = sub->end_display_time - sub->start_display_time;
        int64_t start = av_rescale_q(sub->pts, AV_TIME_BASE_Q, ASS_TIME_BASE);
        start += sub->start_display_time;
        if (!ass_line)
            break;
        ass_process_chunk(ass->track, ass_line, strlen(ass_line), start, duration);
    }
}

static const char * const font_mimetypes[] = {
    "font/ttf",
    "font/otf",
    "font/sfnt",
    "font/woff",
    "font/woff2",
    "application/font-sfnt",
    "application/font-woff",
    "application/x-truetype-font",
    "application/vnd.ms-opentype",
    "application/x-font-ttf",
    NULL
};

void avfilter_inlinesubs_add_attachment(AVFilterContext *context, AVStream *st)
{
    AVDictionaryEntry *filename = NULL;
    AVDictionaryEntry *mimetype = NULL;
    AssContext *ass = (AssContext*)context->priv;
    int i;

    if (!st->codecpar->extradata_size)
        return;

    filename = av_dict_get(st->metadata, "filename", NULL, 0);
    if (!filename) {
        av_log(context, AV_LOG_WARNING,
               "Font attachment has no filename, ignored.\n");
        return;
    }

    mimetype = av_dict_get(st->metadata, "mimetype", NULL, AV_DICT_MATCH_CASE);
    if (mimetype) {
        for (i = 0; font_mimetypes[i]; i++) {
            if (av_strcasecmp(font_mimetypes[i], mimetype->value) == 0) {
                av_log(context, AV_LOG_DEBUG, "Loading attached font: %s\n",
                       filename->value);
                ass_add_font(ass->library, filename->value,
                             st->codecpar->extradata,
                             st->codecpar->extradata_size);
                return;
            }
        }
    }
}

void avfilter_inlinesubs_set_fonts(AVFilterContext *context)
{
    AssContext* ass = context->priv;

    /* Initialize fonts */
    ass_set_fonts(ass->renderer, NULL, NULL, 1, NULL, 1);
}

static const AVFilterPad inlinesubs_inputs[] = {
    {
        .name             = "default",
        .type             = AVMEDIA_TYPE_VIDEO,
        .flags            = AVFILTERPAD_FLAG_NEEDS_WRITABLE,
        .filter_frame     = filter_frame,
        .config_props     = config_input,
    },
};

static const AVFilterPad inlinesubs_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
};

static const AVOption inlinesubs_options[] = {
    {"original_size",  "set the size of the original video (used to scale fonts)", OFFSET(original_w), AV_OPT_TYPE_IMAGE_SIZE, {.str = NULL},  0, 0, FLAGS },
    {"fontsdir",       "set the directory containing the fonts to read",           OFFSET(fontsdir),   AV_OPT_TYPE_STRING,     {.str = NULL},  0, 0, FLAGS },
    {"alpha",          "enable processing of alpha channel",                       OFFSET(alpha),      AV_OPT_TYPE_BOOL,       {.i64 = 0   },  0, 1, FLAGS },
    {"shaping",        "set shaping engine",                                       OFFSET(shaping),    AV_OPT_TYPE_INT,        {.i64 = ASS_SHAPING_COMPLEX }, -1, 1, FLAGS, "shaping_mode"},
        {"auto",       NULL,              0, AV_OPT_TYPE_CONST, {.i64 = -1},                  INT_MIN, INT_MAX, FLAGS, "shaping_mode"},
        {"simple",     "simple shaping",  0, AV_OPT_TYPE_CONST, {.i64 = ASS_SHAPING_SIMPLE},  INT_MIN, INT_MAX, FLAGS, "shaping_mode"},
        {"complex",    "complex shaping", 0, AV_OPT_TYPE_CONST, {.i64 = ASS_SHAPING_COMPLEX}, INT_MIN, INT_MAX, FLAGS, "shaping_mode"},
    {"charenc",      "set input character encoding", OFFSET(charenc),      AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
    {"force_style",  "force subtitle style",         OFFSET(force_style),  AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
    {NULL},
};

AVFILTER_DEFINE_CLASS(inlinesubs);

const AVFilter ff_vf_inlinesubs = {
    .name          = "inlinesubs",
    .description   = NULL_IF_CONFIG_SMALL("Render text subtitles onto input video using the libass library."),
    .priv_size     = sizeof(AssContext),
    .init          = inlinesubs_init,
    .uninit        = inlinesubs_uninit,
    FILTER_INPUTS(inlinesubs_inputs),
    FILTER_OUTPUTS(inlinesubs_outputs),
    FILTER_QUERY_FUNC(query_formats),
    .priv_class    = &inlinesubs_class,
};
