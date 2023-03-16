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

#include "ffmpeg.h"
#include "ffmpeg_subs.h"

#include "config.h"
#include "config_components.h"

#include <sys/types.h>
#include <limits.h>
#include "strings.h"
#include "libavcodec/mpegvideo.h"
#include "libavfilter/vf_inlinesubs.h"
#include "libavutil/timestamp.h"
#include "libavformat/internal.h"

SubsContext subsCtx = {0};

void subs_prepare_setup_input_streams(InputStream* ist)
{
//#if CONFIG_INLINESUBS_FILTER
    int i;

    av_log(NULL, AV_LOG_VERBOSE, "subs_prepare_setup_input_streams!\n");

    for (i = 0; i < subsCtx.nb_inlinesubs_ctxs; i++) {
        InlineSubsContext *ctx = &subsCtx.inlinesubs_ctxs[i];
        if (ist->st->index == ctx->stream_index &&
            ist->file_index == ctx->file_index) {
            ist->discard = 0;
            ist->st->discard = AVDISCARD_NONE;
        }
    }
//#endif
}

void subs_link_subtitles_to_graph(AVFilterGraph* g)
{
//#if CONFIG_INLINESUBS_FILTER
    int contextId = 0;

    av_log(NULL, AV_LOG_VERBOSE, "subs_link_subtitles_to_graph!\n");

    for (int i = 0; i < nb_filtergraphs && contextId < subsCtx.nb_inlinesubs_ctxs; i++) {
        AVFilterGraph* graph = filtergraphs[i]->graph;
        if (!graph)
            continue;
        for (int i = 0; i < graph->nb_filters && contextId < subsCtx.nb_inlinesubs_ctxs; i++) {
            const AVFilterContext* filterCtx = graph->filters[i];
            if (strcmp(filterCtx->filter->name, "inlinesubs") == 0) {
                AVFilterContext *ctx = graph->filters[i];
                InlineSubsContext *inlineSubsCtx = &subsCtx.inlinesubs_ctxs[contextId++];
                inlineSubsCtx->ctx = ctx;

                for (int i = 0; i < nb_input_files; i++) {
                    InputFile *ifile = input_files[i];
                    for (int j = 0; j < ifile->nb_streams; j++) {
                        InputStream *ist = ifile->streams[j];
                        if (ist->st->codecpar->codec_type == AVMEDIA_TYPE_ATTACHMENT)
                            avfilter_inlinesubs_add_attachment(ctx, ist->st);
                        if (ist->file_index == inlineSubsCtx->file_index &&
                            ist->st->index == inlineSubsCtx->stream_index &&
                            ist->sub2video.sub_queue) {
/*
                            while (av_fifo_size(ist->sub2video.sub_queue)) {
                                AVSubtitle tmp;
                                av_fifo_generic_read(ist->sub2video.sub_queue, &tmp, sizeof(tmp), NULL);
                                subs_process_subtitles(ist, &tmp);
                                avsubtitle_free(&tmp);
                            }
*/
                            AVSubtitle sub;
                            while (av_fifo_read(ist->sub2video.sub_queue, &sub, 1) >= 0) {
                                subs_process_subtitles(ist, &sub);
                                avsubtitle_free(&sub);
                            }
                        }
                    }
                }

                avfilter_inlinesubs_set_fonts(ctx);
            }
        }
    }
//#endif
}

int subs_opt_subtitle_stream(void *optctx, const char *opt, const char *arg)
{
//#if CONFIG_INLINESUBS_FILTER
    InlineSubsContext *m = NULL;
    int i, file_idx;
    char *p;
    char *map = av_strdup(arg);

    av_log(NULL, AV_LOG_VERBOSE, "subs_opt_subtitle_stream!\n");

    file_idx = strtol(map, &p, 0);
    if (file_idx >= nb_input_files || file_idx < 0) {
        av_log(NULL, AV_LOG_FATAL, "Invalid subtitle input file index: %d.\n", file_idx);
        goto finish;
    }

    for (i = 0; i < input_files[file_idx]->nb_streams; i++) {
        if (check_stream_specifier(input_files[file_idx]->ctx, input_files[file_idx]->ctx->streams[i],
                    *p == ':' ? p + 1 : p) <= 0)
            continue;
        if (input_files[file_idx]->ctx->streams[i]->codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE) {
            av_log(NULL, AV_LOG_ERROR, "Stream '%s' is not a subtitle stream.\n", arg);
            continue;
        }
        GROW_ARRAY(subsCtx.inlinesubs_ctxs, subsCtx.nb_inlinesubs_ctxs);
        m = &subsCtx.inlinesubs_ctxs[subsCtx.nb_inlinesubs_ctxs - 1];

        m->file_index   = file_idx;
        m->stream_index = i;
        break;
    }

finish:
    if (!m)
        av_log(NULL, AV_LOG_ERROR, "Subtitle stream map '%s' matches no streams.\n", arg);

    av_freep(&map);
//#endif
    return 0;
}

int subs_process_subtitles(const InputStream *ist, AVSubtitle *sub)
{
//#if CONFIG_INLINESUBS_FILTER
    int i;

    av_log(NULL, AV_LOG_VERBOSE, "subs_process_subtitles!\n");

    /* If we're burning subtitles, pass discarded subtitle packets of the
     * appropriate stream to the subtitle renderer */
    for (i = 0; i < subsCtx.nb_inlinesubs_ctxs; i++) {
        InlineSubsContext *ctx = &subsCtx.inlinesubs_ctxs[i];
        if (ist->st->index == ctx->stream_index &&
            ist->file_index == ctx->file_index) {
            if (!ctx->ctx)
                return 1;
            avfilter_inlinesubs_append_data(ctx->ctx, ist->dec_ctx, sub);
            return 2;
        }
    }
//#endif
    return 0;
}
