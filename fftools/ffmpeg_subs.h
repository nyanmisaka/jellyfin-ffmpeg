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

#ifndef FFTOOLS_FFMPEG_SUBS_H
#define FFTOOLS_FFMPEG_SUBS_H

#include "cmdutils.h"
#include "ffmpeg.h"

#include <stdio.h>

#include "libavfilter/avfilter.h"

typedef struct
{
    int file_index;
    int stream_index;
    AVFilterContext *ctx;
} InlineSubsContext;

typedef struct
{
    int nb_inlinesubs_ctxs;
    InlineSubsContext *inlinesubs_ctxs;
} SubsContext;

/* ffmpeg_opt */
int subs_opt_subtitle_stream(void *optctx, const char *opt, const char *arg);

/* ffmpeg_mux_init */
void subs_prepare_setup_input_streams(InputStream* ist);

/* ffmpeg_filter */
void subs_link_subtitles_to_graph(AVFilterGraph* graph);

/* ffmpeg */
int subs_process_subtitles(const InputStream *ist, AVSubtitle *sub);

#endif /* FFTOOLS_FFMPEG_SUBS_H */
