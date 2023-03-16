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

#ifndef AVFILTER_INLINESUBS_H
#define AVFILTER_INLINESUBS_H

#include "avfilter.h"
#include "libavcodec/avcodec.h"

void avfilter_inlinesubs_process_header(AVFilterContext *link,
                                        AVCodecContext *dec_ctx);
void avfilter_inlinesubs_append_data(AVFilterContext *link, AVCodecContext *dec_ctx,
                                     AVSubtitle *sub);
void avfilter_inlinesubs_add_attachment(AVFilterContext *context, AVStream *st);
void avfilter_inlinesubs_set_fonts(AVFilterContext *context);

#endif // AVFILTER_INLINESUBS_H
