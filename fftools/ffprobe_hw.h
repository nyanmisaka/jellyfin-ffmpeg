/*
 * Copyright (C) 2026 NyanMisaka
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

#ifndef FFTOOLS_FFPROBE_HW_H
#define FFTOOLS_FFPROBE_HW_H

#include "textformat/avtextformat.h"
#include "libavutil/hwcontext.h"

#define FFPROBE_HW_FLAG_PRINT_DEV       (1 << 0)
#define FFPROBE_HW_FLAG_PRINT_DEC       (1 << 1)
#define FFPROBE_HW_FLAG_PRINT_ENC       (1 << 2)
#define FFPROBE_HW_FLAG_PRINT_VPP       (1 << 3)
#define FFPROBE_HW_FLAG_PRINT_OPT_OCL   (1 << 4)
#define FFPROBE_HW_FLAG_PRINT_OPT_VK    (1 << 5)
#define FFPROBE_HW_FLAG_PRINT_OPT_DX11  (1 << 6)
#define FFPROBE_HW_FLAG_PRINT_SUB_DEV   (1 << 7)
#define FFPROBE_HW_FLAG_PRINT_UTIL      (1 << 8)

#define FFPROBE_HW_ALL_PRINT_FLAGS \
    (FFPROBE_HW_FLAG_PRINT_DEV | \
     FFPROBE_HW_FLAG_PRINT_DEC | \
     FFPROBE_HW_FLAG_PRINT_ENC | \
     FFPROBE_HW_FLAG_PRINT_VPP | \
     FFPROBE_HW_FLAG_PRINT_OPT_OCL  | \
     FFPROBE_HW_FLAG_PRINT_OPT_VK   | \
     FFPROBE_HW_FLAG_PRINT_OPT_DX11 | \
     FFPROBE_HW_FLAG_PRINT_SUB_DEV  | \
     FFPROBE_HW_FLAG_PRINT_UTIL)

#define FFPROBE_HW_DEFAULT_PRINT_FLAGS \
    (FFPROBE_HW_FLAG_PRINT_DEV | \
     FFPROBE_HW_FLAG_PRINT_DEC | \
     FFPROBE_HW_FLAG_PRINT_ENC | \
     FFPROBE_HW_FLAG_PRINT_VPP)

void ffprobe_show_hwaccel_internal(AVTextFormatContext *tfc,
                                   enum AVHWDeviceType hwaccel_type,
                                   int hwaccel_flags,
                                   const char *hwaccel_device);

#endif /* FFTOOLS_FFPROBE_HW_H */
