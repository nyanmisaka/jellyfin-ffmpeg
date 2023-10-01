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
 * Rockchip RGA (2D Raster Graphic Acceleration) video resizer
 */

#include <unistd.h>
#include <stdint.h>
#include <float.h>
#include <stdio.h>

#include "libavutil/avassert.h"
#include "libavutil/common.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_drm.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "internal.h"
#include "scale_eval.h"
#include "transpose.h"
#include "video.h"

#include "rga/RgaApi.h"
#include "rga/im2d.h"

#define ALIGN_DOWN(a, b) ((a) & ~((b)-1))
#define RK_RGA_YUV_ALIGN 2

static const struct {
    enum AVPixelFormat    pix_fmt;
    enum _Rga_SURF_FORMAT rga_fmt;
} supported_formats[] = {
    { AV_PIX_FMT_YUV420P, RK_FORMAT_YCbCr_420_P },      /* RGA1/RGA2 only */
    { AV_PIX_FMT_YUV422P, RK_FORMAT_YCbCr_422_P },      /* RGA1/RGA2 only */
    { AV_PIX_FMT_NV12,    RK_FORMAT_YCbCr_420_SP },
    { AV_PIX_FMT_NV21,    RK_FORMAT_YCrCb_420_SP },
    { AV_PIX_FMT_NV16,    RK_FORMAT_YCbCr_422_SP },
    { AV_PIX_FMT_P010,    RK_FORMAT_YCbCr_420_SP_10B }, /* RGA3 only */
    { AV_PIX_FMT_NV15,    RK_FORMAT_YCbCr_420_SP_10B }, /* RGA2 only supports input, aka P010 compact */
    { AV_PIX_FMT_YUYV422, RK_FORMAT_YUYV_422 },
    { AV_PIX_FMT_YVYU422, RK_FORMAT_YVYU_422 },
    { AV_PIX_FMT_UYVY422, RK_FORMAT_UYVY_422 },
    { AV_PIX_FMT_RGB565,  RK_FORMAT_BGR_565 },
    { AV_PIX_FMT_BGR565,  RK_FORMAT_RGB_565 },
    { AV_PIX_FMT_RGB24,   RK_FORMAT_RGB_888 },
    { AV_PIX_FMT_BGR24,   RK_FORMAT_BGR_888 },
    { AV_PIX_FMT_RGBA,    RK_FORMAT_RGBA_8888 },
    { AV_PIX_FMT_RGB0,    RK_FORMAT_RGBA_8888 },        /* RK_FORMAT_RGBX_8888 may trigger RGA2 on multiRGA */
    { AV_PIX_FMT_BGRA,    RK_FORMAT_BGRA_8888 },
    { AV_PIX_FMT_BGR0,    RK_FORMAT_BGRA_8888 },        /* RK_FORMAT_BGRX_8888 may trigger RGA2 on multiRGA */
    { AV_PIX_FMT_ARGB,    RK_FORMAT_ARGB_8888 },
    { AV_PIX_FMT_0RGB,    RK_FORMAT_ARGB_8888 },        /* RK_FORMAT_XRGB_8888 may trigger RGA2 on multiRGA */
    { AV_PIX_FMT_ABGR,    RK_FORMAT_ABGR_8888 },
    { AV_PIX_FMT_0BGR,    RK_FORMAT_ABGR_8888 },        /* RK_FORMAT_XBGR_8888 may trigger RGA2 on multiRGA */
};

typedef struct RGAScaleContext {
    const AVClass *class;

    AVBufferRef *frames_ctx;
    AVFrame     *frame;
    AVFrame     *tmp_frame;

    int has_rga1_2, has_rga3;
    int is_rga1_2_used;

    enum _Rga_SURF_FORMAT in_rga_fmt, out_rga_fmt;
    enum AVPixelFormat in_fmt, out_fmt;
    const AVPixFmtDescriptor *in_desc, *out_desc;
    float in_bytes_pp, out_bytes_pp;
    int in_act_w, in_act_h;
    int out_act_w, out_act_h;
    int out_csc_mode, out_bt709_mpeg;
    int in_10b_uncompact_msb;
    int out_10b_uncompact_msb;
    int in_rotate_mode;

    char *w_expr, *h_expr;
    enum AVPixelFormat format;
    int transpose;
    int passthrough;
    int force_original_aspect_ratio;
    int force_divisible_by;

    int scheduler_core;
} RGAScaleContext;

static av_cold int map_av_to_rga_format(enum AVPixelFormat in_format,
                                        enum _Rga_SURF_FORMAT *out_format)
{
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++) {
        if (supported_formats[i].pix_fmt == in_format) {
            if (out_format)
                *out_format = supported_formats[i].rga_fmt;
            return 1;
        }
    }
    return 0;
}

static int get_pixel_stride(const AVDRMObjectDescriptor *object,
                            const AVDRMLayerDescriptor *layer,
                            int is_rgb, int is_planar,
                            float bytes_pp, int *ws, int *hs)
{
    const AVDRMPlaneDescriptor *plane0, *plane1;
    const int is_packed_fmt = is_rgb || (!is_rgb && !is_planar);

    if (!object || !layer || !ws || !hs || bytes_pp <= 0)
        return AVERROR(EINVAL);

    plane0 = &layer->planes[0];
    plane1 = &layer->planes[1];

    *ws = is_packed_fmt ?
        (plane0->pitch / bytes_pp) :
        plane0->pitch;
    *hs = is_packed_fmt ?
        ALIGN_DOWN(object->size / plane0->pitch, is_rgb ? 1 : 2) :
        (plane1->offset / plane0->pitch);

    return (*ws > 0 && *hs > 0) ? 0 : AVERROR(EINVAL);
}

static int is_pixel_stride_rga3_compat(int ws, int hs,
                                       enum _Rga_SURF_FORMAT fmt)
{
    switch (fmt) {
    case RK_FORMAT_YCbCr_420_SP:
    case RK_FORMAT_YCrCb_420_SP:
    case RK_FORMAT_YCbCr_422_SP:     return !(ws % 16) && !(hs % 2);
    case RK_FORMAT_YCbCr_420_SP_10B: return !(ws % 64) && !(hs % 2);
    case RK_FORMAT_YUYV_422:
    case RK_FORMAT_YVYU_422:
    case RK_FORMAT_UYVY_422:         return !(ws % 8) && !(hs % 2);
    case RK_FORMAT_RGB_565:
    case RK_FORMAT_BGR_565:          return !(ws % 8);
    case RK_FORMAT_RGB_888:
    case RK_FORMAT_BGR_888:          return !(ws % 16);
    case RK_FORMAT_RGBA_8888:
    case RK_FORMAT_BGRA_8888:
    case RK_FORMAT_ARGB_8888:
    case RK_FORMAT_ABGR_8888:        return !(ws % 4);
    default:                         return 0;
    }
}

static av_cold int init_hwframe_ctx(RGAScaleContext *r,
                                    AVBufferRef *device_ctx,
                                    int width, int height)
{
    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    int ret;

    out_ref = av_hwframe_ctx_alloc(device_ctx);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext *)out_ref->data;

    out_ctx->format    = AV_PIX_FMT_DRM_PRIME;
    out_ctx->sw_format = r->out_fmt;
    out_ctx->width     = FFALIGN(width,  16);
    out_ctx->height    = FFALIGN(height, 16);

    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
        goto fail;

    av_frame_unref(r->frame);
    ret = av_hwframe_get_buffer(out_ref, r->frame, 0);
    if (ret < 0)
        goto fail;

    r->frame->width  = width;
    r->frame->height = height;

    av_buffer_unref(&r->frames_ctx);
    r->frames_ctx = out_ref;

    return 0;

fail:
    av_buffer_unref(&out_ref);
    return ret;
}

static av_cold int set_format_info(AVFilterContext *ctx,
                                    enum AVPixelFormat in_format,
                                    enum AVPixelFormat out_format)
{
    RGAScaleContext *r = ctx->priv;

    r->in_fmt       = in_format;
    r->out_fmt      = out_format;
    r->in_desc      = av_pix_fmt_desc_get(r->in_fmt);
    r->out_desc     = av_pix_fmt_desc_get(r->out_fmt);
    r->in_bytes_pp  = av_get_padded_bits_per_pixel(r->in_desc) / 8.0f;
    r->out_bytes_pp = av_get_padded_bits_per_pixel(r->out_desc) / 8.0f;

    r->in_10b_uncompact_msb  = r->in_fmt == AV_PIX_FMT_P010;
    r->out_10b_uncompact_msb = r->out_fmt == AV_PIX_FMT_P010;

#if 0
    r->out_csc_mode   = 0;
    r->out_bt709_mpeg = 0;
    if ((r->in_desc->flags & AV_PIX_FMT_FLAG_RGB) &&
        !(r->out_desc->flags & AV_PIX_FMT_FLAG_RGB)) {
        r->out_csc_mode   = 0x3 << 2; /* IM_RGB_TO_YUV_BT709_LIMIT */
        r->out_bt709_mpeg = 1;
    }
    if (!(r->in_desc->flags & AV_PIX_FMT_FLAG_RGB) &&
        (r->out_desc->flags & AV_PIX_FMT_FLAG_RGB)) {
        r->out_csc_mode   = 0x3;      /* IM_YUV_TO_RGB_BT709_LIMIT */
        r->out_bt709_mpeg = 1;
    }
#endif

    /* P010 requires RGA3 */
    if (!r->has_rga3 &&
        (r->in_fmt == AV_PIX_FMT_P010 ||
         r->out_fmt == AV_PIX_FMT_P010)) {
        av_log(ctx, AV_LOG_ERROR, "%s is only supported by RGA3\n",
               av_get_pix_fmt_name(AV_PIX_FMT_P010));
        return AVERROR(ENOSYS);
    }
    /* YUV420P/YUV422P requires RGA1/RGA2 */
    if (!r->has_rga1_2 &&
        (r->in_fmt == AV_PIX_FMT_YUV420P ||
         r->in_fmt == AV_PIX_FMT_YUV422P ||
         r->out_fmt == AV_PIX_FMT_YUV420P ||
         r->out_fmt == AV_PIX_FMT_YUV422P)) {
        av_log(ctx, AV_LOG_ERROR, "%s and %s are only supported by RGA1/RGA2\n",
               av_get_pix_fmt_name(AV_PIX_FMT_YUV420P),
               av_get_pix_fmt_name(AV_PIX_FMT_YUV422P));
        return AVERROR(ENOSYS);
    }
    /* Only RGA3 can handle P010 but it doesn't support YUV420P/YUV422P */
    if (r->in_fmt == AV_PIX_FMT_P010 &&
        (r->out_fmt == AV_PIX_FMT_YUV420P ||
         r->out_fmt == AV_PIX_FMT_YUV422P)) {
        av_log(ctx, AV_LOG_ERROR, "%s to %s is not supported\n",
               av_get_pix_fmt_name(r->in_fmt),
               av_get_pix_fmt_name(r->out_fmt));
        return AVERROR(ENOSYS);
    }
    if (r->out_fmt == AV_PIX_FMT_P010 &&
        (r->in_fmt == AV_PIX_FMT_YUV420P ||
         r->in_fmt == AV_PIX_FMT_YUV422P)) {
        av_log(ctx, AV_LOG_ERROR, "%s to %s is not supported\n",
               av_get_pix_fmt_name(r->in_fmt),
               av_get_pix_fmt_name(r->out_fmt));
        return AVERROR(ENOSYS);
    }

    return 0;
}

static av_cold int set_size_info(AVFilterContext *ctx,
                                 AVFilterLink *inlink,
                                 AVFilterLink *outlink)
{
    RGAScaleContext *r = ctx->priv;
    float scale_ratio_w, scale_ratio_h;
    int w, h, ret;

    if (inlink->w < 2 ||
        inlink->h < 2) {
        av_log(ctx, AV_LOG_ERROR, "Min supported input size is 2x2\n");
        return AVERROR(EINVAL);
    }

    if ((ret = ff_scale_eval_dimensions(r,
                                        r->w_expr, r->h_expr,
                                        inlink, outlink,
                                        &w, &h)) < 0)
        return ret;

    ff_scale_adjust_dimensions(inlink, &w, &h,
                               r->force_original_aspect_ratio, r->force_divisible_by);

    if (((int64_t)h * inlink->w) > INT_MAX  ||
        ((int64_t)w * inlink->h) > INT_MAX) {
        av_log(ctx, AV_LOG_ERROR, "Rescaled value for width or height is too big.\n");
        return AVERROR(EINVAL);
    }

    outlink->w = w;
    outlink->h = h;
    if (outlink->w < 2 ||
        outlink->h < 2) {
        av_log(ctx, AV_LOG_ERROR, "Min supported output size is 2x2\n");
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

    /* The w/h of RGA YUV image needs to be 2 aligned */
    r->in_act_w = inlink->w;
    r->in_act_h = inlink->h;
    if (!(r->in_desc->flags & AV_PIX_FMT_FLAG_RGB)) {
        r->in_act_w = ALIGN_DOWN(r->in_act_w, RK_RGA_YUV_ALIGN);
        r->in_act_h = ALIGN_DOWN(r->in_act_h, RK_RGA_YUV_ALIGN);
    }

    r->out_act_w = outlink->w;
    r->out_act_h = outlink->h;
    if (!(r->out_desc->flags & AV_PIX_FMT_FLAG_RGB)) {
        r->out_act_w = ALIGN_DOWN(r->out_act_w, RK_RGA_YUV_ALIGN);
        r->out_act_h = ALIGN_DOWN(r->out_act_h, RK_RGA_YUV_ALIGN);
    }

    scale_ratio_w = (float)r->out_act_w / (float)r->in_act_w;
    scale_ratio_h = (float)r->out_act_h / (float)r->in_act_h;
    if (scale_ratio_w < 0.0625f || scale_ratio_w > 16.0f ||
        scale_ratio_h < 0.0625f || scale_ratio_h > 16.0f) {
        av_log(ctx, AV_LOG_ERROR, "RGA scale ratio (%.04fx%.04f) exceeds 0.0625 ~ 16.\n",
               scale_ratio_w, scale_ratio_h);
        return AVERROR(EINVAL);
    }

    r->is_rga1_2_used = !r->has_rga3;
    if (r->has_rga3) {
        if (r->in_fmt == AV_PIX_FMT_YUV420P ||
            r->in_fmt == AV_PIX_FMT_YUV422P ||
            r->out_fmt == AV_PIX_FMT_YUV420P ||
            r->out_fmt == AV_PIX_FMT_YUV422P) {
            r->is_rga1_2_used = 1;
        }
        if (scale_ratio_w < 0.125f ||
            scale_ratio_w > 8.0f ||
            scale_ratio_h < 0.125f ||
            scale_ratio_h > 8.0f) {
            r->is_rga1_2_used = 1;
        }
        if (r->in_act_w < 68 ||
            r->in_act_w > 8176 ||
            r->in_act_h > 8176 ||
            r->out_act_w < 68) {
            r->is_rga1_2_used = 1;
        }
    }
    if (r->is_rga1_2_used && !r->has_rga1_2) {
        av_log(ctx, AV_LOG_ERROR, "RGA1/RGA2 is requested but not available\n");
        return AVERROR(ENOSYS);
    }
    if (r->is_rga1_2_used && (r->in_10b_uncompact_msb || r->out_10b_uncompact_msb)) {
        av_log(ctx, AV_LOG_ERROR, "%s is not supported if RGA1/RGA2 is requested\n",
               av_get_pix_fmt_name(AV_PIX_FMT_P010));
        return AVERROR(ENOSYS);
    }
    if (r->is_rga1_2_used && r->out_fmt == AV_PIX_FMT_NV15) {
        av_log(ctx, AV_LOG_ERROR, "%s as output is not supported if RGA1/RGA2 is requested\n",
               av_get_pix_fmt_name(AV_PIX_FMT_NV15));
        return AVERROR(ENOSYS);
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
    int ret;

    if (!inlink->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames_ctx = (AVHWFramesContext *)inlink->hw_frames_ctx->data;
    in_format     = in_frames_ctx->sw_format;
    out_format    = (r->format == AV_PIX_FMT_NONE) ? in_format : r->format;

    if (!map_av_to_rga_format(in_format, &r->in_rga_fmt)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format: %s\n",
               av_get_pix_fmt_name(in_format));
        return AVERROR(ENOSYS);
    }

    if (!map_av_to_rga_format(out_format, &r->out_rga_fmt)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported output format: %s\n",
               av_get_pix_fmt_name(out_format));
        return AVERROR(ENOSYS);
    }

    ret = set_format_info(ctx, in_format, out_format);
    if (ret < 0)
        return ret;

    ret = set_size_info(ctx, inlink, outlink);
    if (ret < 0)
        return ret;

    if (r->passthrough && r->transpose < 0 &&
        inlink->w == outlink->w && inlink->h == outlink->h && in_format == out_format) {
        r->frames_ctx = av_buffer_ref(inlink->hw_frames_ctx);
        if (!r->frames_ctx)
            return AVERROR(ENOMEM);
    } else {
        r->passthrough = 0;

        ret = init_hwframe_ctx(r, in_frames_ctx->device_ref, outlink->w, outlink->h);
        if (ret < 0)
            return ret;
    }

    /* output buffer */
    outlink->hw_frames_ctx = av_buffer_ref(r->frames_ctx);
    if (!outlink->hw_frames_ctx)
        return AVERROR(ENOMEM);

    av_log(ctx, AV_LOG_VERBOSE, "w:%d h:%d fmt:%s -> w:%d h:%d fmt:%s%s\n",
           inlink->w, inlink->h, av_get_pix_fmt_name(r->in_fmt),
           outlink->w, outlink->h, av_get_pix_fmt_name(r->out_fmt),
           r->passthrough ? " (passthrough)" : "");

    return 0;
}

static int rgascale_scale_resize(AVFilterContext *ctx,
                                 AVFrame *out, AVFrame *in)
{
    RGAScaleContext *r = ctx->priv;
    const AVDRMFrameDescriptor *in_drm_desc, *out_drm_desc;
    rga_info_t src = { .mmuFlag = 1, .format = r->in_rga_fmt, };
    rga_info_t dst = { .mmuFlag = 1, .format = r->out_rga_fmt, };
    int w_stride = 0, h_stride = 0;
    int ret;

    if (!in || !out)
        return AVERROR(EINVAL);

    in_drm_desc = (AVDRMFrameDescriptor *)in->data[0];
    out_drm_desc = (AVDRMFrameDescriptor *)out->data[0];
    if (!in_drm_desc || !out_drm_desc)
        return AVERROR(ENOMEM);

    if (in_drm_desc->nb_objects != 1 ||
        in_drm_desc->nb_layers != 1 ||
        out_drm_desc->nb_objects != 1 ||
        out_drm_desc->nb_layers != 1) {
        av_log(ctx, AV_LOG_ERROR, "RGA only supports single DRM object/layer\n");
        return AVERROR(EINVAL);
    }

    ret = get_pixel_stride(&in_drm_desc->objects[0],
                           &in_drm_desc->layers[0],
                           (r->in_desc->flags & AV_PIX_FMT_FLAG_RGB),
                           (r->in_desc->flags & AV_PIX_FMT_FLAG_PLANAR),
                           r->in_bytes_pp, &w_stride, &h_stride);
    if (ret < 0 || !w_stride || !h_stride)
        return AVERROR(EINVAL);

    if (r->in_10b_uncompact_msb) {
        src.is_10b_compact = 1;
        src.is_10b_endian  = 1;
    }
    src.rotation = r->in_rotate_mode;
    src.fd = in_drm_desc->objects[0].fd;
    rga_set_rect(&src.rect, 0, 0,
                 r->in_act_w, r->in_act_h,
                 w_stride, h_stride, r->in_rga_fmt);

    av_log(ctx, AV_LOG_DEBUG, "RGA src | fd:%d mmu:%d | x:%d y:%d w:%d h:%d ws:%d hs:%d fmt:0x%x\n",
           src.fd, src.mmuFlag, src.rect.xoffset, src.rect.yoffset,
           src.rect.width, src.rect.height, src.rect.wstride, src.rect.hstride, (src.rect.format >> 8));

    r->is_rga1_2_used = r->is_rga1_2_used ||
        !is_pixel_stride_rga3_compat(w_stride, h_stride, r->in_rga_fmt);

    if (r->is_rga1_2_used && r->out_fmt == AV_PIX_FMT_NV15) {
        av_log(ctx, AV_LOG_ERROR, "%s as output is not supported if RGA1/RGA2 is requested\n",
               av_get_pix_fmt_name(AV_PIX_FMT_NV15));
        return AVERROR(ENOSYS);
    }

    ret = get_pixel_stride(&out_drm_desc->objects[0],
                           &out_drm_desc->layers[0],
                           (r->out_desc->flags & AV_PIX_FMT_FLAG_RGB),
                           (r->out_desc->flags & AV_PIX_FMT_FLAG_PLANAR),
                           r->out_bytes_pp, &w_stride, &h_stride);
    if (ret < 0 || !w_stride || !h_stride)
        return AVERROR(EINVAL);

    if (r->out_10b_uncompact_msb) {
        dst.is_10b_compact = 1;
        dst.is_10b_endian  = 1;
    }
    dst.color_space_mode = r->out_csc_mode;
    dst.core = r->scheduler_core;
    dst.fd = out_drm_desc->objects[0].fd;
    rga_set_rect(&dst.rect, 0, 0,
                 r->out_act_w, r->out_act_h,
                 w_stride, h_stride, r->out_rga_fmt);

    av_log(ctx, AV_LOG_DEBUG, "RGA dst | fd:%d mmu:%d | x:%d y:%d w:%d h:%d ws:%d hs:%d fmt:0x%x\n",
           dst.fd, dst.mmuFlag, dst.rect.xoffset, dst.rect.yoffset,
           dst.rect.width, dst.rect.height, dst.rect.wstride, dst.rect.hstride, (dst.rect.format >> 8));

    if ((ret = c_RkRgaBlit(&src, &dst, NULL)) != 0) {
        av_log(ctx, AV_LOG_ERROR, "RGA resize failed: %d\n", ret);
        return AVERROR_EXTERNAL;
    }

    return 0;
}

static int rgascale_scale(AVFilterContext *ctx,
                          AVFrame *out, AVFrame *in)
{
    RGAScaleContext *r = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *src = in;
    int ret;

    ret = rgascale_scale_resize(ctx, r->frame, src);
    if (ret < 0)
        return ret;

    src = r->frame;
    ret = av_hwframe_get_buffer(src->hw_frames_ctx, r->tmp_frame, 0);
    if (ret < 0)
        return ret;

    av_frame_move_ref(out, r->frame);
    av_frame_move_ref(r->frame, r->tmp_frame);

    r->frame->width  = outlink->w;
    r->frame->height = outlink->h;

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        return ret;

    if (r->out_bt709_mpeg) {
        out->color_trc       = AVCOL_TRC_UNSPECIFIED;
        out->color_primaries = AVCOL_PRI_UNSPECIFIED;
        out->colorspace      = AVCOL_SPC_BT709;
        out->color_range     = AVCOL_RANGE_MPEG;
    }

    return 0;
}

static int rgascale_filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    RGAScaleContext *r = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *out = NULL;
    int ret = 0;

    if (r->passthrough)
        return ff_filter_frame(outlink, in);

    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    ret = rgascale_scale(ctx, out, in);
    if (ret < 0)
        goto fail;

    av_reduce(&out->sample_aspect_ratio.num, &out->sample_aspect_ratio.den,
              (int64_t)in->sample_aspect_ratio.num * outlink->h * inlink->w,
              (int64_t)in->sample_aspect_ratio.den * outlink->w * inlink->h,
              INT_MAX);

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

static av_cold int rgascale_init(AVFilterContext *ctx)
{
    RGAScaleContext *r = ctx->priv;
    const char *rga_ver = querystring(RGA_VERSION);

    r->has_rga1_2 = !!strstr(rga_ver, "RGA_1") ||
                    !!strstr(rga_ver, "RGA_2");
    r->has_rga3   = !!strstr(rga_ver, "RGA_3");
    if (!(r->has_rga1_2 || r->has_rga3)) {
        av_log(ctx, AV_LOG_ERROR, "No RGA1/RGA2/RGA3 hw available\n");
        return AVERROR(ENOSYS);
    }

    if (r->scheduler_core && !(r->has_rga1_2 && r->has_rga3)) {
        av_log(ctx, AV_LOG_WARNING, "Scheduler core cannot be set on non-multiRGA hw, ignoring\n");
        r->scheduler_core = 0;
    }
    if (r->scheduler_core && r->scheduler_core != (r->scheduler_core & 0x7)) {
        av_log(ctx, AV_LOG_WARNING, "Invalid scheduler core set, ignoring\n");
        r->scheduler_core = 0;
    }
    if (r->scheduler_core && r->scheduler_core == (r->scheduler_core & 0x3))
        r->has_rga1_2 = 0;
    if (r->scheduler_core == 0x4)
        r->has_rga3 = 0;

    r->frame = av_frame_alloc();
    if (!r->frame)
        return AVERROR(ENOMEM);

    r->tmp_frame = av_frame_alloc();
    if (!r->tmp_frame)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold void rgascale_uninit(AVFilterContext *ctx)
{
    RGAScaleContext *r = ctx->priv;

    av_frame_free(&r->frame);
    av_frame_free(&r->tmp_frame);
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
    { "w", "Output video width",  OFFSET(w_expr), AV_OPT_TYPE_STRING, { .str = "iw" }, .flags = FLAGS },
    { "h", "Output video height", OFFSET(h_expr), AV_OPT_TYPE_STRING, { .str = "ih" }, .flags = FLAGS },
    { "format", "Output video pixel format", OFFSET(format), AV_OPT_TYPE_PIXEL_FMT, { .i64 = AV_PIX_FMT_NONE }, INT_MIN, INT_MAX, .flags = FLAGS },
    { "transpose",  "Set transpose direction", OFFSET(transpose), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 6, FLAGS, "transpose" },
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
    { "core", "Set multiRGA scheduler core [use with caution]", OFFSET(scheduler_core), AV_OPT_TYPE_FLAGS, { .i64 = 0 }, 0, INT_MAX, FLAGS, "core" },
        { "default",    NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 0 }, 0, 0, FLAGS, "core" },
        { "rga3_core0", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 1 }, 0, 0, FLAGS, "core" }, /* RGA3_SCHEDULER_CORE0 */
        { "rga3_core1", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 2 }, 0, 0, FLAGS, "core" }, /* RGA3_SCHEDULER_CORE1 */
        { "rga2_core0", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 4 }, 0, 0, FLAGS, "core" }, /* RGA2_SCHEDULER_CORE0 */
    { NULL },
};

AVFILTER_DEFINE_CLASS(rgascale);

static const AVFilterPad rgascale_inputs[] = {
    {
        .name             = "default",
        .type             = AVMEDIA_TYPE_VIDEO,
        .filter_frame     = rgascale_filter_frame,
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
    .description    = NULL_IF_CONFIG_SMALL("Rockchip RGA (2D Raster Graphic Acceleration) video resizer"),
    .priv_size      = sizeof(RGAScaleContext),
    .priv_class     = &rgascale_class,
    .init           = rgascale_init,
    .uninit         = rgascale_uninit,
    FILTER_INPUTS(rgascale_inputs),
    FILTER_OUTPUTS(rgascale_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_DRM_PRIME),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
