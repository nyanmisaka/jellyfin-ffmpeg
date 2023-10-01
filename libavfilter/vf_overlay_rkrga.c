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
#include "libavutil/eval.h"

#include "avfilter.h"
#include "framesync.h"
#include "internal.h"
#include "scale_eval.h"
#include "video.h"

#include "rga/RgaApi.h"
#include "rga/im2d.h"

#define ALIGN_DOWN(a, b) ((a) & ~((b)-1))
#define RK_RGA_YUV_ALIGN 2

typedef struct RgaFormatMap {
    enum AVPixelFormat    pix_fmt;
    enum _Rga_SURF_FORMAT rga_fmt;
} RgaFormatMap;

#define YUV_FORMATS \
    { AV_PIX_FMT_YUV420P, RK_FORMAT_YCbCr_420_P }, \
    { AV_PIX_FMT_YUV422P, RK_FORMAT_YCbCr_422_P }, \
    { AV_PIX_FMT_NV12,    RK_FORMAT_YCbCr_420_SP }, \
    { AV_PIX_FMT_NV21,    RK_FORMAT_YCrCb_420_SP }, \
    { AV_PIX_FMT_NV16,    RK_FORMAT_YCbCr_422_SP }, \
    { AV_PIX_FMT_P010,    RK_FORMAT_YCbCr_420_SP_10B }, \
    { AV_PIX_FMT_NV15,    RK_FORMAT_YCbCr_420_SP_10B }, \
    { AV_PIX_FMT_YUYV422, RK_FORMAT_YUYV_422 }, \
    { AV_PIX_FMT_YVYU422, RK_FORMAT_YVYU_422 }, \
    { AV_PIX_FMT_UYVY422, RK_FORMAT_UYVY_422 },

#define RGB_FORMATS \
    { AV_PIX_FMT_RGB565,  RK_FORMAT_BGR_565 }, \
    { AV_PIX_FMT_BGR565,  RK_FORMAT_RGB_565 }, \
    { AV_PIX_FMT_RGB24,   RK_FORMAT_RGB_888 }, \
    { AV_PIX_FMT_BGR24,   RK_FORMAT_BGR_888 }, \
    { AV_PIX_FMT_RGBA,    RK_FORMAT_RGBA_8888 }, \
    { AV_PIX_FMT_RGB0,    RK_FORMAT_RGBA_8888 }, \
    { AV_PIX_FMT_BGRA,    RK_FORMAT_BGRA_8888 }, \
    { AV_PIX_FMT_BGR0,    RK_FORMAT_BGRA_8888 }, \
    { AV_PIX_FMT_ARGB,    RK_FORMAT_ARGB_8888 }, \
    { AV_PIX_FMT_0RGB,    RK_FORMAT_ARGB_8888 }, \
    { AV_PIX_FMT_ABGR,    RK_FORMAT_ABGR_8888 }, \
    { AV_PIX_FMT_0BGR,    RK_FORMAT_ABGR_8888 },

static const RgaFormatMap supported_formats_main[] = {
    YUV_FORMATS
    RGB_FORMATS
};

static const RgaFormatMap supported_formats_overlay[] = {
    RGB_FORMATS
};
#undef YUV_FORMATS
#undef RGB_FORMATS

typedef struct RGAOverlayContext {
    const AVClass *class;

    AVBufferRef *frames_ctx;
    AVFrame     *frame;
    AVFrame     *tmp_frame;

    AVBufferRef *frames_ctx1;
    AVFrame     *tmp_frame1;
    AVFrame     *tmp_frame2;
    int w_stride_tmp1;
    int h_stride_tmp1;

    AVBufferRef *frames_ctx2;
    AVFrame     *tmp_frame3;
    int w_stride_tmp3;
    int h_stride_tmp3;

    int has_rga1_2, has_rga3;
    int is_rga1_2_used;
    int is_offset_valid;
    int is_resizing;

    enum _Rga_SURF_FORMAT in_rga_fmt_main, in_rga_fmt_overlay, out_rga_fmt;
    enum AVPixelFormat in_fmt_main, in_fmt_overlay, out_fmt;
    const AVPixFmtDescriptor *in_desc_main, *in_desc_overlay, *out_desc;
    float in_bytes_pp_main, in_bytes_pp_overlay, out_bytes_pp;
    int in_act_w_main, in_act_h_main, in_act_w_overlay, in_act_h_overlay;
    int out_act_w, out_act_h;
    int in_blend_mode, out_csc_mode, out_bt709_mpeg;
    int in_10b_uncompact_msb_main, out_10b_uncompact_msb;

    FFFrameSync fs;

    int overlay_x;
    int overlay_y;
    int global_alpha;

    char *w_expr, *h_expr;
    enum AVPixelFormat format;
    int force_original_aspect_ratio;
    int force_divisible_by;

    int scheduler_core;
} RGAOverlayContext;

static av_cold int map_av_to_rga_format(enum AVPixelFormat in_format,
                                        enum _Rga_SURF_FORMAT *out_format, int is_overlay)
{
    int i;

    if (is_overlay)
        goto overlay;

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats_main); i++) {
        if (supported_formats_main[i].pix_fmt == in_format) {
            if (out_format)
                *out_format = supported_formats_main[i].rga_fmt;
            return 1;
        }
    }
    return 0;

overlay:
    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats_overlay); i++) {
        if (supported_formats_overlay[i].pix_fmt == in_format) {
            if (out_format)
                *out_format = supported_formats_overlay[i].rga_fmt;
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

static av_cold int init_hwframe_ctx(RGAOverlayContext *r,
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

static av_cold int init_hwframe_ctx1(RGAOverlayContext *r,
                                     AVBufferRef *device_ctx,
                                     int width, int height)
{
    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    const AVDRMFrameDescriptor *tmp_drm_desc;
    int ret;

    out_ref = av_hwframe_ctx_alloc(device_ctx);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext *)out_ref->data;

    out_ctx->format    = AV_PIX_FMT_DRM_PRIME;
    out_ctx->sw_format = r->in_fmt_overlay;
    out_ctx->width     = FFALIGN(width,  16);
    out_ctx->height    = FFALIGN(height, 16);

    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
        goto fail;

    av_frame_unref(r->tmp_frame1);
    ret = av_hwframe_get_buffer(out_ref, r->tmp_frame1, 0);
    if (ret < 0)
        goto fail;

    av_frame_unref(r->tmp_frame2);
    ret = av_hwframe_get_buffer(out_ref, r->tmp_frame2, 0);
    if (ret < 0)
        goto fail;

    r->tmp_frame1->width  = r->tmp_frame2->width  = width;
    r->tmp_frame1->height = r->tmp_frame2->height = height;

    tmp_drm_desc = (AVDRMFrameDescriptor *)r->tmp_frame1->data[0];
    ret = get_pixel_stride(&tmp_drm_desc->objects[0],
                           &tmp_drm_desc->layers[0],
                           (r->in_desc_overlay->flags & AV_PIX_FMT_FLAG_RGB),
                           (r->in_desc_overlay->flags & AV_PIX_FMT_FLAG_PLANAR),
                           r->in_bytes_pp_overlay, &r->w_stride_tmp1, &r->h_stride_tmp1);
    if (ret < 0 || !r->w_stride_tmp1 || !r->h_stride_tmp1) {
        ret = AVERROR(EINVAL);
        goto fail;
    }

    av_buffer_unref(&r->frames_ctx1);
    r->frames_ctx1 = out_ref;

    return 0;

fail:
    av_buffer_unref(&out_ref);
    return ret;
}

static av_cold int init_hwframe_ctx2(RGAOverlayContext *r,
                                     AVBufferRef *device_ctx,
                                     int width, int height)
{
    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    const AVDRMFrameDescriptor *tmp_drm_desc;
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

    av_frame_unref(r->tmp_frame3);
    ret = av_hwframe_get_buffer(out_ref, r->tmp_frame3, 0);
    if (ret < 0)
        goto fail;

    r->tmp_frame3->width  = width;
    r->tmp_frame3->height = height;

    tmp_drm_desc = (AVDRMFrameDescriptor *)r->tmp_frame3->data[0];
    ret = get_pixel_stride(&tmp_drm_desc->objects[0],
                           &tmp_drm_desc->layers[0],
                           (r->out_desc->flags & AV_PIX_FMT_FLAG_RGB),
                           (r->out_desc->flags & AV_PIX_FMT_FLAG_PLANAR),
                           r->out_bytes_pp, &r->w_stride_tmp3, &r->h_stride_tmp3);
    if (ret < 0 || !r->w_stride_tmp3 || !r->h_stride_tmp3) {
        ret = AVERROR(EINVAL);
        goto fail;
    }

    av_buffer_unref(&r->frames_ctx2);
    r->frames_ctx2 = out_ref;

    return 0;

fail:
    av_buffer_unref(&out_ref);
    return ret;
}

static av_cold int set_format_info(AVFilterContext *ctx,
                                   enum AVPixelFormat in_format_main,
                                   enum AVPixelFormat in_format_overlay,
                                   enum AVPixelFormat out_format)
{
    RGAOverlayContext *r = ctx->priv;

    r->in_fmt_main         = in_format_main;
    r->in_fmt_overlay      = in_format_overlay;
    r->out_fmt             = out_format;
    r->in_desc_main        = av_pix_fmt_desc_get(r->in_fmt_main);
    r->in_desc_overlay     = av_pix_fmt_desc_get(r->in_fmt_overlay);
    r->out_desc            = av_pix_fmt_desc_get(r->out_fmt);
    r->in_bytes_pp_main    = av_get_padded_bits_per_pixel(r->in_desc_main) / 8.0f;
    r->in_bytes_pp_overlay = av_get_padded_bits_per_pixel(r->in_desc_overlay) / 8.0f;
    r->out_bytes_pp        = av_get_padded_bits_per_pixel(r->out_desc) / 8.0f;

    r->in_10b_uncompact_msb_main = r->in_fmt_main == AV_PIX_FMT_P010;
    r->out_10b_uncompact_msb     = r->out_fmt == AV_PIX_FMT_P010;

    /* IM_ALPHA_BLEND_DST_OVER */
    r->in_blend_mode = !!(r->in_desc_overlay->flags & AV_PIX_FMT_FLAG_ALPHA) ? 0x504 : 0x501;
    if (r->global_alpha > 0 && r->global_alpha <= 255)
        r->in_blend_mode ^= (r->global_alpha << 16);
    else
        r->in_blend_mode ^= 0xff0000; /* 255 << 16 */

#if 0
    r->out_csc_mode   = 0;
    r->out_bt709_mpeg = 0;
    if ((r->in_desc_main->flags & AV_PIX_FMT_FLAG_RGB) &&
        !(r->out_desc->flags & AV_PIX_FMT_FLAG_RGB)) {
        r->out_csc_mode   = 0x3 << 2; /* IM_RGB_TO_YUV_BT709_LIMIT */
        r->out_bt709_mpeg = 1;
    }
    if (!(r->in_desc_main->flags & AV_PIX_FMT_FLAG_RGB) &&
        (r->out_desc->flags & AV_PIX_FMT_FLAG_RGB)) {
        r->out_csc_mode   = 0x3;      /* IM_YUV_TO_RGB_BT709_LIMIT */
        r->out_bt709_mpeg = 1;
    }
#endif

    /* P010 requires RGA3 */
    if (!r->has_rga3 &&
        (r->in_fmt_main == AV_PIX_FMT_P010 ||
         r->out_fmt == AV_PIX_FMT_P010)) {
        av_log(ctx, AV_LOG_ERROR, "%s is only supported by RGA3\n",
               av_get_pix_fmt_name(AV_PIX_FMT_P010));
        return AVERROR(ENOSYS);
    }
    /* YUV420P/YUV422P requires RGA1/RGA2 */
    if (!r->has_rga1_2 &&
        (r->in_fmt_main == AV_PIX_FMT_YUV420P ||
         r->in_fmt_main == AV_PIX_FMT_YUV422P ||
         r->out_fmt == AV_PIX_FMT_YUV420P ||
         r->out_fmt == AV_PIX_FMT_YUV422P)) {
        av_log(ctx, AV_LOG_ERROR, "%s and %s are only supported by RGA1/RGA2\n",
               av_get_pix_fmt_name(AV_PIX_FMT_YUV420P),
               av_get_pix_fmt_name(AV_PIX_FMT_YUV422P));
        return AVERROR(ENOSYS);
    }
    /* Only RGA3 can handle P010 but it doesn't support YUV420P/YUV422P */
    if (r->in_fmt_main == AV_PIX_FMT_P010 &&
        (r->out_fmt == AV_PIX_FMT_YUV420P ||
         r->out_fmt == AV_PIX_FMT_YUV422P)) {
        av_log(ctx, AV_LOG_ERROR, "%s to %s is not supported\n",
               av_get_pix_fmt_name(r->in_fmt_main),
               av_get_pix_fmt_name(r->out_fmt));
        return AVERROR(ENOSYS);
    }
    if (r->out_fmt == AV_PIX_FMT_P010 &&
        (r->in_fmt_main == AV_PIX_FMT_YUV420P ||
         r->in_fmt_main == AV_PIX_FMT_YUV422P)) {
        av_log(ctx, AV_LOG_ERROR, "%s to %s is not supported\n",
               av_get_pix_fmt_name(r->in_fmt_main),
               av_get_pix_fmt_name(r->out_fmt));
        return AVERROR(ENOSYS);
    }

    return 0;
}

static av_cold int set_size_info(AVFilterContext *ctx,
                                 AVFilterLink *inlink_main,
                                 AVFilterLink *inlink_overlay,
                                 AVFilterLink *outlink)
{
    RGAOverlayContext *r = ctx->priv;
    float scale_ratio_w, scale_ratio_h;
    int w, h, ret;

    if (inlink_main->w < 2 ||
        inlink_main->h < 2 ||
        inlink_overlay->w < 2 ||
        inlink_overlay->h < 2) {
        av_log(ctx, AV_LOG_ERROR, "Min supported input size is 2x2\n");
        return AVERROR(EINVAL);
    }

    if ((ret = ff_scale_eval_dimensions(r,
                                        r->w_expr, r->h_expr,
                                        inlink_main, outlink,
                                        &w, &h)) < 0)
        return ret;

    ff_scale_adjust_dimensions(inlink_main, &w, &h,
                               r->force_original_aspect_ratio, r->force_divisible_by);

    if (((int64_t)h * inlink_main->w) > INT_MAX  ||
        ((int64_t)w * inlink_main->h) > INT_MAX) {
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

    /* The w/h of RGA YUV image needs to be 2 aligned */
    r->in_act_w_main = inlink_main->w;
    r->in_act_h_main = inlink_main->h;
    if (!(r->in_desc_main->flags & AV_PIX_FMT_FLAG_RGB)) {
        r->in_act_w_main = ALIGN_DOWN(r->in_act_w_main, RK_RGA_YUV_ALIGN);
        r->in_act_h_main = ALIGN_DOWN(r->in_act_h_main, RK_RGA_YUV_ALIGN);
    }

    r->in_act_w_overlay = inlink_overlay->w;
    r->in_act_h_overlay = inlink_overlay->h;
    if (!(r->in_desc_overlay->flags & AV_PIX_FMT_FLAG_RGB)) {
        r->in_act_w_overlay = ALIGN_DOWN(r->in_act_w_overlay, RK_RGA_YUV_ALIGN);
        r->in_act_h_overlay = ALIGN_DOWN(r->in_act_h_overlay, RK_RGA_YUV_ALIGN);
    }

    r->out_act_w = outlink->w;
    r->out_act_h = outlink->h;
    if (!(r->out_desc->flags & AV_PIX_FMT_FLAG_RGB)) {
        r->out_act_w = ALIGN_DOWN(r->out_act_w, RK_RGA_YUV_ALIGN);
        r->out_act_h = ALIGN_DOWN(r->out_act_h, RK_RGA_YUV_ALIGN);
    }

    scale_ratio_w = (float)r->out_act_w / (float)r->in_act_w_main;
    scale_ratio_h = (float)r->out_act_h / (float)r->in_act_h_main;
    if (scale_ratio_w < 0.0625f || scale_ratio_w > 16.0f ||
        scale_ratio_h < 0.0625f || scale_ratio_h > 16.0f) {
        av_log(ctx, AV_LOG_ERROR, "RGA scale ratio (%.04fx%.04f) exceeds 0.0625 ~ 16.\n",
               scale_ratio_w, scale_ratio_h);
        return AVERROR(EINVAL);
    }

    r->is_rga1_2_used = !r->has_rga3;
    if (r->has_rga3) {
        if (r->in_fmt_main == AV_PIX_FMT_YUV420P ||
            r->in_fmt_main == AV_PIX_FMT_YUV422P ||
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
        if (r->in_act_w_main < 68 ||
            r->in_act_w_main > 8176 ||
            r->in_act_h_main > 8176 ||
            r->in_act_w_overlay < 68 ||
            r->in_act_w_overlay > 8176 ||
            r->in_act_h_overlay > 8176 ||
            r->out_act_w < 68) {
            r->is_rga1_2_used = 1;
        }
    }
    if (r->is_rga1_2_used && !r->has_rga1_2) {
        av_log(ctx, AV_LOG_ERROR, "RGA1/RGA2 is requested but not available\n");
        return AVERROR(ENOSYS);
    }
    if (r->is_rga1_2_used && (r->in_10b_uncompact_msb_main || r->out_10b_uncompact_msb)) {
        av_log(ctx, AV_LOG_ERROR, "%s is not supported if RGA1/RGA2 is requested\n",
               av_get_pix_fmt_name(AV_PIX_FMT_P010));
        return AVERROR(ENOSYS);
    }
    if (r->is_rga1_2_used && r->out_fmt == AV_PIX_FMT_NV15) {
        av_log(ctx, AV_LOG_ERROR, "%s as output is not supported if RGA1/RGA2 is requested\n",
               av_get_pix_fmt_name(AV_PIX_FMT_NV15));
        return AVERROR(ENOSYS);
    }

    r->is_offset_valid = (r->overlay_x <= r->in_act_w_main - 2) &&
        (r->overlay_y <= r->in_act_h_main - 2);

    r->is_resizing = r->in_act_w_main != r->out_act_w || r->in_act_h_main != r->out_act_h;

    return 0;
}

static av_cold int rgaoverlay_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    RGAOverlayContext *r = ctx->priv;
    AVFilterLink *inlink_main    = ctx->inputs[0];
    AVFilterLink *inlink_overlay = ctx->inputs[1];
    AVHWFramesContext *frames_ctx_main, *frames_ctx_overlay;
    enum AVPixelFormat in_format_main, in_format_overlay;
    enum AVPixelFormat out_format;
    int ret;

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
    in_format_overlay = frames_ctx_overlay->sw_format;

    if (!map_av_to_rga_format(in_format_main, &r->in_rga_fmt_main, 0)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported main input format: %s\n",
               av_get_pix_fmt_name(in_format_main));
        return AVERROR(ENOSYS);
    }
    if (!map_av_to_rga_format(in_format_overlay, &r->in_rga_fmt_overlay, 1)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported overlay input format: %s\n",
               av_get_pix_fmt_name(in_format_overlay));
        return AVERROR(ENOSYS);
    }
    if (!map_av_to_rga_format(out_format, &r->out_rga_fmt, 0)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported output format: %s\n",
               av_get_pix_fmt_name(out_format));
        return AVERROR(ENOSYS);
    }

    ret = set_format_info(ctx, in_format_main, in_format_overlay, out_format);
    if (ret < 0)
        return ret;

    ret = set_size_info(ctx, inlink_main, inlink_overlay, outlink);
    if (ret < 0)
        return ret;

    /* output buffer */
    ret = init_hwframe_ctx(r, frames_ctx_main->device_ref, outlink->w, outlink->h);
    if (ret < 0)
        return ret;

    outlink->hw_frames_ctx = av_buffer_ref(r->frames_ctx);
    if (!outlink->hw_frames_ctx)
        return AVERROR(ENOMEM);

    if (inlink_main->sample_aspect_ratio.num)
        outlink->sample_aspect_ratio = av_mul_q((AVRational){outlink->h * inlink_main->w,
                                                             outlink->w * inlink_main->h},
                                                inlink_main->sample_aspect_ratio);
    else
        outlink->sample_aspect_ratio = inlink_main->sample_aspect_ratio;

    av_log(ctx, AV_LOG_VERBOSE, "w:%d h:%d fmt:%s + w:%d h:%d fmt:%s -> w:%d h:%d fmt:%s\n",
           inlink_main->w, inlink_main->h, av_get_pix_fmt_name(r->in_fmt_main),
           inlink_overlay->w, inlink_overlay->h, av_get_pix_fmt_name(r->in_fmt_overlay),
           outlink->w, outlink->h, av_get_pix_fmt_name(r->out_fmt));

    /* overlay tmp buffer */
    if (r->is_offset_valid) {
        ret = init_hwframe_ctx1(r, frames_ctx_main->device_ref, inlink_main->w, inlink_main->h);
        if (ret < 0)
            return ret;
    }

    /* output tmp buffer */
    ret = init_hwframe_ctx2(r, frames_ctx_main->device_ref, inlink_main->w, inlink_main->h);
    if (ret < 0)
        return ret;

    ret = ff_framesync_init_dualinput(&r->fs, ctx);
    if (ret < 0)
        return ret;

    r->fs.time_base = outlink->time_base;

    return ff_framesync_configure(&r->fs);
}

static int rgaoverlay_overlay_composite(AVFilterContext *ctx,
                                        AVFrame *out, AVFrame *in_main, AVFrame *in_overlay)
{
    RGAOverlayContext *r = ctx->priv;
    const AVDRMFrameDescriptor *in_drm_desc_main;
    const AVDRMFrameDescriptor *in_drm_desc_overlay;
    const AVDRMFrameDescriptor *out_drm_desc;
    rga_info_t src = { .mmuFlag = 1, .format = r->in_rga_fmt_main, };
    rga_info_t dst = { .mmuFlag = 1, .format = r->out_rga_fmt, };
    rga_info_t pat = { .mmuFlag = 1, .format = r->in_rga_fmt_overlay, };
    rga_info_t pat_tmp1 = { .mmuFlag = 1, .format = r->in_rga_fmt_overlay, };
    rga_info_t pat_tmp2 = { .mmuFlag = 1, .format = r->in_rga_fmt_overlay, };
    rga_info_t *pat_p = NULL;
    rga_info_t dst_tmp1 = { .mmuFlag = 1, .format = r->out_rga_fmt, };
    rga_info_t *dst_p = NULL;
    int w_stride_src = 0, h_stride_src = 0;
    int w_stride_dst = 0, h_stride_dst = 0;
    int w_stride_pat = 0, h_stride_pat = 0;
    int do_overlay = in_overlay && r->is_offset_valid;
    int ret;

    if (!in_main || !out)
        return AVERROR(EINVAL);

    in_drm_desc_main = (AVDRMFrameDescriptor *)in_main->data[0];
    out_drm_desc     = (AVDRMFrameDescriptor *)out->data[0];
    if (!in_drm_desc_main || !out_drm_desc)
        return AVERROR(ENOMEM);

    if (in_drm_desc_main->nb_objects != 1 ||
        in_drm_desc_main->nb_layers != 1 ||
        out_drm_desc->nb_objects != 1 ||
        out_drm_desc->nb_layers != 1) {
        av_log(ctx, AV_LOG_ERROR, "RGA only supports single DRM object/layer\n");
        return AVERROR(EINVAL);
    }
    src.fd = in_drm_desc_main->objects[0].fd;
    dst.fd = out_drm_desc->objects[0].fd;

    if (do_overlay) {
        in_drm_desc_overlay = (AVDRMFrameDescriptor *)in_overlay->data[0];

        if (!in_drm_desc_overlay)
            return AVERROR(ENOMEM);
        if (in_drm_desc_overlay->nb_objects != 1 ||
            in_drm_desc_overlay->nb_layers != 1) {
            av_log(ctx, AV_LOG_ERROR, "RGA only supports single DRM object/layer\n");
            return AVERROR(EINVAL);
        }
        pat.fd = in_drm_desc_overlay->objects[0].fd;
    }

    ret = get_pixel_stride(&in_drm_desc_main->objects[0],
                           &in_drm_desc_main->layers[0],
                           (r->in_desc_main->flags & AV_PIX_FMT_FLAG_RGB),
                           (r->in_desc_main->flags & AV_PIX_FMT_FLAG_PLANAR),
                           r->in_bytes_pp_main, &w_stride_src, &h_stride_src);
    if (ret < 0 || !w_stride_src || !h_stride_src)
        return AVERROR(EINVAL);

    ret = get_pixel_stride(&out_drm_desc->objects[0],
                           &out_drm_desc->layers[0],
                           (r->out_desc->flags & AV_PIX_FMT_FLAG_RGB),
                           (r->out_desc->flags & AV_PIX_FMT_FLAG_PLANAR),
                           r->out_bytes_pp, &w_stride_dst, &h_stride_dst);
    if (ret < 0 || !w_stride_dst || !h_stride_dst)
        return AVERROR(EINVAL);

    if (do_overlay) {
        ret = get_pixel_stride(&in_drm_desc_overlay->objects[0],
                               &in_drm_desc_overlay->layers[0],
                               (r->in_desc_overlay->flags & AV_PIX_FMT_FLAG_RGB),
                               (r->in_desc_overlay->flags & AV_PIX_FMT_FLAG_PLANAR),
                               r->in_bytes_pp_overlay, &w_stride_pat, &h_stride_pat);
        if (ret < 0 || !w_stride_pat || !h_stride_pat)
            return AVERROR(EINVAL);
    }

    r->is_rga1_2_used = r->is_rga1_2_used ||
        !is_pixel_stride_rga3_compat(w_stride_src, h_stride_src, r->in_rga_fmt_main) ||
        !is_pixel_stride_rga3_compat(w_stride_pat, h_stride_pat, r->in_rga_fmt_overlay);

    if (r->is_rga1_2_used && r->out_fmt == AV_PIX_FMT_NV15) {
        av_log(ctx, AV_LOG_ERROR, "%s as output is not supported if RGA1/RGA2 is requested\n",
               av_get_pix_fmt_name(AV_PIX_FMT_NV15));
        return AVERROR(ENOSYS);
    }

    if (r->in_10b_uncompact_msb_main) {
        src.is_10b_compact = 1;
        src.is_10b_endian  = 1;
    }
    if (r->out_10b_uncompact_msb) {
        dst.is_10b_compact = 1;
        dst.is_10b_endian  = 1;
    }
    src.blend = do_overlay ? r->in_blend_mode : 0;
    dst.color_space_mode = r->out_csc_mode;
    dst.core = r->scheduler_core;

    rga_set_rect(&src.rect, 0, 0,
                 r->in_act_w_main, r->in_act_h_main,
                 w_stride_src, h_stride_src, r->in_rga_fmt_main);
    rga_set_rect(&dst.rect, 0, 0,
                 r->out_act_w, r->out_act_h,
                 w_stride_dst, h_stride_dst, r->out_rga_fmt);

    av_log(ctx, AV_LOG_DEBUG, "RGA src | fd:%d mmu:%d | x:%d y:%d w:%d h:%d ws:%d hs:%d fmt:0x%x\n",
           src.fd, src.mmuFlag, src.rect.xoffset, src.rect.yoffset,
           src.rect.width, src.rect.height, src.rect.wstride, src.rect.hstride, (src.rect.format >> 8));
    av_log(ctx, AV_LOG_DEBUG, "RGA dst | fd:%d mmu:%d | x:%d y:%d w:%d h:%d ws:%d hs:%d fmt:0x%x\n",
           dst.fd, dst.mmuFlag, dst.rect.xoffset, dst.rect.yoffset,
           dst.rect.width, dst.rect.height, dst.rect.wstride, dst.rect.hstride, (dst.rect.format >> 8));

    if (do_overlay) {
        const AVDRMFrameDescriptor *tmp1_drm_desc;
        const AVDRMFrameDescriptor *tmp2_drm_desc;

        if (!r->tmp_frame1 || !r->tmp_frame2)
            return AVERROR_BUG;

        tmp1_drm_desc = (AVDRMFrameDescriptor *)r->tmp_frame1->data[0];
        tmp2_drm_desc = (AVDRMFrameDescriptor *)r->tmp_frame2->data[0];
        pat_tmp1.fd = tmp1_drm_desc->objects[0].fd;
        pat_tmp2.fd = tmp2_drm_desc->objects[0].fd;

        pat_p = &pat;
        pat.rect.wstride = w_stride_pat;
        pat.rect.hstride = h_stride_pat;

        /* copy PAT to a new image with the same size of SRC */
        if (r->in_act_w_overlay != r->in_act_w_main ||
            r->in_act_h_overlay != r->in_act_h_main) {
            rga_set_rect(&pat.rect, 0, 0,
                         r->in_act_w_overlay, r->in_act_h_overlay,
                         w_stride_pat, h_stride_pat, r->in_rga_fmt_overlay);
            rga_set_rect(&pat_tmp1.rect, 0, 0,
                         FFMIN(r->in_act_w_overlay, r->in_act_w_main), FFMIN(r->in_act_h_overlay, r->in_act_h_main),
                         r->w_stride_tmp1, r->h_stride_tmp1, r->in_rga_fmt_overlay);
            pat_tmp1.core = r->scheduler_core;

            if ((ret = c_RkRgaBlit(&pat, &pat_tmp1, NULL)) != 0) {
                av_log(ctx, AV_LOG_ERROR, "RGA copy failed: %d\n", ret);
                return AVERROR_EXTERNAL;
            }
            pat_p = &pat_tmp1;
        }

        /* translate PAT from top-left to (x,y) */
        if (r->overlay_x > 0 || r->overlay_y > 0) {
            rga_set_rect(&pat_p->rect, 0, 0,
                         (r->in_act_w_main - r->overlay_x), (r->in_act_h_main - r->overlay_y),
                         r->w_stride_tmp1, r->h_stride_tmp1, r->in_rga_fmt_overlay);
            rga_set_rect(&pat_tmp2.rect, r->overlay_x, r->overlay_y,
                         (r->in_act_w_main - r->overlay_x), (r->in_act_h_main - r->overlay_y),
                         r->w_stride_tmp1, r->h_stride_tmp1, r->in_rga_fmt_overlay);
            pat_tmp2.core = r->scheduler_core;

            if ((ret = c_RkRgaBlit(pat_p, &pat_tmp2, NULL)) != 0) {
                av_log(ctx, AV_LOG_ERROR, "RGA translate failed: %d\n", ret);
                return AVERROR_EXTERNAL;
            }
            pat_p = &pat_tmp2;
        }

        rga_set_rect(&pat_p->rect, 0, 0,
                     r->in_act_w_main, r->in_act_h_main,
                     pat_p->rect.wstride, pat_p->rect.hstride, r->in_rga_fmt_overlay);

        av_log(ctx, AV_LOG_DEBUG, "RGA pat | fd:%d mmu:%d | x:%d y:%d w:%d h:%d ws:%d hs:%d fmt:0x%x\n",
               pat_p->fd, pat_p->mmuFlag, pat_p->rect.xoffset, pat_p->rect.yoffset,
               pat_p->rect.width, pat_p->rect.height, pat_p->rect.wstride, pat_p->rect.hstride, (pat_p->rect.format >> 8));
    }

    dst_p = &dst;
    if (do_overlay && r->is_rga1_2_used && r->is_resizing) {
        const AVDRMFrameDescriptor *tmp3_drm_desc;

        if (!r->tmp_frame3)
            return AVERROR_BUG;

        tmp3_drm_desc = (AVDRMFrameDescriptor *)r->tmp_frame3->data[0];
        dst_tmp1.fd = tmp3_drm_desc->objects[0].fd;

        dst_p = &dst_tmp1;
        rga_set_rect(&dst_tmp1.rect, 0, 0,
                     r->in_act_w_main, r->in_act_h_main,
                     r->w_stride_tmp3, r->h_stride_tmp3, r->out_rga_fmt);
        dst_tmp1.core = r->scheduler_core;
    }

    if ((ret = c_RkRgaBlit(&src, dst_p, pat_p)) != 0) {
        av_log(ctx, AV_LOG_ERROR, "RGA composite failed: %d\n", ret);
        return AVERROR_EXTERNAL;
    }

    /* the DST of RGA1/RGA2 can't do composite and resize in one shot */
    if (do_overlay && r->is_rga1_2_used && r->is_resizing) {
        if ((ret = c_RkRgaBlit(&dst_tmp1, &dst, NULL)) != 0) {
            av_log(ctx, AV_LOG_ERROR, "RGA resize failed: %d\n", ret);
            return AVERROR_EXTERNAL;
        }
    }

    return 0;
}

static int rgaoverlay_overlay(AVFilterContext *ctx,
                              AVFrame *out, AVFrame *in_main, AVFrame *in_overlay)
{
    RGAOverlayContext *r = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *src  = in_main;
    AVFrame *src1 = in_overlay;
    int ret;

    ret = rgaoverlay_overlay_composite(ctx, r->frame, src, src1);
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

    ret = av_frame_copy_props(out, in_main);
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

static int rgaoverlay_on_event(FFFrameSync *fs)
{
    AVFilterContext *ctx = fs->parent;
    AVFilterLink *inlink_main = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *in_main = NULL, *in_overlay = NULL;
    AVFrame *out = NULL;
    int ret = 0;

    ret = ff_framesync_dualinput_get(fs, &in_main, &in_overlay);
    if (ret < 0)
        return ret;

    if (!in_main)
        return AVERROR_BUG;

    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    ret = rgaoverlay_overlay(ctx, out, in_main, in_overlay);
    if (ret < 0)
        goto fail;

    av_reduce(&out->sample_aspect_ratio.num, &out->sample_aspect_ratio.den,
              (int64_t)in_main->sample_aspect_ratio.num * outlink->h * inlink_main->w,
              (int64_t)in_main->sample_aspect_ratio.den * outlink->w * inlink_main->h,
              INT_MAX);

    av_frame_free(&in_main);
    return ff_filter_frame(outlink, out);
fail:
    if (out)
        av_frame_free(&out);
    if (in_main)
        av_frame_free(&in_main);
    if (in_overlay)
        av_frame_free(&in_overlay);
    return ret;
}

static av_cold int rgaoverlay_init(AVFilterContext *ctx)
{
    RGAOverlayContext *r = ctx->priv;
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

    r->fs.on_event = &rgaoverlay_on_event;

    r->frame = av_frame_alloc();
    if (!r->frame)
        return AVERROR(ENOMEM);

    r->tmp_frame = av_frame_alloc();
    if (!r->tmp_frame)
        return AVERROR(ENOMEM);

    r->tmp_frame1 = av_frame_alloc();
    if (!r->tmp_frame1)
        return AVERROR(ENOMEM);

    r->tmp_frame2 = av_frame_alloc();
    if (!r->tmp_frame2)
        return AVERROR(ENOMEM);

    r->tmp_frame3 = av_frame_alloc();
    if (!r->tmp_frame3)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold void rgaoverlay_uninit(AVFilterContext *ctx)
{
    RGAOverlayContext *r = ctx->priv;

    ff_framesync_uninit(&r->fs);

    av_frame_free(&r->frame);
    av_frame_free(&r->tmp_frame);
    av_frame_free(&r->tmp_frame1);
    av_frame_free(&r->tmp_frame2);
    av_frame_free(&r->tmp_frame3);
}

static int rgaoverlay_activate(AVFilterContext *ctx)
{
    RGAOverlayContext *r = ctx->priv;

    return ff_framesync_activate(&r->fs);
}

#define OFFSET(x) offsetof(RGAOverlayContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)

static const AVOption rgaoverlay_options[] = {
    { "x", "Set horizontal offset", OFFSET(overlay_x), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, .flags = FLAGS },
    { "y", "Set vertical offset",   OFFSET(overlay_y), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, .flags = FLAGS },
    { "alpha", "Overlay global alpha", OFFSET(global_alpha), AV_OPT_TYPE_INT, { .i64 = 255 }, 0, 255, .flags = FLAGS },
    { "eof_action", "Action to take when encountering EOF from secondary input ",
        OFFSET(fs.opt_eof_action), AV_OPT_TYPE_INT, { .i64 = EOF_ACTION_REPEAT },
        EOF_ACTION_REPEAT, EOF_ACTION_PASS, .flags = FLAGS, "eof_action" },
        { "repeat", "Repeat the previous frame.",   0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_REPEAT }, .flags = FLAGS, "eof_action" },
        { "endall", "End both streams.",            0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_ENDALL }, .flags = FLAGS, "eof_action" },
        { "pass",   "Pass through the main input.", 0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_PASS },   .flags = FLAGS, "eof_action" },
    { "shortest", "Force termination when the shortest input terminates", OFFSET(fs.opt_shortest), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { "repeatlast", "Repeat overlay of the last overlay frame", OFFSET(fs.opt_repeatlast), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { "w", "Output video width",  OFFSET(w_expr), AV_OPT_TYPE_STRING, { .str = "iw" }, .flags = FLAGS },
    { "h", "Output video height", OFFSET(h_expr), AV_OPT_TYPE_STRING, { .str = "ih" }, .flags = FLAGS },
    { "format", "Output video pixel format", OFFSET(format), AV_OPT_TYPE_PIXEL_FMT, { .i64 = AV_PIX_FMT_NONE }, INT_MIN, INT_MAX, .flags = FLAGS },
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
