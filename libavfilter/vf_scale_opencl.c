/*
 * Copyright (c) 2026 NyanMisaka
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

#include <float.h>

#include "libavutil/avassert.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "opencl.h"
#include "opencl_source.h"
#include "scale_eval.h"
#include "video.h"
#include "dither_matrix.h"

#define OPENCL_SOURCE_NB 2

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUV420P16,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NV15,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P016,
};

enum {
    ALGO_DEFAULT,

    ALGO_NEIGHBOR,
    ALGO_BILINEAR,
    ALGO_BICUBIC,
    ALGO_LANCZOS,

    ALGO_NB,
};

typedef struct ScaleOpenCLContext {
    OpenCLFilterContext ocf;

    cl_command_queue command_queue;
    cl_mem           dither_image;
    cl_kernel        kernel;
    const char      *kernel_name;

    cl_program       program_nv15;
    cl_kernel        kernel_nv15;
    const char      *kernel_name_nv15;
    cl_kernel        kernel_nv15_semiplanar;

    AVBufferRef     *tmp_hwframes_ctx;
    AVFrame         *tmp_frame;

    char *w_expr,  *h_expr;
    int   dst_w,    dst_h;
    int   src_w,    src_h;
    int   passthrough;
    int   algorithm;
    float param;
    int   force_original_aspect_ratio;
    int   force_divisible_by;
    int   reset_sar;
    enum AVPixelFormat format;

    enum AVPixelFormat in_fmt, out_fmt;
    const AVPixFmtDescriptor *in_desc, *out_desc;
    int in_planes, out_planes;

    int   initialised;
} ScaleOpenCLContext;

static av_cold int init_tmp_hwframes_ctx(AVFilterContext *avctx,
                                         enum AVPixelFormat pix_fmt,
                                         int width, int height)
{
    ScaleOpenCLContext *ctx = avctx->priv;
    AVFilterLink       *inlink = avctx->inputs[0];
    FilterLink         *inl    = ff_filter_link(inlink);
    AVHWFramesContext  *hwfc_in;
    AVHWFramesContext  *hwfc_tmp;
    AVBufferRef        *hwfc_tmp_ref;
    AVHWDeviceContext  *device_ctx;
    AVBufferRef        *device_ref;
    int                 ret;

    if (!inl->hw_frames_ctx)
        return AVERROR(EINVAL);

    hwfc_in = (AVHWFramesContext *)inl->hw_frames_ctx->data;
    device_ref = hwfc_in->device_ref;
    device_ctx = (AVHWDeviceContext *)device_ref->data;

    if (!device_ctx || device_ctx->type != AV_HWDEVICE_TYPE_OPENCL) {
        if (avctx->hw_device_ctx) {
            device_ref = avctx->hw_device_ctx;
            device_ctx = (AVHWDeviceContext *)device_ref->data;
        }
        if (!device_ctx || device_ctx->type != AV_HWDEVICE_TYPE_OPENCL) {
            av_log(avctx, AV_LOG_ERROR, "No OpenCL hardware context provided\n");
            return AVERROR(EINVAL);
        }
    }

    hwfc_tmp_ref = av_hwframe_ctx_alloc(device_ref);
    if (!hwfc_tmp_ref)
        return AVERROR(ENOMEM);

    hwfc_tmp = (AVHWFramesContext *)hwfc_tmp_ref->data;
    hwfc_tmp->format    = AV_PIX_FMT_OPENCL;
    hwfc_tmp->sw_format = pix_fmt;
    hwfc_tmp->width     = width;
    hwfc_tmp->height    = height;

    ret = av_hwframe_ctx_init(hwfc_tmp_ref);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "Error creating frames_ctx for tmp frame: %d\n", ret);
        av_buffer_unref(&hwfc_tmp_ref);
        return ret;
    }

    av_buffer_unref(&ctx->tmp_hwframes_ctx);
    ctx->tmp_hwframes_ctx = hwfc_tmp_ref;

    return 0;
}

static int scale_opencl_init(AVFilterContext *avctx)
{
    ScaleOpenCLContext *ctx = avctx->priv;
    AVBPrint header;
    const char *opencl_sources[OPENCL_SOURCE_NB];
    cl_event event = NULL;
    cl_int cle;
    int err;

    av_bprint_init(&header, 512, AV_BPRINT_SIZE_UNLIMITED);

    if (ctx->in_planes > 2) {
        av_bprintf(&header, "#define NON_SEMI_PLANAR_IN\n");
    }
    if (ctx->out_planes > 2) {
        av_bprintf(&header, "#define NON_SEMI_PLANAR_OUT\n");
    }
    if (ctx->in_desc->comp[0].depth > ctx->out_desc->comp[0].depth) {
        av_bprintf(&header, "#define ENABLE_DITHER\n");
        av_bprintf(&header, "__constant float dither_size2 = %.1ff;\n", (float)(ff_fruit_dither_size * ff_fruit_dither_size));
        av_bprintf(&header, "__constant float dither_quantization = %.1ff;\n", (float)((1 << ctx->out_desc->comp[0].depth) - 1));
    }
    if (!isnan(ctx->param)) {
        av_bprintf(&header, "#define SCALE_PARAM\n");
        av_bprintf(&header, "__constant float scale_param = %.13ff;\n", ctx->param);
    }
    if (ctx->in_fmt == AV_PIX_FMT_NV15) {
        av_bprintf(&header, "#define BLIT_NV15\n");
        ctx->kernel_name_nv15 = "blit_nv15";

        opencl_sources[0] = header.str;
        opencl_sources[1] = ff_source_scale_cl;
        err = ff_opencl_filter_load_program(avctx, opencl_sources, OPENCL_SOURCE_NB);

        ctx->program_nv15 = ctx->ocf.program;
        if (err < 0)
            goto fail;
    }

    if (ctx->src_w == ctx->dst_w && ctx->src_h == ctx->dst_h) {
        av_bprintf(&header, "#define SCALE_BUILTIN\n");
        av_bprintf(&header, "#define SCALE_BUILTIN_NEIGHBOR\n");
        ctx->kernel_name = "scale_builtin";
    } else if (ctx->algorithm == ALGO_NEIGHBOR ||
               ctx->algorithm == ALGO_BILINEAR) {
        av_bprintf(&header, "#define SCALE_BUILTIN\n");
        av_bprintf(&header, "#define SCALE_BUILTIN_%s\n",
            ctx->algorithm == ALGO_BILINEAR ? "BILINEAR" : "NEIGHBOR");
        ctx->kernel_name = "scale_builtin";
    } else {
        av_bprintf(&header, "#define SCALE_CONVOLVE\n");
        av_bprintf(&header, "#define COEFFS_FUNCTION %s_coeffs\n",
            ctx->algorithm == ALGO_LANCZOS ? "lanczos" : "bicubic");
        ctx->kernel_name = "scale_convolve";
    }

    av_log(avctx, AV_LOG_DEBUG, "Generated OpenCL header:\n%s\n", header.str);
    opencl_sources[0] = header.str;
    opencl_sources[1] = ff_source_scale_cl;
    err = ff_opencl_filter_load_program(avctx, opencl_sources, OPENCL_SOURCE_NB);

    av_bprint_finalize(&header, NULL);
    if (err < 0)
        goto fail;

    ctx->command_queue = clCreateCommandQueue(ctx->ocf.hwctx->context,
                                              ctx->ocf.hwctx->device_id,
                                              0, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create OpenCL "
                     "command queue %d.\n", cle);

    if (ctx->in_desc->comp[0].depth > ctx->out_desc->comp[0].depth) {
        const size_t m_origin[3] = { 0 };
        const size_t m_region[3] = { ff_fruit_dither_size, ff_fruit_dither_size, 1 };
        const size_t m_row_pitch = ff_fruit_dither_size * sizeof(ff_fruit_dither_matrix[0]);

        cl_image_format image_format = {
            .image_channel_data_type = CL_UNORM_INT16,
            .image_channel_order     = CL_R,
        };
        cl_image_desc image_desc = {
            .image_type      = CL_MEM_OBJECT_IMAGE2D,
            .image_width     = ff_fruit_dither_size,
            .image_height    = ff_fruit_dither_size,
            .image_row_pitch = 0,
        };

        av_assert0(sizeof(ff_fruit_dither_matrix) ==
            sizeof(ff_fruit_dither_matrix[0]) * ff_fruit_dither_size * ff_fruit_dither_size);

        ctx->dither_image = clCreateImage(ctx->ocf.hwctx->context, CL_MEM_READ_ONLY,
                                          &image_format, &image_desc, NULL, &cle);
        if (!ctx->dither_image) {
            av_log(avctx, AV_LOG_ERROR, "Failed to create image for "
                   "dither matrix: %d.\n", cle);
            err = AVERROR(EIO);
            goto fail;
        }

        cle = clEnqueueWriteImage(ctx->command_queue,
                                  ctx->dither_image,
                                  CL_FALSE, m_origin, m_region,
                                  m_row_pitch, 0,
                                  ff_fruit_dither_matrix,
                                  0, NULL, &event);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue write of dither matrix image: %d.\n", cle);

        cle = clWaitForEvents(1, &event);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to wait for event completion: %d.\n", cle);
        if (event) {
            clReleaseEvent(event);
            event = NULL;
        }
    }

    ctx->kernel = clCreateKernel(ctx->ocf.program, ctx->kernel_name, &cle);
    if (!ctx->kernel) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create kernel: %d.\n", cle);
        err = AVERROR(EIO);
        goto fail;
    }

    if (ctx->kernel_name_nv15) {
        enum AVPixelFormat tmp_fmt;
        av_assert0(ctx->in_fmt == AV_PIX_FMT_NV15);

        ctx->kernel_nv15 = clCreateKernel(ctx->program_nv15, ctx->kernel_name_nv15, &cle);
        if (!ctx->kernel_nv15) {
            av_log(avctx, AV_LOG_ERROR, "Failed to create kernel %s: %d.\n", ctx->kernel_name_nv15, cle);
            err = AVERROR(EIO);
            goto fail;
        }
        ctx->kernel_nv15_semiplanar = clCreateKernel(ctx->ocf.program, ctx->kernel_name_nv15, &cle);
        if (!ctx->kernel_nv15_semiplanar) {
            av_log(avctx, AV_LOG_ERROR, "Failed to create kernel %s (tmp): %d.\n", ctx->kernel_name_nv15, cle);
            err = AVERROR(EIO);
            goto fail;
        }
        /* for scaling P010 compact: 1st pass blit & 2nd pass scale */
        tmp_fmt = ctx->out_desc->comp[0].depth <= 8 ? AV_PIX_FMT_NV12 : AV_PIX_FMT_P010;
        if ((err = init_tmp_hwframes_ctx(avctx, tmp_fmt, ctx->src_w, ctx->src_h)) < 0)
            goto fail;

        if (ctx->tmp_frame)
            av_frame_free(&ctx->tmp_frame);

        ctx->tmp_frame = av_frame_alloc();
        if (!ctx->tmp_frame) {
            err = AVERROR(ENOMEM);
            goto fail;
        }
        if ((err = av_hwframe_get_buffer(ctx->tmp_hwframes_ctx, ctx->tmp_frame, 0)) < 0) {
            av_buffer_unref(&ctx->tmp_hwframes_ctx);
            goto fail;
        }
    }

    ctx->initialised = 1;
    return 0;

fail:
    av_bprint_finalize(&header, NULL);
    if (event)
        clReleaseEvent(event);
    if (ctx->kernel)
        clReleaseKernel(ctx->kernel);
    if (ctx->kernel_nv15)
        clReleaseKernel(ctx->kernel_nv15);
    if (ctx->program_nv15)
        clReleaseProgram(ctx->program_nv15);
    if (ctx->dither_image)
        clReleaseMemObject(ctx->dither_image);
    if (ctx->command_queue)
        clReleaseCommandQueue(ctx->command_queue);
    return err;
}

static int format_is_supported(enum AVPixelFormat fmt)
{
    for (int i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

static int scale_opencl_config_output(AVFilterLink *outlink)
{
    FilterLink        *outl = ff_filter_link(outlink);
    AVFilterContext  *avctx = outlink->src;
    AVFilterLink    *inlink = avctx->inputs[0];
    FilterLink         *inl = ff_filter_link(inlink);
    ScaleOpenCLContext *ctx = avctx->priv;
    AVHWFramesContext *in_frames_ctx;
    enum AVPixelFormat in_format;
    enum AVPixelFormat out_format;
    const AVPixFmtDescriptor *in_desc;
    const AVPixFmtDescriptor *out_desc;
    double w_adj = 1.0;
    int ret;

    if (!inl->hw_frames_ctx)
        return AVERROR(EINVAL);

    in_frames_ctx = (AVHWFramesContext*)inl->hw_frames_ctx->data;
    in_format     = in_frames_ctx->sw_format;
    out_format    = (ctx->format == AV_PIX_FMT_NONE) ? in_format : ctx->format;
    in_desc       = av_pix_fmt_desc_get(in_format);
    out_desc      = av_pix_fmt_desc_get(out_format);

    if (!format_is_supported(in_format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format: %s\n",
               av_get_pix_fmt_name(in_format));
        return AVERROR(ENOSYS);
    }
    if (!format_is_supported(out_format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported output format: %s\n",
               av_get_pix_fmt_name(out_format));
        return AVERROR(ENOSYS);
    }

    ctx->in_fmt     = in_format;
    ctx->out_fmt    = out_format;
    ctx->in_desc    = in_desc;
    ctx->out_desc   = out_desc;
    ctx->in_planes  = av_pix_fmt_count_planes(ctx->in_fmt);
    ctx->out_planes = av_pix_fmt_count_planes(ctx->out_fmt);
    ctx->ocf.output_format = out_format;

    if ((ret = ff_scale_eval_dimensions(ctx,
                                        ctx->w_expr, ctx->h_expr,
                                        inlink, outlink,
                                        &ctx->dst_w, &ctx->dst_h)) < 0)
        return ret;

    ctx->dst_w = ctx->dst_w < 0 ? -FFALIGN(-ctx->dst_w, 2) : FFALIGN(ctx->dst_w, 2);
    ctx->dst_h = ctx->dst_h < 0 ? -FFALIGN(-ctx->dst_h, 2) : FFALIGN(ctx->dst_h, 2);

    if (ctx->reset_sar)
        w_adj = inlink->sample_aspect_ratio.num ?
        (double)inlink->sample_aspect_ratio.num / inlink->sample_aspect_ratio.den : 1.0;

    if (ctx->force_divisible_by > 2)
        ctx->force_divisible_by = FFALIGN(ctx->force_divisible_by, 2);

    ff_scale_adjust_dimensions(inlink, &ctx->dst_w, &ctx->dst_h,
                               ctx->force_original_aspect_ratio, ctx->force_divisible_by, w_adj);

    if (((int64_t)ctx->dst_h * inlink->w) > INT_MAX ||
        ((int64_t)ctx->dst_w * inlink->h) > INT_MAX)
        av_log(ctx, AV_LOG_ERROR, "Rescaled value for width or height is too big.\n");

    ctx->src_w = inlink->w;
    ctx->src_h = inlink->h;
    ctx->ocf.output_width  = ctx->dst_w;
    ctx->ocf.output_height = ctx->dst_h;

    if (ctx->passthrough &&
        ctx->src_w == ctx->dst_w &&
        ctx->src_h == ctx->dst_h && ctx->in_fmt == ctx->out_fmt) {
        av_buffer_unref(&outl->hw_frames_ctx);
        outl->hw_frames_ctx = av_buffer_ref(inl->hw_frames_ctx);
        if (!outl->hw_frames_ctx)
            return AVERROR(ENOMEM);
        return 0;
    } else {
        ctx->passthrough = 0;
        if (out_format == AV_PIX_FMT_NV15) {
            av_log(ctx, AV_LOG_ERROR, "Unsupported output format: %s\n",
                   av_get_pix_fmt_name(out_format));
            return AVERROR(ENOSYS);
        }
    }

    ret = ff_opencl_filter_config_output(outlink);
    if (ret < 0)
        return ret;

    if (ctx->reset_sar)
        outlink->sample_aspect_ratio = (AVRational){1, 1};
    else if (inlink->sample_aspect_ratio.num)
        outlink->sample_aspect_ratio = av_mul_q((AVRational){outlink->h * inlink->w,
                                                             outlink->w * inlink->h},
                                                inlink->sample_aspect_ratio);
    else
        outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;

    return 0;
}

static int scale_opencl_run_kernel(AVFilterLink *inlink,
                                   AVFrame *input, AVFrame *output,
                                   int in_planes, int out_planes,
                                   int blit_nv15, int blit_1pass)
{
    AVFilterContext  *avctx = inlink->dst;
    ScaleOpenCLContext *ctx = avctx->priv;
    size_t global_work[2];
    cl_kernel kernel_nv15 = blit_1pass ? ctx->kernel_nv15_semiplanar : ctx->kernel_nv15;
    cl_kernel kernel = blit_nv15 ? kernel_nv15 : ctx->kernel;
    cl_int4 crop_whxy;
    cl_int cle;
    int err, idx_arg = 0;

    if (!output->data[0] || !input->data[0] ||
        !output->data[1] || !input->data[1]) {
        err = AVERROR(EIO);
        goto fail;
    }
    if (out_planes > 2 && !output->data[2]) {
        err = AVERROR(EIO);
        goto fail;
    }
    if (in_planes > 2 && !input->data[2]) {
        err = AVERROR(EIO);
        goto fail;
    }

    CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &output->data[0]);
    CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &input->data[0]);
    CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &output->data[1]);
    CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &input->data[1]);

    if (out_planes > 2) {
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &output->data[2]);
    }
    if (in_planes > 2) {
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &input->data[2]);
    }
    if (ctx->dither_image) {
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &ctx->dither_image);
    }
    if (!blit_nv15) {
        // scale_builtin, scale_convolve
        crop_whxy.s[0] = (input->width - input->crop_right) - input->crop_left;
        crop_whxy.s[1] = (input->height - input->crop_bottom) - input->crop_top;
        crop_whxy.s[2] = input->crop_left;
        crop_whxy.s[3] = input->crop_top;

        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_int4, &crop_whxy);
    }

    err = ff_opencl_filter_work_size_from_image(avctx, global_work, output, 1, 0);
    if (err < 0)
        return err;

    av_log(avctx, AV_LOG_DEBUG, "Run kernel %s (%zu x %zu).\n",
           (blit_nv15 ? ctx->kernel_name_nv15 : ctx->kernel_name),
           global_work[0], global_work[1]);

    cle = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL,
                                 global_work, NULL, 0, NULL, NULL);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue kernel: %d.\n", cle);

    return 0;

fail:
    return err;
}

static int scale_opencl_filter_frame(AVFilterLink *inlink, AVFrame *input)
{
    AVFilterContext     *avctx = inlink->dst;
    AVFilterLink      *outlink = avctx->outputs[0];
    ScaleOpenCLContext    *ctx = avctx->priv;
    AVFrame *output = NULL;
    int has_crop = input->crop_left || input->crop_right ||
                   input->crop_top || input->crop_bottom;
    int has_scale = !(ctx->src_w == ctx->dst_w && ctx->src_h == ctx->dst_h);
    cl_int cle;
    int err;

    av_log(ctx, AV_LOG_DEBUG, "Filter input: %s, %ux%u (%"PRId64").\n",
           av_get_pix_fmt_name(input->format),
           input->width, input->height, input->pts);

    if (ctx->passthrough && !has_scale && ctx->in_fmt == ctx->out_fmt)
        return ff_filter_frame(outlink, input);

    if (!input->hw_frames_ctx)
        return AVERROR(EINVAL);

    if (!ctx->initialised) {
        err = scale_opencl_init(avctx);
        if (err < 0)
            goto fail;
    }

    output = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!output) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    err = av_frame_copy_props(output, input);
    if (err < 0)
        goto fail;
    output->width  = outlink->w;
    output->height = outlink->h;
    if (output->width != input->width || output->height != input->height) {
        av_frame_side_data_remove_by_props(&output->side_data, &output->nb_side_data,
                                           AV_SIDE_DATA_PROP_SIZE_DEPENDENT);
    }
    if (ctx->reset_sar) {
        output->sample_aspect_ratio = (AVRational){1, 1};
    } else {
        av_reduce(&output->sample_aspect_ratio.num, &output->sample_aspect_ratio.den,
                  (int64_t)input->sample_aspect_ratio.num * outlink->h * inlink->w,
                  (int64_t)input->sample_aspect_ratio.den * outlink->w * inlink->h,
                  INT_MAX);
    }

    if (ctx->in_fmt == AV_PIX_FMT_NV15 && (has_crop || has_scale)) {
        /* for scaling P010 compact: 1st pass blit & 2nd pass scale */
        if ((err = av_frame_copy_props(ctx->tmp_frame, input)) < 0)
            goto fail;
        if ((err = scale_opencl_run_kernel(inlink, input, ctx->tmp_frame,
                                           ctx->in_planes, ctx->in_planes, 1, 1)) < 0)
            goto fail;
        if ((err = scale_opencl_run_kernel(inlink, ctx->tmp_frame, output,
                                           ctx->in_planes, ctx->out_planes, 0, 0)) < 0)
            goto fail;
    } else {
        if ((err = scale_opencl_run_kernel(inlink, input, output,
                                           ctx->in_planes, ctx->out_planes,
                                           ctx->in_fmt == AV_PIX_FMT_NV15, 0)) < 0)
            goto fail;
    }

    cle = clFinish(ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue: %d.\n", cle);

    av_frame_free(&input);

    av_log(ctx, AV_LOG_DEBUG, "Filter output: %s, %ux%u (%"PRId64").\n",
           av_get_pix_fmt_name(output->format),
           output->width, output->height, output->pts);

    return ff_filter_frame(outlink, output);

fail:
    clFinish(ctx->command_queue);
    av_frame_free(&input);
    av_frame_free(&output);
    return err;
}

static av_cold void scale_opencl_uninit(AVFilterContext *avctx)
{
    ScaleOpenCLContext *ctx = avctx->priv;
    cl_int cle;

    if (ctx->tmp_frame)
        av_frame_free(&ctx->tmp_frame);

    if (ctx->tmp_hwframes_ctx)
        av_buffer_unref(&ctx->tmp_hwframes_ctx);

    if (ctx->kernel) {
        cle = clReleaseKernel(ctx->kernel);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel: %d.\n", cle);
    }

    if (ctx->kernel_nv15) {
        cle = clReleaseKernel(ctx->kernel_nv15);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel_nv15: %d.\n", cle);
    }

    if (ctx->program_nv15) {
        cle = clReleaseProgram(ctx->program_nv15);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "program_nv15: %d.\n", cle);
    }

    if (ctx->dither_image) {
        cle = clReleaseMemObject(ctx->dither_image);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
            "dither image: %d.\n", cle);
    }

    if (ctx->command_queue) {
        cle = clReleaseCommandQueue(ctx->command_queue);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "command queue: %d.\n", cle);
    }

    ff_opencl_filter_uninit(avctx);
}

static AVFrame *scale_opencl_get_video_buffer(AVFilterLink *inlink, int w, int h)
{
    ScaleOpenCLContext *ctx = inlink->dst->priv;

    return ctx->passthrough ? ff_null_get_video_buffer(inlink, w, h) :
                              ff_default_get_video_buffer(inlink, w, h);
}

#define OFFSET(x) offsetof(ScaleOpenCLContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption scale_opencl_options[] = {
    { "w",           "Output video width",                               OFFSET(w_expr),      AV_OPT_TYPE_STRING,    { .str = "iw" }, .flags = FLAGS },
    { "h",           "Output video height",                              OFFSET(h_expr),      AV_OPT_TYPE_STRING,    { .str = "ih" }, .flags = FLAGS },
    { "format",      "Output pixel format",                              OFFSET(format),      AV_OPT_TYPE_PIXEL_FMT, { .i64 = AV_PIX_FMT_NONE }, AV_PIX_FMT_NONE, INT_MAX, FLAGS, .unit = "fmt" },
    { "passthrough", "Do not process frames at all if parameters match", OFFSET(passthrough), AV_OPT_TYPE_BOOL,      { .i64 = 0 }, 0, 1, FLAGS },
    { "algo",        "Scaling algorithm",                                OFFSET(algorithm),   AV_OPT_TYPE_INT,       { .i64 = ALGO_DEFAULT }, 0, ALGO_NB-1, FLAGS, .unit = "algo" },
        { "neighbor",     "Nearest Neighbor", 0, AV_OPT_TYPE_CONST, { .i64 = ALGO_NEIGHBOR }, 0, 0, FLAGS, .unit = "algo" },
        { "bilinear",     "Bilinear",         0, AV_OPT_TYPE_CONST, { .i64 = ALGO_BILINEAR }, 0, 0, FLAGS, .unit = "algo" },
        { "bicubic",      "Bicubic",          0, AV_OPT_TYPE_CONST, { .i64 = ALGO_BICUBIC  }, 0, 0, FLAGS, .unit = "algo" },
        { "lanczos",      "Lanczos",          0, AV_OPT_TYPE_CONST, { .i64 = ALGO_LANCZOS  }, 0, 0, FLAGS, .unit = "algo" },
    { "param",       "Algorithm-Specific parameter",                     OFFSET(param),       AV_OPT_TYPE_FLOAT,     { .dbl = NAN }, -FLT_MAX, FLT_MAX, FLAGS },
    { "force_original_aspect_ratio", "Decrease or increase w/h if necessary to keep the original AR", OFFSET(force_original_aspect_ratio), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, SCALE_FORCE_OAR_NB-1, FLAGS, .unit = "force_oar" },
        { "disable",       NULL,              0, AV_OPT_TYPE_CONST, {.i64 = SCALE_FORCE_OAR_DISABLE  }, 0, 0, FLAGS, .unit = "force_oar" },
        { "decrease",      NULL,              0, AV_OPT_TYPE_CONST, {.i64 = SCALE_FORCE_OAR_DECREASE }, 0, 0, FLAGS, .unit = "force_oar" },
        { "increase",      NULL,              0, AV_OPT_TYPE_CONST, {.i64 = SCALE_FORCE_OAR_INCREASE }, 0, 0, FLAGS, .unit = "force_oar" },
    { "force_divisible_by", "Enforce that the output resolution is divisible by a defined integer when force_original_aspect_ratio is used", OFFSET(force_divisible_by), AV_OPT_TYPE_INT, { .i64 = 1 }, 1, 256, FLAGS },
    { "reset_sar",   "Reset SAR to 1 and scale to square pixels if scaling proportionally", OFFSET(reset_sar), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(scale_opencl);

static const AVFilterPad scale_opencl_inputs[] = {
    {
        .name             = "default",
        .type             = AVMEDIA_TYPE_VIDEO,
        .filter_frame     = &scale_opencl_filter_frame,
        .get_buffer.video = &scale_opencl_get_video_buffer,
        .config_props     = &ff_opencl_filter_config_input,
    },
};

static const AVFilterPad scale_opencl_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &scale_opencl_config_output,
    },
};

const FFFilter ff_vf_scale_opencl = {
    .p.name         = "scale_opencl",
    .p.description  = NULL_IF_CONFIG_SMALL("Scale the input video size through OpenCL."),
    .priv_size      = sizeof(ScaleOpenCLContext),
    .p.priv_class   = &scale_opencl_class,
    .init           = &ff_opencl_filter_init,
    .uninit         = &scale_opencl_uninit,
    FILTER_INPUTS(scale_opencl_inputs),
    FILTER_OUTPUTS(scale_opencl_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_OPENCL),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
    .p.flags        = AVFILTER_FLAG_HWDEVICE,
};
