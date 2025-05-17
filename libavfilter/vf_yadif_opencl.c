/*
 * Copyright (c) 2025 NyanMisaka
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
 
#include "libavutil/avassert.h"
#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "filters.h"
#include "opencl.h"
#include "opencl_source.h"
#include "video.h"
#include "yadif.h"

typedef struct DeintOpenCLContext {
    YADIFContext yadif;

    OpenCLFilterContext ocf;

    cl_command_queue command_queue;
    cl_kernel        kernel;

    int initialised;
} DeintOpenCLContext;

static int deint_opencl_kernel_init(AVFilterContext *avctx)
{
    DeintOpenCLContext *ctx = avctx->priv;
    cl_int cle;
    int err;

    err = ff_opencl_filter_load_program2(avctx, &ctx->ocf, &ff_source_yadif_cl, 1);
    if (err < 0)
        goto fail;

    ctx->command_queue = clCreateCommandQueue(ctx->ocf.hwctx->context,
                                              ctx->ocf.hwctx->device_id,
                                              0, &cle);
    if (!ctx->command_queue) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create OpenCL command queue: %d.\n", cle);
        err = AVERROR(EIO);
        goto fail;
    }

    ctx->kernel = clCreateKernel(ctx->ocf.program, "yadif", &cle);
    if (!ctx->kernel) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create kernel: %d.\n", cle);
        err = AVERROR(EIO);
        goto fail;
    }

    ctx->initialised = 1;
    return 0;

fail:
    if (ctx->command_queue)
        clReleaseCommandQueue(ctx->command_queue);
    if (ctx->kernel)
        clReleaseKernel(ctx->kernel);
    return err;
}

static void deint_opencl_filter(AVFilterContext *avctx, AVFrame *dst,
                                int parity, int tff)
{
    DeintOpenCLContext *ctx = avctx->priv;
    YADIFContext *y = &ctx->yadif;
    size_t global_work[2];
    size_t local_work[2] = { 16, 16 };
    int i, err;
    cl_int cle;

    if (!ctx->initialised) {
        err = deint_opencl_kernel_init(avctx);
        if (err < 0)
            goto fail;
    }

    for (i = 0; i < y->csp->nb_components; i++) {
        int pixel_size, channels;
        int skip_spatial_check = !!(y->mode & 2);
        int is_second_field = !(parity ^ tff);
        const AVComponentDescriptor *comp = &y->csp->comp[i];

        if (comp->plane < i) {
            // We process planes as a whole, so don't reprocess
            // them for additional components
            continue;
        }

        pixel_size = (comp->depth + comp->shift) / 8;
        channels = comp->step / pixel_size;
        if (pixel_size > 2 || channels > 2) {
            av_log(ctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n", y->csp->name);
            goto fail;
        }
        if (!(pixel_size == 1 || pixel_size == 2)) {
            av_log(ctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n", y->csp->name);
            goto fail;
        }
        av_log(ctx, AV_LOG_TRACE,
               "Deinterlacing plane %d: pixel_size: %d channels: %d\n",
               comp->plane, pixel_size, channels);

        if (!dst->data[i] || !y->prev->data[i] || !y->cur->data[i] || !y->next->data[i]) {
            err = AVERROR(EIO);
            goto fail;
        }

        CL_SET_KERNEL_ARG(ctx->kernel, 0, cl_mem, &dst->data[i]);
        CL_SET_KERNEL_ARG(ctx->kernel, 1, cl_mem, &y->prev->data[i]);
        CL_SET_KERNEL_ARG(ctx->kernel, 2, cl_mem, &y->cur->data[i]);
        CL_SET_KERNEL_ARG(ctx->kernel, 3, cl_mem, &y->next->data[i]);
        CL_SET_KERNEL_ARG(ctx->kernel, 4, cl_int, &channels);
        CL_SET_KERNEL_ARG(ctx->kernel, 5, cl_int, &parity);
        CL_SET_KERNEL_ARG(ctx->kernel, 6, cl_int, &is_second_field);
        CL_SET_KERNEL_ARG(ctx->kernel, 7, cl_int, &skip_spatial_check);

        err = ff_opencl_filter_work_size_from_image(avctx, global_work, dst, i, 16);
        if (err < 0)
            goto fail;

        cle = clEnqueueNDRangeKernel(ctx->command_queue, ctx->kernel,
                                     2, NULL, global_work, local_work,
                                     0, NULL, NULL);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue kernel: %d.\n", cle);
    }

    clFinish(ctx->command_queue);
    return;

fail:
    clFinish(ctx->command_queue);
    av_frame_free(&dst);
    return;
}

static int deint_opencl_config_input(AVFilterLink *inlink)
{
    FilterLink            *l = ff_filter_link(inlink);
    AVFilterContext   *avctx = inlink->dst;
    DeintOpenCLContext  *ctx = avctx->priv;
    OpenCLFilterContext *ocf = &ctx->ocf;
    AVHWFramesContext *input_frames;

    if (!l->hw_frames_ctx) {
        av_log(avctx, AV_LOG_ERROR, "OpenCL filtering requires a "
               "hardware frames context on the input.\n");
        return AVERROR(EINVAL);
    }

    // Extract the device and default output format from the first input.
    if (avctx->inputs[0] != inlink)
        return 0;

    input_frames = (AVHWFramesContext*)l->hw_frames_ctx->data;
    if (input_frames->format != AV_PIX_FMT_OPENCL)
        return AVERROR(EINVAL);

    av_buffer_unref(&ocf->device_ref);

    ocf->device_ref = av_buffer_ref(input_frames->device_ref);
    if (!ocf->device_ref)
        return AVERROR(ENOMEM);

    ocf->device = (AVHWDeviceContext*)ocf->device_ref->data;
    ocf->hwctx  = ocf->device->hwctx;

    // Default output parameters match input parameters.
    if (ocf->output_format == AV_PIX_FMT_NONE)
        ocf->output_format = input_frames->sw_format;
    if (!ocf->output_width)
        ocf->output_width  = inlink->w;
    if (!ocf->output_height)
        ocf->output_height = inlink->h;

    return 0;
}

static int deint_opencl_config_output(AVFilterLink *outlink)
{
    AVFilterContext  *avctx = outlink->src;
    AVFilterLink    *inlink = avctx->inputs[0];
    FilterLink         *inl = ff_filter_link(inlink);
    DeintOpenCLContext *ctx = avctx->priv;
    YADIFContext *y = &ctx->yadif;
    AVHWFramesContext *in_frames_ctx;
    enum AVPixelFormat in_format;
    const AVPixFmtDescriptor *in_desc;
    int ret;

    if (!inl->hw_frames_ctx)
        return AVERROR(EINVAL);
    in_frames_ctx = (AVHWFramesContext*)inl->hw_frames_ctx->data;
    in_format     = in_frames_ctx->sw_format;
    in_desc       = av_pix_fmt_desc_get(in_format);
    ctx->ocf.output_format = in_format;
    ctx->ocf.output_width  = inlink->w;
    ctx->ocf.output_height = inlink->h;

    ret = ff_opencl_filter_config_output2(outlink, &ctx->ocf);
    if (ret < 0)
        return ret;

    y->csp = in_desc;
    y->filter = deint_opencl_filter;

    ret = ff_yadif_config_output_common(outlink);
    if (ret < 0)
        return ret;

    return 0;
}

static int deint_opencl_init(AVFilterContext *avctx)
{
    DeintOpenCLContext *ctx = avctx->priv;
    OpenCLFilterContext *ocf = &ctx->ocf;

    ocf->output_format = AV_PIX_FMT_NONE;

    return 0;
}

static av_cold void deint_opencl_uninit(AVFilterContext *avctx)
{
    DeintOpenCLContext *ctx = avctx->priv;
    OpenCLFilterContext *ocf = &ctx->ocf;
    cl_int cle;

    ff_yadif_uninit(avctx);

    if (ctx->kernel) {
        cle = clReleaseKernel(ctx->kernel);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel: %d.\n", cle);
    }

    if (ctx->command_queue) {
        cle = clReleaseCommandQueue(ctx->command_queue);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "command queue: %d.\n", cle);
    }

    ff_opencl_filter_uninit2(avctx, ocf);
}

#define OFFSET(x) offsetof(YADIFContext, x)
#define FLAGS (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM)
#define CONST(name, help, val, unit) { name, help, 0, AV_OPT_TYPE_CONST, { .i64 = val }, INT_MIN, INT_MAX, FLAGS, unit }

static const AVOption yadif_opencl_options[] = {
    { "mode",   "specify the interlacing mode", OFFSET(mode), AV_OPT_TYPE_INT, { .i64 = YADIF_MODE_SEND_FRAME }, 0, 3, FLAGS, .unit = "mode" },
    CONST("send_frame",           "send one frame for each frame",                                     YADIF_MODE_SEND_FRAME,           .unit = "mode"),
    CONST("send_field",           "send one frame for each field",                                     YADIF_MODE_SEND_FIELD,           .unit = "mode"),
    CONST("send_frame_nospatial", "send one frame for each frame, but skip spatial interlacing check", YADIF_MODE_SEND_FRAME_NOSPATIAL, .unit = "mode"),
    CONST("send_field_nospatial", "send one frame for each field, but skip spatial interlacing check", YADIF_MODE_SEND_FIELD_NOSPATIAL, .unit = "mode"),

    { "parity", "specify the assumed picture field parity", OFFSET(parity), AV_OPT_TYPE_INT, { .i64 = YADIF_PARITY_AUTO }, -1, 1, FLAGS, .unit = "parity" },
    CONST("tff",  "assume top field first",    YADIF_PARITY_TFF,  .unit = "parity"),
    CONST("bff",  "assume bottom field first", YADIF_PARITY_BFF,  .unit = "parity"),
    CONST("auto", "auto detect parity",        YADIF_PARITY_AUTO, .unit = "parity"),

    { "deint", "specify which frames to deinterlace", OFFSET(deint), AV_OPT_TYPE_INT, { .i64 = YADIF_DEINT_ALL }, 0, 1, FLAGS, .unit = "deint" },
    CONST("all",        "deinterlace all frames",                       YADIF_DEINT_ALL,        .unit = "deint"),
    CONST("interlaced", "only deinterlace frames marked as interlaced", YADIF_DEINT_INTERLACED, .unit = "deint"),

    { NULL }
};

AVFILTER_DEFINE_CLASS(yadif_opencl);

static const AVFilterPad deint_opencl_inputs[] = {
    {
        .name             = "default",
        .type             = AVMEDIA_TYPE_VIDEO,
        .filter_frame     = ff_yadif_filter_frame,
        .config_props     = &deint_opencl_config_input,
    },
};

static const AVFilterPad deint_opencl_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .request_frame = ff_yadif_request_frame,
        .config_props = &deint_opencl_config_output,
    },
};

const AVFilter ff_vf_yadif_opencl = {
    .name           = "yadif_opencl",
    .description    = NULL_IF_CONFIG_SMALL("Deinterlace (YADIF) the video through OpenCL."),
    .priv_size      = sizeof(DeintOpenCLContext),
    .priv_class     = &yadif_opencl_class,
    .init           = &deint_opencl_init,
    .uninit         = &deint_opencl_uninit,
    FILTER_INPUTS(deint_opencl_inputs),
    FILTER_OUTPUTS(deint_opencl_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_OPENCL),
    .flags          = AVFILTER_FLAG_HWDEVICE |
                      AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
