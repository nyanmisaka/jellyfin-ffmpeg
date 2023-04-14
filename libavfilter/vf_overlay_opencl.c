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

#include "libavutil/log.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "framesync.h"
#include "internal.h"
#include "opencl.h"
#include "opencl_source.h"
#include "video.h"

static const enum AVPixelFormat supported_main_formats[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P016,
    AV_PIX_FMT_NONE,
};

static const enum AVPixelFormat supported_overlay_formats[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUVA420P,
    AV_PIX_FMT_NONE,
};

typedef struct OverlayOpenCLContext {
    OpenCLFilterContext ocf;

    enum AVPixelFormat in_fmt_main, in_fmt_overlay;
    const AVPixFmtDescriptor *in_desc_main, *in_desc_overlay;
    int in_planes_main, in_planes_overlay;

    int              initialised;
    cl_kernel        kernel;
    cl_kernel        kernel_pass;
    cl_kernel        kernel_uv;
    const char      *kernel_name;
    const char      *kernel_name_pass;
    const char      *kernel_name_uv;
    cl_command_queue command_queue;

    FFFrameSync      fs;

    int              x_subsample;
    int              y_subsample;
    int              alpha;

    int              x_position;
    int              y_position;

    int              opt_repeatlast;
    int              opt_shortest;
    int              opt_eof_action;
} OverlayOpenCLContext;

static int format_is_supported(const enum AVPixelFormat fmts[], enum AVPixelFormat fmt)
{
    for (int i = 0; fmts[i] != AV_PIX_FMT_NONE; i++)
        if (fmts[i] == fmt)
            return 1;
    return 0;
}

static int formats_match(const enum AVPixelFormat fmt_main, const enum AVPixelFormat fmt_overlay) {
    switch(fmt_main) {
    case AV_PIX_FMT_NV12:
    case AV_PIX_FMT_P010:
    case AV_PIX_FMT_P016:
        return fmt_overlay == AV_PIX_FMT_NV12 ||
               fmt_overlay == AV_PIX_FMT_YUV420P ||
               fmt_overlay == AV_PIX_FMT_YUVA420P;
    case AV_PIX_FMT_YUV420P:
        return fmt_overlay == AV_PIX_FMT_YUV420P ||
               fmt_overlay == AV_PIX_FMT_YUVA420P;
    default:
        return 0;
    }
}

static int overlay_opencl_load(AVFilterContext *avctx)
{
    OverlayOpenCLContext *ctx = avctx->priv;
    cl_int cle;
    int err;

    ctx->x_subsample = 1 << ctx->in_desc_main->log2_chroma_w;
    ctx->y_subsample = 1 << ctx->in_desc_main->log2_chroma_h;

    if (ctx->x_position % ctx->x_subsample ||
        ctx->y_position % ctx->y_subsample) {
        av_log(avctx, AV_LOG_WARNING, "Overlay position (%d, %d) "
               "does not match subsampling (%d, %d).\n",
               ctx->x_position, ctx->y_position,
               ctx->x_subsample, ctx->y_subsample);
    }

    switch(ctx->in_fmt_overlay) {
    case AV_PIX_FMT_NV12:
    case AV_PIX_FMT_YUV420P:
        ctx->alpha = 0;
        ctx->kernel_name = "overlay_noalpha";
        break;
    case AV_PIX_FMT_YUVA420P:
        ctx->alpha = 1;
        ctx->kernel_name = "overlay_alpha";
        break;
    default:
        err = AVERROR_BUG;
        goto fail;
    }

    if (ctx->in_planes_main == 2 && ctx->in_planes_overlay > 2) {
        if (ctx->alpha)
            ctx->kernel_name_uv = "overlay_alpha_uv";
        else
            ctx->kernel_name_uv = "overlay_noalpha_uv";
    }

    av_log(avctx, AV_LOG_DEBUG, "Using kernel %s.\n", ctx->kernel_name);

    err = ff_opencl_filter_load_program(avctx, &ff_opencl_source_overlay, 1);
    if (err < 0)
        goto fail;

    ctx->command_queue = clCreateCommandQueue(ctx->ocf.hwctx->context,
                                              ctx->ocf.hwctx->device_id,
                                              0, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create OpenCL "
                     "command queue %d.\n", cle);

    ctx->kernel = clCreateKernel(ctx->ocf.program, ctx->kernel_name, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel %d.\n", cle);

    ctx->kernel_name_pass = "overlay_pass";
    ctx->kernel_pass = clCreateKernel(ctx->ocf.program, ctx->kernel_name_pass, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel_pass %d.\n", cle);

    if (ctx->kernel_name_uv) {
        ctx->kernel_uv = clCreateKernel(ctx->ocf.program, ctx->kernel_name_uv, &cle);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel_uv %d.\n", cle);
    }

    ctx->initialised = 1;

    return 0;

fail:
    if (ctx->command_queue)
        clReleaseCommandQueue(ctx->command_queue);
    if (ctx->kernel)
        clReleaseKernel(ctx->kernel);
    if (ctx->kernel_pass)
        clReleaseKernel(ctx->kernel_pass);
    if (ctx->kernel_uv)
        clReleaseKernel(ctx->kernel_uv);
    return err;
}

static int launch_kernel(AVFilterContext *avctx, AVFrame *output, AVFrame *input_main,
                         AVFrame *input_overlay, int plane, int passthrough) {
    OverlayOpenCLContext *ctx = avctx->priv;
    cl_mem mem;
    cl_int cle, x, y;
    cl_kernel kernel;
    size_t global_work[2];
    int idx_arg = 0;
    int err;

    if (passthrough)
        kernel = ctx->kernel_pass;
    else if (plane == 1 && ctx->in_planes_main == 2 && ctx->in_planes_overlay > 2)
        kernel = ctx->kernel_uv;
    else
        kernel = ctx->kernel;

    // dst
    mem = (cl_mem)output->data[plane];
    if (!mem) {
        err = AVERROR(EIO);
        goto fail;
    }
    CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &mem);

    // main
    mem = (cl_mem)input_main->data[plane];
    if (!mem) {
        err = AVERROR(EIO);
        goto fail;
    }
    CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &mem);

    if (!passthrough) {
        // overlay
        mem = (cl_mem)input_overlay->data[plane];
        if (!mem) {
            err = AVERROR(EIO);
            goto fail;
        }
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &mem);

        // non-semi planar on top of the semi planar
        if (plane == 1 && ctx->in_planes_main == 2 && ctx->in_planes_overlay > 2) {
            mem = (cl_mem)input_overlay->data[plane + 1];
            if (!mem) {
                err = AVERROR(EIO);
                goto fail;
            }
            CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &mem);
        }

        // alpha
        if (ctx->alpha) {
            mem = (cl_mem)input_overlay->data[ctx->in_planes_overlay - 1];
            if (!mem) {
                err = AVERROR(EIO);
                goto fail;
            }
            CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &mem);
        }

        x = ctx->x_position / (plane == 0 ? 1 : ctx->x_subsample);
        y = ctx->y_position / (plane == 0 ? 1 : ctx->y_subsample);
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_int, &x);
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_int, &y);

        if (ctx->alpha) {
            cl_int alpha_adj_x = plane == 0 ? 1 : ctx->x_subsample;
            cl_int alpha_adj_y = plane == 0 ? 1 : ctx->y_subsample;
            CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_int, &alpha_adj_x);
            CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_int, &alpha_adj_y);
        }
    }

    err = ff_opencl_filter_work_size_from_image(avctx, global_work,
                                                input_main, plane, 0);
    if (err < 0)
        goto fail;

    cle = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL,
                                 global_work, NULL, 0, NULL, NULL);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue overlay kernel "
                     "for plane %d: %d.\n", plane, cle);
    return 0;

fail:
    return err;
}

static int overlay_opencl_blend(FFFrameSync *fs)
{
    AVFilterContext *avctx = fs->parent;
    AVFilterLink    *outlink = avctx->outputs[0];
    OverlayOpenCLContext *ctx = avctx->priv;
    AVFrame *input_main, *input_overlay;
    AVFrame *output;
    cl_int cle;
    int passthrough = 0;
    int err, p;

    err = ff_framesync_get_frame(fs, 0, &input_main, 0);
    if (err < 0)
        return err;
    err = ff_framesync_get_frame(fs, 1, &input_overlay, 0);
    if (err < 0)
        return err;

    if (!input_main)
        return AVERROR_BUG;

    if (!input_overlay)
        passthrough = 1;

    if (!ctx->initialised) {
        err = overlay_opencl_load(avctx);
        if (err < 0)
            return err;
    }

    output = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!output) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    for (p = 0; p < ctx->in_planes_main; p++) {
        err = launch_kernel(avctx, output, input_main, input_overlay, p, passthrough);
        if (err < 0)
            return err;
    }

    cle = clFinish(ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue: %d.\n", cle);

    err = av_frame_copy_props(output, input_main);

    av_log(avctx, AV_LOG_DEBUG, "Filter output: %s, %ux%u (%"PRId64").\n",
           av_get_pix_fmt_name(output->format),
           output->width, output->height, output->pts);

    return ff_filter_frame(outlink, output);

fail:
    clFinish(ctx->command_queue);
    av_frame_free(&input_main);
    av_frame_free(&input_overlay);
    av_frame_free(&output);
    return err;
}

static int overlay_opencl_config_output(AVFilterLink *outlink)
{
    AVFilterContext *avctx = outlink->src;
    OverlayOpenCLContext *ctx = avctx->priv;

    AVFilterLink *inlink = avctx->inputs[0];
    AVHWFramesContext *frames_ctx_main = (AVHWFramesContext*)inlink->hw_frames_ctx->data;

    AVFilterLink *inlink_overlay = avctx->inputs[1];
    AVHWFramesContext *frames_ctx_overlay = (AVHWFramesContext*)inlink_overlay->hw_frames_ctx->data;

    int err;

    if (!frames_ctx_main) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on main input\n");
        return AVERROR(EINVAL);
    }

    ctx->in_fmt_main = frames_ctx_main->sw_format;
    ctx->in_desc_main = av_pix_fmt_desc_get(frames_ctx_main->sw_format);
    ctx->in_planes_main = av_pix_fmt_count_planes(frames_ctx_main->sw_format);
    if (!format_is_supported(supported_main_formats, ctx->in_fmt_main)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported main input format: %s\n",
               av_get_pix_fmt_name(ctx->in_fmt_main));
        return AVERROR(ENOSYS);
    }

    if (!frames_ctx_overlay) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on overlay input\n");
        return AVERROR(EINVAL);
    }

    ctx->in_fmt_overlay = frames_ctx_overlay->sw_format;
    ctx->in_desc_overlay = av_pix_fmt_desc_get(frames_ctx_overlay->sw_format);
    ctx->in_planes_overlay = av_pix_fmt_count_planes(frames_ctx_overlay->sw_format);
    if (!format_is_supported(supported_overlay_formats, ctx->in_fmt_overlay)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported overlay input format: %s\n",
            av_get_pix_fmt_name(ctx->in_fmt_overlay));
        return AVERROR(ENOSYS);
    }

    if (!formats_match(ctx->in_fmt_main, ctx->in_fmt_overlay)) {
        av_log(ctx, AV_LOG_ERROR, "Can't overlay %s on %s \n",
            av_get_pix_fmt_name(ctx->in_fmt_overlay), av_get_pix_fmt_name(ctx->in_fmt_main));
        return AVERROR(EINVAL);
    }

    err = ff_opencl_filter_config_output(outlink);
    if (err < 0)
        return err;

    err = ff_framesync_init_dualinput(&ctx->fs, avctx);
    if (err < 0)
        return err;

    ctx->fs.opt_repeatlast = ctx->opt_repeatlast;
    ctx->fs.opt_shortest = ctx->opt_shortest;
    ctx->fs.opt_eof_action = ctx->opt_eof_action;
    ctx->fs.time_base = outlink->time_base = inlink->time_base;

    return ff_framesync_configure(&ctx->fs);
}

static av_cold int overlay_opencl_init(AVFilterContext *avctx)
{
    OverlayOpenCLContext *ctx = avctx->priv;

    ctx->fs.on_event = &overlay_opencl_blend;

    return ff_opencl_filter_init(avctx);
}

static int overlay_opencl_activate(AVFilterContext *avctx)
{
    OverlayOpenCLContext *ctx = avctx->priv;

    return ff_framesync_activate(&ctx->fs);
}

static av_cold void overlay_opencl_uninit(AVFilterContext *avctx)
{
    OverlayOpenCLContext *ctx = avctx->priv;
    cl_int cle;

    if (ctx->kernel) {
        cle = clReleaseKernel(ctx->kernel);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel: %d.\n", cle);
    }

    if (ctx->kernel_pass) {
        cle = clReleaseKernel(ctx->kernel_pass);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel_pass: %d.\n", cle);
    }

    if (ctx->kernel_uv) {
        cle = clReleaseKernel(ctx->kernel_uv);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel_uv: %d.\n", cle);
    }

    if (ctx->command_queue) {
        cle = clReleaseCommandQueue(ctx->command_queue);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "command queue: %d.\n", cle);
    }

    ff_opencl_filter_uninit(avctx);

    ff_framesync_uninit(&ctx->fs);
}

#define OFFSET(x) offsetof(OverlayOpenCLContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)

static const AVOption overlay_opencl_options[] = {
    { "x", "Overlay x position",
      OFFSET(x_position), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, .flags = FLAGS },
    { "y", "Overlay y position",
      OFFSET(y_position), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, .flags = FLAGS },
    { "eof_action", "Action to take when encountering EOF from secondary input ",
        OFFSET(opt_eof_action), AV_OPT_TYPE_INT, { .i64 = EOF_ACTION_REPEAT },
        EOF_ACTION_REPEAT, EOF_ACTION_PASS, .flags = FLAGS, "eof_action" },
        { "repeat", "Repeat the previous frame.",   0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_REPEAT }, .flags = FLAGS, "eof_action" },
        { "endall", "End both streams.",            0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_ENDALL }, .flags = FLAGS, "eof_action" },
        { "pass",   "Pass through the main input.", 0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_PASS },   .flags = FLAGS, "eof_action" },
    { "shortest", "force termination when the shortest input terminates", OFFSET(opt_shortest), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { "repeatlast", "repeat overlay of the last overlay frame", OFFSET(opt_repeatlast), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { NULL },
};

AVFILTER_DEFINE_CLASS(overlay_opencl);

static const AVFilterPad overlay_opencl_inputs[] = {
    {
        .name         = "main",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &ff_opencl_filter_config_input,
    },
    {
        .name         = "overlay",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &ff_opencl_filter_config_input,
    },
};

static const AVFilterPad overlay_opencl_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = &overlay_opencl_config_output,
    },
};

const AVFilter ff_vf_overlay_opencl = {
    .name            = "overlay_opencl",
    .description     = NULL_IF_CONFIG_SMALL("Overlay one video on top of another"),
    .priv_size       = sizeof(OverlayOpenCLContext),
    .priv_class      = &overlay_opencl_class,
    .init            = &overlay_opencl_init,
    .uninit          = &overlay_opencl_uninit,
    .activate        = &overlay_opencl_activate,
    FILTER_INPUTS(overlay_opencl_inputs),
    FILTER_OUTPUTS(overlay_opencl_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_OPENCL),
    .flags_internal  = FF_FILTER_FLAG_HWFRAME_AWARE,
};
