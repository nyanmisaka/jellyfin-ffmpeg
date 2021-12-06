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

#include <float.h>

#include "libavutil/avassert.h"
#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "internal.h"
#include "opencl.h"
#include "opencl_source.h"
#include "video.h"
#include "colorspace.h"

#define OPENCL_SOURCE_NB 3

#define FLOAT_EPS 1.175494351e-38f

#define ST2084_MAX_LUMINANCE 10000.0f
#define REF_WHITE_BT2390 203.0f
#define REF_WHITE_DEFAULT 100.0f

#define ST2084_M1 0.1593017578125f
#define ST2084_M2 78.84375f
#define ST2084_C1 0.8359375f
#define ST2084_C2 18.8515625f
#define ST2084_C3 18.6875f

#define ARIB_B67_A 0.17883277f
#define ARIB_B67_B 0.28466892f
#define ARIB_B67_C 0.55991073f

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(a, b, c) MIN(MAX((a), (b)), (c))
#define MIX(x,y,a) (x) * (1 - (a)) + (y) * (a)

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUV420P16,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P016,
};

enum TonemapAlgorithm {
    TONEMAP_NONE,
    TONEMAP_LINEAR,
    TONEMAP_GAMMA,
    TONEMAP_CLIP,
    TONEMAP_REINHARD,
    TONEMAP_HABLE,
    TONEMAP_MOBIUS,
    TONEMAP_BT2390,
    TONEMAP_MAX,
};

typedef struct TonemapOpenCLContext {
    OpenCLFilterContext ocf;

    enum AVColorSpace colorspace, colorspace_in, colorspace_out;
    enum AVColorTransferCharacteristic trc, trc_in, trc_out;
    enum AVColorPrimaries primaries, primaries_in, primaries_out;
    enum AVColorRange range, range_in, range_out;
    enum AVChromaLocation chroma_loc;
    enum AVPixelFormat in_fmt, out_fmt;
    const AVPixFmtDescriptor *in_desc, *out_desc;
    int in_planes, out_planes;

    float *lin_lut, *delin_lut;
    float *pqlin_lut, *pqdelin_lut;

    enum TonemapAlgorithm tonemap;
    enum AVPixelFormat    format;
    double                ref_white;
    double                peak;
    double                param;
    double                desat_param;
    double                target_peak;
    double                scene_threshold;
    int                   lut_trc;
    int                   initialised;
    cl_kernel             kernel;
    cl_command_queue      command_queue;
} TonemapOpenCLContext;

static const char *const linearize_funcs[AVCOL_TRC_NB] = {
    [AVCOL_TRC_SMPTE2084]    = "eotf_st2084",
    [AVCOL_TRC_ARIB_STD_B67] = "eotf_arib_b67",
};

static const char *const delinearize_funcs[AVCOL_TRC_NB] = {
    [AVCOL_TRC_BT709]     = "inverse_eotf_bt1886",
    [AVCOL_TRC_BT2020_10] = "inverse_eotf_bt1886",
};

static const struct PrimaryCoefficients primaries_table[AVCOL_PRI_NB] = {
    [AVCOL_PRI_BT709]  = { 0.640, 0.330, 0.300, 0.600, 0.150, 0.060 },
    [AVCOL_PRI_BT2020] = { 0.708, 0.292, 0.170, 0.797, 0.131, 0.046 },
};

static const struct WhitepointCoefficients whitepoint_table[AVCOL_PRI_NB] = {
    [AVCOL_PRI_BT709]  = { 0.3127, 0.3290 },
    [AVCOL_PRI_BT2020] = { 0.3127, 0.3290 },
};

static const char *const tonemap_func[TONEMAP_MAX] = {
    [TONEMAP_NONE]     = "direct",
    [TONEMAP_LINEAR]   = "linear",
    [TONEMAP_GAMMA]    = "gamma",
    [TONEMAP_CLIP]     = "clip",
    [TONEMAP_REINHARD] = "reinhard",
    [TONEMAP_HABLE]    = "hable",
    [TONEMAP_MOBIUS]   = "mobius",
    [TONEMAP_BT2390]   = "bt2390",
};

// linearizer for PQ/ST2084
static float eotf_st2084(float x, float ref_white)
{
    x = FFMAX(x, 0.0f);
    float xpow = powf(x, 1.0f / ST2084_M2);
    float num = FFMAX(xpow - ST2084_C1, 0.0f);
    float den = FFMAX(ST2084_C2 - ST2084_C3 * xpow, FLOAT_EPS);
    x = powf(num / den, 1.0f / ST2084_M1);
    return x * ST2084_MAX_LUMINANCE / ref_white;
}

// delinearizer for PQ/ST2084
static float inverse_eotf_st2084(float x, float ref_white)
{
    x = FFMAX(x, 0.0f);
    x *= ref_white / ST2084_MAX_LUMINANCE;
    float xpow = powf(x, ST2084_M1);
#if 0
    // Original formulation from SMPTE ST 2084:2014 publication.
    float num = ST2084_C1 + ST2084_C2 * xpow;
    float den = 1.0f + ST2084_C3 * xpow;
    return powf(num / den, ST2084_M2);
#else
    // More stable arrangement that avoids some cancellation error.
    float num = (ST2084_C1 - 1.0f) + (ST2084_C2 - ST2084_C3) * xpow;
    float den = 1.0f + ST2084_C3 * xpow;
    return powf(1.0f + num / den, ST2084_M2);
#endif
}

static float ootf_1_2(float x) {
    return x > 0.0f ? powf(x, 1.2f) : x;
}

static float inverse_ootf_1_2(float x) {
    return x > 0.0f ? powf(x, 1.0f / 1.2f) : x;
}

static float oetf_arib_b67(float x) {
    x = FFMAX(x, 0.0f);
    return x <= (1.0f / 12.0f)
           ? sqrtf(3.0f * x)
           : (ARIB_B67_A * logf(12.0f * x - ARIB_B67_B) + ARIB_B67_C);
}

static float inverse_oetf_arib_b67(float x) {
    x = FFMAX(x, 0.0f);
    return x <= 0.5f
           ? (x * x) * (1.0f / 3.0f)
           : (expf((x - ARIB_B67_C) / ARIB_B67_A) + ARIB_B67_B) * (1.0f / 12.0f);
}

// linearizer for HLG/ARIB-B67
static float eotf_arib_b67(float x) {
    return ootf_1_2(inverse_oetf_arib_b67(x));
}

// delinearizer for HLG/ARIB-B67
static float inverse_eotf_arib_b67(float x) {
    return oetf_arib_b67(inverse_ootf_1_2(x));
}

// delinearizer for BT709, BT2020-10
static float inverse_eotf_bt1886(float x) {
    return x > 0.0f ? powf(x, 1.0f / 2.4f) : 0.0f;
}

static float linearize(float x, float ref_white, enum AVColorTransferCharacteristic trc_in)
{
    if (trc_in == AVCOL_TRC_SMPTE2084)
        return eotf_st2084(x, ref_white);
    else if (trc_in == AVCOL_TRC_ARIB_STD_B67)
        return eotf_arib_b67(x);
    else
        return x;
}

static float delinearize(float x, float ref_white, enum AVColorTransferCharacteristic trc_out)
{
    if (trc_out == AVCOL_TRC_BT709 || trc_out == AVCOL_TRC_BT2020_10)
        return inverse_eotf_bt1886(x);
    if (trc_out == AVCOL_TRC_SMPTE2084)
        return inverse_eotf_st2084(x, ref_white);
    else
        return x;
}

static int compute_trc_luts(AVFilterContext *avctx)
{
    TonemapOpenCLContext *ctx = avctx->priv;
    int lut_pq = ctx->tonemap == TONEMAP_BT2390 && ctx->trc_in != AVCOL_TRC_SMPTE2084;
    int i;

    if (!ctx->lin_lut && !(ctx->lin_lut = av_calloc(1024, sizeof(float))))
        return AVERROR(ENOMEM);
    if (!ctx->delin_lut && !(ctx->delin_lut = av_calloc(1024, sizeof(float))))
        return AVERROR(ENOMEM);
    if (lut_pq) {
        if (!ctx->pqlin_lut && !(ctx->pqlin_lut = av_calloc(1024, sizeof(float))))
            return AVERROR(ENOMEM);
        if (!ctx->pqdelin_lut && !(ctx->pqdelin_lut = av_calloc(1024, sizeof(float))))
            return AVERROR(ENOMEM);
    }

    for (i = 0; i < 1024; i++) {
        float x = i / 1023.0f;
        ctx->lin_lut[i] = FFMAX(linearize(x, ctx->ref_white, ctx->trc_in), 0.0f);
        ctx->delin_lut[i] = FFMAX(delinearize(x, ctx->ref_white, ctx->trc_out), 0.0f);
        if (lut_pq) {
            ctx->pqlin_lut[i] = FFMAX(linearize(x, ctx->ref_white, AVCOL_TRC_SMPTE2084), 0.0f);
            ctx->pqdelin_lut[i] = FFMAX(delinearize(x, ctx->ref_white, AVCOL_TRC_SMPTE2084), 0.0f);
        }
    }

    return 0;
}

static void print_opencl_const_trc_luts(AVFilterContext *avctx, AVBPrint *buf)
{
    TonemapOpenCLContext *ctx = avctx->priv;
    int i;

    if (ctx->lin_lut) {
        av_bprintf(buf, "__constant float lin_lut[1024] = {\n");
        for (i = 0; i < 1024; i++) {
            av_bprintf(buf, " %.5ff,", ctx->lin_lut[i]);
        }
        av_bprintf(buf, "};\n");
    }
    if (ctx->delin_lut) {
        av_bprintf(buf, "__constant float delin_lut[1024] = {\n");
        for (i = 0; i < 1024; i++) {
            av_bprintf(buf, " %.5ff,", ctx->delin_lut[i]);
        }
        av_bprintf(buf, "};\n");
    }
    if (ctx->pqlin_lut) {
        av_bprintf(buf, "__constant float pqlin_lut[1024] = {\n");
        for (i = 0; i < 1024; i++) {
            av_bprintf(buf, " %.5ff,", ctx->pqlin_lut[i]);
        }
        av_bprintf(buf, "};\n");
    }
    if (ctx->pqdelin_lut) {
        av_bprintf(buf, "__constant float pqdelin_lut[1024] = {\n");
        for (i = 0; i < 1024; i++) {
            av_bprintf(buf, " %.5ff,", ctx->pqdelin_lut[i]);
        }
        av_bprintf(buf, "};\n");
    }
}

static void get_rgb2rgb_matrix(enum AVColorPrimaries in, enum AVColorPrimaries out,
                               double rgb2rgb[3][3]) {
    double rgb2xyz[3][3], xyz2rgb[3][3];

    ff_fill_rgb2xyz_table(&primaries_table[out], &whitepoint_table[out], rgb2xyz);
    ff_matrix_invert_3x3(rgb2xyz, xyz2rgb);
    ff_fill_rgb2xyz_table(&primaries_table[in], &whitepoint_table[in], rgb2xyz);
    ff_matrix_mul_3x3(rgb2rgb, rgb2xyz, xyz2rgb);
}

static int tonemap_opencl_init(AVFilterContext *avctx)
{
    TonemapOpenCLContext *ctx = avctx->priv;
    AVBPrint header;
    const char *opencl_sources[OPENCL_SOURCE_NB];
    int rgb2rgb_passthrough = 1;
    double rgb2rgb[3][3], rgb2yuv[3][3], yuv2rgb[3][3];
    const struct LumaCoefficients *luma_src, *luma_dst;
    cl_int cle;
    int lut_pq = ctx->tonemap == TONEMAP_BT2390 && ctx->trc_in != AVCOL_TRC_SMPTE2084;
    int err;

    switch(ctx->tonemap) {
    case TONEMAP_GAMMA:
        if (isnan(ctx->param))
            ctx->param = 1.8f;
        break;
    case TONEMAP_REINHARD:
        if (!isnan(ctx->param))
            ctx->param = (1.0f - ctx->param) / ctx->param;
        break;
    case TONEMAP_MOBIUS:
        if (isnan(ctx->param))
            ctx->param = 0.3f;
        break;
    }

    if (isnan(ctx->param))
        ctx->param = 1.0f;

    ctx->ref_white = ctx->tonemap == TONEMAP_BT2390 ? REF_WHITE_BT2390
                                                    : REF_WHITE_DEFAULT;

    // SDR peak is 1.0f
    ctx->target_peak = 1.0f;

    av_log(ctx, AV_LOG_DEBUG, "Tonemapping transfer from %s to %s\n",
           av_color_transfer_name(ctx->trc_in),
           av_color_transfer_name(ctx->trc_out));
    av_log(ctx, AV_LOG_DEBUG, "Mapping colorspace from %s to %s\n",
           av_color_space_name(ctx->colorspace_in),
           av_color_space_name(ctx->colorspace_out));
    av_log(ctx, AV_LOG_DEBUG, "Mapping primaries from %s to %s\n",
           av_color_primaries_name(ctx->primaries_in),
           av_color_primaries_name(ctx->primaries_out));
    av_log(ctx, AV_LOG_DEBUG, "Mapping range from %s to %s\n",
           av_color_range_name(ctx->range_in),
           av_color_range_name(ctx->range_out));

    av_assert0(ctx->trc_out == AVCOL_TRC_BT709 ||
               ctx->trc_out == AVCOL_TRC_BT2020_10);
    av_assert0(ctx->trc_in == AVCOL_TRC_SMPTE2084||
               ctx->trc_in == AVCOL_TRC_ARIB_STD_B67);
    av_assert0(ctx->colorspace_in == AVCOL_SPC_BT2020_NCL ||
               ctx->colorspace_in == AVCOL_SPC_BT709);
    av_assert0(ctx->primaries_in == AVCOL_PRI_BT2020 ||
               ctx->primaries_in == AVCOL_PRI_BT709);

    av_bprint_init(&header, 2048, AV_BPRINT_SIZE_UNLIMITED);

    av_bprintf(&header, "__constant float tone_param = %.4ff;\n",
               ctx->param);
    av_bprintf(&header, "__constant float desat_param = %.4ff;\n",
               ctx->desat_param);
    av_bprintf(&header, "__constant float target_peak = %.4ff;\n",
               ctx->target_peak);
    av_bprintf(&header, "__constant float scene_threshold = %.4ff;\n",
               ctx->scene_threshold);

    av_bprintf(&header, "#define TONE_FUNC %s\n", tonemap_func[ctx->tonemap]);

    if (ctx->in_planes > 2)
        av_bprintf(&header, "#define NON_SEMI_PLANAR_IN\n");

    if (ctx->out_planes > 2)
        av_bprintf(&header, "#define NON_SEMI_PLANAR_OUT\n");

    if (ctx->primaries_out != ctx->primaries_in) {
        get_rgb2rgb_matrix(ctx->primaries_in, ctx->primaries_out, rgb2rgb);
        rgb2rgb_passthrough = 0;
    }

    if (ctx->range_in == AVCOL_RANGE_JPEG)
        av_bprintf(&header, "#define FULL_RANGE_IN\n");

    if (ctx->range_out == AVCOL_RANGE_JPEG)
        av_bprintf(&header, "#define FULL_RANGE_OUT\n");

    av_bprintf(&header, "#define chroma_loc %d\n", (int)ctx->chroma_loc);

    if (rgb2rgb_passthrough)
        av_bprintf(&header, "#define RGB2RGB_PASSTHROUGH\n");
    else
        ff_opencl_print_const_matrix_3x3(&header, "rgb2rgb", rgb2rgb);

    luma_src = ff_get_luma_coefficients(ctx->colorspace_in);
    if (!luma_src) {
        err = AVERROR(EINVAL);
        av_log(avctx, AV_LOG_ERROR, "Unsupported input colorspace %d (%s)\n",
               ctx->colorspace_in, av_color_space_name(ctx->colorspace_in));
        goto fail;
    }

    luma_dst = ff_get_luma_coefficients(ctx->colorspace_out);
    if (!luma_dst) {
        err = AVERROR(EINVAL);
        av_log(avctx, AV_LOG_ERROR, "Unsupported output colorspace %d (%s)\n",
               ctx->colorspace_out, av_color_space_name(ctx->colorspace_out));
        goto fail;
    }

    ff_fill_rgb2yuv_table(luma_dst, rgb2yuv);
    ff_opencl_print_const_matrix_3x3(&header, "yuv_matrix", rgb2yuv);

    ff_fill_rgb2yuv_table(luma_src, rgb2yuv);
    ff_matrix_invert_3x3(rgb2yuv, yuv2rgb);
    ff_opencl_print_const_matrix_3x3(&header, "rgb_matrix", yuv2rgb);

    av_bprintf(&header, "__constant float3 luma_src = {%.4ff, %.4ff, %.4ff};\n",
               luma_src->cr, luma_src->cg, luma_src->cb);
    av_bprintf(&header, "__constant float3 luma_dst = {%.4ff, %.4ff, %.4ff};\n",
               luma_dst->cr, luma_dst->cg, luma_dst->cb);

    if (ctx->lut_trc) {
        if (!ctx->lin_lut || !ctx->delin_lut) {
            err = compute_trc_luts(avctx);
            if (err < 0)
                goto fail;
        }
        print_opencl_const_trc_luts(avctx, &header);
        if (lut_pq)
            av_bprintf(&header, "#define TRC_LUT_PQ\n");
        av_bprintf(&header, "#define TRC_LUT\n");
        av_bprintf(&header, "#define linearize %s\n", "linearize_lut");
        av_bprintf(&header, "#define delinearize %s\n", "delinearize_lut");
    } else {
        av_bprintf(&header, "#define linearize %s\n", linearize_funcs[ctx->trc_in]);
        av_bprintf(&header, "#define delinearize %s\n", delinearize_funcs[ctx->trc_out]);
    }

    av_log(avctx, AV_LOG_DEBUG, "Generated OpenCL header:\n%s\n", header.str);
    opencl_sources[0] = header.str;
    opencl_sources[1] = ff_opencl_source_tonemap;
    opencl_sources[2] = ff_opencl_source_colorspace_common;
    err = ff_opencl_filter_load_program(avctx, opencl_sources, OPENCL_SOURCE_NB);

    av_bprint_finalize(&header, NULL);
    if (err < 0)
        goto fail;

    ctx->command_queue = clCreateCommandQueue(ctx->ocf.hwctx->context,
                                              ctx->ocf.hwctx->device_id,
                                              0, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create OpenCL "
                     "command queue %d.\n", cle);

    ctx->kernel = clCreateKernel(ctx->ocf.program, "tonemap", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel %d.\n", cle);

    ctx->initialised = 1;
    return 0;

fail:
    av_bprint_finalize(&header, NULL);
    if (ctx->command_queue)
        clReleaseCommandQueue(ctx->command_queue);
    if (ctx->kernel)
        clReleaseKernel(ctx->kernel);
    if (ctx->lin_lut)
        av_freep(&ctx->lin_lut);
    if (ctx->delin_lut)
        av_freep(&ctx->delin_lut);
    if (ctx->pqlin_lut)
        av_freep(&ctx->pqlin_lut);
    if (ctx->pqdelin_lut)
        av_freep(&ctx->pqdelin_lut);
    return err;
}

static int format_is_supported(enum AVPixelFormat fmt)
{
    for (int i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

static int tonemap_opencl_config_output(AVFilterLink *outlink)
{
    AVFilterContext    *avctx = outlink->src;
    AVFilterLink      *inlink = avctx->inputs[0];
    TonemapOpenCLContext *ctx = avctx->priv;
    AVHWFramesContext *in_frames_ctx;
    enum AVPixelFormat in_format;
    enum AVPixelFormat out_format;
    const AVPixFmtDescriptor *in_desc;
    const AVPixFmtDescriptor *out_desc;
    int ret;

    if (!inlink->hw_frames_ctx)
        return AVERROR(EINVAL);
    in_frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
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
    if (in_desc->comp[0].depth != 10 && in_desc->comp[0].depth != 16) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format depth: %d\n",
               in_desc->comp[0].depth);
        return AVERROR(ENOSYS);
    }

    ctx->in_fmt     = in_format;
    ctx->out_fmt    = out_format;
    ctx->in_desc    = in_desc;
    ctx->out_desc   = out_desc;
    ctx->in_planes  = av_pix_fmt_count_planes(in_format);
    ctx->out_planes = av_pix_fmt_count_planes(out_format);
    ctx->ocf.output_format = out_format;

    ret = ff_opencl_filter_config_output(outlink);
    if (ret < 0)
        return ret;

    return 0;
}

static int launch_kernel(AVFilterContext *avctx, cl_kernel kernel,
                         AVFrame *output, AVFrame *input, float peak) {
    TonemapOpenCLContext *ctx = avctx->priv;
    int err = AVERROR(ENOSYS);
    size_t global_work[2];
    size_t local_work[2];
    cl_int cle;
    int idx_arg;

    if (!output->data[0] || !input->data[0] || !output->data[1] || !input->data[1]) {
        err = AVERROR(EIO);
        goto fail;
    }

    if (ctx->out_planes > 2 && !output->data[2]) {
        err = AVERROR(EIO);
        goto fail;
    }

    if (ctx->in_planes > 2 && !input->data[2]) {
        err = AVERROR(EIO);
        goto fail;
    }

    CL_SET_KERNEL_ARG(kernel, 0, cl_mem, &output->data[0]);
    CL_SET_KERNEL_ARG(kernel, 1, cl_mem, &input->data[0]);
    CL_SET_KERNEL_ARG(kernel, 2, cl_mem, &output->data[1]);
    CL_SET_KERNEL_ARG(kernel, 3, cl_mem, &input->data[1]);

    idx_arg = 4;
    if (ctx->out_planes > 2)
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &output->data[2]);

    if (ctx->in_planes > 2)
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &input->data[2]);

    CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_float, &peak);

    local_work[0]  = 16;
    local_work[1]  = 16;
    // Note the work size based on uv plane, as we process a 2x2 quad in one workitem
    err = ff_opencl_filter_work_size_from_image(avctx, global_work, output,
                                                1, 16);
    if (err < 0)
        return err;

    cle = clEnqueueNDRangeKernel(ctx->command_queue, kernel, 2, NULL,
                                 global_work, local_work,
                                 0, NULL, NULL);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue kernel: %d.\n", cle);
    return 0;
fail:
    return err;
}

static int tonemap_opencl_filter_frame(AVFilterLink *inlink, AVFrame *input)
{
    AVFilterContext    *avctx = inlink->dst;
    AVFilterLink     *outlink = avctx->outputs[0];
    TonemapOpenCLContext *ctx = avctx->priv;
    AVFrame *output = NULL;
    cl_int cle;
    int err;

    av_log(ctx, AV_LOG_DEBUG, "Filter input: %s, %ux%u (%"PRId64").\n",
           av_get_pix_fmt_name(input->format),
           input->width, input->height, input->pts);

    if (!input->hw_frames_ctx)
        return AVERROR(EINVAL);

    output = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!output) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    err = av_frame_copy_props(output, input);
    if (err < 0)
        goto fail;

    if (!ctx->peak) {
        ctx->peak = ff_determine_signal_peak(input);
        av_log(ctx, AV_LOG_DEBUG, "Computed signal peak: %f\n", ctx->peak);
    }

    if (ctx->trc != -1)
        output->color_trc = ctx->trc;
    if (ctx->primaries != -1)
        output->color_primaries = ctx->primaries;
    if (ctx->colorspace != -1)
        output->colorspace = ctx->colorspace;
    if (ctx->range != -1)
        output->color_range = ctx->range;

    ctx->trc_in = input->color_trc;
    ctx->trc_out = output->color_trc;
    ctx->colorspace_in = input->colorspace;
    ctx->colorspace_out = output->colorspace;
    ctx->primaries_in = input->color_primaries;
    ctx->primaries_out = output->color_primaries;
    ctx->range_in = input->color_range;
    ctx->range_out = output->color_range;
    ctx->chroma_loc = output->chroma_location;

    if (!ctx->initialised) {
        if (!(input->color_trc == AVCOL_TRC_SMPTE2084 ||
            input->color_trc == AVCOL_TRC_ARIB_STD_B67)) {
            av_log(ctx, AV_LOG_ERROR, "Unsupported transfer function characteristic: %s\n",
                   av_color_transfer_name(input->color_trc));
            err = AVERROR(ENOSYS);
            goto fail;
        }

        err = tonemap_opencl_init(avctx);
        if (err < 0)
            goto fail;
    }

    err = launch_kernel(avctx, ctx->kernel, output, input, ctx->peak);
    if (err < 0)
        goto fail;

    cle = clFinish(ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue: %d.\n", cle);

    av_frame_free(&input);

    ff_update_hdr_metadata(output, ctx->target_peak);

    av_log(ctx, AV_LOG_DEBUG, "Tonemapping output: %s, %ux%u (%"PRId64").\n",
           av_get_pix_fmt_name(output->format),
           output->width, output->height, output->pts);

    return ff_filter_frame(outlink, output);

fail:
    clFinish(ctx->command_queue);
    av_frame_free(&input);
    av_frame_free(&output);
    return err;
}

static av_cold void tonemap_opencl_uninit(AVFilterContext *avctx)
{
    TonemapOpenCLContext *ctx = avctx->priv;
    cl_int cle;

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

    if (ctx->lin_lut)
        av_freep(&ctx->lin_lut);
    if (ctx->delin_lut)
        av_freep(&ctx->delin_lut);
    if (ctx->pqlin_lut)
        av_freep(&ctx->pqlin_lut);
    if (ctx->pqdelin_lut)
        av_freep(&ctx->pqdelin_lut);

    ff_opencl_filter_uninit(avctx);
}

#define OFFSET(x) offsetof(TonemapOpenCLContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption tonemap_opencl_options[] = {
    { "tonemap",      "Tonemap algorithm selection", OFFSET(tonemap), AV_OPT_TYPE_INT, { .i64 = TONEMAP_NONE }, TONEMAP_NONE, TONEMAP_MAX - 1, FLAGS, "tonemap" },
        { "none",     0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_NONE },              0, 0, FLAGS, "tonemap" },
        { "linear",   0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_LINEAR },            0, 0, FLAGS, "tonemap" },
        { "gamma",    0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_GAMMA },             0, 0, FLAGS, "tonemap" },
        { "clip",     0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_CLIP },              0, 0, FLAGS, "tonemap" },
        { "reinhard", 0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_REINHARD },          0, 0, FLAGS, "tonemap" },
        { "hable",    0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_HABLE },             0, 0, FLAGS, "tonemap" },
        { "mobius",   0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_MOBIUS },            0, 0, FLAGS, "tonemap" },
        { "bt2390",   0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_BT2390 },            0, 0, FLAGS, "tonemap" },
    { "transfer", "Set transfer characteristic", OFFSET(trc), AV_OPT_TYPE_INT, { .i64 = AVCOL_TRC_BT709 }, -1, INT_MAX, FLAGS, "transfer" },
    { "t",        "Set transfer characteristic", OFFSET(trc), AV_OPT_TYPE_INT, { .i64 = AVCOL_TRC_BT709 }, -1, INT_MAX, FLAGS, "transfer" },
        { "bt709",            0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_TRC_BT709 },         0, 0, FLAGS, "transfer" },
        { "bt2020",           0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_TRC_BT2020_10 },     0, 0, FLAGS, "transfer" },
    { "matrix", "Set colorspace matrix", OFFSET(colorspace), AV_OPT_TYPE_INT, { .i64 = AVCOL_SPC_BT709 }, -1, INT_MAX, FLAGS, "matrix" },
    { "m",      "Set colorspace matrix", OFFSET(colorspace), AV_OPT_TYPE_INT, { .i64 = AVCOL_SPC_BT709 }, -1, INT_MAX, FLAGS, "matrix" },
        { "bt709",            0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_SPC_BT709 },         0, 0, FLAGS, "matrix" },
        { "bt2020",           0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_SPC_BT2020_NCL },    0, 0, FLAGS, "matrix" },
    { "primaries", "Set color primaries", OFFSET(primaries), AV_OPT_TYPE_INT, { .i64 = AVCOL_PRI_BT709 }, -1, INT_MAX, FLAGS, "primaries" },
    { "p",         "Set color primaries", OFFSET(primaries), AV_OPT_TYPE_INT, { .i64 = AVCOL_PRI_BT709 }, -1, INT_MAX, FLAGS, "primaries" },
        { "bt709",            0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_PRI_BT709 },         0, 0, FLAGS, "primaries" },
        { "bt2020",           0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_PRI_BT2020 },        0, 0, FLAGS, "primaries" },
    { "range",         "Set color range", OFFSET(range), AV_OPT_TYPE_INT, { .i64 = AVCOL_RANGE_MPEG }, -1, INT_MAX, FLAGS, "range" },
    { "r",             "Set color range", OFFSET(range), AV_OPT_TYPE_INT, { .i64 = AVCOL_RANGE_MPEG }, -1, INT_MAX, FLAGS, "range" },
        { "tv",            0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_RANGE_MPEG },         0, 0, FLAGS, "range" },
        { "pc",            0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_RANGE_JPEG },         0, 0, FLAGS, "range" },
        { "limited",       0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_RANGE_MPEG },         0, 0, FLAGS, "range" },
        { "full",          0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_RANGE_JPEG },         0, 0, FLAGS, "range" },
    { "format",    "Output pixel format", OFFSET(format), AV_OPT_TYPE_PIXEL_FMT, { .i64 = AV_PIX_FMT_NONE }, AV_PIX_FMT_NONE, INT_MAX, FLAGS, "fmt" },
    { "peak",      "Signal peak override", OFFSET(peak), AV_OPT_TYPE_DOUBLE, { .dbl = 0 }, 0, DBL_MAX, FLAGS },
    { "param",     "Tonemap parameter",   OFFSET(param), AV_OPT_TYPE_DOUBLE, { .dbl = NAN }, DBL_MIN, DBL_MAX, FLAGS },
    { "desat",     "Desaturation parameter",   OFFSET(desat_param), AV_OPT_TYPE_DOUBLE, { .dbl = 0.5}, 0, DBL_MAX, FLAGS },
    { "threshold", "Scene detection threshold",   OFFSET(scene_threshold), AV_OPT_TYPE_DOUBLE, { .dbl = 0.2 }, 0, DBL_MAX, FLAGS },
    { "luttrc",    "Enable LUT for de/linearize",   OFFSET(lut_trc), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(tonemap_opencl);

static const AVFilterPad tonemap_opencl_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = &tonemap_opencl_filter_frame,
        .config_props = &ff_opencl_filter_config_input,
    },
    { NULL }
};

static const AVFilterPad tonemap_opencl_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &tonemap_opencl_config_output,
    },
    { NULL }
};

AVFilter ff_vf_tonemap_opencl = {
    .name           = "tonemap_opencl",
    .description    = NULL_IF_CONFIG_SMALL("Perform HDR to SDR conversion with tonemapping."),
    .priv_size      = sizeof(TonemapOpenCLContext),
    .priv_class     = &tonemap_opencl_class,
    .init           = &ff_opencl_filter_init,
    .uninit         = &tonemap_opencl_uninit,
    .query_formats  = &ff_opencl_filter_query_formats,
    .inputs         = tonemap_opencl_inputs,
    .outputs        = tonemap_opencl_outputs,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
