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
#include <stdio.h>
#include <string.h>

#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/bprint.h"
#include "libavutil/common.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "colorspace.h"
#include "cuda/host_util.h"
#include "cuda/shared.h"
#include "cuda/tonemap.h"
#include "formats.h"
#include "internal.h"
#include "scale_eval.h"
#include "video.h"

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P016
};

#define REF_WHITE_BT2390 203.0f
#define REF_WHITE_DEFAULT 100.0f

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#define ALIGN_UP(a, b) (((a) + (b) - 1) & ~((b) - 1))
#define NUM_BUFFERS 2
#define BLOCKX 32
#define BLOCKY 16

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)

typedef struct TonemapCUDAContext {
    const AVClass *class;

    AVCUDADeviceContext *hwctx;

    enum AVPixelFormat in_fmt, out_fmt;

    enum AVColorTransferCharacteristic trc, in_trc, out_trc;
    enum AVColorSpace spc, in_spc, out_spc;
    enum AVColorPrimaries pri, in_pri, out_pri;
    enum AVColorRange range, in_range, out_range;
    enum AVChromaLocation in_chroma_loc, out_chroma_loc;

    AVBufferRef *frames_ctx;
    AVFrame     *frame;

    AVFrame *tmp_frame;

    /**
     * Output sw format. AV_PIX_FMT_NONE for no conversion.
     */
    enum AVPixelFormat format;
    char *format_str;

    CUcontext   cu_ctx;
    CUmodule    cu_module;

    CUfunction  cu_func;

    CUdeviceptr srcBuffer;
    CUdeviceptr dstBuffer;

    enum TonemapAlgorithm tonemap;
    double ref_white;
    double param;
    double desat_param;
    double peak;
    double dst_peak;
    double scene_threshold;

    const AVPixFmtDescriptor *in_desc, *out_desc;
} TonemapCUDAContext;

static av_cold int init(AVFilterContext *ctx)
{
    TonemapCUDAContext *s = ctx->priv;

    if (!strcmp(s->format_str, "same")) {
        s->format = AV_PIX_FMT_NONE;
    } else {
        s->format = av_get_pix_fmt(s->format_str);
        if (s->format == AV_PIX_FMT_NONE) {
            av_log(ctx, AV_LOG_ERROR, "Unrecognized pixel format: %s\n", s->format_str);
            return AVERROR(EINVAL);
        }
    }

    s->frame = av_frame_alloc();
    if (!s->frame)
        return AVERROR(ENOMEM);

    s->tmp_frame = av_frame_alloc();
    if (!s->tmp_frame)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    TonemapCUDAContext *s = ctx->priv;

    if (s->hwctx) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy, cuda_ctx = s->hwctx->cuda_ctx;

        CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));

        if (s->cu_module) {
            CHECK_CU(cu->cuModuleUnload(s->cu_module));
            s->cu_func = NULL;
            s->cu_module = NULL;
        }

        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }

    av_frame_free(&s->frame);
    av_buffer_unref(&s->frames_ctx);
    av_frame_free(&s->tmp_frame);
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE,
    };
    AVFilterFormats *pix_fmts = ff_make_format_list(pixel_formats);

    return ff_set_common_formats(ctx, pix_fmts);
}

static av_cold int init_stage(TonemapCUDAContext *s, AVBufferRef *device_ctx,
                              AVFilterLink *outlink)
{
    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    int ret;

    out_ref = av_hwframe_ctx_alloc(device_ctx);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext*)out_ref->data;

    out_ctx->format    = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = s->out_fmt;
    out_ctx->width     = FFALIGN(outlink->w, 32);
    out_ctx->height    = FFALIGN(outlink->h, 32);

    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
        goto fail;

    av_frame_unref(s->frame);
    ret = av_hwframe_get_buffer(out_ref, s->frame, 0);
    if (ret < 0)
        goto fail;

    s->frame->width  = outlink->w;
    s->frame->height = outlink->h;

    av_buffer_unref(&s->frames_ctx);
    s->frames_ctx = out_ref;

    return 0;
fail:
    av_buffer_unref(&out_ref);
    return ret;
}

static int format_is_supported(enum AVPixelFormat fmt)
{
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

static av_cold int init_processing_chain(AVFilterContext *ctx, AVFilterLink *outlink)
{
    TonemapCUDAContext *s = ctx->priv;

    AVHWFramesContext *in_frames_ctx;

    enum AVPixelFormat in_format;
    enum AVPixelFormat out_format;
    const AVPixFmtDescriptor *in_desc;
    const AVPixFmtDescriptor *out_desc;
    int ret;

    /* check that we have a hw context */
    if (!ctx->inputs[0]->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames_ctx = (AVHWFramesContext*)ctx->inputs[0]->hw_frames_ctx->data;
    in_format     = in_frames_ctx->sw_format;
    out_format    = (s->format == AV_PIX_FMT_NONE) ? in_format : s->format;
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
    if (!(in_desc->comp[0].depth == 10 ||
        in_desc->comp[0].depth == 16)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format depth: %d\n",
               in_desc->comp[0].depth);
        return AVERROR(ENOSYS);
    }

    s->in_fmt = in_format;
    s->out_fmt = out_format;
    s->in_desc  = in_desc;
    s->out_desc = out_desc;

    ret = init_stage(s, in_frames_ctx->device_ref, outlink);
    if (ret < 0)
        return ret;

    ctx->outputs[0]->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!ctx->outputs[0]->hw_frames_ctx)
        return AVERROR(ENOMEM);

    return 0;
}

static const struct PrimaryCoefficients primaries_table[AVCOL_PRI_NB] = {
    [AVCOL_PRI_BT709]  = { 0.640, 0.330, 0.300, 0.600, 0.150, 0.060 },
    [AVCOL_PRI_BT2020] = { 0.708, 0.292, 0.170, 0.797, 0.131, 0.046 },
};

static const struct WhitepointCoefficients whitepoint_table[AVCOL_PRI_NB] = {
    [AVCOL_PRI_BT709]  = { 0.3127, 0.3290 },
    [AVCOL_PRI_BT2020] = { 0.3127, 0.3290 },
};

static int get_rgb2rgb_matrix(enum AVColorPrimaries in, enum AVColorPrimaries out,
                              double rgb2rgb[3][3]) {
    double rgb2xyz[3][3], xyz2rgb[3][3];

    ff_fill_rgb2xyz_table(&primaries_table[out], &whitepoint_table[out], rgb2xyz);
    ff_matrix_invert_3x3(rgb2xyz, xyz2rgb);
    ff_fill_rgb2xyz_table(&primaries_table[in], &whitepoint_table[in], rgb2xyz);
    ff_matrix_mul_3x3(rgb2rgb, rgb2xyz, xyz2rgb);

    return 0;
}

static av_cold int compile(AVFilterLink *inlink)
{
    int ret = 0;
    AVFilterContext  *ctx = inlink->dst;
    TonemapCUDAContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy, cuda_ctx = s->hwctx->cuda_ctx;
    AVBPrint constants;
    CUlinkState link_state;
    void *cubin;
    size_t cubin_size;
    double rgb_matrix[3][3], yuv_matrix[3][3], rgb2rgb_matrix[3][3];
    const struct LumaCoefficients *in_coeffs, *out_coeffs;
    enum AVColorTransferCharacteristic in_trc = s->in_trc, out_trc = s->out_trc;
    enum AVColorSpace in_spc = s->in_spc, out_spc = s->out_spc;
    enum AVColorPrimaries in_pri = s->in_pri, out_pri = s->out_pri;
    enum AVColorRange in_range = s->in_range, out_range = s->out_range;
    char info_log[4096], error_log[4096];
    CUjit_option options[] = {CU_JIT_INFO_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void *option_values[]  = {&info_log,              &error_log,              (void*)(intptr_t)sizeof(info_log), (void*)(intptr_t)sizeof(error_log)};

    extern char tonemap_ptx[];

    switch(s->tonemap) {
    case TONEMAP_GAMMA:
        if (isnan(s->param))
            s->param = 1.8f;
        break;
    case TONEMAP_REINHARD:
        if (!isnan(s->param))
            s->param = (1.0f - s->param) / s->param;
        break;
    case TONEMAP_MOBIUS:
        if (isnan(s->param))
            s->param = 0.3f;
        break;
    }

    if (isnan(s->param))
        s->param = 1.0f;

    s->ref_white = s->tonemap == TONEMAP_BT2390 ? REF_WHITE_BT2390
                                                : REF_WHITE_DEFAULT;

    s->dst_peak = 1.0f;

    if (in_trc == AVCOL_TRC_UNSPECIFIED)
        in_trc = AVCOL_TRC_SMPTE2084;
    if (out_trc == AVCOL_TRC_UNSPECIFIED)
        out_trc = AVCOL_TRC_BT709;

    if (in_spc == AVCOL_SPC_UNSPECIFIED)
        in_spc = AVCOL_SPC_BT2020_NCL;
    if (out_spc == AVCOL_SPC_UNSPECIFIED)
        out_spc = AVCOL_SPC_BT709;

    if (in_pri == AVCOL_PRI_UNSPECIFIED)
        in_pri = AVCOL_PRI_BT2020;
    if (out_pri == AVCOL_PRI_UNSPECIFIED)
        out_pri = AVCOL_PRI_BT709;

    if (in_range == AVCOL_RANGE_UNSPECIFIED)
        in_range = AVCOL_RANGE_MPEG;
    if (out_range == AVCOL_RANGE_UNSPECIFIED)
        out_range = AVCOL_RANGE_MPEG;

    av_log(ctx, AV_LOG_DEBUG, "Tonemapping transfer from %s to %s\n",
           av_color_transfer_name(in_trc),
           av_color_transfer_name(out_trc));
    av_log(ctx, AV_LOG_DEBUG, "Mapping colorspace from %s to %s\n",
           av_color_space_name(in_spc),
           av_color_space_name(out_spc));
    av_log(ctx, AV_LOG_DEBUG, "Mapping primaries from %s to %s\n",
           av_color_primaries_name(in_pri),
           av_color_primaries_name(out_pri));
    av_log(ctx, AV_LOG_DEBUG, "Mapping range from %s to %s\n",
           av_color_range_name(in_range),
           av_color_range_name(out_range));

    if (!(in_coeffs = ff_get_luma_coefficients(in_spc)))
        return AVERROR(EINVAL);

    ff_fill_rgb2yuv_table(in_coeffs, yuv_matrix);
    ff_matrix_invert_3x3(yuv_matrix, rgb_matrix);

    if (!(out_coeffs = ff_get_luma_coefficients(out_spc)))
        return AVERROR(EINVAL);

    ff_fill_rgb2yuv_table(out_coeffs, yuv_matrix);

    if ((ret = get_rgb2rgb_matrix(in_pri, out_pri, rgb2rgb_matrix)) < 0)
        return ret;

    av_bprint_init(&constants, 2048, AV_BPRINT_SIZE_UNLIMITED);

    av_bprintf(&constants, ".version 3.2\n");
    av_bprintf(&constants, ".target sm_30\n");
    av_bprintf(&constants, ".address_size %zu\n", sizeof(void*) * 8);

#define CONSTANT_A(decl, align, ...) \
    av_bprintf(&constants, ".visible .const .align " #align " " decl ";\n", __VA_ARGS__)
#define CONSTANT(decl, ...) CONSTANT_A(decl, 4, __VA_ARGS__)
#define CONSTANT_M(a, b) \
    CONSTANT(".f32 " a "[] = {%f, %f, %f, %f, %f, %f, %f, %f, %f}", \
             b[0][0], b[0][1], b[0][2], \
             b[1][0], b[1][1], b[1][2], \
             b[2][0], b[2][1], b[2][2])
#define CONSTANT_C(a, b) \
    CONSTANT(".f32 " a "[] = {%f, %f, %f}", \
             b->cr, b->cg, b->cb)

    CONSTANT(".u32 depth_src      = %i", (int)s->in_desc->comp[0].depth);
    CONSTANT(".u32 depth_dst      = %i", (int)s->out_desc->comp[0].depth);
    CONSTANT(".u32 fmt_src        = %i", (int)s->in_fmt);
    CONSTANT(".u32 fmt_dst        = %i", (int)s->out_fmt);
    CONSTANT(".u32 range_src      = %i", (int)in_range);
    CONSTANT(".u32 range_dst      = %i", (int)out_range);
    CONSTANT(".u32 trc_src        = %i", (int)in_trc);
    CONSTANT(".u32 trc_dst        = %i", (int)out_trc);
    CONSTANT(".u32 chroma_loc_src = %i", (int)s->in_chroma_loc);
    CONSTANT(".u32 chroma_loc_dst = %i", (int)s->out_chroma_loc);
    CONSTANT(".u32 tonemap_func   = %i", (int)s->tonemap);
    CONSTANT(".f32 ref_white      = %f", s->ref_white);
    CONSTANT(".f32 tone_param     = %f", s->param);
    CONSTANT(".f32 desat_param    = %f", s->desat_param);
    CONSTANT_M("rgb_matrix", rgb_matrix);
    CONSTANT_M("yuv_matrix", yuv_matrix);
    CONSTANT_A(".u8 rgb2rgb_passthrough = %i", 1, in_pri == out_pri);
    CONSTANT_M("rgb2rgb_matrix", rgb2rgb_matrix);
    CONSTANT_C("luma_src", in_coeffs);
    CONSTANT_C("luma_dst", out_coeffs);

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        return ret;

    if (s->cu_module) {
        ret = CHECK_CU(cu->cuModuleUnload(s->cu_module));
        if (ret < 0)
            goto fail;

        s->cu_func = NULL;
        s->cu_module = NULL;
    }

    ret = CHECK_CU(cu->cuLinkCreate(sizeof(options) / sizeof(options[0]), options, option_values, &link_state));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuLinkAddData(link_state, CU_JIT_INPUT_PTX, constants.str,
                                     constants.len, "constants", 0, NULL, NULL));
    if (ret < 0)
        goto fail2;

    ret = CHECK_CU(cu->cuLinkAddData(link_state, CU_JIT_INPUT_PTX, tonemap_ptx,
                                     strlen(tonemap_ptx), "tonemap.ptx", 0, NULL, NULL));
    if (ret < 0)
        goto fail2;

    ret = CHECK_CU(cu->cuLinkComplete(link_state, &cubin, &cubin_size));
    if (ret < 0)
        goto fail2;

    ret = CHECK_CU(cu->cuModuleLoadData(&s->cu_module, cubin));
    if (ret < 0)
        goto fail2;

    CHECK_CU(cu->cuModuleGetFunction(&s->cu_func, s->cu_module, "tonemap"));
    if (ret < 0)
        goto fail2;

fail2:
    CHECK_CU(cu->cuLinkDestroy(link_state));

fail:
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    av_bprint_finalize(&constants, NULL);

    if ((intptr_t)option_values[2] > 0)
        av_log(ctx, AV_LOG_INFO, "CUDA linker output: %.*s\n", (int)(intptr_t)option_values[2], info_log);

    if ((intptr_t)option_values[3] > 0)
        av_log(ctx, AV_LOG_ERROR, "CUDA linker output: %.*s\n", (int)(intptr_t)option_values[3], error_log);

    return ret;
}

static av_cold int config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = outlink->src->inputs[0];
    AVHWFramesContext *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;
    TonemapCUDAContext *s  = ctx->priv;
    int ret;

    s->hwctx = device_hwctx;

    outlink->w = inlink->w;
    outlink->h = inlink->h;

    ret = init_processing_chain(ctx, outlink);
    if (ret < 0)
        return ret;

    outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;

    return 0;
}

static int run_kernel(AVFilterContext *ctx,
                      AVFrame *out, AVFrame *in)
{
    TonemapCUDAContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    FFCUDAFrame src, dst;
    void *args_uchar[] = { &src, &dst };
    int ret;

    ret = ff_make_cuda_frame(&src, in);
    if (ret < 0)
        goto fail;

    ret = ff_make_cuda_frame(&dst, out);
    if (ret < 0)
        goto fail;

    src.peak = s->peak;
    if (!src.peak) {
        src.peak = ff_determine_signal_peak(in);
        av_log(s, AV_LOG_DEBUG, "Computed signal peak: %f\n", src.peak);
    }

    dst.peak = s->dst_peak;

    ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func,
                                      DIV_UP(src.width / 2, BLOCKX), DIV_UP(src.height / 2, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, s->hwctx->stream, args_uchar, NULL));

fail:
    return ret;
}

static int do_tonemap(AVFilterContext *ctx, AVFrame *out, AVFrame *in)
{
    TonemapCUDAContext *s = ctx->priv;
    AVFrame *src = in;
    int ret;

    ret = run_kernel(ctx, s->frame, src);
    if (ret < 0)
        return ret;

    src = s->frame;
    ret = av_hwframe_get_buffer(src->hw_frames_ctx, s->tmp_frame, 0);
    if (ret < 0)
        return ret;

    av_frame_move_ref(out, s->frame);
    av_frame_move_ref(s->frame, s->tmp_frame);

    s->frame->width  = in->width;
    s->frame->height = in->height;

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        return ret;

    if (s->out_trc        != out->color_trc ||
        s->out_spc        != out->colorspace ||
        s->out_pri        != out->color_primaries ||
        s->out_range      != out->color_range ||
        s->out_chroma_loc != out->chroma_location) {
        out->color_trc       = s->out_trc;
        out->colorspace      = s->out_spc;
        out->color_primaries = s->out_pri;
        out->color_range     = s->out_range;
        out->chroma_location = s->out_chroma_loc;
    }

    return 0;
}

static int filter_frame(AVFilterLink *link, AVFrame *in)
{
    AVFilterContext       *ctx = link->dst;
    TonemapCUDAContext      *s = ctx->priv;
    AVFilterLink      *outlink = ctx->outputs[0];
    CudaFunctions          *cu = s->hwctx->internal->cuda_dl;

    AVFrame *out = NULL;
    CUcontext dummy;
    int ret = 0;

    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    if (!(in->color_trc == AVCOL_TRC_SMPTE2084 ||
        in->color_trc == AVCOL_TRC_ARIB_STD_B67)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input transfer characteristic: %s\n",
               av_color_transfer_name(in->color_trc));
        ret = AVERROR(EINVAL);
        goto fail;
    }

    if (!s->cu_func ||
        s->in_trc        != in->color_trc ||
        s->in_spc        != in->colorspace ||
        s->in_pri        != in->color_primaries ||
        s->in_range      != in->color_range ||
        s->in_chroma_loc != in->chroma_location) {
        s->in_trc        = in->color_trc;
        s->in_spc        = in->colorspace;
        s->in_pri        = in->color_primaries;
        s->in_range      = in->color_range;
        s->in_chroma_loc = in->chroma_location;

        s->out_trc        = s->trc;
        s->out_spc        = s->spc;
        s->out_pri        = s->pri;
        s->out_range      = s->range;
        s->out_chroma_loc = s->in_chroma_loc;

        if ((ret = compile(link)) < 0)
            goto fail;
    }

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;

    ret = do_tonemap(ctx, out, in);

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        goto fail;

    av_frame_free(&in);

    ff_update_hdr_metadata(out, s->dst_peak);

    return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

#define OFFSET(x) offsetof(TonemapCUDAContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption options[] = {
    { "tonemap",      "tonemap algorithm selection", OFFSET(tonemap), AV_OPT_TYPE_INT, {.i64 = TONEMAP_NONE}, TONEMAP_NONE, TONEMAP_MAX - 1, FLAGS, "tonemap" },
    {     "none",     0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_NONE},              0, 0, FLAGS, "tonemap" },
    {     "linear",   0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_LINEAR},            0, 0, FLAGS, "tonemap" },
    {     "gamma",    0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_GAMMA},             0, 0, FLAGS, "tonemap" },
    {     "clip",     0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_CLIP},              0, 0, FLAGS, "tonemap" },
    {     "reinhard", 0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_REINHARD},          0, 0, FLAGS, "tonemap" },
    {     "hable",    0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_HABLE},             0, 0, FLAGS, "tonemap" },
    {     "mobius",   0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_MOBIUS},            0, 0, FLAGS, "tonemap" },
    {     "bt2390",   0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_BT2390},            0, 0, FLAGS, "tonemap" },
    { "transfer",     "set transfer characteristic", OFFSET(trc), AV_OPT_TYPE_INT, {.i64 = AVCOL_TRC_BT709}, -1, INT_MAX, FLAGS, "transfer" },
    { "t",            "set transfer characteristic", OFFSET(trc), AV_OPT_TYPE_INT, {.i64 = AVCOL_TRC_BT709}, -1, INT_MAX, FLAGS, "transfer" },
    {     "bt709",    0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_TRC_BT709},           0, 0, FLAGS, "transfer" },
    {     "bt2020",   0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_TRC_BT2020_10},       0, 0, FLAGS, "transfer" },
    { "matrix",       "set colorspace matrix", OFFSET(spc), AV_OPT_TYPE_INT, {.i64 = AVCOL_SPC_BT709}, -1, INT_MAX, FLAGS, "matrix" },
    { "m",            "set colorspace matrix", OFFSET(spc), AV_OPT_TYPE_INT, {.i64 = AVCOL_SPC_BT709}, -1, INT_MAX, FLAGS, "matrix" },
    {     "bt709",    0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_SPC_BT709},           0, 0, FLAGS, "matrix" },
    {     "bt2020",   0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_SPC_BT2020_NCL},      0, 0, FLAGS, "matrix" },
    { "primaries",    "set color primaries", OFFSET(pri), AV_OPT_TYPE_INT, {.i64 = AVCOL_PRI_BT709}, -1, INT_MAX, FLAGS, "primaries" },
    { "p",            "set color primaries", OFFSET(pri), AV_OPT_TYPE_INT, {.i64 = AVCOL_PRI_BT709}, -1, INT_MAX, FLAGS, "primaries" },
    {     "bt709",    0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_PRI_BT709},           0, 0, FLAGS, "primaries" },
    {     "bt2020",   0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_PRI_BT2020},          0, 0, FLAGS, "primaries" },
    { "range",        "set color range", OFFSET(range), AV_OPT_TYPE_INT, {.i64 = AVCOL_RANGE_MPEG}, -1, INT_MAX, FLAGS, "range" },
    { "r",            "set color range", OFFSET(range), AV_OPT_TYPE_INT, {.i64 = AVCOL_RANGE_MPEG}, -1, INT_MAX, FLAGS, "range" },
    {     "tv",       0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_RANGE_MPEG},          0, 0, FLAGS, "range" },
    {     "pc",       0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_RANGE_JPEG},          0, 0, FLAGS, "range" },
    {     "limited",  0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_RANGE_MPEG},          0, 0, FLAGS, "range" },
    {     "full",     0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_RANGE_JPEG},          0, 0, FLAGS, "range" },
    { "format",       "Output format",       OFFSET(format_str), AV_OPT_TYPE_STRING, { .str = "same" }, .flags = FLAGS },
    { "peak",         "signal peak override", OFFSET(peak), AV_OPT_TYPE_DOUBLE, {.dbl = 0}, 0, DBL_MAX, FLAGS },
    { "param",        "tonemap parameter",   OFFSET(param), AV_OPT_TYPE_DOUBLE, {.dbl = NAN}, DBL_MIN, DBL_MAX, FLAGS },
    { "desat",        "desaturation parameter",   OFFSET(desat_param), AV_OPT_TYPE_DOUBLE, {.dbl = 0.5}, 0, DBL_MAX, FLAGS },
    { "threshold",    "scene detection threshold",   OFFSET(scene_threshold), AV_OPT_TYPE_DOUBLE, {.dbl = 0.2}, 0, DBL_MAX, FLAGS },
    { NULL },
};

static const AVClass tonemap_cuda_class = {
    .class_name = "tonemap_cuda",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad inputs[] = {
    {
        .name        = "default",
        .type        = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_props,
    },
    { NULL }
};

AVFilter ff_vf_tonemap_cuda = {
    .name        = "tonemap_cuda",
    .description = NULL_IF_CONFIG_SMALL("GPU accelerated HDR to SDR tonemapping"),

    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,

    .priv_size  = sizeof(TonemapCUDAContext),
    .priv_class = &tonemap_cuda_class,

    .inputs  = inputs,
    .outputs = outputs,

    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
