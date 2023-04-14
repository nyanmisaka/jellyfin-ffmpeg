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
#include "dither_matrix.h"

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P016
};

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

    CUfunction  cu_func_tm;
    CUfunction  cu_func_dovi;
    CUfunction  cu_func_dovi_pq;

    CUdeviceptr ditherBuffer;
    CUtexObject ditherTex;

#define params_cnt 8
#define pivots_cnt (7+1)
#define coeffs_cnt 8*4
#define mmr_cnt 8*6*4
#define params_sz params_cnt*sizeof(float)
#define pivots_sz pivots_cnt*sizeof(float)
#define coeffs_sz coeffs_cnt*sizeof(float)
#define mmr_sz mmr_cnt*sizeof(float)
    CUdeviceptr doviBuffer;
    struct DoviMetadata *dovi;
    float *dovi_pbuf;

    enum TonemapAlgorithm tonemap;
    enum TonemapMode tonemap_mode;
    int apply_dovi;
    int tradeoff;
    int init_with_dovi;
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

    s->dovi = NULL;
    s->doviBuffer = NULL;

    return 0;
}

static av_cold void uninit_dovi(AVFilterContext *ctx)
{
    TonemapCUDAContext *s = ctx->priv;

    if (s->hwctx) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy, cuda_ctx = s->hwctx->cuda_ctx;

        CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));

        if (s->doviBuffer) {
            CHECK_CU(cu->cuMemFree(s->doviBuffer));
            s->doviBuffer = 0;
        }

        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }

    if (s->dovi)
        av_freep(&s->dovi);
    if (s->dovi_pbuf)
        av_freep(&s->dovi_pbuf);

    s->init_with_dovi = 0;
}

static av_cold void uninit_common(AVFilterContext *ctx)
{
    TonemapCUDAContext *s = ctx->priv;

    if (s->hwctx) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy, cuda_ctx = s->hwctx->cuda_ctx;

        CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));

        if (s->ditherTex) {
            CHECK_CU(cu->cuTexObjectDestroy(s->ditherTex));
            s->ditherTex = 0;
        }
        if (s->ditherBuffer) {
            CHECK_CU(cu->cuMemFree(s->ditherBuffer));
            s->ditherBuffer = 0;
        }
        if (s->cu_module) {
            CHECK_CU(cu->cuModuleUnload(s->cu_module));
            s->cu_func_tm = NULL;
            s->cu_func_dovi = NULL;
            s->cu_func_dovi_pq = NULL;
            s->cu_module = NULL;
        }

        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }
}

static av_cold void uninit(AVFilterContext *ctx)
{
    TonemapCUDAContext *s = ctx->priv;

    uninit_common(ctx);
    uninit_dovi(ctx);

    av_frame_free(&s->frame);
    av_buffer_unref(&s->frames_ctx);
    av_frame_free(&s->tmp_frame);
}

static av_cold int setup_dither(AVFilterContext *ctx)
{
    TonemapCUDAContext  *s = ctx->priv;
    AVFilterLink        *inlink = ctx->inputs[0];
    AVHWFramesContext   *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;
    CudaFunctions       *cu = device_hwctx->internal->cuda_dl;
    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;
    int ret = 0;

    CUDA_MEMCPY2D cpy = {
        .srcMemoryType = CU_MEMORYTYPE_HOST,
        .dstMemoryType = CU_MEMORYTYPE_DEVICE,
        .srcHost       = ff_fruit_dither_matrix,
        .dstDevice     = 0,
        .srcPitch      = ff_fruit_dither_size * sizeof(ff_fruit_dither_matrix[0]),
        .dstPitch      = ff_fruit_dither_size * sizeof(ff_fruit_dither_matrix[0]),
        .WidthInBytes  = ff_fruit_dither_size * sizeof(ff_fruit_dither_matrix[0]),
        .Height        = ff_fruit_dither_size,
    };

#ifndef CU_TRSF_NORMALIZED_COORDINATES
  #define CU_TRSF_NORMALIZED_COORDINATES 2
#endif
    CUDA_TEXTURE_DESC tex_desc = {
        .addressMode = CU_TR_ADDRESS_MODE_WRAP,
        .filterMode = CU_TR_FILTER_MODE_POINT,
        .flags = CU_TRSF_NORMALIZED_COORDINATES,
    };

    CUDA_RESOURCE_DESC res_desc = {
        .resType = CU_RESOURCE_TYPE_PITCH2D,
        .res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT16,
        .res.pitch2D.numChannels = 1,
        .res.pitch2D.width = ff_fruit_dither_size,
        .res.pitch2D.height = ff_fruit_dither_size,
        .res.pitch2D.pitchInBytes = ff_fruit_dither_size * sizeof(ff_fruit_dither_matrix[0]),
        .res.pitch2D.devPtr = 0,
    };

    av_assert0(sizeof(ff_fruit_dither_matrix) == sizeof(ff_fruit_dither_matrix[0]) * ff_fruit_dither_size * ff_fruit_dither_size);

    if ((ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx))) < 0)
        return ret;

    if ((ret = CHECK_CU(cu->cuMemAlloc(&s->ditherBuffer, sizeof(ff_fruit_dither_matrix)))) < 0)
        goto fail;

    res_desc.res.pitch2D.devPtr = cpy.dstDevice = s->ditherBuffer;

    if ((ret = CHECK_CU(cu->cuMemcpy2D(&cpy))) < 0)
        goto fail;

    if ((ret = CHECK_CU(cu->cuTexObjectCreate(&s->ditherTex, &res_desc, &tex_desc, NULL))) < 0)
        goto fail;

fail:
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return ret;
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

static const double dovi_lms2rgb_matrix[3][3] =
{
    { 3.06441879, -2.16597676,  0.10155818},
    {-0.65612108,  1.78554118, -0.12943749},
    { 0.01736321, -0.04725154,  1.03004253},
};

static int get_rgb2rgb_matrix(enum AVColorPrimaries in, enum AVColorPrimaries out,
                              double rgb2rgb[3][3]) {
    double rgb2xyz[3][3], xyz2rgb[3][3];

    const AVColorPrimariesDesc *in_primaries = av_csp_primaries_desc_from_id(in);
    const AVColorPrimariesDesc *out_primaries = av_csp_primaries_desc_from_id(out);

    if (!in_primaries || !out_primaries)
        return AVERROR(EINVAL);

    ff_fill_rgb2xyz_table(&out_primaries->prim, &out_primaries->wp, rgb2xyz);
    ff_matrix_invert_3x3(rgb2xyz, xyz2rgb);
    ff_fill_rgb2xyz_table(&in_primaries->prim, &in_primaries->wp, rgb2xyz);
    ff_matrix_mul_3x3(rgb2rgb, rgb2xyz, xyz2rgb);

    return 0;
}

static void update_dovi_buf(AVFilterContext *ctx)
{
    TonemapCUDAContext *s = ctx->priv;
    float coeffs_data[8][4] = {0};
    float mmr_packed_data[8*6][4] = {0};
    int c, i, j, k;

    for (c = 0; c < 3; c++) {
        int has_poly = 0, has_mmr = 0, mmr_single = 1;
        int mmr_idx = 0, min_order = 3, max_order = 1;
        const struct ReshapeData *comp = &s->dovi->comp[c];
        if (!comp->num_pivots)
            continue;
        av_assert0(comp->num_pivots >= 2 && comp->num_pivots <= 9);

        memset(coeffs_data, 0, sizeof(coeffs_data));
        for (i = 0; i < comp->num_pivots - 1; i++) {
            switch (comp->method[i]) {
            case 0: // polynomial
                has_poly = 1;
                coeffs_data[i][3] = 0.0f; // order=0 signals polynomial
                for (k = 0; k < 3; k++)
                    coeffs_data[i][k] = comp->poly_coeffs[i][k];
                break;
            case 1:
                min_order = FFMIN(min_order, comp->mmr_order[i]);
                max_order = FFMAX(max_order, comp->mmr_order[i]);
                mmr_single = !has_mmr;
                has_mmr = 1;
                coeffs_data[i][3] = (float)comp->mmr_order[i];
                coeffs_data[i][0] = comp->mmr_constant[i];
                coeffs_data[i][1] = (float)mmr_idx;
                for (j = 0; j < comp->mmr_order[i]; j++) {
                    // store weights per order as two packed vec4s
                    float *mmr = &mmr_packed_data[mmr_idx][0];
                    mmr[0] = comp->mmr_coeffs[i][j][0];
                    mmr[1] = comp->mmr_coeffs[i][j][1];
                    mmr[2] = comp->mmr_coeffs[i][j][2];
                    mmr[3] = 0.0f; // unused
                    mmr[4] = comp->mmr_coeffs[i][j][3];
                    mmr[5] = comp->mmr_coeffs[i][j][4];
                    mmr[6] = comp->mmr_coeffs[i][j][5];
                    mmr[7] = comp->mmr_coeffs[i][j][6];
                    mmr_idx += 2;
                }
                break;
            default:
                av_assert0(0);
            }
        }

        av_assert0(has_poly || has_mmr);

        if (has_mmr)
            av_assert0(min_order <= max_order);

        // dovi_params
        {
            float params[8] = {
                comp->num_pivots, !!has_mmr, !!has_poly,
                mmr_single, min_order, max_order,
                comp->pivots[0], comp->pivots[comp->num_pivots - 1]
            };
            memcpy(s->dovi_pbuf + c*params_cnt, params, params_sz);
        }

        // dovi_pivots
        if (c == 0 && comp->num_pivots > 2) {
            // Skip the (irrelevant) lower and upper bounds
            float pivots_data[7+1] = {0};
            memcpy(pivots_data, comp->pivots + 1,
                   (comp->num_pivots - 2) * sizeof(pivots_data[0]));
            // Fill the remainder with a quasi-infinite sentinel pivot
            for (i = comp->num_pivots - 2; i < FF_ARRAY_ELEMS(pivots_data); i++)
                pivots_data[i] = 1e9f;
            memcpy(s->dovi_pbuf + 3*params_cnt + c*pivots_cnt, pivots_data, pivots_sz);
        }

        // dovi_coeffs
        memcpy(s->dovi_pbuf + 3*(params_cnt+pivots_cnt) + c*coeffs_cnt, &coeffs_data[0], coeffs_sz);

        // dovi_mmr
        if (has_mmr)
            memcpy(s->dovi_pbuf + 3*(params_cnt+pivots_cnt+coeffs_cnt) + c*mmr_cnt, &mmr_packed_data[0], mmr_sz);
    }
}

static av_cold int compile(AVFilterLink *inlink)
{
    AVFilterContext  *ctx = inlink->dst;
    TonemapCUDAContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy, cuda_ctx = s->hwctx->cuda_ctx;
    AVBPrint constants;
    CUlinkState link_state;
    int i, j, ret = 0;
    void *cubin;
    size_t cubin_size;
    double ycc2rgb_offset[3] = {0};
    double lms2rgb_matrix[3][3] = {0};
    double rgb_matrix[3][3], yuv_matrix[3][3], rgb2rgb_matrix[3][3];
    const AVLumaCoefficients *in_coeffs, *out_coeffs;
    enum AVColorTransferCharacteristic in_trc = s->in_trc, out_trc = s->out_trc;
    enum AVColorSpace in_spc = s->in_spc, out_spc = s->out_spc;
    enum AVColorPrimaries in_pri = s->in_pri, out_pri = s->out_pri;
    enum AVColorRange in_range = s->in_range, out_range = s->out_range;
    int rgb = s->tonemap_mode == TONEMAP_MODE_RGB;
    int d = s->in_desc->comp[0].depth > s->out_desc->comp[0].depth && s->ditherTex;
    char info_log[4096], error_log[4096];
    CUjit_option options[] = { CU_JIT_INFO_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
    void *option_values[]  = { &info_log,
                               &error_log,
                               (void*)(intptr_t)sizeof(info_log),
                               (void*)(intptr_t)sizeof(error_log) };

    extern const unsigned char ff_tonemap_ptx_data[];
    extern const unsigned int ff_tonemap_ptx_len;

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

    s->ref_white = s->tonemap == TONEMAP_BT2390 ? REFERENCE_WHITE_ALT
                                                : REFERENCE_WHITE;

    if (s->tonemap == TONEMAP_BT2390 && s->peak)
        s->peak = FFMAX(s->peak / 10.0f, 1.1f);

    s->dst_peak = 1.0f;

    if (in_trc == AVCOL_TRC_UNSPECIFIED)
        in_trc = AVCOL_TRC_SMPTE2084;
    if (out_trc == AVCOL_TRC_UNSPECIFIED)
        out_trc = AVCOL_TRC_BT709;

    if (!s->dovi && in_spc == AVCOL_SPC_UNSPECIFIED)
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

    if (out_trc == AVCOL_TRC_SMPTE2084) {
        int is_10_or_16b_out = s->out_desc->comp[0].depth == 10 ||
                               s->out_desc->comp[0].depth == 16;
        if (!(is_10_or_16b_out &&
            out_pri == AVCOL_PRI_BT2020 &&
            out_spc == AVCOL_SPC_BT2020_NCL)) {
            av_log(ctx, AV_LOG_ERROR, "HDR passthrough requires BT.2020 "
                   "colorspace and 10/16 bit output format depth.\n");
            return AVERROR(EINVAL);
        }
    }

    av_log(ctx, AV_LOG_DEBUG, "Tonemapping transfer from %s to %s\n",
           av_color_transfer_name(in_trc),
           av_color_transfer_name(out_trc));
    av_log(ctx, AV_LOG_DEBUG, "Mapping colorspace from %s to %s\n",
           s->dovi ? "dolby_vision" : av_color_space_name(in_spc),
           av_color_space_name(out_spc));
    av_log(ctx, AV_LOG_DEBUG, "Mapping primaries from %s to %s\n",
           av_color_primaries_name(in_pri),
           av_color_primaries_name(out_pri));
    av_log(ctx, AV_LOG_DEBUG, "Mapping range from %s to %s\n",
           av_color_range_name(in_range),
           av_color_range_name(out_range));

    if (s->dovi) {
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                ycc2rgb_offset[i] -= s->dovi->nonlinear[i][j] * s->dovi->nonlinear_offset[j];
        }
        ff_matrix_mul_3x3(lms2rgb_matrix, dovi_lms2rgb_matrix, s->dovi->linear);
    } else {
        if (!(in_coeffs = av_csp_luma_coeffs_from_avcsp(in_spc)))
            return AVERROR(EINVAL);

        ff_fill_rgb2yuv_table(in_coeffs, yuv_matrix);
        ff_matrix_invert_3x3(yuv_matrix, rgb_matrix);
    }

    if (!(out_coeffs = av_csp_luma_coeffs_from_avcsp(out_spc)))
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
#define CONSTANT_C(a, b, c, d) \
    CONSTANT(".f32 " a "[] = {%f, %f, %f}", \
             b, c, d)

    CONSTANT(".u32 depth_src           = %i", (int)s->in_desc->comp[0].depth);
    CONSTANT(".u32 depth_dst           = %i", (int)s->out_desc->comp[0].depth);
    CONSTANT(".u32 fmt_src             = %i", (int)s->in_fmt);
    CONSTANT(".u32 fmt_dst             = %i", (int)s->out_fmt);
    CONSTANT(".u32 range_src           = %i", (int)in_range);
    CONSTANT(".u32 range_dst           = %i", (int)out_range);
    CONSTANT(".u32 trc_src             = %i", (int)in_trc);
    CONSTANT(".u32 trc_dst             = %i", (int)out_trc);
    CONSTANT(".u32 chroma_loc_src      = %i", (int)s->in_chroma_loc);
    CONSTANT(".u32 chroma_loc_dst      = %i", (int)s->out_chroma_loc);
    CONSTANT(".u32 tonemap_func        = %i", (int)s->tonemap);
    CONSTANT(".u32 enable_dither       = %i", (int)(s->in_desc->comp[0].depth > s->out_desc->comp[0].depth));
    CONSTANT(".f32 dither_size         = %f", (float)ff_fruit_dither_size);
    CONSTANT(".f32 dither_quantization = %f", (float)((1 << s->out_desc->comp[0].depth) - 1));
    CONSTANT(".f32 ref_white           = %f", s->ref_white);
    CONSTANT(".f32 tone_param          = %f", s->param);
    CONSTANT(".f32 desat_param         = %f", s->desat_param);
    CONSTANT(".f32 pq_max_lum_div_ref_white = %f", (float)(ST2084_MAX_LUMINANCE / s->ref_white));
    CONSTANT(".f32 ref_white_div_pq_max_lum = %f", (float)(s->ref_white / ST2084_MAX_LUMINANCE));
    CONSTANT_M("rgb_matrix", (s->dovi ? s->dovi->nonlinear : rgb_matrix));
    CONSTANT_M("yuv_matrix", yuv_matrix);
    CONSTANT_A(".u8 rgb2rgb_passthrough = %i", 1, in_pri == out_pri);
    CONSTANT_M("rgb2rgb_matrix", rgb2rgb_matrix);
    CONSTANT_M("lms2rgb_matrix", lms2rgb_matrix);
    CONSTANT_C("luma_dst", av_q2d(out_coeffs->cr), av_q2d(out_coeffs->cg), av_q2d(out_coeffs->cb));
    CONSTANT_C("ycc2rgb_offset", ycc2rgb_offset[0], ycc2rgb_offset[1], ycc2rgb_offset[2]);

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        return ret;

    if (s->dovi) {
        s->dovi_pbuf = av_mallocz(3*(params_sz+pivots_sz+coeffs_sz+mmr_sz));
        ret = CHECK_CU(cu->cuMemAlloc(&s->doviBuffer, 3*(params_sz+pivots_sz+coeffs_sz+mmr_sz)));
        if (ret < 0)
            goto fail;
    }

    if (s->dovi && s->tradeoff == -1) {
        int major, minor, mp;
        s->tradeoff = 0;

        ret = CHECK_CU(cu->cuDeviceComputeCapability(&major, &minor, s->hwctx->internal->cuda_device));
        if (ret < 0)
            return ret;

        ret = CHECK_CU(cu->cuDeviceGetAttribute(&mp,
                                                CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                                s->hwctx->internal->cuda_device));
        if (ret < 0)
            return ret;

        switch (major) {
        case 1:
        case 2:
            s->tradeoff = 1; break;
        case 3:
            s->tradeoff = mp * 192 < 1024; break;
        case 5:
            s->tradeoff = mp * 128 < 1024; break;
        case 6:
            if (minor == 0) s->tradeoff = mp * 64 < 1024;
            if (minor == 1 || minor == 2) s->tradeoff = mp * 128 < 1024;
            break;
        case 7:
            s->tradeoff = mp * 64 < 512; break;
        }

        if (!s->tradeoff)
            av_log(ctx, AV_LOG_DEBUG, "Disabled dovi tradeoff on high perf GPU.\n");
    }

    if (s->cu_module) {
        ret = CHECK_CU(cu->cuModuleUnload(s->cu_module));
        if (ret < 0)
            goto fail;

        s->cu_func_tm = NULL;
        s->cu_func_dovi = NULL;
        s->cu_func_dovi_pq = NULL;
        s->cu_module = NULL;
    }

    ret = CHECK_CU(cu->cuLinkCreate(sizeof(options) / sizeof(options[0]), options, option_values, &link_state));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuLinkAddData(link_state, CU_JIT_INPUT_PTX, constants.str,
                                     constants.len, "constants", 0, NULL, NULL));
    if (ret < 0)
        goto fail2;

    ret = CHECK_CU(cu->cuLinkAddData(link_state, CU_JIT_INPUT_PTX, ff_tonemap_ptx_data,
                                     ff_tonemap_ptx_len, "ff_tonemap_ptx_data", 0, NULL, NULL));
    if (ret < 0)
        goto fail2;

    ret = CHECK_CU(cu->cuLinkComplete(link_state, &cubin, &cubin_size));
    if (ret < 0)
        goto fail2;

    ret = CHECK_CU(cu->cuModuleLoadData(&s->cu_module, cubin));
    if (ret < 0)
        goto fail2;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_tm, s->cu_module,
                                           rgb ? (d ? "tonemap_rgb_d" : "tonemap_rgb")
                                               : (d ? "tonemap_d" : "tonemap")));
    if (ret < 0)
        goto fail2;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_dovi, s->cu_module,
                                           s->tradeoff == 1 ? (rgb ? (d ? "tonemap_dovi_rgb_d_f" : "tonemap_dovi_rgb_f")
                                                                   : (d ? "tonemap_dovi_d_f" : "tonemap_dovi_f"))
                                                            : (rgb ? (d ? "tonemap_dovi_rgb_d" : "tonemap_dovi_rgb")
                                                                   : (d ? "tonemap_dovi_d" : "tonemap_dovi"))));
    if (ret < 0)
        goto fail2;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_dovi_pq, s->cu_module,
                                           s->tradeoff == 1 ? "tonemap_dovi_pq_f"
                                                            : "tonemap_dovi_pq"));
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

    if (s->in_desc->comp[0].depth > s->out_desc->comp[0].depth) {
        if ((ret = setup_dither(ctx)) < 0)
            return ret;
    }

    outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;

    return 0;
}

static int run_kernel(AVFilterContext *ctx,
                      AVFrame *out, AVFrame *in)
{
    TonemapCUDAContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    FFCUDAFrame src, dst;
    void *args[] = { &src, &dst, &s->ditherTex, &s->doviBuffer };
    int ret, pq_out = s->out_trc == AVCOL_TRC_SMPTE2084;

    ret = ff_make_cuda_frame(ctx, cu, 1,
                             &src, in, s->in_desc);
    if (ret < 0)
        goto fail;

    ret = ff_make_cuda_frame(ctx, cu, 0,
                             &dst, out, s->out_desc);
    if (ret < 0)
        goto fail;

    src.peak = s->peak;
    dst.peak = s->dst_peak;

    ret = CHECK_CU(cu->cuLaunchKernel(s->dovi ? (pq_out ? s->cu_func_dovi_pq : s->cu_func_dovi) : s->cu_func_tm,
                                      DIV_UP(src.width / 2, BLOCKX), DIV_UP(src.height / 2, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, s->hwctx->stream, args, NULL));

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
    AVFrameSideData *dovi_sd = NULL;
    CUcontext dummy;
    int ret = 0;

    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    if (s->apply_dovi)
        dovi_sd = av_frame_get_side_data(in, AV_FRAME_DATA_DOVI_METADATA);

    // check DOVI->HDR10/HLG
    if (!dovi_sd) {
        if (in->color_trc != AVCOL_TRC_SMPTE2084 &&
            in->color_trc != AVCOL_TRC_ARIB_STD_B67) {
            av_log(ctx, AV_LOG_ERROR, "No DOVI metadata and "
                   "unsupported input transfer characteristic: %s\n",
                   av_color_transfer_name(in->color_trc));
            ret = AVERROR(EINVAL);
            goto fail;
        }
    }

    if (!s->peak) {
        if (dovi_sd) {
            const AVDOVIMetadata *metadata = (AVDOVIMetadata *) dovi_sd->data;
            s->peak = ff_determine_dovi_signal_peak(metadata);
        } else {
            s->peak = ff_determine_signal_peak(in);
        }
        av_log(ctx, AV_LOG_DEBUG, "Computed signal peak: %f\n", s->peak);
    }

    if (dovi_sd) {
        const AVDOVIMetadata *metadata = (AVDOVIMetadata *) dovi_sd->data;
        const AVDOVIRpuDataHeader *rpu = av_dovi_get_header(metadata);
        // only map dovi rpus that don't require an EL
        if (rpu->disable_residual_flag) {
            struct DoviMetadata *dovi = av_malloc(sizeof(*dovi));
            s->dovi = dovi;
            if (!s->dovi)
                goto fail;

            ff_map_dovi_metadata(s->dovi, metadata);
            in->color_trc = AVCOL_TRC_SMPTE2084;
            in->colorspace = AVCOL_SPC_UNSPECIFIED;
            in->color_primaries = AVCOL_PRI_BT2020;
        }
    }

    if (!s->init_with_dovi && s->dovi && s->cu_func_tm)
        uninit_common(ctx);

    if (!s->cu_func_tm ||
        !s->cu_func_dovi ||
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

        s->init_with_dovi = !!s->dovi;
    }

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;

    if (s->dovi) {
        update_dovi_buf(ctx);

        ret = CHECK_CU(cu->cuMemcpyHtoDAsync(s->doviBuffer, s->dovi_pbuf,
                                             3*(params_sz+pivots_sz+coeffs_sz+mmr_sz), s->hwctx->stream));
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Failed to update dovi buf.\n");
            goto fail;
        }
    }

    ret = do_tonemap(ctx, out, in);

    if (s->dovi)
        av_freep(&s->dovi);

    ret = CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        goto fail;

    av_frame_free(&in);

    if (s->out_trc != AVCOL_TRC_SMPTE2084) {
        av_frame_remove_side_data(out, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
        av_frame_remove_side_data(out, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    }

    av_frame_remove_side_data(out, AV_FRAME_DATA_DOVI_RPU_BUFFER);
    av_frame_remove_side_data(out, AV_FRAME_DATA_DOVI_METADATA);

    return ff_filter_frame(outlink, out);
fail:
    if (s->dovi)
        av_freep(&s->dovi);
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

#define OFFSET(x) offsetof(TonemapCUDAContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption options[] = {
    { "tonemap",       "Tonemap algorithm selection", OFFSET(tonemap), AV_OPT_TYPE_INT, {.i64 = TONEMAP_NONE}, TONEMAP_NONE, TONEMAP_COUNT - 1, FLAGS, "tonemap" },
    {     "none",      0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_NONE},              0, 0, FLAGS, "tonemap" },
    {     "linear",    0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_LINEAR},            0, 0, FLAGS, "tonemap" },
    {     "gamma",     0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_GAMMA},             0, 0, FLAGS, "tonemap" },
    {     "clip",      0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_CLIP},              0, 0, FLAGS, "tonemap" },
    {     "reinhard",  0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_REINHARD},          0, 0, FLAGS, "tonemap" },
    {     "hable",     0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_HABLE},             0, 0, FLAGS, "tonemap" },
    {     "mobius",    0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_MOBIUS},            0, 0, FLAGS, "tonemap" },
    {     "bt2390",    0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_BT2390},            0, 0, FLAGS, "tonemap" },
    { "tonemap_mode",  "Tonemap mode selection", OFFSET(tonemap_mode), AV_OPT_TYPE_INT, {.i64 = TONEMAP_MODE_MAX}, TONEMAP_MODE_MAX, TONEMAP_MODE_COUNT - 1, FLAGS, "tonemap_mode" },
    {     "max",       0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_MODE_MAX},          0, 0, FLAGS, "tonemap_mode" },
    {     "rgb",       0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_MODE_RGB},          0, 0, FLAGS, "tonemap_mode" },
    { "transfer",      "Set transfer characteristic", OFFSET(trc), AV_OPT_TYPE_INT, {.i64 = AVCOL_TRC_BT709}, -1, INT_MAX, FLAGS, "transfer" },
    { "t",             "Set transfer characteristic", OFFSET(trc), AV_OPT_TYPE_INT, {.i64 = AVCOL_TRC_BT709}, -1, INT_MAX, FLAGS, "transfer" },
    {     "bt709",     0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_TRC_BT709},           0, 0, FLAGS, "transfer" },
    {     "bt2020",    0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_TRC_BT2020_10},       0, 0, FLAGS, "transfer" },
    {     "smpte2084", 0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_TRC_SMPTE2084},       0, 0, FLAGS, "transfer" },
    { "matrix",        "Set colorspace matrix", OFFSET(spc), AV_OPT_TYPE_INT, {.i64 = AVCOL_SPC_BT709}, -1, INT_MAX, FLAGS, "matrix" },
    { "m",             "Set colorspace matrix", OFFSET(spc), AV_OPT_TYPE_INT, {.i64 = AVCOL_SPC_BT709}, -1, INT_MAX, FLAGS, "matrix" },
    {     "bt709",     0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_SPC_BT709},           0, 0, FLAGS, "matrix" },
    {     "bt2020",    0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_SPC_BT2020_NCL},      0, 0, FLAGS, "matrix" },
    { "primaries",     "Set color primaries", OFFSET(pri), AV_OPT_TYPE_INT, {.i64 = AVCOL_PRI_BT709}, -1, INT_MAX, FLAGS, "primaries" },
    { "p",             "Set color primaries", OFFSET(pri), AV_OPT_TYPE_INT, {.i64 = AVCOL_PRI_BT709}, -1, INT_MAX, FLAGS, "primaries" },
    {     "bt709",     0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_PRI_BT709},           0, 0, FLAGS, "primaries" },
    {     "bt2020",    0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_PRI_BT2020},          0, 0, FLAGS, "primaries" },
    { "range",         "Set color range", OFFSET(range), AV_OPT_TYPE_INT, {.i64 = AVCOL_RANGE_MPEG}, -1, INT_MAX, FLAGS, "range" },
    { "r",             "Set color range", OFFSET(range), AV_OPT_TYPE_INT, {.i64 = AVCOL_RANGE_MPEG}, -1, INT_MAX, FLAGS, "range" },
    {     "tv",        0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_RANGE_MPEG},          0, 0, FLAGS, "range" },
    {     "pc",        0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_RANGE_JPEG},          0, 0, FLAGS, "range" },
    {     "limited",   0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_RANGE_MPEG},          0, 0, FLAGS, "range" },
    {     "full",      0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_RANGE_JPEG},          0, 0, FLAGS, "range" },
    { "format",        "Output format",       OFFSET(format_str), AV_OPT_TYPE_STRING, { .str = "same" }, .flags = FLAGS },
    { "apply_dovi",    "Apply Dolby Vision metadata if possible", OFFSET(apply_dovi), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { "tradeoff",      "Apply tradeoffs to offload computing", OFFSET(tradeoff), AV_OPT_TYPE_INT, {.i64 = -1}, -1, 1, FLAGS, "tradeoff" },
    {     "auto",      0, 0, AV_OPT_TYPE_CONST, {.i64 = -1},                        0, 0, FLAGS, "tradeoff" },
    {     "disabled",  0, 0, AV_OPT_TYPE_CONST, {.i64 = 0},                         0, 0, FLAGS, "tradeoff" },
    {     "enabled",   0, 0, AV_OPT_TYPE_CONST, {.i64 = 1},                         0, 0, FLAGS, "tradeoff" },
    { "peak",          "Signal peak override", OFFSET(peak), AV_OPT_TYPE_DOUBLE, {.dbl = 0}, 0, DBL_MAX, FLAGS },
    { "param",         "Tonemap parameter",   OFFSET(param), AV_OPT_TYPE_DOUBLE, {.dbl = NAN}, DBL_MIN, DBL_MAX, FLAGS },
    { "desat",         "Desaturation parameter",   OFFSET(desat_param), AV_OPT_TYPE_DOUBLE, {.dbl = 0.5}, 0, DBL_MAX, FLAGS },
    { "threshold",     "Scene detection threshold",   OFFSET(scene_threshold), AV_OPT_TYPE_DOUBLE, {.dbl = 0.2}, 0, DBL_MAX, FLAGS },
    { NULL },
};

static const AVClass tonemap_cuda_class = {
    .class_name = "tonemap_cuda",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad tonemap_cuda_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
};

static const AVFilterPad tonemap_cuda_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_props,
    },
};

const AVFilter ff_vf_tonemap_cuda = {
    .name           = "tonemap_cuda",
    .description    = NULL_IF_CONFIG_SMALL("GPU accelerated HDR to SDR tonemapping"),

    .init           = init,
    .uninit         = uninit,

    .priv_size      = sizeof(TonemapCUDAContext),
    .priv_class     = &tonemap_cuda_class,

    FILTER_INPUTS(tonemap_cuda_inputs),
    FILTER_OUTPUTS(tonemap_cuda_outputs),

    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_CUDA),

    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
