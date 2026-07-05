/*
 * Copyright (c) 2024 Gnattu OC <gnattuoc@me.com>
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
#include "libavutil/objc.h"
#include "libavutil/hwcontext.h"

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "video.h"
#include "colorspace.h"
#include "dither_matrix.h"
#include "metal/utils.h"
#include "libavutil/hwcontext_videotoolbox.h"

#define params_cnt 8
#define pivots_cnt (7+1)
#define coeffs_cnt (8*4)
#define mmr_cnt (8*6*4)
#define params_sz params_cnt*sizeof(float)
#define pivots_sz pivots_cnt*sizeof(float)
#define coeffs_sz coeffs_cnt*sizeof(float)
#define mmr_sz mmr_cnt*sizeof(float)

#define REF_WHITE_SCALE (REFERENCE_WHITE / REFERENCE_WHITE_ALT)

extern char ff_vf_tonemap_videotoolbox_metallib_data[];
extern unsigned int ff_vf_tonemap_videotoolbox_metallib_len;

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
};

static const int colorspaces_out[] = {
    AVCOL_SPC_UNSPECIFIED,
    AVCOL_SPC_BT709,
    AVCOL_SPC_BT2020_NCL,
    -1
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
    TONEMAP_COUNT,
};

enum TonemapMode {
    TONEMAP_MODE_MAX,
    TONEMAP_MODE_RGB,
    TONEMAP_MODE_LUM,
    TONEMAP_MODE_ITP,
    TONEMAP_MODE_COUNT,
};

typedef struct TonemapVideoToolboxContext {
    const AVClass *class;
    enum AVColorSpace colorspace, colorspace_in, colorspace_out;
    enum AVColorTransferCharacteristic trc, trc_in, trc_out;
    enum AVColorPrimaries primaries, primaries_in, primaries_out;
    enum AVColorRange range, range_in, range_out;
    enum AVChromaLocation chroma_loc;
    enum AVPixelFormat in_fmt, out_fmt;
    const AVPixFmtDescriptor *in_desc, *out_desc;
    int in_planes, out_planes;
    struct FFDOVIMetadataRemap *dovi;
    /* enum TonemapAlgorithm */
    int                         tonemap;
    /* enum TonemapMode */
    int                         tonemap_mode;
    enum AVPixelFormat          format;
    int                         apply_dovi;
    double                      peak;
    double                      src_peak;
    double                      target_peak;
    double                      param;
    double                      final_param;
    double                      desat_param;
    double                      scene_threshold;
    int                         initialised;
    int                         init_with_dovi;

    id<MTLTexture>              dither_texture;
    id<MTLDevice>               mtl_device;
    id<MTLLibrary>              mtl_library;
    id<MTLCommandQueue>         mtl_queue;
    id<MTLComputePipelineState> mtl_pipeline;
    id<MTLFunction>             mtl_function;
    id<MTLBuffer>               mtl_dovi_buffer;
    id<MTLBuffer>               mtl_peak_buffer;
    CVMetalTextureCacheRef      texture_cache;
} TonemapVideoToolboxContext;

static const short linearize_funcs[] = {
    [AVCOL_TRC_SMPTE2084]    = 1, //"eotf_st2084",
    [AVCOL_TRC_ARIB_STD_B67] = 2, //"eotf_arib_b67",
};

static const short delinearize_funcs[] = {
    [AVCOL_TRC_BT709]     = 1, //"inverse_eotf_bt1886",
    [AVCOL_TRC_BT2020_10] = 1, //"inverse_eotf_bt1886",
};

static const double dovi_lms2rgb_matrix[3][3] =
    {
        { 3.06441879, -2.16597676,  0.10155818},
        {-0.65612108,  1.78554118, -0.12943749},
        { 0.01736321, -0.04725154,  1.03004253},
    };

static int format_is_supported(enum AVPixelFormat fmt)
{
    for (int i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

static int get_rgb2rgb_matrix(enum AVColorPrimaries in, enum AVColorPrimaries out,
                              double rgb2rgb[3][3])
{
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

static MTLPixelFormat get_plane_texture_format(TonemapVideoToolboxContext* ctx, int plane, bool is_output)
{
    int pixel_size, channels;
    const AVComponentDescriptor *comp;
    MTLPixelFormat format;

    comp = is_output ? &ctx->out_desc->comp[plane] : &ctx->in_desc->comp[plane];
    pixel_size = (comp->depth + comp->shift) / 8;
    channels = comp->step / pixel_size;
    if (pixel_size > 2 || channels > 2) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n", ctx->in_desc->name);
        return MTLPixelFormatInvalid;
    }
    switch (pixel_size) {
        case 1:
            format = channels == 1 ? MTLPixelFormatR8Unorm : MTLPixelFormatRG8Unorm;
            break;
        case 2:
            format = channels == 1 ? MTLPixelFormatR16Unorm : MTLPixelFormatRG16Unorm;
            break;
        default:
            av_log(ctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n", ctx->in_desc->name);
            return MTLPixelFormatInvalid;
    }
    return format;
}

static void tonemap_videotoolbox_update_dovi_buf(AVFilterContext *avctx)
{
    TonemapVideoToolboxContext *ctx = avctx->priv;
    float *dovi_buf = ctx->mtl_dovi_buffer.contents;
    float coeffs_data[8][4] = {0};
    float mmr_packed_data[8*6][4] = {0};
    int c, i, j, k;

    av_assert0(dovi_buf);

    for (c = 0; c < 3; c++) {
        int has_poly = 0, has_mmr = 0, mmr_single = 1;
        int mmr_idx = 0, min_order = 3, max_order = 1;
        const struct FFDOVIReshapeData *comp = &ctx->dovi->comp[c];
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
            memcpy(dovi_buf + c * params_cnt, params, params_sz);
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
            memcpy(dovi_buf + 3 * params_cnt + c * pivots_cnt, pivots_data, pivots_sz);
        }

        // dovi_coeffs
        memcpy(dovi_buf + 3 * (params_cnt + pivots_cnt) + c * coeffs_cnt, &coeffs_data[0], coeffs_sz);

        // dovi_mmr
        if (has_mmr)
            memcpy(dovi_buf + 3 * (params_cnt + pivots_cnt + coeffs_cnt) + c * mmr_cnt, &mmr_packed_data[0], mmr_sz);
    }
}

static av_cold int tonemap_videotoolbox_preinit(AVFilterContext *avctx)
{
    TonemapVideoToolboxContext *ctx = avctx->priv;
    ctx->final_param = NAN;
    return 0;
}

static av_cold void tonemap_videotoolbox_uninit_common(AVFilterContext *avctx)
{
    TonemapVideoToolboxContext *ctx = avctx->priv;

    ff_objc_release(&ctx->dither_texture);
    ff_objc_release(&ctx->mtl_peak_buffer);
    ff_objc_release(&ctx->mtl_function);
    ff_objc_release(&ctx->mtl_pipeline);
    ff_objc_release(&ctx->mtl_queue);
    ff_objc_release(&ctx->mtl_library);
    ff_objc_release(&ctx->mtl_device);
    if (ctx->texture_cache) {
        CFRelease(ctx->texture_cache);
        ctx->texture_cache = NULL;
    }
    ctx->initialised = 0;
}

static av_cold void tonemap_videotoolbox_uninit_dovi(AVFilterContext *avctx)
{
    TonemapVideoToolboxContext *ctx = avctx->priv;
    ff_objc_release(&ctx->mtl_dovi_buffer);
    if (ctx->dovi) {
        av_freep(&ctx->dovi);
    }
    ctx->init_with_dovi = 0;
}

static av_cold void tonemap_videotoolbox_uninit(AVFilterContext *avctx)
{
    tonemap_videotoolbox_uninit_common(avctx);
    tonemap_videotoolbox_uninit_dovi(avctx);
}

static int tonemap_videotoolbox_init(AVFilterContext *avctx)
{
    TonemapVideoToolboxContext *ctx = avctx->priv;
    int rgb2rgb_passthrough = 1;
    double rgb2rgb[3][3], rgb2yuv[3][3], yuv2rgb[3][3];
    double lms2rgb[3][3];
    float ycc2rgb_offset[3] = {0};
    float rgb2rgb_matrix_1[3], rgb2rgb_matrix_2[3], rgb2rgb_matrix_3[3];
    float rgb_matrix_1[3], rgb_matrix_2[3], rgb_matrix_3[3];
    float yuv_matrix_1[3], yuv_matrix_2[3], yuv_matrix_3[3];
    float lms2rgb_matrix_1[3], lms2rgb_matrix_2[3], lms2rgb_matrix_3[3];
    float mtl_luma_src[3], mtl_luma_dst[3];
    const AVLumaCoefficients *luma_src, *luma_dst;

    MTLFunctionConstantValues* constant_values = [MTLFunctionConstantValues new];
    dispatch_data_t lib_data;
    float tone_param;
    float desat_param;
    float target_peak;
    float scene_threshold;
    short tonemap_func_type;
    bool is_tone_func_bt2390;
    bool is_tone_mode_rgb;
    bool is_tone_mode_max;
    bool is_tone_mode_itp;
    bool is_non_semi_planar_in;
    bool is_non_semi_planar_out;
    bool enable_dither;
    float dither_size2;
    float dither_quantization;
    bool is_full_range_in;
    bool is_full_range_out;
    int chroma_loc;
    bool skip_tonemap;
    bool hlg_eotf_bt2446b;
    bool dovi_reshape;
    bool map_in_src_space;

    int i, j, err;
    NSError* ns_error = nil;
    CVReturn ret;

    if (ctx->primaries_out != ctx->primaries_in) {
        if ((err = get_rgb2rgb_matrix(ctx->primaries_in, ctx->primaries_out, rgb2rgb)) < 0)
            goto fail;
        rgb2rgb_passthrough = 0;
    }

    switch(ctx->tonemap) {
    case TONEMAP_GAMMA:
        if (isnan(ctx->param))
            ctx->final_param = 1.8f;
        break;
    case TONEMAP_REINHARD:
        if (!isnan(ctx->param))
            ctx->final_param = (1.0f - ctx->param) / ctx->param;
        break;
    case TONEMAP_MOBIUS:
        if (isnan(ctx->param))
            ctx->final_param = 0.3f;
        break;
    case TONEMAP_BT2390:
        if (isnan(ctx->param))
            ctx->final_param = 1.0f; // diff from the spec-defined 0.5f
        else
            ctx->final_param = FFMIN(FFMAX(ctx->param, 0.5f), 2.0f);
        break;
    }

    if (isnan(ctx->final_param))
        ctx->final_param = 1.0f;

    if (ctx->peak)
        ctx->src_peak = ctx->peak / 10.0f * REF_WHITE_SCALE;

    // sanity check
    if (ctx->src_peak <= REF_WHITE_SCALE)
        ctx->src_peak = 10.0f * REF_WHITE_SCALE;

    // SDR peak is 1.0f
    ctx->target_peak = 1.0f;

    av_log(ctx, AV_LOG_DEBUG, "Tone-mapping transfer from %s to %s\n",
           av_color_transfer_name(ctx->trc_in),
           av_color_transfer_name(ctx->trc_out));
    av_log(ctx, AV_LOG_DEBUG, "Mapping colorspace from %s to %s\n",
           ctx->dovi ? "dolby_vision" : av_color_space_name(ctx->colorspace_in),
           av_color_space_name(ctx->colorspace_out));
    av_log(ctx, AV_LOG_DEBUG, "Mapping primaries from %s to %s\n",
           av_color_primaries_name(ctx->primaries_in),
           av_color_primaries_name(ctx->primaries_out));
    av_log(ctx, AV_LOG_DEBUG, "Mapping range from %s to %s\n",
           av_color_range_name(ctx->range_in),
           av_color_range_name(ctx->range_out));

    av_assert0(ctx->trc_out == AVCOL_TRC_BT709 ||
               ctx->trc_out == AVCOL_TRC_BT2020_10 ||
               ctx->trc_out == AVCOL_TRC_SMPTE2084);

    av_assert0(ctx->trc_in == AVCOL_TRC_SMPTE2084||
               ctx->trc_in == AVCOL_TRC_ARIB_STD_B67);
    av_assert0(ctx->dovi ||
               ctx->colorspace_in == AVCOL_SPC_BT2020_NCL ||
               ctx->colorspace_in == AVCOL_SPC_BT709);
    av_assert0(ctx->primaries_in == AVCOL_PRI_BT2020 ||
               ctx->primaries_in == AVCOL_PRI_BT709);

    if (ctx->trc_out == AVCOL_TRC_SMPTE2084) {
        int is_10b_out = ctx->out_desc->comp[0].depth == 10;
        if (!(is_10b_out &&
              ctx->primaries_out == AVCOL_PRI_BT2020 &&
              ctx->colorspace_out == AVCOL_SPC_BT2020_NCL)) {
            av_log(avctx, AV_LOG_ERROR, "HDR passthrough requires BT.2020 "
                                        "colorspace and 10 bit output format depth.\n");
            return AVERROR(EINVAL);
        }
    }

    ctx->mtl_device = MTLCreateSystemDefaultDevice();
    if (!ctx->mtl_device) {
        av_log(ctx, AV_LOG_ERROR, "Unable to find Metal device\n");
        err = AVERROR_EXTERNAL;
        goto fail;
    }
    av_log(ctx, AV_LOG_INFO, "Using Metal device: %s\n", ctx->mtl_device.name.UTF8String);

    lib_data = dispatch_data_create(
        ff_vf_tonemap_videotoolbox_metallib_data,
        ff_vf_tonemap_videotoolbox_metallib_len,
        nil,
        nil);
    ctx->mtl_library = [ctx->mtl_device newLibraryWithData:lib_data error:&ns_error];
    dispatch_release(lib_data);
    lib_data = nil;
    if (ns_error) {
        av_log(ctx, AV_LOG_ERROR, "Failed to load Metal library: %s\n", ns_error.description.UTF8String);
        err = AVERROR_EXTERNAL;
        goto fail;
    }

    ctx->mtl_queue = ctx->mtl_device.newCommandQueue;
    if (!ctx->mtl_queue) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create Metal command queue!\n");
        err = AVERROR_EXTERNAL;
        goto fail;
    }

    ret = CVMetalTextureCacheCreate(
        NULL,
        NULL,
        ctx->mtl_device,
        NULL,
        &ctx->texture_cache
    );
    if (ret != kCVReturnSuccess) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create CVMetalTextureCache: %d\n", ret);
        err = AVERROR_EXTERNAL;
        goto fail;
    }

    tone_param = (float)ctx->final_param;
    desat_param = (float)ctx->desat_param;
    target_peak = (float)ctx->target_peak;
    scene_threshold = (float)ctx->scene_threshold;
    tonemap_func_type = (short)ctx->tonemap;
    is_tone_func_bt2390 = ctx->tonemap == TONEMAP_BT2390;
    is_tone_mode_rgb = ctx->tonemap_mode == TONEMAP_MODE_RGB;
    is_tone_mode_max = ctx->tonemap_mode == TONEMAP_MODE_MAX;
    is_tone_mode_itp = ctx->tonemap_mode == TONEMAP_MODE_ITP;
    is_non_semi_planar_in = ctx->in_planes > 2;
    is_non_semi_planar_out = ctx->out_planes > 2;
    enable_dither = ctx->in_desc->comp[0].depth > ctx->out_desc->comp[0].depth;
    dither_size2 = (float)(ff_fruit_dither_size * ff_fruit_dither_size);
    dither_quantization = (float)((1 << ctx->out_desc->comp[0].depth) - 1);
    is_full_range_in = ctx->range_in == AVCOL_RANGE_JPEG;
    is_full_range_out = ctx->range_out == AVCOL_RANGE_JPEG;
    chroma_loc = (int)ctx->chroma_loc;
    skip_tonemap = ctx->trc_out == AVCOL_TRC_SMPTE2084;
    hlg_eotf_bt2446b = ctx->trc_in == AVCOL_TRC_ARIB_STD_B67 &&
                       ctx->trc_out != AVCOL_TRC_SMPTE2084;
    dovi_reshape = !!ctx->dovi;
    map_in_src_space = !is_tone_mode_rgb && !is_tone_mode_max;

    [constant_values setConstantValue:&tone_param type:MTLDataTypeFloat withName:@"tone_param"];
    [constant_values setConstantValue:&desat_param type:MTLDataTypeFloat withName:@"desat_param"];
    [constant_values setConstantValue:&target_peak type:MTLDataTypeFloat withName:@"target_peak"];
    [constant_values setConstantValue:&scene_threshold type:MTLDataTypeFloat withName:@"scene_threshold"];

    [constant_values setConstantValue:&tonemap_func_type type:MTLDataTypeShort withName:@"tonemap_func_type"];
    [constant_values setConstantValue:&is_tone_func_bt2390 type:MTLDataTypeBool withName:@"is_tone_func_bt2390"];
    [constant_values setConstantValue:&is_tone_mode_rgb type:MTLDataTypeBool withName:@"is_tone_mode_rgb"];
    [constant_values setConstantValue:&is_tone_mode_max type:MTLDataTypeBool withName:@"is_tone_mode_max"];
    [constant_values setConstantValue:&is_tone_mode_itp type:MTLDataTypeBool withName:@"is_tone_mode_itp"];

    [constant_values setConstantValue:&is_non_semi_planar_in type:MTLDataTypeBool withName:@"is_non_semi_planar_in"];
    [constant_values setConstantValue:&is_non_semi_planar_out type:MTLDataTypeBool withName:@"is_non_semi_planar_out"];

    [constant_values setConstantValue:&enable_dither type:MTLDataTypeBool withName:@"enable_dither"];
    [constant_values setConstantValue:&dither_size2 type:MTLDataTypeFloat withName:@"dither_size2"];
    [constant_values setConstantValue:&dither_quantization type:MTLDataTypeFloat withName:@"dither_quantization"];

    [constant_values setConstantValue:&is_full_range_in type:MTLDataTypeBool withName:@"is_full_range_in"];
    [constant_values setConstantValue:&is_full_range_out type:MTLDataTypeBool withName:@"is_full_range_out"];
    [constant_values setConstantValue:&chroma_loc type:MTLDataTypeInt withName:@"chroma_loc"];

    [constant_values setConstantValue:&rgb2rgb_passthrough type:MTLDataTypeBool withName:@"is_rgb2rgb_passthrough"];
    if (!rgb2rgb_passthrough) {
        rgb2rgb_matrix_1[0] = (float)rgb2rgb[0][0];
        rgb2rgb_matrix_1[1] = (float)rgb2rgb[0][1];
        rgb2rgb_matrix_1[2] = (float)rgb2rgb[0][2];

        rgb2rgb_matrix_2[0] = (float)rgb2rgb[1][0];
        rgb2rgb_matrix_2[1] = (float)rgb2rgb[1][1];
        rgb2rgb_matrix_2[2] = (float)rgb2rgb[1][2];

        rgb2rgb_matrix_3[0] = (float)rgb2rgb[2][0];
        rgb2rgb_matrix_3[1] = (float)rgb2rgb[2][1];
        rgb2rgb_matrix_3[2] = (float)rgb2rgb[2][2];

        [constant_values setConstantValue:&rgb2rgb_matrix_1 type:MTLDataTypeFloat3 withName:@"rgb2rgb_matrix_1"];
        [constant_values setConstantValue:&rgb2rgb_matrix_2 type:MTLDataTypeFloat3 withName:@"rgb2rgb_matrix_2"];
        [constant_values setConstantValue:&rgb2rgb_matrix_3 type:MTLDataTypeFloat3 withName:@"rgb2rgb_matrix_3"];
    }

    luma_src = av_csp_luma_coeffs_from_avcsp(ctx->colorspace_in);
    if (!luma_src) {
        err = AVERROR(EINVAL);
        av_log(avctx, AV_LOG_ERROR, "Unsupported input colorspace %d (%s)\n",
               ctx->colorspace_in, av_color_space_name(ctx->colorspace_in));
        goto fail;
    }

    mtl_luma_src[0] = (float)av_q2d(luma_src->cr);
    mtl_luma_src[1] = (float)av_q2d(luma_src->cg);
    mtl_luma_src[2] = (float)av_q2d(luma_src->cb);
    [constant_values setConstantValue:&mtl_luma_src type:MTLDataTypeFloat3 withName:@"luma_src"];

    luma_dst = av_csp_luma_coeffs_from_avcsp(ctx->colorspace_out);
    if (!luma_dst) {
        err = AVERROR(EINVAL);
        av_log(avctx, AV_LOG_ERROR, "Unsupported output colorspace %d (%s)\n",
               ctx->colorspace_out, av_color_space_name(ctx->colorspace_out));
        goto fail;
    }

    mtl_luma_dst[0] = (float)av_q2d(luma_dst->cr);
    mtl_luma_dst[1] = (float)av_q2d(luma_dst->cg);
    mtl_luma_dst[2] = (float)av_q2d(luma_dst->cb);
    [constant_values setConstantValue:&mtl_luma_dst type:MTLDataTypeFloat3 withName:@"luma_dst"];

    [constant_values setConstantValue:&skip_tonemap type:MTLDataTypeBool withName:@"skip_tonemap"];
    [constant_values setConstantValue:&hlg_eotf_bt2446b type:MTLDataTypeBool withName:@"hlg_eotf_bt2446b"];
    [constant_values setConstantValue:&dovi_reshape type:MTLDataTypeBool withName:@"dovi_reshape"];
    if (dovi_reshape) {
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                ycc2rgb_offset[i] -= (float)(ctx->dovi->nonlinear[i][j] * ctx->dovi->nonlinear_offset[j]);
        }
        [constant_values setConstantValue:&ycc2rgb_offset type:MTLDataTypeFloat3 withName:@"ycc2rgb_offset"];
        ff_matrix_mul_3x3(lms2rgb, dovi_lms2rgb_matrix, ctx->dovi->linear);
        // ycc2rgb
        rgb_matrix_1[0] = (float)ctx->dovi->nonlinear[0][0];
        rgb_matrix_1[1] = (float)ctx->dovi->nonlinear[0][1];
        rgb_matrix_1[2] = (float)ctx->dovi->nonlinear[0][2];

        rgb_matrix_2[0] = (float)ctx->dovi->nonlinear[1][0];
        rgb_matrix_2[1] = (float)ctx->dovi->nonlinear[1][1];
        rgb_matrix_2[2] = (float)ctx->dovi->nonlinear[1][2];

        rgb_matrix_3[0] = (float)ctx->dovi->nonlinear[2][0];
        rgb_matrix_3[1] = (float)ctx->dovi->nonlinear[2][1];
        rgb_matrix_3[2] = (float)ctx->dovi->nonlinear[2][2];

        [constant_values setConstantValue:&rgb_matrix_1 type:MTLDataTypeFloat3 withName:@"rgb_matrix_1"];
        [constant_values setConstantValue:&rgb_matrix_2 type:MTLDataTypeFloat3 withName:@"rgb_matrix_2"];
        [constant_values setConstantValue:&rgb_matrix_3 type:MTLDataTypeFloat3 withName:@"rgb_matrix_3"];
        //lms2rgb
        lms2rgb_matrix_1[0] = (float)lms2rgb[0][0];
        lms2rgb_matrix_1[1] = (float)lms2rgb[0][1];
        lms2rgb_matrix_1[2] = (float)lms2rgb[0][2];

        lms2rgb_matrix_2[0] = (float)lms2rgb[1][0];
        lms2rgb_matrix_2[1] = (float)lms2rgb[1][1];
        lms2rgb_matrix_2[2] = (float)lms2rgb[1][2];

        lms2rgb_matrix_3[0] = (float)lms2rgb[2][0];
        lms2rgb_matrix_3[1] = (float)lms2rgb[2][1];
        lms2rgb_matrix_3[2] = (float)lms2rgb[2][2];

        [constant_values setConstantValue:&lms2rgb_matrix_1 type:MTLDataTypeFloat3 withName:@"lms2rgb_matrix_1"];
        [constant_values setConstantValue:&lms2rgb_matrix_2 type:MTLDataTypeFloat3 withName:@"lms2rgb_matrix_2"];
        [constant_values setConstantValue:&lms2rgb_matrix_3 type:MTLDataTypeFloat3 withName:@"lms2rgb_matrix_3"];
    } else {
        ff_fill_rgb2yuv_table(luma_src, rgb2yuv);
        ff_matrix_invert_3x3(rgb2yuv, yuv2rgb);

        rgb_matrix_1[0] = (float)yuv2rgb[0][0];
        rgb_matrix_1[1] = (float)yuv2rgb[0][1];
        rgb_matrix_1[2] = (float)yuv2rgb[0][2];

        rgb_matrix_2[0] = (float)yuv2rgb[1][0];
        rgb_matrix_2[1] = (float)yuv2rgb[1][1];
        rgb_matrix_2[2] = (float)yuv2rgb[1][2];

        rgb_matrix_3[0] = (float)yuv2rgb[2][0];
        rgb_matrix_3[1] = (float)yuv2rgb[2][1];
        rgb_matrix_3[2] = (float)yuv2rgb[2][2];

        [constant_values setConstantValue:&rgb_matrix_1 type:MTLDataTypeFloat3 withName:@"rgb_matrix_1"];
        [constant_values setConstantValue:&rgb_matrix_2 type:MTLDataTypeFloat3 withName:@"rgb_matrix_2"];
        [constant_values setConstantValue:&rgb_matrix_3 type:MTLDataTypeFloat3 withName:@"rgb_matrix_3"];
    }

    ff_fill_rgb2yuv_table(luma_dst, rgb2yuv);
    yuv_matrix_1[0] = (float)rgb2yuv[0][0];
    yuv_matrix_1[1] = (float)rgb2yuv[0][1];
    yuv_matrix_1[2] = (float)rgb2yuv[0][2];

    yuv_matrix_2[0] = (float)rgb2yuv[1][0];
    yuv_matrix_2[1] = (float)rgb2yuv[1][1];
    yuv_matrix_2[2] = (float)rgb2yuv[1][2];

    yuv_matrix_3[0] = (float)rgb2yuv[2][0];
    yuv_matrix_3[1] = (float)rgb2yuv[2][1];
    yuv_matrix_3[2] = (float)rgb2yuv[2][2];

    [constant_values setConstantValue:&yuv_matrix_1 type:MTLDataTypeFloat3 withName:@"yuv_matrix_1"];
    [constant_values setConstantValue:&yuv_matrix_2 type:MTLDataTypeFloat3 withName:@"yuv_matrix_2"];
    [constant_values setConstantValue:&yuv_matrix_3 type:MTLDataTypeFloat3 withName:@"yuv_matrix_3"];

    if (ctx->trc_out != AVCOL_TRC_SMPTE2084) {
        [constant_values setConstantValue:&linearize_funcs[ctx->trc_in] type:MTLDataTypeShort withName:@"linearize_type"];
        [constant_values setConstantValue:&delinearize_funcs[ctx->trc_in] type:MTLDataTypeShort withName:@"delinearize_type"];
    }

    if (enable_dither) {
        uint bytes_per_row = 2 * ff_fruit_dither_size;
        uint bytes_per_image = 2 * ff_fruit_dither_size2;
        MTLTextureDescriptor *texture_descriptor = [[MTLTextureDescriptor alloc] init];
        MTLRegion region = {
            { 0, 0, 0 }, // MTLOrigin
            {ff_fruit_dither_size, ff_fruit_dither_size, 1} // MTLSize
        };
        id <MTLBuffer> source_buffer;
        id <MTLCommandBuffer> command_buffer;
        id <MTLBlitCommandEncoder> blit_command_encoder;

        source_buffer = [ctx->mtl_device newBufferWithBytes: ff_fruit_dither_matrix
                                                     length: bytes_per_image
                                                    options: MTLResourceStorageModeShared];

        texture_descriptor.pixelFormat = MTLPixelFormatR16Unorm;
        texture_descriptor.width = ff_fruit_dither_size;
        texture_descriptor.height = ff_fruit_dither_size;
        texture_descriptor.storageMode = MTLStorageModePrivate;

        ctx->dither_texture = [ctx->mtl_device newTextureWithDescriptor:texture_descriptor];

        command_buffer = [ctx->mtl_queue commandBuffer];

        blit_command_encoder = [command_buffer blitCommandEncoder];
        [blit_command_encoder copyFromBuffer: source_buffer
                                sourceOffset: 0
                           sourceBytesPerRow: bytes_per_row
                         sourceBytesPerImage: bytes_per_image
                                  sourceSize: region.size
                                   toTexture: ctx->dither_texture
                            destinationSlice: 0
                            destinationLevel: 0
                           destinationOrigin: region.origin];
        [blit_command_encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }

    [constant_values setConstantValue:&map_in_src_space type:MTLDataTypeBool withName:@"map_in_src_space"];

    ctx->mtl_function = [ctx->mtl_library newFunctionWithName:@"tonemap" constantValues:constant_values error:&ns_error];
    if (ns_error) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create Metal function: %s\n", ns_error.description.UTF8String);
        err = AVERROR_EXTERNAL;
        goto fail;
    }

    ctx->mtl_pipeline = [ctx->mtl_device newComputePipelineStateWithFunction:ctx->mtl_function error:&ns_error];
    if (ns_error) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create Metal compute pipeline: %s\n", ns_error.description.UTF8String);
        err = AVERROR_EXTERNAL;
        goto fail;
    }

    if (dovi_reshape) {
        ctx->mtl_dovi_buffer = [ctx->mtl_device newBufferWithLength: 3*(params_sz+pivots_sz+coeffs_sz+mmr_sz)
                                                            options: MTLResourceStorageModeShared];
        if (!ctx->mtl_dovi_buffer) {
            av_log(ctx, AV_LOG_ERROR, "Failed to create Metal buffer for Dolby Vision data\n");
            err = AVERROR_EXTERNAL;
            goto fail;
        }
    }

    ctx->mtl_peak_buffer = [ctx->mtl_device newBufferWithLength: sizeof(float)
                                                        options: MTLResourceStorageModeShared];
    if (!ctx->mtl_peak_buffer) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create Metal buffer for Peak data\n");
        err = AVERROR_EXTERNAL;
        goto fail;
    }

    ctx->initialised = 1;
    return 0;

fail:
    tonemap_videotoolbox_uninit(avctx);
    return err;
}

static int tonemap_videotoolbox_config_output(AVFilterLink *outlink)
{
    FilterLink          *outl = ff_filter_link(outlink);
    AVFilterContext    *avctx = outlink->src;
    AVFilterLink      *inlink = avctx->inputs[0];
    FilterLink           *inl = ff_filter_link(inlink);
    TonemapVideoToolboxContext *ctx = avctx->priv;
    AVHWFramesContext *in_frames_ctx, *out_frames_ctx;
    enum AVPixelFormat in_format;
    enum AVPixelFormat out_format;
    const AVPixFmtDescriptor *in_desc;
    const AVPixFmtDescriptor *out_desc;
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

    av_buffer_unref(&outl->hw_frames_ctx);
    outl->hw_frames_ctx = av_hwframe_ctx_alloc(in_frames_ctx->device_ref);
    outlink->w = inlink->w;
    outlink->h = inlink->h;
    out_frames_ctx = (AVHWFramesContext *)outl->hw_frames_ctx->data;
    out_frames_ctx->format = AV_PIX_FMT_VIDEOTOOLBOX;
    out_frames_ctx->sw_format = out_format;
    out_frames_ctx->width = outlink->w;
    out_frames_ctx->height = outlink->h;

    if (ctx->range != -1) {
        ((AVVTFramesContext *)out_frames_ctx->hwctx)->color_range = ctx->range;
    } else {
        ((AVVTFramesContext *)out_frames_ctx->hwctx)->color_range = ((AVVTFramesContext *)in_frames_ctx->hwctx)->color_range;
    }

    ret = ff_filter_init_hw_frames(avctx, outlink, 1);
    if (ret < 0)
        return ret;

    ret = av_hwframe_ctx_init(outl->hw_frames_ctx);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR,
               "Failed to init videotoolbox frame context, %s\n",
               av_err2str(ret));
        return ret;
    }

    if (ctx->trc != AVCOL_TRC_SMPTE2084) {
        av_frame_side_data_remove(&outlink->side_data, &outlink->nb_side_data,
                                  AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
        av_frame_side_data_remove(&outlink->side_data, &outlink->nb_side_data,
                                  AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    }

    return 0;
}

static void call_kernel(AVFilterContext *avctx,
                        id<MTLTexture> in_y_tex,
                        id<MTLTexture> in_u_uv_tex,
                        id<MTLTexture> out_y_tex,
                        id<MTLTexture> out_u_uv_tex,
                        id<MTLTexture> in_v_tex,
                        id<MTLTexture> out_v_tex,
                        float peak)
{
    TonemapVideoToolboxContext *ctx = avctx->priv;
    id<MTLCommandBuffer> buffer = ctx->mtl_queue.commandBuffer;
    id<MTLComputeCommandEncoder> encoder = buffer.computeCommandEncoder;
    float* peak_ptr = ctx->mtl_peak_buffer.contents;
    *peak_ptr = peak;

    [encoder setTexture:out_y_tex  atIndex:0];
    [encoder setTexture:in_y_tex atIndex:1];
    [encoder setTexture:out_u_uv_tex  atIndex:2];
    [encoder setTexture:in_u_uv_tex atIndex:3];
    if (ctx->out_planes > 2) {
        [encoder setTexture:out_v_tex atIndex:4];
    }
    if (ctx->in_planes > 2) {
        [encoder setTexture:in_v_tex atIndex:5];
    }
    if (ctx->dither_texture) {
        [encoder setTexture:ctx->dither_texture atIndex:6];
    }
    if (ctx->mtl_dovi_buffer) {
        [encoder setBuffer:ctx->mtl_dovi_buffer offset:0 atIndex:7];
    }
    [encoder setBuffer:ctx->mtl_peak_buffer offset:0 atIndex:8];

    ff_metal_compute_encoder_dispatch(ctx->mtl_device, ctx->mtl_pipeline, encoder, out_u_uv_tex.width, out_u_uv_tex.height);

    [encoder endEncoding];

    [buffer commit];
    [buffer waitUntilCompleted];
}

static int tonemap_videotoolbox_filter_frame(AVFilterLink *inlink, AVFrame *input)
{
    AVFilterContext    *avctx = inlink->dst;
    AVFilterLink     *outlink = avctx->outputs[0];
    TonemapVideoToolboxContext *ctx = avctx->priv;
    AVFrameSideData  *dovi_sd = NULL;
    AVFrame *output = NULL;

    CVMetalTextureRef in_y, in_u_uv, in_v;
    id<MTLTexture> in_y_tex, in_u_uv_tex, in_v_tex = NULL;

    CVMetalTextureRef out_y, out_u_uv, out_v;
    id<MTLTexture> out_y_tex, out_u_uv_tex, out_v_tex = NULL;

    MTLPixelFormat format;

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

    if (ctx->apply_dovi)
        dovi_sd = av_frame_get_side_data(input, AV_FRAME_DATA_DOVI_METADATA);

    // check DOVI->HDR10/HLG
    if (!dovi_sd) {
        if (input->color_trc != AVCOL_TRC_SMPTE2084 &&
            input->color_trc != AVCOL_TRC_ARIB_STD_B67) {
            av_log(ctx, AV_LOG_ERROR, "No DOVI metadata and "
                                      "unsupported transfer function characteristic: %s\n",
                   av_color_transfer_name(input->color_trc));
            err = AVERROR(ENOSYS);
            goto fail;
        }
    }

    if (!ctx->peak) {
        if (dovi_sd) {
            const AVDOVIMetadata *metadata = (AVDOVIMetadata *) dovi_sd->data;
            ctx->src_peak = ff_determine_dovi_signal_peak(metadata, 0);
            ctx->src_peak *= REF_WHITE_SCALE;
        } else if (!ctx->src_peak) {
            ctx->src_peak = ff_determine_signal_peak(input);
            ctx->src_peak *= REF_WHITE_SCALE;
        }
        av_log(ctx, AV_LOG_DEBUG, "Computed signal peak: %f "
               "at pts %"PRId64"\n", ctx->src_peak, input->pts);
    }

    if (dovi_sd) {
        const AVDOVIMetadata *metadata = (AVDOVIMetadata *) dovi_sd->data;
        const AVDOVIRpuDataHeader *rpu = av_dovi_get_header(metadata);
        // only map dovi rpus that don't require an EL
        if (rpu->disable_residual_flag) {
            struct FFDOVIMetadataRemap *dovi = av_malloc(sizeof(*dovi));
            ctx->dovi = dovi;
            if (!ctx->dovi)
                goto fail;

            ff_map_dovi_metadata(ctx->dovi, metadata);
            output->color_trc = input->color_trc = AVCOL_TRC_SMPTE2084;
            output->colorspace = input->colorspace = AVCOL_SPC_BT2020_NCL;
            output->color_primaries = input->color_primaries = AVCOL_PRI_BT2020;
            if (rpu->bl_video_full_range_flag)
                input->color_range = AVCOL_RANGE_JPEG;
        }
    }

    if (ctx->trc != -1)
        output->color_trc = ctx->trc;
    if (ctx->primaries != -1)
        output->color_primaries = ctx->primaries;

    output->colorspace = outlink->colorspace;
    output->color_range = outlink->color_range;

    ctx->trc_in = input->color_trc;
    ctx->trc_out = output->color_trc;
    ctx->colorspace_in = input->colorspace;
    ctx->colorspace_out = output->colorspace;
    ctx->primaries_in = input->color_primaries;
    ctx->primaries_out = output->color_primaries;
    ctx->range_in = input->color_range;
    ctx->range_out = output->color_range;
    ctx->chroma_loc = output->chroma_location;

    // Some DOVI video does not carry metadata in the first few frames, and we have to reset the pipeline.
    if (!ctx->init_with_dovi && ctx->dovi && ctx->initialised) {
        tonemap_videotoolbox_uninit_common(avctx);
    }

    if (!ctx->initialised) {
        err = tonemap_videotoolbox_init(avctx);
        if (err < 0)
            goto fail;

        ctx->init_with_dovi = ctx->dovi != NULL;
    }

    if (ctx->dovi) {
        tonemap_videotoolbox_update_dovi_buf(avctx);
        av_freep(&ctx->dovi);
    }

    // First Input Plane
    format = get_plane_texture_format(ctx, 0, false);
    if (format == MTLPixelFormatInvalid) {
        err = AVERROR(EIO);
        goto fail;
    }
    in_y = ff_metal_texture_from_pixbuf(ctx, ctx->texture_cache, (CVPixelBufferRef)input->data[3], 0, format);
    in_y_tex = CVMetalTextureGetTexture(in_y);

    // Second Input Plane
    format = get_plane_texture_format(ctx, 1, false);
    if (format == MTLPixelFormatInvalid) {
        err = AVERROR(EIO);
        goto fail;
    }
    in_u_uv = ff_metal_texture_from_pixbuf(ctx, ctx->texture_cache, (CVPixelBufferRef)input->data[3], 1, format);
    in_u_uv_tex = CVMetalTextureGetTexture(in_u_uv);

    // First Output Plane
    format = get_plane_texture_format(ctx, 0, true);
    if (format == MTLPixelFormatInvalid) {
        err = AVERROR(EIO);
        goto fail;
    }
    out_y = ff_metal_texture_from_pixbuf(ctx, ctx->texture_cache, (CVPixelBufferRef)output->data[3], 0, format);
    out_y_tex = CVMetalTextureGetTexture(out_y);

    // Second Output Plane
    format = get_plane_texture_format(ctx, 1, true);
    if (format == MTLPixelFormatInvalid) {
        err = AVERROR(EIO);
        goto fail;
    }
    out_u_uv = ff_metal_texture_from_pixbuf(ctx, ctx->texture_cache, (CVPixelBufferRef)output->data[3], 1, format);
    out_u_uv_tex = CVMetalTextureGetTexture(out_u_uv);

    if (ctx->in_planes > 2) {
        // Third Input Plane
        format = get_plane_texture_format(ctx, 2, false);
        if (format == MTLPixelFormatInvalid) {
            err = AVERROR(EIO);
            goto fail;
        }
        in_v = ff_metal_texture_from_pixbuf(ctx, ctx->texture_cache, (CVPixelBufferRef)input->data[3], 2, format);
        in_v_tex = CVMetalTextureGetTexture(in_v);
    }

    if (ctx->out_planes > 2) {
        // Third Output Plane
        format = get_plane_texture_format(ctx, 2, true);
        if (format == MTLPixelFormatInvalid) {
            err = AVERROR(EIO);
            goto fail;
        }
        out_v = ff_metal_texture_from_pixbuf(ctx, ctx->texture_cache, (CVPixelBufferRef)output->data[3], 2, format);
        out_v_tex = CVMetalTextureGetTexture(out_v);
    }

    call_kernel(avctx,
                in_y_tex,
                in_u_uv_tex,
                out_y_tex,
                out_u_uv_tex,
                in_v_tex,
                out_v_tex,
                (float)ctx->src_peak);

    CFRelease(in_y);
    CFRelease(in_u_uv);
    CFRelease(out_y);
    CFRelease(out_u_uv);
    if(in_v_tex) {
        CFRelease(in_v);
    }
    if(out_v_tex) {
        CFRelease(out_v);
    }

    CVBufferPropagateAttachments((CVPixelBufferRef)input->data[3], (CVPixelBufferRef)output->data[3]);
    av_frame_free(&input);

    {
        CGColorSpaceRef colorspace = NULL;
        CFStringRef colormatrix = av_map_videotoolbox_color_matrix_from_av(ctx->colorspace_out);
        CFStringRef colorpri = av_map_videotoolbox_color_primaries_from_av(ctx->primaries_out);
        CFStringRef colortrc = av_map_videotoolbox_color_trc_from_av(ctx->trc_out);
        CFMutableDictionaryRef attachments = CFDictionaryCreateMutable(NULL, 4,
                                                                       &kCFTypeDictionaryKeyCallBacks,
                                                                       &kCFTypeDictionaryValueCallBacks);
        if (!attachments) {
            err = AVERROR(ENOMEM);
            goto fail;
        }
        CFDictionarySetValue(attachments, kCVImageBufferYCbCrMatrixKey, colormatrix);
        CFDictionarySetValue(attachments, kCVImageBufferColorPrimariesKey, colorpri);
        CFDictionarySetValue(attachments, kCVImageBufferTransferFunctionKey, colortrc);
        colorspace = CVImageBufferCreateColorSpaceFromAttachments(attachments);
        if (colorspace) {
            CFDictionarySetValue(attachments, kCVImageBufferCGColorSpaceKey, colorspace);
            CFRelease(colorspace);
        } else {
            av_log(avctx, AV_LOG_WARNING, "Unable to set proper colorspace for the CVImageBuffer.\n");
        }
        CVBufferSetAttachments(
            (CVPixelBufferRef)output->data[3],
            attachments,
            kCVAttachmentMode_ShouldPropagate);
        CFRelease(attachments);
        if (ctx->trc_out != AVCOL_TRC_SMPTE2084) {
            av_frame_remove_side_data(output, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
            av_frame_remove_side_data(output, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
        } else {
            ff_update_hdr_metadata(output, 100.0f);
        }
    }

    av_frame_remove_side_data(output, AV_FRAME_DATA_DOVI_RPU_BUFFER);
    av_frame_remove_side_data(output, AV_FRAME_DATA_DOVI_METADATA);
    return ff_filter_frame(outlink, output);

fail:
    if (ctx->dovi)
        av_freep(&ctx->dovi);
    av_frame_free(&input);
    av_frame_free(&output);
    return err;
}

static int tonemap_videotoolbox_query_formats(AVFilterContext *avctx)
{
    TonemapVideoToolboxContext *ctx = avctx->priv;
    AVFilterFormats *formats;
    int ret;
    const enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_VIDEOTOOLBOX, AV_PIX_FMT_NONE };

    // single format
    formats = ff_make_format_list(pix_fmts);
    ret = ff_formats_ref(formats, &avctx->inputs[0]->outcfg.formats);
    if (ret < 0)
        return ret;

    ret = ff_formats_ref(formats, &avctx->outputs[0]->incfg.formats);
    if (ret < 0)
        return ret;

    // colorspaces and ranges
    if ((ret = ff_formats_ref(ff_all_color_spaces(),
                              &avctx->inputs[0]->outcfg.color_spaces)) < 0)
        return ret;

    if ((ret = ff_formats_ref(ff_all_color_ranges(),
                              &avctx->inputs[0]->outcfg.color_ranges)) < 0)
        return ret;

    formats = ctx->colorspace != -1
        ? ff_make_formats_list_singleton(ctx->colorspace)
        : ff_make_format_list(colorspaces_out);
    if ((ret = ff_formats_ref(formats, &avctx->outputs[0]->incfg.color_spaces)) < 0)
        return ret;

    formats = ctx->range != -1
        ? ff_make_formats_list_singleton(ctx->range)
        : ff_all_color_ranges();
    if ((ret = ff_formats_ref(formats, &avctx->outputs[0]->incfg.color_ranges)) < 0)
        return ret;

    return 0;
}

#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
#define OFFSET(x) offsetof(TonemapVideoToolboxContext, x)

static const AVOption tonemap_videotoolbox_options[] = {
    { "tonemap", "Tonemap algorithm selection", OFFSET(tonemap), AV_OPT_TYPE_INT, { .i64 = TONEMAP_BT2390 }, TONEMAP_NONE, TONEMAP_COUNT - 1, FLAGS, .unit = "tonemap" },
        { "none",     0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_NONE },              0, 0, FLAGS, .unit = "tonemap" },
        { "linear",   0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_LINEAR },            0, 0, FLAGS, .unit = "tonemap" },
        { "gamma",    0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_GAMMA },             0, 0, FLAGS, .unit = "tonemap" },
        { "clip",     0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_CLIP },              0, 0, FLAGS, .unit = "tonemap" },
        { "reinhard", 0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_REINHARD },          0, 0, FLAGS, .unit = "tonemap" },
        { "hable",    0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_HABLE },             0, 0, FLAGS, .unit = "tonemap" },
        { "mobius",   0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_MOBIUS },            0, 0, FLAGS, .unit = "tonemap" },
        { "bt2390",   0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_BT2390 },            0, 0, FLAGS, .unit = "tonemap" },
    { "tonemap_mode", "Tonemap mode selection", OFFSET(tonemap_mode), AV_OPT_TYPE_INT, { .i64 = TONEMAP_MODE_ITP }, TONEMAP_MODE_MAX, TONEMAP_MODE_COUNT - 1, FLAGS, .unit = "tonemap_mode" },
        { "max",      "Brightest channel based tonemap",  0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_MODE_MAX },     0, 0, FLAGS, .unit = "tonemap_mode" },
        { "rgb",      "Per-channel based tonemap",        0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_MODE_RGB },     0, 0, FLAGS, .unit = "tonemap_mode" },
        { "lum",      "Relative luminance based tonemap", 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_MODE_LUM },     0, 0, FLAGS, .unit = "tonemap_mode" },
        { "itp",      "ICtCp intensity based tonemap",    0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_MODE_ITP },     0, 0, FLAGS, .unit = "tonemap_mode" },
    { "transfer", "Set transfer characteristic", OFFSET(trc), AV_OPT_TYPE_INT, { .i64 = AVCOL_TRC_BT709 }, -1, INT_MAX, FLAGS, .unit = "transfer" },
    { "t",        "Set transfer characteristic", OFFSET(trc), AV_OPT_TYPE_INT, { .i64 = AVCOL_TRC_BT709 }, -1, INT_MAX, FLAGS, .unit = "transfer" },
        { "bt709",            0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_TRC_BT709 },         0, 0, FLAGS, .unit = "transfer" },
        { "bt2020",           0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_TRC_BT2020_10 },     0, 0, FLAGS, .unit = "transfer" },
        { "smpte2084",        0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_TRC_SMPTE2084 },     0, 0, FLAGS, .unit = "transfer" },
    { "matrix", "Set colorspace matrix", OFFSET(colorspace), AV_OPT_TYPE_INT, { .i64 = AVCOL_SPC_BT709 }, -1, INT_MAX, FLAGS, .unit = "matrix" },
    { "m",      "Set colorspace matrix", OFFSET(colorspace), AV_OPT_TYPE_INT, { .i64 = AVCOL_SPC_BT709 }, -1, INT_MAX, FLAGS, .unit = "matrix" },
        { "bt709",            0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_SPC_BT709 },         0, 0, FLAGS, .unit = "matrix" },
        { "bt2020",           0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_SPC_BT2020_NCL },    0, 0, FLAGS, .unit = "matrix" },
    { "primaries", "Set color primaries", OFFSET(primaries), AV_OPT_TYPE_INT, { .i64 = AVCOL_PRI_BT709 }, -1, INT_MAX, FLAGS, .unit = "primaries" },
    { "p",         "Set color primaries", OFFSET(primaries), AV_OPT_TYPE_INT, { .i64 = AVCOL_PRI_BT709 }, -1, INT_MAX, FLAGS, .unit = "primaries" },
        { "bt709",            0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_PRI_BT709 },         0, 0, FLAGS, .unit = "primaries" },
        { "bt2020",           0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_PRI_BT2020 },        0, 0, FLAGS, .unit = "primaries" },
    { "range",         "Set color range", OFFSET(range), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, INT_MAX, FLAGS, .unit = "range" },
    { "r",             "Set color range", OFFSET(range), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, INT_MAX, FLAGS, .unit = "range" },
        { "tv",            0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_RANGE_MPEG },         0, 0, FLAGS, .unit = "range" },
        { "pc",            0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_RANGE_JPEG },         0, 0, FLAGS, .unit = "range" },
        { "limited",       0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_RANGE_MPEG },         0, 0, FLAGS, .unit = "range" },
        { "full",          0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_RANGE_JPEG },         0, 0, FLAGS, .unit = "range" },
    { "format",      "Output pixel format", OFFSET(format), AV_OPT_TYPE_PIXEL_FMT, { .i64 = AV_PIX_FMT_NONE }, AV_PIX_FMT_NONE, INT_MAX, FLAGS },
    { "apply_dovi",  "Apply Dolby Vision metadata if possible", OFFSET(apply_dovi), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { "peak",        "Signal peak override", OFFSET(peak), AV_OPT_TYPE_DOUBLE, { .dbl = 0 }, 0, DBL_MAX, FLAGS },
    { "param",       "Tonemap parameter",   OFFSET(param), AV_OPT_TYPE_DOUBLE, { .dbl = NAN }, DBL_MIN, DBL_MAX, FLAGS },
    { "desat",       "Desaturation parameter",   OFFSET(desat_param), AV_OPT_TYPE_DOUBLE, { .dbl = 0 }, 0, DBL_MAX, FLAGS },
    { "threshold",   "Scene detection threshold",   OFFSET(scene_threshold), AV_OPT_TYPE_DOUBLE, { .dbl = 0.2 }, 0, DBL_MAX, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(tonemap_videotoolbox);

static const AVFilterPad tonemap_videotoolbox_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = tonemap_videotoolbox_filter_frame,
    },
};

static const AVFilterPad tonemap_videotoolbox_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = tonemap_videotoolbox_config_output,
    },
};

const FFFilter ff_vf_tonemap_videotoolbox = {
    .p.name         = "tonemap_videotoolbox",
    .p.description  = NULL_IF_CONFIG_SMALL("Perform HDR to SDR conversion with Metal."),
    .priv_size      = sizeof(TonemapVideoToolboxContext),
    .p.priv_class   = &tonemap_videotoolbox_class,
    .preinit        = tonemap_videotoolbox_preinit,
    .uninit         = tonemap_videotoolbox_uninit,
    FILTER_INPUTS(tonemap_videotoolbox_inputs),
    FILTER_OUTPUTS(tonemap_videotoolbox_outputs),
    FILTER_QUERY_FUNC(tonemap_videotoolbox_query_formats),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
    .p.flags        = AVFILTER_FLAG_HWDEVICE,
};
