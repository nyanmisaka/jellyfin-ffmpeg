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

#ifdef __APPLE__
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl_ext.h>
#endif

#include "libavutil/avassert.h"
#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "internal.h"
#include "opencl.h"
#include "opencl_source.h"
#include "video.h"
#include "colorspace.h"
#include "dither_matrix.h"

#define OPENCL_SOURCE_NB 3

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

    float *lin_lut;

#define params_cnt 8
#define pivots_cnt (7+1)
#define coeffs_cnt 8*4
#define mmr_cnt 8*6*4
#define params_sz params_cnt*sizeof(float)
#define pivots_sz pivots_cnt*sizeof(float)
#define coeffs_sz coeffs_cnt*sizeof(float)
#define mmr_sz mmr_cnt*sizeof(float)
    struct DoviMetadata *dovi;
    cl_mem dovi_buf;

    enum TonemapAlgorithm tonemap;
    enum AVPixelFormat    format;
    int                   apply_dovi;
    double                ref_white;
    double                peak;
    double                target_peak;
    double                param;
    double                desat_param;
    double                scene_threshold;
    int                   tradeoff;
    int                   initialised;
    int                   init_with_dovi;
    cl_kernel             kernel;
    cl_mem                dither_image;
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

static const double dovi_lms2rgb_matrix[3][3] =
{
    { 3.06441879, -2.16597676,  0.10155818},
    {-0.65612108,  1.78554118, -0.12943749},
    { 0.01736321, -0.04725154,  1.03004253},
};

static float linearize(float x, float ref_white, enum AVColorTransferCharacteristic trc_in)
{
    if (trc_in == AVCOL_TRC_SMPTE2084)
        return eotf_st2084(x, ref_white);
    if (trc_in == AVCOL_TRC_ARIB_STD_B67)
        return eotf_arib_b67(x);
    return x;
}

#define LUT_SIZE (1 << 10)
static int compute_trc_luts(AVFilterContext *avctx)
{
    TonemapOpenCLContext *ctx = avctx->priv;
    int i;

    if (!ctx->lin_lut && !(ctx->lin_lut = av_calloc(LUT_SIZE, sizeof(float))))
        return AVERROR(ENOMEM);
    for (i = 0; i < LUT_SIZE; i++) {
        float x = (float)i / (LUT_SIZE - 1);
        ctx->lin_lut[i] = FFMAX(linearize(x, ctx->ref_white, ctx->trc_in), 0.0f);
    }

    return 0;
}

static void print_opencl_const_trc_luts(AVFilterContext *avctx, AVBPrint *buf)
{
    TonemapOpenCLContext *ctx = avctx->priv;
    int i;

    if (ctx->lin_lut) {
        av_bprintf(buf, "__constant float lin_lut[%d] = {\n", LUT_SIZE);
        for (i = 0; i < LUT_SIZE; i++)
            av_bprintf(buf, " %ff,", ctx->lin_lut[i]);
        av_bprintf(buf, "};\n");
    }
}

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

static int tonemap_opencl_update_dovi_buf(AVFilterContext *avctx)
{
    TonemapOpenCLContext *ctx = avctx->priv;
    float *pbuf = NULL;
    float coeffs_data[8][4] = {0};
    float mmr_packed_data[8*6][4] = {0};
    int c, i, j, k, err;
    cl_int cle;

    pbuf = (float *)clEnqueueMapBuffer(ctx->command_queue, ctx->dovi_buf,
                                       CL_TRUE, CL_MAP_WRITE, 0,
                                       3*(params_sz+pivots_sz+coeffs_sz+mmr_sz),
                                       0, NULL, NULL, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to map dovi buf: %d.\n", cle);

    av_assert0(pbuf);

    for (c = 0; c < 3; c++) {
        int has_poly = 0, has_mmr = 0, mmr_single = 1;
        int mmr_idx = 0, min_order = 3, max_order = 1;
        const struct ReshapeData *comp = &ctx->dovi->comp[c];
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
            memcpy(pbuf + c*params_cnt, params, params_sz);
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
            memcpy(pbuf + 3*params_cnt + c*pivots_cnt, pivots_data, pivots_sz);
        }

        // dovi_coeffs
        memcpy(pbuf + 3*(params_cnt+pivots_cnt) + c*coeffs_cnt, &coeffs_data[0], coeffs_sz);

        // dovi_mmr
        if (has_mmr)
            memcpy(pbuf + 3*(params_cnt+pivots_cnt+coeffs_cnt) + c*mmr_cnt, &mmr_packed_data[0], mmr_sz);
    }

    cle = clEnqueueUnmapMemObject(ctx->command_queue, ctx->dovi_buf, pbuf, 0, NULL, NULL);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to unmap dovi buf: %d.\n", cle);

fail:
    return cle;
}

static char *check_opencl_device_str(cl_device_id device_id,
                                     cl_device_info key)
{
    char *str;
    size_t size;
    cl_int cle;
    cle = clGetDeviceInfo(device_id, key, 0, NULL, &size);
    if (cle != CL_SUCCESS)
        return NULL;
    str = av_malloc(size);
    if (!str)
        return NULL;
    cle = clGetDeviceInfo(device_id, key, size, str, &size);
    if (cle != CL_SUCCESS) {
        av_free(str);
        return NULL;
    }
    av_assert0(strlen(str) + 1== size);
    return str;
}

static int tonemap_opencl_init(AVFilterContext *avctx)
{
    TonemapOpenCLContext *ctx = avctx->priv;
    AVBPrint header;
    const char *opencl_sources[OPENCL_SOURCE_NB];
    size_t m_origin[3] = {0};
    size_t m_region[3] = {ff_fruit_dither_size, ff_fruit_dither_size, 1};
    size_t m_row_pitch = ff_fruit_dither_size * sizeof(ff_fruit_dither_matrix[0]);
    int rgb2rgb_passthrough = 1;
    double rgb2rgb[3][3], rgb2yuv[3][3], yuv2rgb[3][3];
    const AVLumaCoefficients *luma_src, *luma_dst;
    cl_event event;
    cl_bool device_is_integrated;
    cl_uint max_compute_units, device_vendor_id;
    cl_int cle;
    cl_mem_flags dovi_buf_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
    char *device_name = NULL;
    char *device_exts = NULL;
    int i, j, err;

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

    ctx->ref_white = ctx->tonemap == TONEMAP_BT2390 ? REFERENCE_WHITE_ALT
                                                    : REFERENCE_WHITE;

    if (ctx->tonemap == TONEMAP_BT2390 && ctx->peak)
        ctx->peak = FFMAX(ctx->peak / 10.0f, 1.1f);

    // SDR peak is 1.0f
    ctx->target_peak = 1.0f;

    cle = clGetDeviceInfo(ctx->ocf.hwctx->device_id, CL_DEVICE_VENDOR_ID,
                          sizeof(cl_uint), &device_vendor_id,
                          NULL);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to check OpenCL "
                     "device vendor id %d.\n", cle);

#ifndef CL_MEM_FORCE_HOST_MEMORY_INTEL
  #define CL_MEM_FORCE_HOST_MEMORY_INTEL (1 << 20)
#endif
    // zero-copy buffer requires this extension on Intel dGPUs
    if (device_vendor_id == 0x8086) {
        device_exts = check_opencl_device_str(ctx->ocf.hwctx->device_id, CL_DEVICE_EXTENSIONS);
        if (device_exts && strstr(device_exts, "cl_intel_mem_force_host_memory"))
            dovi_buf_flags |= CL_MEM_FORCE_HOST_MEMORY_INTEL;
        if (device_exts)
            av_free(device_exts);
    }

    if (ctx->tradeoff == -1) {
        ctx->tradeoff = 1;
        cle = clGetDeviceInfo(ctx->ocf.hwctx->device_id, CL_DEVICE_HOST_UNIFIED_MEMORY,
                              sizeof(cl_bool), &device_is_integrated,
                              NULL);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to check if OpenCL "
                         "device is integrated %d.\n", cle);
        cle = clGetDeviceInfo(ctx->ocf.hwctx->device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
                              sizeof(cl_uint), &max_compute_units,
                              NULL);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to check OpenCL "
                         "device max compute units %d.\n", cle);
        if (device_vendor_id == 0x8086 && device_is_integrated == CL_TRUE) {
            if (max_compute_units >= 40)
                ctx->tradeoff = 0;
            if (device_name = check_opencl_device_str(ctx->ocf.hwctx->device_id, CL_DEVICE_NAME)) {
                const char *excluded_devices[4] = { "Iris", "Xe", "770", "750" };
                for (i = 0; i < FF_ARRAY_ELEMS(excluded_devices); i++) {
                    if (strstr(device_name, excluded_devices[i])) {
                        ctx->tradeoff = 0;
                        break;
                    }
                }
                av_free(device_name);
            }
        } else {
            ctx->tradeoff = 0;
        }

        if (!ctx->tradeoff)
            av_log(avctx, AV_LOG_DEBUG, "Disabled tradeoffs on high performance device.\n");
    }

    av_log(ctx, AV_LOG_DEBUG, "Tonemapping transfer from %s to %s\n",
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
        int is_10_or_16b_out = ctx->out_desc->comp[0].depth == 10 ||
                               ctx->out_desc->comp[0].depth == 16;
        if (!(is_10_or_16b_out &&
            ctx->primaries_out == AVCOL_PRI_BT2020 &&
            ctx->colorspace_out == AVCOL_SPC_BT2020_NCL)) {
            av_log(avctx, AV_LOG_ERROR, "HDR passthrough requires BT.2020 "
                   "colorspace and 10/16 bit output format depth.\n");
            return AVERROR(EINVAL);
        }
    }

    av_bprint_init(&header, 2048, AV_BPRINT_SIZE_UNLIMITED);

    av_bprintf(&header, "__constant float ref_white = %.4ff;\n",
               ctx->ref_white);
    av_bprintf(&header, "__constant float tone_param = %.4ff;\n",
               ctx->param);
    av_bprintf(&header, "__constant float desat_param = %.4ff;\n",
               ctx->desat_param);
    av_bprintf(&header, "__constant float target_peak = %.4ff;\n",
               ctx->target_peak);
    av_bprintf(&header, "__constant float scene_threshold = %.4ff;\n",
               ctx->scene_threshold);

    av_bprintf(&header, "__constant float pq_max_lum_div_ref_white = %ff;\n",
               (ST2084_MAX_LUMINANCE / ctx->ref_white));
    av_bprintf(&header, "__constant float ref_white_div_pq_max_lum = %ff;\n",
               (ctx->ref_white / ST2084_MAX_LUMINANCE));

    av_bprintf(&header, "#define TONE_FUNC %s\n", tonemap_func[ctx->tonemap]);
    if (ctx->tonemap == TONEMAP_BT2390)
        av_bprintf(&header, "#define TONE_FUNC_BT2390\n");

    if (ctx->in_planes > 2)
        av_bprintf(&header, "#define NON_SEMI_PLANAR_IN\n");

    if (ctx->out_planes > 2)
        av_bprintf(&header, "#define NON_SEMI_PLANAR_OUT\n");

    if (ctx->in_desc->comp[0].depth > ctx->out_desc->comp[0].depth) {
        av_bprintf(&header, "#define ENABLE_DITHER\n");
        av_bprintf(&header, "__constant float dither_size2 = %.1ff;\n", (float)(ff_fruit_dither_size * ff_fruit_dither_size));
        av_bprintf(&header, "__constant float dither_quantization = %.1ff;\n", (float)((1 << ctx->out_desc->comp[0].depth) - 1));
    }

    if (ctx->primaries_out != ctx->primaries_in) {
        if ((err = get_rgb2rgb_matrix(ctx->primaries_in, ctx->primaries_out, rgb2rgb)) < 0)
            goto fail;
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

    if (ctx->trc_out == AVCOL_TRC_SMPTE2084)
        av_bprintf(&header, "#define SKIP_TONEMAP\n");

    if (ctx->dovi) {
        double ycc2rgb_offset[3] = {0};
        double lms2rgb[3][3];
        av_bprintf(&header, "#define DOVI_RESHAPE\n");
        if (ctx->tradeoff)
            av_bprintf(&header, "#define DOVI_PERF_TRADEOFF\n");
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                ycc2rgb_offset[i] -= ctx->dovi->nonlinear[i][j] * ctx->dovi->nonlinear_offset[j];
        }
        av_bprintf(&header, "__constant float3 ycc2rgb_offset = {%ff, %ff, %ff};\n",
                   ycc2rgb_offset[0], ycc2rgb_offset[1], ycc2rgb_offset[2]);
        ff_matrix_mul_3x3(lms2rgb, dovi_lms2rgb_matrix, ctx->dovi->linear);
        ff_opencl_print_const_matrix_3x3(&header, "rgb_matrix", ctx->dovi->nonlinear); //ycc2rgb
        ff_opencl_print_const_matrix_3x3(&header, "lms2rgb_matrix", lms2rgb); //lms2rgb
    } else {
        luma_src = av_csp_luma_coeffs_from_avcsp(ctx->colorspace_in);
        if (!luma_src) {
            err = AVERROR(EINVAL);
            av_log(avctx, AV_LOG_ERROR, "Unsupported input colorspace %d (%s)\n",
                   ctx->colorspace_in, av_color_space_name(ctx->colorspace_in));
            goto fail;
        }
        ff_fill_rgb2yuv_table(luma_src, rgb2yuv);
        ff_matrix_invert_3x3(rgb2yuv, yuv2rgb);
        ff_opencl_print_const_matrix_3x3(&header, "rgb_matrix", yuv2rgb);
    }

    luma_dst = av_csp_luma_coeffs_from_avcsp(ctx->colorspace_out);
    if (!luma_dst) {
        err = AVERROR(EINVAL);
        av_log(avctx, AV_LOG_ERROR, "Unsupported output colorspace %d (%s)\n",
               ctx->colorspace_out, av_color_space_name(ctx->colorspace_out));
        goto fail;
    }

    ff_fill_rgb2yuv_table(luma_dst, rgb2yuv);
    ff_opencl_print_const_matrix_3x3(&header, "yuv_matrix", rgb2yuv);

    av_bprintf(&header, "__constant float3 luma_dst = {%ff, %ff, %ff};\n",
               av_q2d(luma_dst->cr), av_q2d(luma_dst->cg), av_q2d(luma_dst->cb));

    if (ctx->tradeoff) {
        av_bprintf(&header, "#define LUT_TRC %d\n", LUT_SIZE - 1);
        if (ctx->trc_out != AVCOL_TRC_SMPTE2084) {
            av_bprintf(&header, "#define linearize %s\n", "linearize_lut");
            av_bprintf(&header, "#define delinearize %s\n", delinearize_funcs[ctx->trc_out]);
        }
        if (!ctx->lin_lut)
            if ((err = compute_trc_luts(avctx)) < 0)
                goto fail;
        print_opencl_const_trc_luts(avctx, &header);
    } else if (ctx->trc_out != AVCOL_TRC_SMPTE2084) {
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

    if (ctx->in_desc->comp[0].depth > ctx->out_desc->comp[0].depth) {
        av_assert0(sizeof(ff_fruit_dither_matrix) == sizeof(ff_fruit_dither_matrix[0]) * ff_fruit_dither_size * ff_fruit_dither_size);

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
    }

    ctx->kernel = clCreateKernel(ctx->ocf.program, "tonemap", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel %d.\n", cle);

    if (ctx->dovi)
        CL_CREATE_BUFFER_FLAGS(ctx, dovi_buf, dovi_buf_flags,
                               3*(params_sz+pivots_sz+coeffs_sz+mmr_sz), NULL);

    ctx->initialised = 1;
    return 0;

fail:
    av_bprint_finalize(&header, NULL);
    if (ctx->dovi_buf)
        clReleaseMemObject(ctx->dovi_buf);
    if (ctx->command_queue)
        clReleaseCommandQueue(ctx->command_queue);
    if (ctx->kernel)
        clReleaseKernel(ctx->kernel);
    if (event)
        clReleaseEvent(event);
    if (ctx->dither_image)
        clReleaseMemObject(ctx->dither_image);
    if (ctx->lin_lut)
        av_freep(&ctx->lin_lut);
    return err;
}

static av_cold void tonemap_opencl_uninit_dovi(AVFilterContext *avctx)
{
    TonemapOpenCLContext *ctx = avctx->priv;
    cl_int cle;

    if (ctx->dovi)
        av_freep(&ctx->dovi);

    if (ctx->dovi_buf) {
        cle = clReleaseMemObject(ctx->dovi_buf);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
            "dovi buf: %d.\n", cle);
    }

    ctx->init_with_dovi = 0;
}

static av_cold void tonemap_opencl_uninit_common(AVFilterContext *avctx)
{
    TonemapOpenCLContext *ctx = avctx->priv;
    cl_int cle;

    if (ctx->lin_lut)
        av_freep(&ctx->lin_lut);

    if (ctx->kernel) {
        cle = clReleaseKernel(ctx->kernel);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel: %d.\n", cle);
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

    ctx->initialised = 0;
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

    if (ctx->dither_image)
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &ctx->dither_image);

    if (ctx->dovi_buf)
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &ctx->dovi_buf);

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
    AVFrameSideData  *dovi_sd = NULL;
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
            ctx->peak = ff_determine_dovi_signal_peak(metadata);
        } else {
            ctx->peak = ff_determine_signal_peak(input);
        }
        av_log(ctx, AV_LOG_DEBUG, "Computed signal peak: %f\n", ctx->peak);
    }

    if (dovi_sd) {
        const AVDOVIMetadata *metadata = (AVDOVIMetadata *) dovi_sd->data;
        const AVDOVIRpuDataHeader *rpu = av_dovi_get_header(metadata);
        // only map dovi rpus that don't require an EL
        if (rpu->disable_residual_flag) {
            struct DoviMetadata *dovi = av_malloc(sizeof(*dovi));
            ctx->dovi = dovi;
            if (!ctx->dovi)
                goto fail;

            ff_map_dovi_metadata(ctx->dovi, metadata);
            ctx->trc_in = AVCOL_TRC_SMPTE2084;
            ctx->colorspace_in = AVCOL_SPC_UNSPECIFIED;
            ctx->primaries_in = AVCOL_PRI_BT2020;
        }
    }

    if (!ctx->init_with_dovi && ctx->dovi && ctx->initialised)
        tonemap_opencl_uninit_common(avctx);

    if (!ctx->initialised) {
        err = tonemap_opencl_init(avctx);
        if (err < 0)
            goto fail;

        ctx->init_with_dovi = !!ctx->dovi;
    }

    if (ctx->dovi) {
        cle = tonemap_opencl_update_dovi_buf(avctx);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to update dovi buf: %d.\n", cle);
        av_freep(&ctx->dovi);
    }

    err = launch_kernel(avctx, ctx->kernel, output, input, ctx->peak);
    if (err < 0)
        goto fail;

    cle = clFinish(ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue: %d.\n", cle);

    av_frame_free(&input);

    if (ctx->trc_out != AVCOL_TRC_SMPTE2084) {
        av_frame_remove_side_data(output, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
        av_frame_remove_side_data(output, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    }

    av_frame_remove_side_data(output, AV_FRAME_DATA_DOVI_RPU_BUFFER);
    av_frame_remove_side_data(output, AV_FRAME_DATA_DOVI_METADATA);

    av_log(ctx, AV_LOG_DEBUG, "Tonemapping output: %s, %ux%u (%"PRId64").\n",
           av_get_pix_fmt_name(output->format),
           output->width, output->height, output->pts);

    return ff_filter_frame(outlink, output);

fail:
    clFinish(ctx->command_queue);
    if (ctx->dovi)
        av_freep(&ctx->dovi);
    av_frame_free(&input);
    av_frame_free(&output);
    return err;
}

static av_cold void tonemap_opencl_uninit(AVFilterContext *avctx)
{
    tonemap_opencl_uninit_common(avctx);

    tonemap_opencl_uninit_dovi(avctx);

    ff_opencl_filter_uninit(avctx);
}

#define OFFSET(x) offsetof(TonemapOpenCLContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption tonemap_opencl_options[] = {
    { "tonemap", "Tonemap algorithm selection", OFFSET(tonemap), AV_OPT_TYPE_INT, { .i64 = TONEMAP_NONE }, TONEMAP_NONE, TONEMAP_MAX - 1, FLAGS, "tonemap" },
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
        { "smpte2084",        0,       0,                 AV_OPT_TYPE_CONST, { .i64 = AVCOL_TRC_SMPTE2084 },     0, 0, FLAGS, "transfer" },
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
    { "format",      "Output pixel format", OFFSET(format), AV_OPT_TYPE_PIXEL_FMT, { .i64 = AV_PIX_FMT_NONE }, AV_PIX_FMT_NONE, INT_MAX, FLAGS, "fmt" },
    { "apply_dovi",  "Apply Dolby Vision metadata if possible", OFFSET(apply_dovi), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { "tradeoff",    "Apply tradeoffs to offload computing", OFFSET(tradeoff), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 1, FLAGS, "tradeoff" },
        { "auto",          0,       0,                 AV_OPT_TYPE_CONST, { .i64 = -1 }, 0, 0, FLAGS, "tradeoff" },
        { "disabled",      0,       0,                 AV_OPT_TYPE_CONST, { .i64 = 0  }, 0, 0, FLAGS, "tradeoff" },
        { "enabled",       0,       0,                 AV_OPT_TYPE_CONST, { .i64 = 1  }, 0, 0, FLAGS, "tradeoff" },
    { "peak",        "Signal peak override", OFFSET(peak), AV_OPT_TYPE_DOUBLE, { .dbl = 0 }, 0, DBL_MAX, FLAGS },
    { "param",       "Tonemap parameter",   OFFSET(param), AV_OPT_TYPE_DOUBLE, { .dbl = NAN }, DBL_MIN, DBL_MAX, FLAGS },
    { "desat",       "Desaturation parameter",   OFFSET(desat_param), AV_OPT_TYPE_DOUBLE, { .dbl = 0.5}, 0, DBL_MAX, FLAGS },
    { "threshold",   "Scene detection threshold",   OFFSET(scene_threshold), AV_OPT_TYPE_DOUBLE, { .dbl = 0.2 }, 0, DBL_MAX, FLAGS },
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
};

static const AVFilterPad tonemap_opencl_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &tonemap_opencl_config_output,
    },
};

const AVFilter ff_vf_tonemap_opencl = {
    .name           = "tonemap_opencl",
    .description    = NULL_IF_CONFIG_SMALL("Perform HDR to SDR conversion with tonemapping."),
    .priv_size      = sizeof(TonemapOpenCLContext),
    .priv_class     = &tonemap_opencl_class,
    .init           = &ff_opencl_filter_init,
    .uninit         = &tonemap_opencl_uninit,
    FILTER_INPUTS(tonemap_opencl_inputs),
    FILTER_OUTPUTS(tonemap_opencl_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_OPENCL),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
