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

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl_ext.h>
#endif

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
#include "video.h"
#include "colorspace.h"
#include "dither_matrix.h"

#define OPENCL_SOURCE_NB 3

#define REF_WHITE_SCALE (REFERENCE_WHITE / REFERENCE_WHITE_ALT)

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUV420P16,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NV15,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P016,
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
    TONEMAP_MODE_AUTO,
    TONEMAP_MODE_COUNT,
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

#define params_cnt 8
#define pivots_cnt (7+1)
#define coeffs_cnt 8*4
#define mmr_cnt 8*6*4
#define params_sz params_cnt*sizeof(cl_float)
#define pivots_sz pivots_cnt*sizeof(cl_float)
#define coeffs_sz coeffs_cnt*sizeof(cl_float)
#define mmr_sz mmr_cnt*sizeof(cl_float)
    struct FFDOVIMetadataRemap *dovi;
    cl_mem dovi_buf;
    unsigned dovi_use_fp16;
    unsigned is_pure_dovi;
    unsigned is_hlg_dovi;

#define LUT_SIZE 65
    /* enum TonemapAlgorithm */
    int                   tonemap;
    /* enum TonemapMode */
    int                   tonemap_mode;
    enum AVPixelFormat    format;
    int                   apply_dovi;
    double                peak;
    double                src_peak;
    double                target_peak;
    double                param;
    double                final_param;
    double                desat_param;
    double                scene_threshold;
    int                   tradeoff;
    int                   use_image3d;
    int                   initialised;
    int                   init_with_dovi;
    cl_kernel             kernel;
    cl_kernel             lut_generation_kernel;
    cl_mem                dither_image;
    cl_mem                lut_buffer;
    cl_mem                lut_image;
    cl_command_queue      command_queue;
} TonemapOpenCLContext;

static const char *const linearize_funcs[] = {
    [AVCOL_TRC_SMPTE2084]    = "eotf_st2084x3",
    [AVCOL_TRC_ARIB_STD_B67] = "eotf_arib_b67x3",
};

static const char *const delinearize_funcs[] = {
    [AVCOL_TRC_BT709]     = "inverse_eotf_bt1886x3",
    [AVCOL_TRC_BT2020_10] = "inverse_eotf_bt1886x3",
};

static const char *const tonemap_func[TONEMAP_COUNT] = {
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

static unsigned as_unsigned(const float x) {
    const float *px = &x;
    return *(unsigned*)px;
}

// IEEE-754 16-bit floating-point format (without infinity):
// 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
static cl_half cl_float2half(const cl_float x) {
    // round-to-nearest-even: add last bit after truncated mantissa
    const unsigned b = as_unsigned(x)+0x00001000;
    // exponent
    const unsigned e = (b&0x7F800000)>>23;
    // mantissa; in line below:
    // 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
    const unsigned m = b&0x007FFFFF;

    // sign : normalized : denormalized : saturate
    return (b&0x80000000)>>16                                   |
           (e>112)*((((e-112)<<10)&0x7C00)|m>>13)               |
           ((e<113)&(e>101))*((((0x007FF000+m)>>(125-e))+1)>>1) |
           (e>143)*0x7FFF;
}

static int tonemap_opencl_update_dovi_buf(AVFilterContext *avctx)
{
    TonemapOpenCLContext *ctx = avctx->priv;
    const unsigned fp16 = !!ctx->dovi_use_fp16;
    const size_t buf_sz = 3*(params_sz+pivots_sz+coeffs_sz+mmr_sz) >> fp16;
    void *pbuf = NULL;
    cl_float coeffs_dataf[8][4] = {0};
    cl_float mmr_packed_dataf[8*6][4] = {0};
    cl_half coeffs_datah[8][4] = {0};
    cl_half mmr_packed_datah[8*6][4] = {0};
    int c, i, j, k, err av_unused;
    cl_int cle;

    pbuf = clEnqueueMapBuffer(ctx->command_queue, ctx->dovi_buf,
                              CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0,
                              buf_sz, 0, NULL, NULL, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to map dovi buf: %d.\n", cle);

    av_assert0(pbuf);

    for (c = 0; c < 3; c++) {
        int has_poly = 0, has_mmr = 0, mmr_single = 1;
        int mmr_idx = 0, min_order = 3, max_order = 1;
        const struct FFDOVIReshapeData *comp = &ctx->dovi->comp[c];
        if (!comp->num_pivots)
            continue;
        av_assert0(comp->num_pivots >= 2 && comp->num_pivots <= 9);

        memset(coeffs_dataf, 0, sizeof(coeffs_dataf));
        for (i = 0; i < comp->num_pivots - 1; i++) {
            switch (comp->method[i]) {
            case 0: // polynomial
                has_poly = 1;
                coeffs_dataf[i][3] = 0.0f; // order=0 signals polynomial
                for (k = 0; k < 3; k++)
                    coeffs_dataf[i][k] = comp->poly_coeffs[i][k];
                break;
            case 1:
                min_order = FFMIN(min_order, comp->mmr_order[i]);
                max_order = FFMAX(max_order, comp->mmr_order[i]);
                mmr_single = !has_mmr;
                has_mmr = 1;
                coeffs_dataf[i][3] = (float)comp->mmr_order[i];
                coeffs_dataf[i][0] = comp->mmr_constant[i];
                coeffs_dataf[i][1] = (float)mmr_idx;
                for (j = 0; j < comp->mmr_order[i]; j++) {
                    // store weights per order as two packed vec4s
                    cl_float *mmr = &mmr_packed_dataf[mmr_idx][0];
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

        if (fp16) {
            for (i = 0; i < 8; i++)
                for (j = 0; j < 4; j++)
                    coeffs_datah[i][j] = cl_float2half(coeffs_dataf[i][j]);

            for (i = 0; i < 8*6; i++)
                for (j = 0; j < 4; j++)
                    mmr_packed_datah[i][j] = cl_float2half(mmr_packed_dataf[i][j]);
        }

        // dovi_params
        {
            const cl_float paramsf[8] = {
                comp->num_pivots, !!has_mmr, !!has_poly,
                mmr_single, min_order, max_order,
                comp->pivots[0], comp->pivots[comp->num_pivots - 1]
            };

            if (fp16) {
                cl_half paramsh[8] = {0};
                for (i = 0; i < 8; i++)
                    paramsh[i] = cl_float2half(paramsf[i]);

                memcpy((cl_half*)pbuf + c*params_cnt, paramsh, params_sz>>1);
            } else
                memcpy((cl_float*)pbuf + c*params_cnt, paramsf, params_sz);
        }

        // dovi_pivots
        if (c == 0 && comp->num_pivots > 2) {
            // Skip the (irrelevant) lower and upper bounds
            cl_float pivots_dataf[7+1] = {0};
            memcpy(pivots_dataf, comp->pivots + 1,
                   (comp->num_pivots - 2) * sizeof(pivots_dataf[0]));
            // Fill the remainder with a quasi-infinite sentinel pivot
            for (i = comp->num_pivots - 2; i < FF_ARRAY_ELEMS(pivots_dataf); i++)
                pivots_dataf[i] = 1e9f;

            if (fp16) {
                cl_half pivots_datah[7+1] = {0};
                for (i = 0; i < 7+1; i++)
                    pivots_datah[i] = cl_float2half(pivots_dataf[i]);

                memcpy((cl_half*)pbuf + 3*params_cnt + c*pivots_cnt, pivots_datah, pivots_sz>>1);
            } else
                memcpy((cl_float*)pbuf + 3*params_cnt + c*pivots_cnt, pivots_dataf, pivots_sz);
        }

        // dovi_coeffs
        if (fp16)
            memcpy((cl_half*)pbuf + 3*(params_cnt+pivots_cnt) + c*coeffs_cnt, &coeffs_datah[0], coeffs_sz>>1);
        else
            memcpy((cl_float*)pbuf + 3*(params_cnt+pivots_cnt) + c*coeffs_cnt, &coeffs_dataf[0], coeffs_sz);

        // dovi_mmr
        if (has_mmr) {
            if (fp16)
                memcpy((cl_half*)pbuf + 3*(params_cnt+pivots_cnt+coeffs_cnt) + c*mmr_cnt, &mmr_packed_datah[0], mmr_sz>>1);
            else
                memcpy((cl_float*)pbuf + 3*(params_cnt+pivots_cnt+coeffs_cnt) + c*mmr_cnt, &mmr_packed_dataf[0], mmr_sz);
        }
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
    int rgb2rgb_passthrough = 1;
    double rgb2rgb[3][3], rgb2yuv[3][3], yuv2rgb[3][3];
    const AVLumaCoefficients *luma_src, *luma_dst;
    cl_event event_in = NULL, event_out = NULL;
    cl_mem_flags dovi_buf_flags = CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY;
    cl_uint device_vendor_id;
    cl_int cle;
    char *device_vendor = NULL;
    char *device_name = NULL;
    char *device_exts = NULL;
    int is_qcom_proprietary = 0;
    int i, j, err;

    if (ctx->tonemap_mode == TONEMAP_MODE_AUTO)
        ctx->tonemap_mode = TONEMAP_MODE_ITP;

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

    cle = clGetDeviceInfo(ctx->ocf.hwctx->device_id,
                          CL_DEVICE_VENDOR_ID,
                          sizeof(cl_uint), &device_vendor_id, NULL);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to check OpenCL "
                     "device vendor id %d.\n", cle);

    device_exts = check_opencl_device_str(ctx->ocf.hwctx->device_id,
                                          CL_DEVICE_EXTENSIONS);

    ctx->use_image3d = 0;
    if (ctx->tradeoff) {
        cl_bool device_is_uma = 0;
        int is_intel = 0, is_arm = 0, is_qcom = 0;
        int is_tradeoff_auto = ctx->tradeoff == -1;

        cle = clGetDeviceInfo(ctx->ocf.hwctx->device_id,
                              CL_DEVICE_HOST_UNIFIED_MEMORY,
                              sizeof(cl_bool), &device_is_uma, NULL);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to check if OpenCL "
                         "device is UMA %d.\n", cle);

        device_vendor = check_opencl_device_str(ctx->ocf.hwctx->device_id,
                                                CL_DEVICE_VENDOR);
        device_name = check_opencl_device_str(ctx->ocf.hwctx->device_id,
                                              CL_DEVICE_NAME);

        is_intel = device_vendor_id == 0x8086;
        is_arm   = device_vendor_id == 0x13b5 ||
                   (device_vendor && strstr(device_vendor, "ARM")) ||
                   (device_name && strstr(device_name, "Mali"));
        is_qcom  = device_vendor_id == 0x5143 ||
                   device_vendor_id == MKTAG('Q', 'C', 'O', 'M');

        ctx->tradeoff = 1;
        if (is_intel && device_is_uma && is_tradeoff_auto) {
            // Use tradeoff on low perf Intel iGPUs
            cl_uint max_compute_units = 0;

            cle = clGetDeviceInfo(ctx->ocf.hwctx->device_id,
                                  CL_DEVICE_MAX_COMPUTE_UNITS,
                                  sizeof(cl_uint), &max_compute_units, NULL);
            CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to check OpenCL "
                             "device max compute units %d.\n", cle);

            if (max_compute_units >= 40)
                ctx->tradeoff = 0;
            else if (device_name) {
                const char *excluded_devices[5] = { "Arc", "Iris", "Xe", "770", "750" };
                for (i = 0; i < FF_ARRAY_ELEMS(excluded_devices); i++) {
                    if (strstr(device_name, excluded_devices[i])) {
                        ctx->tradeoff = 0; break;
                    }
                }
            }
        } else if (is_arm && device_is_uma) {
            // Use tradeoff and check image3d_t support for lut on ARM Mali Valhall+
            cl_uint nb_formats = 0;
            cl_image_format *formats = NULL;

            if (!(device_exts && strstr(device_exts, "cl_arm_job_slot_selection"))) {
                cle = clGetSupportedImageFormats(ctx->ocf.hwctx->context,
                                                 CL_MEM_READ_ONLY,
                                                 CL_MEM_OBJECT_IMAGE3D,
                                                 0, NULL, &nb_formats);
                if (cle == CL_SUCCESS && nb_formats > 0) {
                    formats = av_malloc_array(nb_formats, sizeof(*formats));
                    if (!formats) {
                        err = AVERROR(ENOMEM);
                        goto fail;
                    }
                    cle = clGetSupportedImageFormats(ctx->ocf.hwctx->context,
                                                     CL_MEM_READ_ONLY,
                                                     CL_MEM_OBJECT_IMAGE3D,
                                                     nb_formats, formats, NULL);
                    for (i = 0; cle == CL_SUCCESS && i < nb_formats; i++) {
                        if (formats[i].image_channel_order == CL_RGBA &&
                            formats[i].image_channel_data_type == CL_FLOAT) {
                            ctx->use_image3d = 1; break;
                        }
                    }
                }
                av_freep(&formats);
            }
            if (ctx->use_image3d) {
                size_t value = 0;
                cl_device_info params[] = {
                    CL_DEVICE_IMAGE3D_MAX_WIDTH,
                    CL_DEVICE_IMAGE3D_MAX_HEIGHT,
                    CL_DEVICE_IMAGE3D_MAX_DEPTH
                };

                for (i = 0; i < FF_ARRAY_ELEMS(params); i++) {
                    cle = clGetDeviceInfo(ctx->ocf.hwctx->device_id, params[i],
                                          sizeof(value), &value, NULL);
                    if (cle != CL_SUCCESS || value < LUT_SIZE) {
                        ctx->use_image3d = 0; break;
                    }
                }
            }
            if (!ctx->use_image3d)
                av_log(avctx, AV_LOG_DEBUG,
                       "Disabled image3d for lut due to lack of support.\n");
        } else if (is_qcom) {
            // Always use tradeoff on Qualcomm due to inconsistent performance
        } else if (is_tradeoff_auto) {
            ctx->tradeoff = 0;
        }

        if (is_tradeoff_auto && !ctx->tradeoff)
            av_log(avctx, AV_LOG_DEBUG,
                   "Disabled tradeoffs on high performance device.\n");

        av_freep(&device_vendor);
        av_freep(&device_name);
    }

    // for low perf device, only do reshaping for pure dovi
    if (ctx->tradeoff && ctx->dovi && !ctx->is_pure_dovi) {
        av_freep(&ctx->dovi);
        ctx->apply_dovi = 0;
        if (ctx->is_hlg_dovi) {
            ctx->trc_in = AVCOL_TRC_ARIB_STD_B67;
            ctx->colorspace_in = AVCOL_SPC_BT2020_NCL;
            ctx->primaries_in = AVCOL_PRI_BT2020;
        }
    }

    // use FP16 for dovi reshaping only when tradeoff is enabled and it's supported
    ctx->dovi_use_fp16 = 0;
    if (ctx->tradeoff && ctx->dovi && device_exts && strstr(device_exts, "cl_khr_fp16")) {
        ctx->dovi_use_fp16 = 1;
        av_log(avctx, AV_LOG_DEBUG, "FP16 is enabled for DOVI reshaping.\n");
    }

    // zero-copy buffer requires this extension on Intel dGPUs
    if (device_vendor_id == 0x8086 && device_exts && strstr(device_exts, "cl_intel_mem_force_host_memory"))
        dovi_buf_flags |= (1 << 20); /* CL_MEM_FORCE_HOST_MEMORY_INTEL */

    av_freep(&device_exts);

    if (device_vendor_id == 0x5143) {
        // Qualcomm has two device IDs: 0x5143 and 0x4d4f4351 ('Q' | 'C' << 8 | 'O' << 16 | 'M' << 24)
        // The former is reported by Qualcomm's official OpenCL driver
        // and the latter is reported by Microsoft's compatability layer
        // The former has better performance if the kernel is written in a way its compiler handles properly
        // The latter one has more predictable performance and compiler behaves a bit more like other GPU
        // Only use the workaround on Qualcomm native OpenCL driver
        is_qcom_proprietary = 1;
        av_log(avctx, AV_LOG_DEBUG, "Qualcomm driver in use, vendor specific workarounds applied.\n");
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

    if (is_qcom_proprietary)
        av_bprintf(&header, "#define IS_QCOM_GPU\n");

    av_bprintf(&header, "#define LUT_SIZE %d\n", LUT_SIZE);
    if (ctx->use_image3d)
        av_bprintf(&header, "#define LUT_PERF_IMAGE3D\n");

    av_bprintf(&header, "__constant float tone_param = %.4ff;\n",
               ctx->final_param);
    av_bprintf(&header, "__constant float desat_param = %.4ff;\n",
               ctx->desat_param);
    av_bprintf(&header, "__constant float target_peak = %.4ff;\n",
               ctx->target_peak);
    av_bprintf(&header, "__constant float scene_threshold = %.4ff;\n",
               ctx->scene_threshold);

    av_bprintf(&header, "#define TONE_FUNC %s\n", tonemap_func[ctx->tonemap]);
    if (ctx->tonemap == TONEMAP_BT2390)
        av_bprintf(&header, "#define TONE_FUNC_BT2390\n");

    if (ctx->tonemap_mode == TONEMAP_MODE_RGB) {
        av_bprintf(&header, "#define TONE_MODE_RGB\n");
        av_bprintf(&header, "#define MAP_IN_DST_SPACE\n");
    }
    else if (ctx->tonemap_mode == TONEMAP_MODE_MAX) {
        av_bprintf(&header, "#define TONE_MODE_MAX\n");
        av_bprintf(&header, "#define MAP_IN_DST_SPACE\n");
    }
    else if (ctx->tonemap_mode == TONEMAP_MODE_ITP)
        av_bprintf(&header, "#define TONE_MODE_ITP\n");

    if (ctx->in_planes > 2)
        av_bprintf(&header, "#define NON_SEMI_PLANAR_IN\n");

    if (ctx->out_planes > 2)
        av_bprintf(&header, "#define NON_SEMI_PLANAR_OUT\n");

    if (ctx->in_fmt == AV_PIX_FMT_NV15)
        av_bprintf(&header, "#define P010LE_COMPACT_IN\n");

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

    if (ctx->in_desc->comp[0].depth == 16) {
        // Assume 16bit is actually 12bit for now as that is what the hardware decoders producing
        // and what videos are actually encoded in
        av_bprintf(&header, "__constant float input_quantization_offset = %.13lff;\n", QUANTIZATION_OFFSET(12));
        av_bprintf(&header, "__constant float input_y_scale = %.13lff;\n", INPUT_Y_SCALE(12));
        av_bprintf(&header, "__constant float input_uv_scale = %.13lff;\n", INPUT_UV_SCALE(12));
    } else {
        av_bprintf(&header, "__constant float input_quantization_offset = %.13lff;\n", QUANTIZATION_OFFSET(ctx->in_desc->comp[0].depth));
        av_bprintf(&header, "__constant float input_y_scale = %.13lff;\n", INPUT_Y_SCALE(ctx->in_desc->comp[0].depth));
        av_bprintf(&header, "__constant float input_uv_scale = %.13lff;\n", INPUT_UV_SCALE(ctx->in_desc->comp[0].depth));
    }

    if (ctx->out_desc->comp[0].depth > 8) {
        av_bprintf(&header, "#define RESCALE_LIMITED_RANGE_OUTPUT\n");
    }

    if (ctx->out_desc->comp[0].depth == 10) {
        av_bprintf(&header, "__constant float output_quantization_offset = %.13lff;\n", QUANTIZATION_OFFSET(10));
    } else {
        // Don't handle 12b offset for now and assume 16b output is real 16b out to make it consistent with other filters
        av_bprintf(&header, "__constant float output_quantization_offset = 0.0f;\n");
    }

    av_bprintf(&header, "#define chroma_loc %d\n", (int)ctx->chroma_loc);

    if (rgb2rgb_passthrough)
        av_bprintf(&header, "#define RGB2RGB_PASSTHROUGH\n");
    else
        ff_opencl_print_const_matrix_3x3(&header, "rgb2rgb", rgb2rgb);

    if (ctx->trc_out == AVCOL_TRC_SMPTE2084)
        av_bprintf(&header, "#define SKIP_TONEMAP\n");

    if (ctx->trc_in == AVCOL_TRC_ARIB_STD_B67 &&
        ctx->trc_out != AVCOL_TRC_SMPTE2084)
        av_bprintf(&header, "#define HLG_EOTF_BT2446B\n");

    luma_src = av_csp_luma_coeffs_from_avcsp(ctx->colorspace_in);
    if (!luma_src) {
        err = AVERROR(EINVAL);
        av_log(avctx, AV_LOG_ERROR, "Unsupported input colorspace %d (%s)\n",
               ctx->colorspace_in, av_color_space_name(ctx->colorspace_in));
        goto fail;
    }
    av_bprintf(&header, "__constant float3 luma_src = {%.13ff, %.13ff, %.13ff};\n",
               av_q2d(luma_src->cr), av_q2d(luma_src->cg), av_q2d(luma_src->cb));

    luma_dst = av_csp_luma_coeffs_from_avcsp(ctx->colorspace_out);
    if (!luma_dst) {
        err = AVERROR(EINVAL);
        av_log(avctx, AV_LOG_ERROR, "Unsupported output colorspace %d (%s)\n",
               ctx->colorspace_out, av_color_space_name(ctx->colorspace_out));
        goto fail;
    }
    av_bprintf(&header, "__constant float3 luma_dst = {%.13ff, %.13ff, %.13ff};\n",
               av_q2d(luma_dst->cr), av_q2d(luma_dst->cg), av_q2d(luma_dst->cb));

    if (ctx->dovi) {
        const size_t buf_sz = 3*(params_sz+pivots_sz+coeffs_sz+mmr_sz) >> !!ctx->dovi_use_fp16;
        double ycc2rgb_offset[3] = {0};
        double lms2rgb[3][3];
        av_bprintf(&header, "#define DOVI_RESHAPE\n");
        if (ctx->dovi_use_fp16)
            av_bprintf(&header, "#define DOVI_PERF_FP16\n");
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                ycc2rgb_offset[i] -= ctx->dovi->nonlinear[i][j] * ctx->dovi->nonlinear_offset[j];
        }
        av_bprintf(&header, "__constant float3 ycc2rgb_offset = {%.13ff, %.13ff, %.13ff};\n",
                   ycc2rgb_offset[0], ycc2rgb_offset[1], ycc2rgb_offset[2]);
        ff_matrix_mul_3x3(lms2rgb, dovi_lms2rgb_matrix, ctx->dovi->linear);
        ff_opencl_print_const_matrix_3x3(&header, "rgb_matrix", ctx->dovi->nonlinear); //ycc2rgb
        ff_opencl_print_const_matrix_3x3(&header, "lms2rgb_matrix", lms2rgb); //lms2rgb

        CL_CREATE_BUFFER_FLAGS(ctx, dovi_buf, dovi_buf_flags, buf_sz, NULL);
    } else {
        ff_fill_rgb2yuv_table(luma_src, rgb2yuv);
        ff_matrix_invert_3x3(rgb2yuv, yuv2rgb);
        ff_opencl_print_const_matrix_3x3(&header, "rgb_matrix", yuv2rgb);
    }

    ff_fill_rgb2yuv_table(luma_dst, rgb2yuv);
    ff_opencl_print_const_matrix_3x3(&header, "yuv_matrix", rgb2yuv);

    if (ctx->trc_out != AVCOL_TRC_SMPTE2084) {
        av_bprintf(&header, "#define linearize %s\n", linearize_funcs[ctx->trc_in]);
        av_bprintf(&header, "#define delinearize %s\n", delinearize_funcs[ctx->trc_out]);
    }

    av_log(avctx, AV_LOG_DEBUG, "Generated OpenCL header:\n%s\n", header.str);
    opencl_sources[0] = header.str;
    opencl_sources[1] = ff_source_tonemap_cl;
    opencl_sources[2] = ff_source_colorspace_common_cl;
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
                                  0, NULL, &event_out);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue write of dither matrix image: %d.\n", cle);

        cle = clWaitForEvents(1, &event_out);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to wait for event completion: %d.\n", cle);
        if (event_out) {
            clReleaseEvent(event_out);
            event_out = NULL;
        }
    }

    if (ctx->tradeoff) {
        const size_t lut_size = LUT_SIZE;
        const size_t lut_size_3d = lut_size * lut_size * lut_size;
        const size_t lut_buffer_size = lut_size_3d * sizeof(cl_float4);
        const float peak = (float)ctx->src_peak;
        const size_t m_origin[3] = { 0 };
        const size_t m_region[3] = { lut_size, lut_size, lut_size };
        cl_mem_flags mem_flags = CL_MEM_HOST_NO_ACCESS;

        cl_image_format image_format = {
            .image_channel_order     = CL_RGBA,
            .image_channel_data_type = CL_FLOAT,
        };
        cl_image_desc image_desc = {
            .image_type        = CL_MEM_OBJECT_IMAGE3D,
            .image_width       = lut_size,
            .image_height      = lut_size,
            .image_depth       = lut_size,
            .image_row_pitch   = 0,
            .image_slice_pitch = 0,
        };

        ctx->lut_generation_kernel = clCreateKernel(ctx->ocf.program, "build_lut", &cle);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel %d.\n", cle);

        CL_CREATE_BUFFER_FLAGS(ctx, lut_buffer, CL_MEM_READ_WRITE | mem_flags, lut_buffer_size, NULL);

        CL_SET_KERNEL_ARG(ctx->lut_generation_kernel, 0, cl_mem, &ctx->lut_buffer);
        CL_SET_KERNEL_ARG(ctx->lut_generation_kernel, 1, cl_float, &peak);
        cle = clEnqueueNDRangeKernel(ctx->command_queue, ctx->lut_generation_kernel, 1, NULL,
                                     &lut_size_3d, NULL,
                                     0, NULL, ctx->use_image3d ? &event_in : &event_out);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue build_lut kernel: %d.\n", cle);

        if (ctx->use_image3d) {
            ctx->lut_image = clCreateImage(ctx->ocf.hwctx->context, CL_MEM_READ_ONLY | mem_flags,
                                            &image_format, &image_desc, NULL, &err);
            if (!ctx->lut_image) {
                av_log(avctx, AV_LOG_ERROR, "Failed to create image for "
                       "lut image: %d.\n", cle);
                err = AVERROR(EIO);
                goto fail;
            }
            cle = clEnqueueCopyBufferToImage(ctx->command_queue,
                                             ctx->lut_buffer, ctx->lut_image,
                                             0, m_origin, m_region,
                                             1, &event_in, &event_out);
            CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue copy of lut buffer to image: %d.\n", cle);
        }

        cle = clWaitForEvents(1, &event_out);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to wait for event completion: %d.\n", cle);
        if (event_in) {
            clReleaseEvent(event_in);
            event_in = NULL;
        }
        if (event_out) {
            clReleaseEvent(event_out);
            event_out = NULL;
        }
    }

    ctx->kernel = clCreateKernel(ctx->ocf.program, ctx->tradeoff ? "tonemap_lut" : "tonemap", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel %d.\n", cle);

    ctx->initialised = 1;
    return 0;

fail:
    av_bprint_finalize(&header, NULL);
    av_freep(&device_vendor);
    av_freep(&device_name);
    av_freep(&device_exts);
    if (event_in)
        clReleaseEvent(event_in);
    if (event_out)
        clReleaseEvent(event_out);
    if (ctx->kernel)
        clReleaseKernel(ctx->kernel);
    if (ctx->lut_generation_kernel)
        clReleaseKernel(ctx->lut_generation_kernel);
    if (ctx->dither_image)
        clReleaseMemObject(ctx->dither_image);
    if (ctx->lut_buffer)
        clReleaseMemObject(ctx->lut_buffer);
    if (ctx->lut_image)
        clReleaseMemObject(ctx->lut_image);
    if (ctx->dovi_buf)
        clReleaseMemObject(ctx->dovi_buf);
    if (ctx->command_queue)
        clReleaseCommandQueue(ctx->command_queue);
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

    if (ctx->kernel) {
        cle = clReleaseKernel(ctx->kernel);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel: %d.\n", cle);
    }

    if (ctx->lut_generation_kernel) {
        cle = clReleaseKernel(ctx->lut_generation_kernel);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                                        "lut_generation_kernel: %d.\n", cle);
    }

    if (ctx->ocf.program) {
        cle = clReleaseProgram(ctx->ocf.program);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "program: %d.\n", cle);
        ctx->ocf.program = NULL;
    }

    if (ctx->dither_image) {
        cle = clReleaseMemObject(ctx->dither_image);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
            "dither image: %d.\n", cle);
    }

    if (ctx->lut_buffer) {
        cle = clReleaseMemObject(ctx->lut_buffer);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                                        "lut buffer: %d.\n", cle);
    }

    if (ctx->lut_image) {
        cle = clReleaseMemObject(ctx->lut_image);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                                        "lut image: %d.\n", cle);
    }

    if (ctx->command_queue) {
        cle = clReleaseCommandQueue(ctx->command_queue);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "command queue: %d.\n", cle);
    }

    ctx->initialised = 0;
}

static av_cold int tonemap_opencl_preinit(AVFilterContext *avctx)
{
    TonemapOpenCLContext *ctx = avctx->priv;
    ctx->final_param = NAN;
    return 0;
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
    FilterLink           *inl = ff_filter_link(inlink);
    TonemapOpenCLContext *ctx = avctx->priv;
    AVHWFramesContext *in_frames_ctx;
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
    if (!format_is_supported(out_format) || out_format == AV_PIX_FMT_NV15) {
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

    if (ctx->trc != AVCOL_TRC_SMPTE2084) {
        av_frame_side_data_remove(&outlink->side_data, &outlink->nb_side_data,
                                  AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
        av_frame_side_data_remove(&outlink->side_data, &outlink->nb_side_data,
                                  AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    }

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

    idx_arg = 0;
    if (ctx->tradeoff) {
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem,
            ctx->use_image3d ? &ctx->lut_image : &ctx->lut_buffer);
    }
    CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &output->data[0]);
    CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &input->data[0]);
    CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &output->data[1]);
    CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &input->data[1]);

    if (ctx->out_planes > 2) {
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &output->data[2]);
    }

    if (ctx->in_planes > 2) {
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &input->data[2]);
    }

    if (ctx->dither_image) {
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &ctx->dither_image);
    }

    if (ctx->dovi_buf) {
        CL_SET_KERNEL_ARG(kernel, idx_arg++, cl_mem, &ctx->dovi_buf);
    }

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
            const int l0_only = ctx->tradeoff == 1 || !ctx->src_peak;
            if (!ctx->src_peak || ctx->tradeoff != 1) {
                ctx->src_peak = ff_determine_dovi_signal_peak(metadata, l0_only);
                ctx->src_peak *= REF_WHITE_SCALE;
            }
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

            ctx->is_pure_dovi = rpu->vdr_rpu_profile == 0;
            ctx->is_hlg_dovi = input->color_trc == AVCOL_TRC_ARIB_STD_B67;

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

    err = launch_kernel(avctx, ctx->kernel, output, input, ctx->src_peak);
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

static int tonemap_opencl_query_formats(AVFilterContext *avctx)
{
    TonemapOpenCLContext *ctx = avctx->priv;
    AVFilterFormats *formats;
    int ret;
    const enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_OPENCL, AV_PIX_FMT_NONE };

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

#define OFFSET(x) offsetof(TonemapOpenCLContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption tonemap_opencl_options[] = {
    { "tonemap", "Tonemap algorithm selection", OFFSET(tonemap), AV_OPT_TYPE_INT, { .i64 = TONEMAP_BT2390 }, TONEMAP_NONE, TONEMAP_COUNT - 1, FLAGS, "tonemap" },
        { "none",     0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_NONE },              0, 0, FLAGS, "tonemap" },
        { "linear",   0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_LINEAR },            0, 0, FLAGS, "tonemap" },
        { "gamma",    0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_GAMMA },             0, 0, FLAGS, "tonemap" },
        { "clip",     0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_CLIP },              0, 0, FLAGS, "tonemap" },
        { "reinhard", 0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_REINHARD },          0, 0, FLAGS, "tonemap" },
        { "hable",    0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_HABLE },             0, 0, FLAGS, "tonemap" },
        { "mobius",   0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_MOBIUS },            0, 0, FLAGS, "tonemap" },
        { "bt2390",   0, 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_BT2390 },            0, 0, FLAGS, "tonemap" },
    { "tonemap_mode", "Tonemap mode selection", OFFSET(tonemap_mode), AV_OPT_TYPE_INT, { .i64 = TONEMAP_MODE_AUTO }, TONEMAP_MODE_MAX, TONEMAP_MODE_COUNT - 1, FLAGS, "tonemap_mode" },
        { "max",  "Brightest channel based tonemap",  0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_MODE_MAX },  0, 0, FLAGS, "tonemap_mode" },
        { "rgb",  "Per-channel based tonemap",        0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_MODE_RGB },  0, 0, FLAGS, "tonemap_mode" },
        { "lum",  "Relative luminance based tonemap", 0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_MODE_LUM },  0, 0, FLAGS, "tonemap_mode" },
        { "itp",  "ICtCp intensity based tonemap",    0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_MODE_ITP },  0, 0, FLAGS, "tonemap_mode" },
        { "auto", "Select the preferred mode",        0, AV_OPT_TYPE_CONST, { .i64 = TONEMAP_MODE_AUTO }, 0, 0, FLAGS, "tonemap_mode" },
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
    { "range",         "Set color range", OFFSET(range), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, INT_MAX, FLAGS, "range" },
    { "r",             "Set color range", OFFSET(range), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, INT_MAX, FLAGS, "range" },
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
    { "desat",       "Desaturation parameter",   OFFSET(desat_param), AV_OPT_TYPE_DOUBLE, { .dbl = 0 }, 0, DBL_MAX, FLAGS },
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

const FFFilter ff_vf_tonemap_opencl = {
    .p.name         = "tonemap_opencl",
    .p.description  = NULL_IF_CONFIG_SMALL("Perform HDR to SDR conversion with tonemapping."),
    .priv_size      = sizeof(TonemapOpenCLContext),
    .p.priv_class   = &tonemap_opencl_class,
    .preinit        = &tonemap_opencl_preinit,
    .init           = &ff_opencl_filter_init,
    .uninit         = &tonemap_opencl_uninit,
    FILTER_INPUTS(tonemap_opencl_inputs),
    FILTER_OUTPUTS(tonemap_opencl_outputs),
    FILTER_QUERY_FUNC(tonemap_opencl_query_formats),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
    .p.flags        = AVFILTER_FLAG_HWDEVICE,
};
