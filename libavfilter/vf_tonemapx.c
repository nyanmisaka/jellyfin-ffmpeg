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

/**
 * @file
 * tonemap algorithms
 */

#include <float.h>
#include <string.h>

#include "libavutil/avassert.h"
#include "libavutil/imgutils.h"
#include "libavutil/internal.h"
#include "libavutil/mem.h"
#include "libavutil/mem_internal.h"
#include "libavutil/opt.h"
#include "libavutil/cpu.h"

#include "vf_tonemapx.h"

#ifdef CC_SUPPORTS_TONEMAPX_INTRINSICS
#    if ARCH_AARCH64
#        if HAVE_INTRINSICS_NEON
#            include "libavutil/aarch64/cpu.h"
#            include "aarch64/vf_tonemapx_intrin_neon.h"
#        endif
#    endif // ARCH_AARCH64
#    if ARCH_X86
#        include "libavutil/x86/cpu.h"
#        if HAVE_INTRINSICS_SSE42
#            include "x86/vf_tonemapx_intrin_sse.h"
#        endif
#        if HAVE_INTRINSICS_AVX2 && HAVE_INTRINSICS_FMA3
#            include "x86/vf_tonemapx_intrin_avx.h"
#        endif
#    endif // ARCH_X86
#endif // CC_SUPPORTS_TONEMAPX_INTRINSICS

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "video.h"

#define MIX(x, y, a) ((x) + ((y) - (x)) * (a))
#define CLAMP(a, b, c) (FFMIN(FFMAX((a), (b)), (c)))

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

typedef struct TonemapxContext {
    const AVClass *class;

    enum TonemapAlgorithm tonemap;
    enum AVColorTransferCharacteristic trc;
    enum AVColorSpace spc;
    enum AVColorPrimaries pri;
    enum AVColorRange range;
    enum AVPixelFormat format;
    char *format_str;
    double param;
    double desat;
    double peak;
    int apply_dovi;

    const AVLumaCoefficients *coeffs, *ocoeffs;

    double lut_peak;
    float *lin_lut;
    float *tonemap_lut;
    uint16_t *delin_lut;
    int in_yuv_off, out_yuv_off;

    struct DoviMetadata *dovi;

    DECLARE_ALIGNED(16, float,   dovi_pbuf)[3*(params_sz+pivots_sz+coeffs_sz+mmr_sz)];
    DECLARE_ALIGNED(16, int,     yuv2rgb_coeffs)[3][3][8];
    DECLARE_ALIGNED(16, int,     rgb2yuv_coeffs)[3][3][8];
    DECLARE_ALIGNED(16, double,  rgb2rgb_coeffs)[3][3];
    DECLARE_ALIGNED(16, double,  lms2rgb_matrix)[3][3];
    DECLARE_ALIGNED(16, float,   ycc_offset)[3];

    int (*filter_slice) (AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs);

    void (*tonemap_func_biplanar8) (uint8_t *dsty, uint8_t *dstuv,
                                    const uint16_t *srcy, const uint16_t *srcuv,
                                    const int *dstlinesize, const int *srclinesize,
                                    int dstdepth, int srcdepth,
                                    int width, int height,
                                    const struct TonemapIntParams *params);

    void (*tonemap_func_planar8) (uint8_t *dsty, uint8_t *dstu, uint8_t *dstv,
                                  const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                  const int *dstlinesize, const int *srclinesize,
                                  int dstdepth, int srcdepth,
                                  int width, int height,
                                  const struct TonemapIntParams *params);

    void (*tonemap_func_biplanar10) (uint16_t *dsty, uint16_t *dstuv,
                                     const uint16_t *srcy, const uint16_t *srcuv,
                                     const int *dstlinesize, const int *srclinesize,
                                     int dstdepth, int srcdepth,
                                     int width, int height,
                                     const struct TonemapIntParams *params);

    void (*tonemap_func_planar10) (uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                   const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                   const int *dstlinesize, const int *srclinesize,
                                   int dstdepth, int srcdepth,
                                   int width, int height,
                                   const struct TonemapIntParams *params);

    void (*tonemap_func_dovi8) (uint8_t *dsty, uint8_t *dstu, uint8_t *dstv,
                                const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                const int *dstlinesize, const int *srclinesize,
                                int dstdepth, int srcdepth,
                                int width, int height,
                                const struct TonemapIntParams *params);

    void (*tonemap_func_dovi10) (uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                 const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                 const int *dstlinesize, const int *srclinesize,
                                 int dstdepth, int srcdepth,
                                 int width, int height,
                                 const struct TonemapIntParams *params);

} TonemapxContext;

typedef struct ThreadData {
    AVFrame *in, *out;
    const AVPixFmtDescriptor *desc, *odesc;
    double peak;
} ThreadData;

static const enum AVPixelFormat in_pix_fmts[] = {
    AV_PIX_FMT_YUV420P10,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_NONE,
};

static const enum AVPixelFormat out_pix_fmts[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUV420P10,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
};

static const int colorspaces_out[] = {
    AVCOL_SPC_UNSPECIFIED,
    AVCOL_SPC_BT709,
    AVCOL_SPC_BT2020_NCL,
    -1
};

const double dovi_lms2rgb_matrix[3][3] =
{
    { 3.06441879, -2.16597676,  0.10155818},
    {-0.65612108,  1.78554118, -0.12943749},
    { 0.01736321, -0.04725154,  1.03004253},
};

static void update_dovi_buf(AVFilterContext *ctx)
{
    TonemapxContext *s = ctx->priv;
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

inline static float dot(const float* x, const float* y, int len)
{
    int i;
    float result = 0;
    for (i = 0; i < len; i++) {
        result += x[i] * y[i];
    }
    return result;
}

inline static float reshape_poly(float s, float* coeffs) {
    return (coeffs[2] * s + coeffs[1]) * s + coeffs[0];
}

inline static float reshape_mmr(const float* sig, const float* coeffs, const float* mmr,
                                int mmr_single, int min_order, int max_order)
{
    int mmr_idx = mmr_single ? 0 : (int)coeffs[1];
    int order = (int)coeffs[3];
    float s = coeffs[0];
    float sigX[7+1] = {sig[0], sig[1], sig[2], 0,
                       sig[0]*sig[1], sig[0]*sig[2], sig[1]*sig[2], sig[0]*sig[1]*sig[2]};

    s += dot(&mmr[mmr_idx + 0*4], sigX, 7+1);
    if (max_order >= 2 && (min_order >= 2 || order >= 2)) {
        float sigX2[7+1] = {sig[0]*sig[0], sig[1]*sig[1], sig[2]*sig[2], 0,
                            sigX[4]*sigX[4], sigX[5]*sigX[5], sigX[6]*sigX[6], sigX[7]*sigX[7]};
        s += dot(&mmr[mmr_idx + 2*4], sigX2, 7+1);

        if (max_order == 3 && (min_order == 3 || order >= 3)) {
            float sigX3[7+1] = {sig[0]*sig[0]*sig[0], sig[1]*sig[1]*sig[1], sig[2]*sig[2]*sig[2], 0,
                                sigX2[4]*sigX[4], sigX2[5]*sigX[5], sigX2[6]*sigX[6], sigX2[7]*sigX[7]};
            s += dot(&mmr[mmr_idx + 4*4], sigX3, 7+1);
        }
    }

    return s;
}

inline static void ycc2rgb(float* dest, float y, float cb, float cr, const double nonlinear[3][3], const float ycc_offset[3])
{
    dest[0] = (y * (float)nonlinear[0][0] + cb * (float)nonlinear[0][1] + cr * (float)nonlinear[0][2]) - ycc_offset[0];
    dest[1] = (y * (float)nonlinear[1][0] + cb * (float)nonlinear[1][1] + cr * (float)nonlinear[1][2]) - ycc_offset[1];
    dest[2] = (y * (float)nonlinear[2][0] + cb * (float)nonlinear[2][1] + cr * (float)nonlinear[2][2]) - ycc_offset[2];
}

// This implementation does not do the costly linearization and de-linearization for performance reasons
// The output color accuracy will be affected due to this
inline static void lms2rgb(float* dest, float l, float m, float s, const double linear[3][3], const double lms2rgb_matrix[3][3])
{
    dest[0] = l * (float)lms2rgb_matrix[0][0] + m * (float)lms2rgb_matrix[0][1] + s * (float)lms2rgb_matrix[0][2];
    dest[1] = l * (float)lms2rgb_matrix[1][0] + m * (float)lms2rgb_matrix[1][1] + s * (float)lms2rgb_matrix[1][2];
    dest[2] = l * (float)lms2rgb_matrix[2][0] + m * (float)lms2rgb_matrix[2][1] + s * (float)lms2rgb_matrix[2][2];
}

inline static void reshape_dovi_yuv(float* dest, float* src, const TonemapIntParams *ctx)
{
    int i;
    float s;
    float coeffs[4] = {0, 0, 0, 0};
    float sig_arr[3] = {src[0],src[1],src[2]};

    int dovi_num_pivots, dovi_has_mmr, dovi_has_poly;
    int dovi_mmr_single, dovi_min_order, dovi_max_order;
    int has_mmr_poly;
    float dovi_lo, dovi_hi;
    float *dovi_params;
    float *dovi_pivots;
    float *dovi_coeffs, *dovi_mmr; //float4*

    float *src_dovi_params = ctx->dovi_pbuf;
    float *src_dovi_pivots = ctx->dovi_pbuf + 24;
    float *src_dovi_coeffs = ctx->dovi_pbuf + 48; //float4*
    float *src_dovi_mmr = ctx->dovi_pbuf + 144; //float4*

    for (i = 0; i < 3; i++) {
        dovi_params = src_dovi_params + i*8;
        dovi_pivots = src_dovi_pivots + i*8;
        dovi_coeffs = src_dovi_coeffs + i*8*4; //float4*
        dovi_mmr = src_dovi_mmr + i*48*4; //float4*
        dovi_num_pivots = dovi_params[0];
        dovi_has_mmr = dovi_params[1];
        dovi_has_poly = dovi_params[2];
        dovi_mmr_single = dovi_params[3];
        dovi_min_order = dovi_params[4];
        dovi_max_order = dovi_params[5];
        dovi_lo = dovi_params[6];
        dovi_hi = dovi_params[7];

        s = sig_arr[i];
        coeffs[0] = dovi_coeffs[0*4+0];
        coeffs[1] = dovi_coeffs[0*4+1];
        coeffs[2] = dovi_coeffs[0*4+2];
        coeffs[3] = dovi_coeffs[0*4+3];

        if (i == 0 && dovi_num_pivots > 2) {
            const int t0 = s >= dovi_pivots[0], t1 = s >= dovi_pivots[1];
            const int t2 = s >= dovi_pivots[2], t3 = s >= dovi_pivots[3];
            const int t4 = s >= dovi_pivots[4], t5 = s >= dovi_pivots[5], t6 = s >= dovi_pivots[6];

            float m01[4] = {
                t0 ? dovi_coeffs[1*4 + 0] : dovi_coeffs[0*4 + 0],
                t0 ? dovi_coeffs[1*4 + 1] : dovi_coeffs[0*4 + 1],
                t0 ? dovi_coeffs[1*4 + 2] : dovi_coeffs[0*4 + 2],
                t0 ? dovi_coeffs[1*4 + 3] : dovi_coeffs[0*4 + 3]
            };

            float m23[4] = {
                t2 ? dovi_coeffs[3*4 + 0] : dovi_coeffs[2*4 + 0],
                t2 ? dovi_coeffs[3*4 + 1] : dovi_coeffs[2*4 + 1],
                t2 ? dovi_coeffs[3*4 + 2] : dovi_coeffs[2*4 + 2],
                t2 ? dovi_coeffs[3*4 + 3] : dovi_coeffs[2*4 + 3]
            };

            float m0123[4] = {
                t1 ? m23[0] : m01[0],
                t1 ? m23[1] : m01[1],
                t1 ? m23[2] : m01[2],
                t1 ? m23[3] : m01[3]
            };

            float m45[4] = {
                t4 ? dovi_coeffs[5*4 + 0] : dovi_coeffs[4*4 + 0],
                t4 ? dovi_coeffs[5*4 + 1] : dovi_coeffs[4*4 + 1],
                t4 ? dovi_coeffs[5*4 + 2] : dovi_coeffs[4*4 + 2],
                t4 ? dovi_coeffs[5*4 + 3] : dovi_coeffs[4*4 + 3]
            };

            float m67[4] = {
                t6 ? dovi_coeffs[7*4 + 0] : dovi_coeffs[6*4 + 0],
                t6 ? dovi_coeffs[7*4 + 1] : dovi_coeffs[6*4 + 1],
                t6 ? dovi_coeffs[7*4 + 2] : dovi_coeffs[6*4 + 2],
                t6 ? dovi_coeffs[7*4 + 3] : dovi_coeffs[6*4 + 3]
            };

            float m4567[4] = {
                t5 ? m67[0] : m45[0],
                t5 ? m67[1] : m45[1],
                t5 ? m67[2] : m45[2],
                t5 ? m67[3] : m45[3]
            };

            coeffs[0] = t3 ? m4567[0] : m0123[0];
            coeffs[1] = t3 ? m4567[1] : m0123[1];
            coeffs[2] = t3 ? m4567[2] : m0123[2];
            coeffs[3] = t3 ? m4567[3] : m0123[3];
        }

        has_mmr_poly = dovi_has_mmr && dovi_has_poly;

        if ((has_mmr_poly && coeffs[3] == 0.0f) || (!has_mmr_poly && dovi_has_poly))
            s = reshape_poly(s, coeffs);
        else
            s = reshape_mmr(sig_arr, coeffs, dovi_mmr,
                            dovi_mmr_single, dovi_min_order, dovi_max_order);

        sig_arr[i] = CLAMP(s, dovi_lo, dovi_hi);
    }

    dest[0] = sig_arr[0];
    dest[1] = sig_arr[1];
    dest[2] = sig_arr[2];
}

static int out_format_is_supported(enum AVPixelFormat fmt)
{
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(out_pix_fmts); i++)
        if (out_pix_fmts[i] == fmt)
            return 1;
    return 0;
}

static float hable(float in)
{
    float a = 0.15f, b = 0.50f, c = 0.10f, d = 0.20f, e = 0.02f, f = 0.30f;
    return (in * (in * a + b * c) + d * e) / (in * (in * a + b) + d * f) - e / f;
}

static float mobius(float in, float j, double peak)
{
    float a, b;

    if (in <= j)
        return in;

    a = -j * j * (peak - 1.0f) / (j * j - 2.0f * j + peak);
    b = (j * j - 2.0f * j * peak + peak) / FFMAX(peak - 1.0f, FLOAT_EPS);

    return (b * b + 2.0f * b * j + j * j) / (b - a) * (in + a) / (in + b);
}

static float bt2390(float s, float knee_offset, float peak)
{
    float peak_pq = inverse_eotf_st2084(peak, REFERENCE_WHITE_ALT);
    float scale = peak_pq > 0.0f ? (1.0f / peak_pq) : 1.0f;

    // SDR peak
    float dst_peak = 1.0f;
    float s_pq = inverse_eotf_st2084(s, REFERENCE_WHITE_ALT) * scale;
    float max_lum = inverse_eotf_st2084(dst_peak, REFERENCE_WHITE_ALT) * scale;

    float ks = (1.0f + knee_offset) * max_lum - knee_offset;
    float tb = (s_pq - ks) / (1.0f - ks);
    float tb2 = tb * tb;
    float tb3 = tb2 * tb;
    float pb = (2.0f * tb3 - 3.0f * tb2 + 1.0f) * ks +
               (tb3 - 2.0f * tb2 + tb) * (1.0f - ks) +
               (-2.0f * tb3 + 3.0f * tb2) * max_lum;
    float sig = MIX(pb, s_pq, s_pq < ks);

    return eotf_st2084(sig * peak_pq, REFERENCE_WHITE_ALT);
}

static float mapsig(enum TonemapAlgorithm alg, float sig, double peak, double param)
{
    switch(alg) {
    default:
    case TONEMAP_NONE:
        // do nothing
        break;
    case TONEMAP_LINEAR:
        sig = sig * param / peak;
        break;
    case TONEMAP_GAMMA:
        sig = sig > 0.05f
              ? pow(sig / peak, 1.0f / param)
              : sig * pow(0.05f / peak, 1.0f / param) / 0.05f;
        break;
    case TONEMAP_CLIP:
        sig = av_clipf(sig * param, 0, 1.0f);
        break;
    case TONEMAP_HABLE:
        sig = hable(sig) / hable(peak);
        break;
    case TONEMAP_REINHARD:
        sig = sig / (sig + param) * (peak + param) / peak;
        break;
    case TONEMAP_MOBIUS:
        sig = mobius(sig, param, peak);
        break;
    case TONEMAP_BT2390:
        sig = bt2390(sig, param, peak);
        break;
    }

    return sig;
}

static float linearize(float x, enum AVColorTransferCharacteristic trc_src)
{
    if (trc_src == AVCOL_TRC_SMPTE2084)
        return eotf_st2084(x, REFERENCE_WHITE_ALT);
    else if (trc_src == AVCOL_TRC_ARIB_STD_B67)
        return eotf_arib_b67(x);
    else
        return x;
}

static float delinearize(float x, enum AVColorTransferCharacteristic trc_dst)
{
    if (trc_dst == AVCOL_TRC_BT709 || trc_dst == AVCOL_TRC_BT2020_10)
        return inverse_eotf_bt1886(x);
    else
        return x;
}

static int compute_trc_luts(TonemapxContext *s, enum AVColorTransferCharacteristic trc_src,
                            enum AVColorTransferCharacteristic trc_dst)
{
    int i;

    if (!s->lin_lut && !(s->lin_lut = av_calloc(32768, sizeof(float))))
        return AVERROR(ENOMEM);
    if (!s->delin_lut && !(s->delin_lut = av_calloc(32768, sizeof(uint16_t))))
        return AVERROR(ENOMEM);

    for (i = 0; i < 32768; i++) {
        float v = (float)i / JPEG_SCALE;
        s->lin_lut[i] = FFMAX(linearize(v, trc_src), 0);
        s->delin_lut[i] = av_clip_int16((int)rintf(delinearize(v, trc_dst) * JPEG_SCALE));
    }

    return 0;
}

static int compute_tonemap_lut(TonemapxContext *s, enum AVColorTransferCharacteristic trc_src)
{
    int i;
    double peak = s->lut_peak;

    if (!s->tonemap_lut && !(s->tonemap_lut = av_calloc(32768, sizeof(float))))
        return AVERROR(ENOMEM);

    for (i = 0; i < 32768; i++) {
        float v = (float)i / JPEG_SCALE;
        float sig = linearize(v, trc_src);
        float mapped = mapsig(s->tonemap, sig, peak, s->param);
        s->tonemap_lut[i] = (sig > 0.0f && mapped > 0.0f) ? mapped / sig : 0.0f;
    }

    return 0;
}

static int compute_yuv_coeffs(TonemapxContext *s,
                              const AVLumaCoefficients *coeffs,
                              const AVLumaCoefficients *ocoeffs,
                              const AVPixFmtDescriptor *idesc,
                              const AVPixFmtDescriptor *odesc,
                              enum AVColorRange irng,
                              enum AVColorRange orng)
{
    double rgb2yuv[3][3], yuv2rgb[3][3];
    int res;
    int y_rng, uv_rng;

    res = ff_get_range_off(&s->in_yuv_off, &y_rng, &uv_rng,
                           irng, idesc->comp[0].depth);
    if (res < 0) {
        av_log(s, AV_LOG_ERROR,
               "Unsupported input color range %d (%s)\n",
               irng, av_color_range_name(irng));
        return res;
    }

    ff_fill_rgb2yuv_table(coeffs, rgb2yuv);
    ff_matrix_invert_3x3(rgb2yuv, yuv2rgb);
    ff_fill_rgb2yuv_table(ocoeffs, rgb2yuv);

    ff_get_yuv_coeffs(s->yuv2rgb_coeffs, yuv2rgb, idesc->comp[0].depth,
                      y_rng, uv_rng, 1);

    res = ff_get_range_off(&s->out_yuv_off, &y_rng, &uv_rng,
                           orng, odesc->comp[0].depth);
    if (res < 0) {
        av_log(s, AV_LOG_ERROR,
               "Unsupported output color range %d (%s)\n",
               orng, av_color_range_name(orng));
        return res;
    }

    ff_get_yuv_coeffs(s->rgb2yuv_coeffs, rgb2yuv, odesc->comp[0].depth,
                      y_rng, uv_rng, 0);

    return 0;
}

static int compute_rgb_coeffs(TonemapxContext *s,
                              enum AVColorPrimaries iprm,
                              enum AVColorPrimaries oprm)
{
    double rgb2xyz[3][3], xyz2rgb[3][3];
    const AVColorPrimariesDesc *iprm_desc = av_csp_primaries_desc_from_id(iprm);
    const AVColorPrimariesDesc *oprm_desc = av_csp_primaries_desc_from_id(oprm);

    if (!iprm_desc) {
        av_log(s, AV_LOG_ERROR,
               "Unsupported input color primaries %d (%s)\n",
               iprm, av_color_primaries_name(iprm));
        return AVERROR(EINVAL);
    }
    if (!oprm_desc) {
        av_log(s, AV_LOG_ERROR,
               "Unsupported output color primaries %d (%s)\n",
               oprm, av_color_primaries_name(oprm));
        return AVERROR(EINVAL);
    }

    ff_fill_rgb2xyz_table(&oprm_desc->prim, &oprm_desc->wp, rgb2xyz);
    ff_matrix_invert_3x3(rgb2xyz, xyz2rgb);
    ff_fill_rgb2xyz_table(&iprm_desc->prim, &iprm_desc->wp, rgb2xyz);
    ff_matrix_mul_3x3(s->rgb2rgb_coeffs, rgb2xyz, xyz2rgb);

    return 0;
}

__attribute__((always_inline))
static inline void dovi2rgb(int y00, int y01, int y10, int y11, int u, int v,
                            const struct TonemapIntParams *params,
                            const float in_rng,
                            int16_t r[4], int16_t g[4], int16_t b[4])
{
    float yuv1[3], yuv2[3], yuv3[3], yuv4[3];
    float c1[3], c2[3], c3[3], c4[3];

    yuv1[0] = CLAMP(y00 / in_rng, 0.0f, 1.0f);
    yuv2[0] = CLAMP(y01 / in_rng, 0.0f, 1.0f);
    yuv3[0] = CLAMP(y10 / in_rng, 0.0f, 1.0f);
    yuv4[0] = CLAMP(y11 / in_rng, 0.0f, 1.0f);
    yuv1[1] = yuv2[1] = yuv3[1] = yuv4[1] = CLAMP(u / in_rng, 0.0f, 1.0f);
    yuv1[2] = yuv2[2] = yuv3[2] = yuv4[2] = CLAMP(v / in_rng, 0.0f, 1.0f);

    reshape_dovi_yuv(yuv1, yuv1, params);
    reshape_dovi_yuv(yuv2, yuv2, params);
    reshape_dovi_yuv(yuv3, yuv3, params);
    reshape_dovi_yuv(yuv4, yuv4, params);

    ycc2rgb(c1, yuv1[0], yuv1[1], yuv1[2], params->dovi->nonlinear, *params->ycc_offset);
    ycc2rgb(c2, yuv2[0], yuv2[1], yuv2[2], params->dovi->nonlinear, *params->ycc_offset);
    ycc2rgb(c3, yuv3[0], yuv3[1], yuv3[2], params->dovi->nonlinear, *params->ycc_offset);
    ycc2rgb(c4, yuv4[0], yuv4[1], yuv4[2], params->dovi->nonlinear, *params->ycc_offset);

    lms2rgb(c1, c1[0], c1[1], c1[2], params->dovi->linear, *params->lms2rgb_matrix);
    lms2rgb(c2, c2[0], c2[1], c2[2], params->dovi->linear, *params->lms2rgb_matrix);
    lms2rgb(c3, c3[0], c3[1], c3[2], params->dovi->linear, *params->lms2rgb_matrix);
    lms2rgb(c4, c4[0], c4[1], c4[2], params->dovi->linear, *params->lms2rgb_matrix);

    // DoVi always uses full range
    r[0] = av_clip_uintp2((int)(c1[0] * JPEG_SCALE), 15);
    r[1] = av_clip_uintp2((int)(c2[0] * JPEG_SCALE), 15);
    r[2] = av_clip_uintp2((int)(c3[0] * JPEG_SCALE), 15);
    r[3] = av_clip_uintp2((int)(c4[0] * JPEG_SCALE), 15);

    g[0] = av_clip_uintp2((int)(c1[1] * JPEG_SCALE), 15);
    g[1] = av_clip_uintp2((int)(c2[1] * JPEG_SCALE), 15);
    g[2] = av_clip_uintp2((int)(c3[1] * JPEG_SCALE), 15);
    g[3] = av_clip_uintp2((int)(c4[1] * JPEG_SCALE), 15);

    b[0] = av_clip_uintp2((int)(c1[2] * JPEG_SCALE), 15);
    b[1] = av_clip_uintp2((int)(c2[2] * JPEG_SCALE), 15);
    b[2] = av_clip_uintp2((int)(c3[2] * JPEG_SCALE), 15);
    b[3] = av_clip_uintp2((int)(c4[2] * JPEG_SCALE), 15);
}

inline static void tonemap_int16(int16_t r_in, int16_t g_in, int16_t b_in,
                                 int16_t *r_out, int16_t *g_out, int16_t *b_out,
                                 float *lin_lut, float *tonemap_lut, uint16_t *delin_lut,
                                 const AVLumaCoefficients *coeffs,
                                 const AVLumaCoefficients *ocoeffs, double desat,
                                 double (*rgb2rgb)[3][3],
                                 int rgb2rgb_passthrough)
{
    int16_t sig;
    float mapval, r_lin, g_lin, b_lin;

    r_in = av_clip_uintp2(r_in, 15);
    g_in = av_clip_uintp2(g_in, 15);
    b_in = av_clip_uintp2(b_in, 15);

    /* load values */
    *r_out = r_in;
    *g_out = g_in;
    *b_out = b_in;

    /* pick the brightest component, reducing the value range as necessary
     * to keep the entire signal in range and preventing discoloration due to
     * out-of-bounds clipping */
    sig = FFMAX3(r_in, g_in, b_in);

    mapval = tonemap_lut[sig];

    r_lin = lin_lut[r_in];
    g_lin = lin_lut[g_in];
    b_lin = lin_lut[b_in];

    if (!rgb2rgb_passthrough) {
        r_lin = (*rgb2rgb)[0][0] * r_lin + (*rgb2rgb)[0][1] * g_lin + (*rgb2rgb)[0][2] * b_lin;
        g_lin = (*rgb2rgb)[1][0] * r_lin + (*rgb2rgb)[1][1] * g_lin + (*rgb2rgb)[1][2] * b_lin;
        b_lin = (*rgb2rgb)[2][0] * r_lin + (*rgb2rgb)[2][1] * g_lin + (*rgb2rgb)[2][2] * b_lin;
    }

    /* desaturate to prevent unnatural colors */
    if (desat > 0) {
        float luma = av_q2d(coeffs->cr) * r_lin + av_q2d(coeffs->cg) * g_lin + av_q2d(coeffs->cb) * b_lin;
        float overbright = FFMAX(luma - desat, FLOAT_EPS) / FFMAX(luma, FLOAT_EPS);
        r_lin = MIX(r_lin, luma, overbright);
        g_lin = MIX(g_lin, luma, overbright);
        b_lin = MIX(b_lin, luma, overbright);
    }

    r_lin *= mapval;
    g_lin *= mapval;
    b_lin *= mapval;

    *r_out = delin_lut[av_clip_uintp2(r_lin * 32767 + 0.5, 15)];
    *g_out = delin_lut[av_clip_uintp2(g_lin * 32767 + 0.5, 15)];
    *b_out = delin_lut[av_clip_uintp2(b_lin * 32767 + 0.5, 15)];
}

// See also libavfilter/colorspacedsp_template.c
void tonemap_frame_p010_2_nv12(uint8_t *dsty, uint8_t *dstuv,
                               const uint16_t *srcy, const uint16_t *srcuv,
                               const int *dstlinesize, const int *srclinesize,
                               int dstdepth, int srcdepth,
                               int width, int height,
                               const struct TonemapIntParams *params)
{
    const int in_depth = srcdepth;
    const int in_uv_offset = 128 << (in_depth - 8);
    const int in_sh = in_depth - 1;
    const int in_rnd = 1 << (in_sh - 1);
    const int in_sh2 = 16 - in_depth;

    const int out_depth = dstdepth;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);

    int cy  = (*params->yuv2rgb_coeffs)[0][0][0];
    int crv = (*params->yuv2rgb_coeffs)[0][2][0];
    int cgu = (*params->yuv2rgb_coeffs)[1][1][0];
    int cgv = (*params->yuv2rgb_coeffs)[1][2][0];
    int cbu = (*params->yuv2rgb_coeffs)[2][1][0];

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    int r00, g00, b00;
    int r01, g01, b01;
    int r10, g10, b10;
    int r11, g11, b11;

    int16_t r[4], g[4], b[4];
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0] * 2, dstuv += dstlinesize[1],
                       srcy += srclinesize[0], srcuv += srclinesize[1] / 2) {
        for (int x = 0; x < width; x += 2) {
            int y00 = (srcy[x]                          >> in_sh2) - params->in_yuv_off;
            int y01 = (srcy[x + 1]                      >> in_sh2) - params->in_yuv_off;
            int y10 = (srcy[srclinesize[0] / 2 + x]     >> in_sh2) - params->in_yuv_off;
            int y11 = (srcy[srclinesize[0] / 2 + x + 1] >> in_sh2) - params->in_yuv_off;
            int u = (srcuv[x]     >> in_sh2) - in_uv_offset;
            int v = (srcuv[x + 1] >> in_sh2) - in_uv_offset;

            r[0] = av_clip_int16((y00 * cy + crv * v + in_rnd) >> in_sh);
            r[1] = av_clip_int16((y01 * cy + crv * v + in_rnd) >> in_sh);
            r[2] = av_clip_int16((y10 * cy + crv * v + in_rnd) >> in_sh);
            r[3] = av_clip_int16((y11 * cy + crv * v + in_rnd) >> in_sh);

            g[0] = av_clip_int16((y00 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[1] = av_clip_int16((y01 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[2] = av_clip_int16((y10 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[3] = av_clip_int16((y11 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);

            b[0] = av_clip_int16((y00 * cy + cbu * u + in_rnd) >> in_sh);
            b[1] = av_clip_int16((y01 * cy + cbu * u + in_rnd) >> in_sh);
            b[2] = av_clip_int16((y10 * cy + cbu * u + in_rnd) >> in_sh);
            b[3] = av_clip_int16((y11 * cy + cbu * u + in_rnd) >> in_sh);

            tonemap_int16(r[0], g[0], b[0], &r[0], &g[0], &b[0],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[1], g[1], b[1], &r[1], &g[1], &b[1],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[2], g[2], b[2], &r[2], &g[2], &b[2],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[3], g[3], b[3], &r[3], &g[3], &b[3],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);

            r00 = r[0], g00 = g[0], b00 = b[0];
            r01 = r[1], g01 = g[1], b01 = b[1];
            r10 = r[2], g10 = g[2], b10 = b[2];
            r11 = r[3], g11 = g[3], b11 = b[3];

            dsty[x]                      = av_clip_uint8(params->out_yuv_off + ((r00 * cry + g00 * cgy + b00 * cby + out_rnd) >> out_sh));
            dsty[x + 1]                  = av_clip_uint8(params->out_yuv_off + ((r01 * cry + g01 * cgy + b01 * cby + out_rnd) >> out_sh));
            dsty[dstlinesize[0] + x]     = av_clip_uint8(params->out_yuv_off + ((r10 * cry + g10 * cgy + b10 * cby + out_rnd) >> out_sh));
            dsty[dstlinesize[0] + x + 1] = av_clip_uint8(params->out_yuv_off + ((r11 * cry + g11 * cgy + b11 * cby + out_rnd) >> out_sh));

#define AVG(a,b,c,d) (((a) + (b) + (c) + (d) + 2) >> 2)
            dstuv[x]     = av_clip_uint8(out_uv_offset + ((AVG(r00, r01, r10, r11) * cru + AVG(g00, g01, g10, g11) * ocgu + AVG(b00, b01, b10, b11) * cburv + out_rnd) >> out_sh));
            dstuv[x + 1] = av_clip_uint8(out_uv_offset + ((AVG(r00, r01, r10, r11) * cburv + AVG(g00, g01, g10, g11) * ocgv + AVG(b00, b01, b10, b11) * cbv + out_rnd) >> out_sh));
#undef AVG
        }
    }
}

void tonemap_frame_p010_2_p010(uint16_t *dsty, uint16_t *dstuv,
                               const uint16_t *srcy, const uint16_t *srcuv,
                               const int *dstlinesize, const int *srclinesize,
                               int dstdepth, int srcdepth,
                               int width, int height,
                               const struct TonemapIntParams *params)
{
    const int in_depth = srcdepth;
    const int in_uv_offset = 128 << (in_depth - 8);
    const int in_sh = in_depth - 1;
    const int in_rnd = 1 << (in_sh - 1);
    const int in_sh2 = 16 - in_depth;

    const int out_depth = dstdepth;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);
    const int out_sh2 = 16 - out_depth;

    int cy  = (*params->yuv2rgb_coeffs)[0][0][0];
    int crv = (*params->yuv2rgb_coeffs)[0][2][0];
    int cgu = (*params->yuv2rgb_coeffs)[1][1][0];
    int cgv = (*params->yuv2rgb_coeffs)[1][2][0];
    int cbu = (*params->yuv2rgb_coeffs)[2][1][0];

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    int r00, g00, b00;
    int r01, g01, b01;
    int r10, g10, b10;
    int r11, g11, b11;

    int16_t r[4], g[4], b[4];
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstuv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcuv += srclinesize[1] / 2) {
        for (int x = 0; x < width; x += 2) {
            int y00 = (srcy[x]                          >> in_sh2) - params->in_yuv_off;
            int y01 = (srcy[x + 1]                      >> in_sh2) - params->in_yuv_off;
            int y10 = (srcy[srclinesize[0] / 2 + x]     >> in_sh2) - params->in_yuv_off;
            int y11 = (srcy[srclinesize[0] / 2 + x + 1] >> in_sh2) - params->in_yuv_off;
            int u = (srcuv[x]     >> in_sh2) - in_uv_offset;
            int v = (srcuv[x + 1] >> in_sh2) - in_uv_offset;

            r[0] = av_clip_int16((y00 * cy + crv * v + in_rnd) >> in_sh);
            r[1] = av_clip_int16((y01 * cy + crv * v + in_rnd) >> in_sh);
            r[2] = av_clip_int16((y10 * cy + crv * v + in_rnd) >> in_sh);
            r[3] = av_clip_int16((y11 * cy + crv * v + in_rnd) >> in_sh);

            g[0] = av_clip_int16((y00 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[1] = av_clip_int16((y01 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[2] = av_clip_int16((y10 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[3] = av_clip_int16((y11 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);

            b[0] = av_clip_int16((y00 * cy + cbu * u + in_rnd) >> in_sh);
            b[1] = av_clip_int16((y01 * cy + cbu * u + in_rnd) >> in_sh);
            b[2] = av_clip_int16((y10 * cy + cbu * u + in_rnd) >> in_sh);
            b[3] = av_clip_int16((y11 * cy + cbu * u + in_rnd) >> in_sh);

            tonemap_int16(r[0], g[0], b[0], &r[0], &g[0], &b[0],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[1], g[1], b[1], &r[1], &g[1], &b[1],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[2], g[2], b[2], &r[2], &g[2], &b[2],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[3], g[3], b[3], &r[3], &g[3], &b[3],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);

            r00 = r[0], g00 = g[0], b00 = b[0];
            r01 = r[1], g01 = g[1], b01 = b[1];
            r10 = r[2], g10 = g[2], b10 = b[2];
            r11 = r[3], g11 = g[3], b11 = b[3];

            dsty[x]                          = av_clip_uintp2((params->out_yuv_off + ((r00 * cry + g00 * cgy + b00 * cby + out_rnd) >> out_sh)) << out_sh2, 16);
            dsty[x + 1]                      = av_clip_uintp2((params->out_yuv_off + ((r01 * cry + g01 * cgy + b01 * cby + out_rnd) >> out_sh)) << out_sh2, 16);
            dsty[dstlinesize[0] / 2 + x]     = av_clip_uintp2((params->out_yuv_off + ((r10 * cry + g10 * cgy + b10 * cby + out_rnd) >> out_sh)) << out_sh2, 16);
            dsty[dstlinesize[0] / 2 + x + 1] = av_clip_uintp2((params->out_yuv_off + ((r11 * cry + g11 * cgy + b11 * cby + out_rnd) >> out_sh)) << out_sh2, 16);

#define AVG(a,b,c,d) (((a) + (b) + (c) + (d) + 2) >> 2)
            dstuv[x]     = av_clip_uintp2((out_uv_offset + ((AVG(r00, r01, r10, r11) * cru + AVG(g00, g01, g10, g11) * ocgu + AVG(b00, b01, b10, b11) * cburv + out_rnd) >> out_sh)) << out_sh2, 16);
            dstuv[x + 1] = av_clip_uintp2((out_uv_offset + ((AVG(r00, r01, r10, r11) * cburv + AVG(g00, g01, g10, g11) * ocgv + AVG(b00, b01, b10, b11) * cbv + out_rnd) >> out_sh)) << out_sh2, 16);
#undef AVG
        }
    }
}

void tonemap_frame_dovi_2_420p(uint8_t *dsty, uint8_t *dstu, uint8_t *dstv,
                               const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                               const int *dstlinesize, const int *srclinesize,
                               int dstdepth, int srcdepth,
                               int width, int height,
                               const struct TonemapIntParams *params)
{
    const int in_depth = srcdepth;
    const int out_depth = dstdepth;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    int r00, g00, b00;
    int r01, g01, b01;
    int r10, g10, b10;
    int r11, g11, b11;

    const float in_rng = (float)((1 << in_depth) - 1);

    int16_t r[4], g[4], b[4];
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0] * 2, dstu += dstlinesize[1], dstv += dstlinesize[2],
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[2] / 2) {
        for (int x = 0; x < width; x += 2) {
            int y00 = (srcy[x]                         );
            int y01 = (srcy[x + 1]                     );
            int y10 = (srcy[srclinesize[0] / 2 + x]    );
            int y11 = (srcy[srclinesize[0] / 2 + x + 1]);
            int u = (srcu[x >> 1]);
            int v = (srcv[x >> 1]);

            dovi2rgb(y00, y01, y10, y11, u, v, params, in_rng, r, g, b);

            tonemap_int16(r[0], g[0], b[0], &r[0], &g[0], &b[0],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[1], g[1], b[1], &r[1], &g[1], &b[1],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[2], g[2], b[2], &r[2], &g[2], &b[2],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[3], g[3], b[3], &r[3], &g[3], &b[3],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);

            r00 = r[0], g00 = g[0], b00 = b[0];
            r01 = r[1], g01 = g[1], b01 = b[1];
            r10 = r[2], g10 = g[2], b10 = b[2];
            r11 = r[3], g11 = g[3], b11 = b[3];

            dsty[x]                      = av_clip_uint8(params->out_yuv_off + ((r00 * cry + g00 * cgy + b00 * cby + out_rnd) >> out_sh));
            dsty[x + 1]                  = av_clip_uint8(params->out_yuv_off + ((r01 * cry + g01 * cgy + b01 * cby + out_rnd) >> out_sh));
            dsty[dstlinesize[0] + x]     = av_clip_uint8(params->out_yuv_off + ((r10 * cry + g10 * cgy + b10 * cby + out_rnd) >> out_sh));
            dsty[dstlinesize[0] + x + 1] = av_clip_uint8(params->out_yuv_off + ((r11 * cry + g11 * cgy + b11 * cby + out_rnd) >> out_sh));

#define AVG(a,b,c,d) (((a) + (b) + (c) + (d) + 2) >> 2)
            dstu[x >> 1] = av_clip_uint8(out_uv_offset + ((AVG(r00, r01, r10, r11) * cru + AVG(g00, g01, g10, g11) * ocgu + AVG(b00, b01, b10, b11) * cburv + out_rnd) >> out_sh));
            dstv[x >> 1] = av_clip_uint8(out_uv_offset + ((AVG(r00, r01, r10, r11) * cburv + AVG(g00, g01, g10, g11) * ocgv + AVG(b00, b01, b10, b11) * cbv + out_rnd) >> out_sh));
#undef AVG
        }
    }
}

void tonemap_frame_dovi_2_420p10(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                 const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                 const int *dstlinesize, const int *srclinesize,
                                 int dstdepth, int srcdepth,
                                 int width, int height,
                                 const struct TonemapIntParams *params)
{
    const int in_depth = srcdepth;
    const int out_depth = dstdepth;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    int r00, g00, b00;
    int r01, g01, b01;
    int r10, g10, b10;
    int r11, g11, b11;

    const float in_rng = (float)((1 << in_depth) - 1);

    int16_t r[4], g[4], b[4];
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstu += dstlinesize[1] / 2, dstv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[1] / 2) {
        for (int x = 0; x < width; x += 2) {
            int y00 = (srcy[x]                         );
            int y01 = (srcy[x + 1]                     );
            int y10 = (srcy[srclinesize[0] / 2 + x]    );
            int y11 = (srcy[srclinesize[0] / 2 + x + 1]);
            int u = (srcu[x >> 1]);
            int v = (srcv[x >> 1]);

            dovi2rgb(y00, y01, y10, y11, u, v, params, in_rng, r, g, b);

            tonemap_int16(r[0], g[0], b[0], &r[0], &g[0], &b[0],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[1], g[1], b[1], &r[1], &g[1], &b[1],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[2], g[2], b[2], &r[2], &g[2], &b[2],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[3], g[3], b[3], &r[3], &g[3], &b[3],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);

            r00 = r[0], g00 = g[0], b00 = b[0];
            r01 = r[1], g01 = g[1], b01 = b[1];
            r10 = r[2], g10 = g[2], b10 = b[2];
            r11 = r[3], g11 = g[3], b11 = b[3];

            dsty[x]                          = av_clip_uintp2((params->out_yuv_off + ((r00 * cry + g00 * cgy + b00 * cby + out_rnd) >> out_sh)), 10);
            dsty[x + 1]                      = av_clip_uintp2((params->out_yuv_off + ((r01 * cry + g01 * cgy + b01 * cby + out_rnd) >> out_sh)), 10);
            dsty[dstlinesize[0] / 2 + x]     = av_clip_uintp2((params->out_yuv_off + ((r10 * cry + g10 * cgy + b10 * cby + out_rnd) >> out_sh)), 10);
            dsty[dstlinesize[0] / 2 + x + 1] = av_clip_uintp2((params->out_yuv_off + ((r11 * cry + g11 * cgy + b11 * cby + out_rnd) >> out_sh)), 10);

#define AVG(a,b,c,d) (((a) + (b) + (c) + (d) + 2) >> 2)
            dstu[x >> 1] = av_clip_uintp2((out_uv_offset + ((AVG(r00, r01, r10, r11) * cru + AVG(g00, g01, g10, g11) * ocgu + AVG(b00, b01, b10, b11) * cburv + out_rnd) >> out_sh)), 10);
            dstv[x >> 1] = av_clip_uintp2((out_uv_offset + ((AVG(r00, r01, r10, r11) * cburv + AVG(g00, g01, g10, g11) * ocgv + AVG(b00, b01, b10, b11) * cbv + out_rnd) >> out_sh)), 10);
#undef AVG
        }
    }
}

void tonemap_frame_dovi_2_420hdr(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                 const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                 const int *dstlinesize, const int *srclinesize,
                                 int dstdepth, int srcdepth,
                                 int width, int height,
                                 const struct TonemapIntParams *params)
{
    const int in_depth = srcdepth;
    const int out_depth = dstdepth;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    int r00, g00, b00;
    int r01, g01, b01;
    int r10, g10, b10;
    int r11, g11, b11;

    const float in_rng = (float)((1 << in_depth) - 1);

    int16_t r[4], g[4], b[4];
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstu += dstlinesize[1] / 2, dstv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[1] / 2) {
        for (int x = 0; x < width; x += 2) {
            int y00 = (srcy[x]                         );
            int y01 = (srcy[x + 1]                     );
            int y10 = (srcy[srclinesize[0] / 2 + x]    );
            int y11 = (srcy[srclinesize[0] / 2 + x + 1]);
            int u = (srcu[x >> 1]);
            int v = (srcv[x >> 1]);

            dovi2rgb(y00, y01, y10, y11, u, v, params, in_rng, r, g, b);

            r00 = r[0], g00 = g[0], b00 = b[0];
            r01 = r[1], g01 = g[1], b01 = b[1];
            r10 = r[2], g10 = g[2], b10 = b[2];
            r11 = r[3], g11 = g[3], b11 = b[3];

            dsty[x]                          = av_clip_uintp2((params->out_yuv_off + ((r00 * cry + g00 * cgy + b00 * cby + out_rnd) >> out_sh)), 10);
            dsty[x + 1]                      = av_clip_uintp2((params->out_yuv_off + ((r01 * cry + g01 * cgy + b01 * cby + out_rnd) >> out_sh)), 10);
            dsty[dstlinesize[0] / 2 + x]     = av_clip_uintp2((params->out_yuv_off + ((r10 * cry + g10 * cgy + b10 * cby + out_rnd) >> out_sh)), 10);
            dsty[dstlinesize[0] / 2 + x + 1] = av_clip_uintp2((params->out_yuv_off + ((r11 * cry + g11 * cgy + b11 * cby + out_rnd) >> out_sh)), 10);

#define AVG(a,b,c,d) (((a) + (b) + (c) + (d) + 2) >> 2)
            dstu[x >> 1] = av_clip_uintp2((out_uv_offset + ((AVG(r00, r01, r10, r11) * cru + AVG(g00, g01, g10, g11) * ocgu + AVG(b00, b01, b10, b11) * cburv + out_rnd) >> out_sh)), 10);
            dstv[x >> 1] = av_clip_uintp2((out_uv_offset + ((AVG(r00, r01, r10, r11) * cburv + AVG(g00, g01, g10, g11) * ocgv + AVG(b00, b01, b10, b11) * cbv + out_rnd) >> out_sh)), 10);
#undef AVG
        }
    }
}

void tonemap_frame_420p10_2_420p(uint8_t *dsty, uint8_t *dstu, uint8_t *dstv,
                                 const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                 const int *dstlinesize, const int *srclinesize,
                                 int dstdepth, int srcdepth,
                                 int width, int height,
                                 const struct TonemapIntParams *params)
{
    const int in_depth = srcdepth;
    const int in_uv_offset = 128 << (in_depth - 8);
    const int in_sh = in_depth - 1;
    const int in_rnd = 1 << (in_sh - 1);

    const int out_depth = dstdepth;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);

    int cy  = (*params->yuv2rgb_coeffs)[0][0][0];
    int crv = (*params->yuv2rgb_coeffs)[0][2][0];
    int cgu = (*params->yuv2rgb_coeffs)[1][1][0];
    int cgv = (*params->yuv2rgb_coeffs)[1][2][0];
    int cbu = (*params->yuv2rgb_coeffs)[2][1][0];

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    int r00, g00, b00;
    int r01, g01, b01;
    int r10, g10, b10;
    int r11, g11, b11;

    int16_t r[4], g[4], b[4];
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0] * 2, dstu += dstlinesize[1], dstv += dstlinesize[2],
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[2] / 2) {
        for (int x = 0; x < width; x += 2) {
            int y00 = (srcy[x]                         ) - params->in_yuv_off;
            int y01 = (srcy[x + 1]                     ) - params->in_yuv_off;
            int y10 = (srcy[srclinesize[0] / 2 + x]    ) - params->in_yuv_off;
            int y11 = (srcy[srclinesize[0] / 2 + x + 1]) - params->in_yuv_off;
            int u = (srcu[x >> 1]) - in_uv_offset;
            int v = (srcv[x >> 1]) - in_uv_offset;

            r[0] = av_clip_int16((y00 * cy + crv * v + in_rnd) >> in_sh);
            r[1] = av_clip_int16((y01 * cy + crv * v + in_rnd) >> in_sh);
            r[2] = av_clip_int16((y10 * cy + crv * v + in_rnd) >> in_sh);
            r[3] = av_clip_int16((y11 * cy + crv * v + in_rnd) >> in_sh);

            g[0] = av_clip_int16((y00 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[1] = av_clip_int16((y01 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[2] = av_clip_int16((y10 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[3] = av_clip_int16((y11 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);

            b[0] = av_clip_int16((y00 * cy + cbu * u + in_rnd) >> in_sh);
            b[1] = av_clip_int16((y01 * cy + cbu * u + in_rnd) >> in_sh);
            b[2] = av_clip_int16((y10 * cy + cbu * u + in_rnd) >> in_sh);
            b[3] = av_clip_int16((y11 * cy + cbu * u + in_rnd) >> in_sh);

            tonemap_int16(r[0], g[0], b[0], &r[0], &g[0], &b[0],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[1], g[1], b[1], &r[1], &g[1], &b[1],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[2], g[2], b[2], &r[2], &g[2], &b[2],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[3], g[3], b[3], &r[3], &g[3], &b[3],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);

            r00 = r[0], g00 = g[0], b00 = b[0];
            r01 = r[1], g01 = g[1], b01 = b[1];
            r10 = r[2], g10 = g[2], b10 = b[2];
            r11 = r[3], g11 = g[3], b11 = b[3];

            dsty[x]                      = av_clip_uint8(params->out_yuv_off + ((r00 * cry + g00 * cgy + b00 * cby + out_rnd) >> out_sh));
            dsty[x + 1]                  = av_clip_uint8(params->out_yuv_off + ((r01 * cry + g01 * cgy + b01 * cby + out_rnd) >> out_sh));
            dsty[dstlinesize[0] + x]     = av_clip_uint8(params->out_yuv_off + ((r10 * cry + g10 * cgy + b10 * cby + out_rnd) >> out_sh));
            dsty[dstlinesize[0] + x + 1] = av_clip_uint8(params->out_yuv_off + ((r11 * cry + g11 * cgy + b11 * cby + out_rnd) >> out_sh));

#define AVG(a,b,c,d) (((a) + (b) + (c) + (d) + 2) >> 2)
            dstu[x >> 1] = av_clip_uint8(out_uv_offset + ((AVG(r00, r01, r10, r11) * cru + AVG(g00, g01, g10, g11) * ocgu + AVG(b00, b01, b10, b11) * cburv + out_rnd) >> out_sh));
            dstv[x >> 1] = av_clip_uint8(out_uv_offset + ((AVG(r00, r01, r10, r11) * cburv + AVG(g00, g01, g10, g11) * ocgv + AVG(b00, b01, b10, b11) * cbv + out_rnd) >> out_sh));
#undef AVG
        }
    }
}

void tonemap_frame_420p10_2_420p10(uint16_t *dsty, uint16_t *dstu, uint16_t *dstv,
                                   const uint16_t *srcy, const uint16_t *srcu, const uint16_t *srcv,
                                   const int *dstlinesize, const int *srclinesize,
                                   int dstdepth, int srcdepth,
                                   int width, int height,
                                   const struct TonemapIntParams *params)
{
    const int in_depth = srcdepth;
    const int in_uv_offset = 128 << (in_depth - 8);
    const int in_sh = in_depth - 1;
    const int in_rnd = 1 << (in_sh - 1);

    const int out_depth = dstdepth;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);

    int cy  = (*params->yuv2rgb_coeffs)[0][0][0];
    int crv = (*params->yuv2rgb_coeffs)[0][2][0];
    int cgu = (*params->yuv2rgb_coeffs)[1][1][0];
    int cgv = (*params->yuv2rgb_coeffs)[1][2][0];
    int cbu = (*params->yuv2rgb_coeffs)[2][1][0];

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    int r00, g00, b00;
    int r01, g01, b01;
    int r10, g10, b10;
    int r11, g11, b11;

    int16_t r[4], g[4], b[4];
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0], dstu += dstlinesize[1] / 2, dstv += dstlinesize[1] / 2,
                       srcy += srclinesize[0], srcu += srclinesize[1] / 2, srcv += srclinesize[1] / 2) {
        for (int x = 0; x < width; x += 2) {
            int y00 = (srcy[x]                         ) - params->in_yuv_off;
            int y01 = (srcy[x + 1]                     ) - params->in_yuv_off;
            int y10 = (srcy[srclinesize[0] / 2 + x]    ) - params->in_yuv_off;
            int y11 = (srcy[srclinesize[0] / 2 + x + 1]) - params->in_yuv_off;
            int u = (srcu[x >> 1]) - in_uv_offset;
            int v = (srcv[x >> 1]) - in_uv_offset;

            r[0] = av_clip_int16((y00 * cy + crv * v + in_rnd) >> in_sh);
            r[1] = av_clip_int16((y01 * cy + crv * v + in_rnd) >> in_sh);
            r[2] = av_clip_int16((y10 * cy + crv * v + in_rnd) >> in_sh);
            r[3] = av_clip_int16((y11 * cy + crv * v + in_rnd) >> in_sh);

            g[0] = av_clip_int16((y00 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[1] = av_clip_int16((y01 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[2] = av_clip_int16((y10 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[3] = av_clip_int16((y11 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);

            b[0] = av_clip_int16((y00 * cy + cbu * u + in_rnd) >> in_sh);
            b[1] = av_clip_int16((y01 * cy + cbu * u + in_rnd) >> in_sh);
            b[2] = av_clip_int16((y10 * cy + cbu * u + in_rnd) >> in_sh);
            b[3] = av_clip_int16((y11 * cy + cbu * u + in_rnd) >> in_sh);

            tonemap_int16(r[0], g[0], b[0], &r[0], &g[0], &b[0],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[1], g[1], b[1], &r[1], &g[1], &b[1],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[2], g[2], b[2], &r[2], &g[2], &b[2],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);
            tonemap_int16(r[3], g[3], b[3], &r[3], &g[3], &b[3],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs, params->rgb2rgb_passthrough);

            r00 = r[0], g00 = g[0], b00 = b[0];
            r01 = r[1], g01 = g[1], b01 = b[1];
            r10 = r[2], g10 = g[2], b10 = b[2];
            r11 = r[3], g11 = g[3], b11 = b[3];

            dsty[x]                          = av_clip_uintp2((params->out_yuv_off + ((r00 * cry + g00 * cgy + b00 * cby + out_rnd) >> out_sh)), 10);
            dsty[x + 1]                      = av_clip_uintp2((params->out_yuv_off + ((r01 * cry + g01 * cgy + b01 * cby + out_rnd) >> out_sh)), 10);
            dsty[dstlinesize[0] / 2 + x]     = av_clip_uintp2((params->out_yuv_off + ((r10 * cry + g10 * cgy + b10 * cby + out_rnd) >> out_sh)), 10);
            dsty[dstlinesize[0] / 2 + x + 1] = av_clip_uintp2((params->out_yuv_off + ((r11 * cry + g11 * cgy + b11 * cby + out_rnd) >> out_sh)), 10);

#define AVG(a,b,c,d) (((a) + (b) + (c) + (d) + 2) >> 2)
            dstu[x >> 1] = av_clip_uintp2((out_uv_offset + ((AVG(r00, r01, r10, r11) * cru + AVG(g00, g01, g10, g11) * ocgu + AVG(b00, b01, b10, b11) * cburv + out_rnd) >> out_sh)), 10);
            dstv[x >> 1] = av_clip_uintp2((out_uv_offset + ((AVG(r00, r01, r10, r11) * cburv + AVG(g00, g01, g10, g11) * ocgv + AVG(b00, b01, b10, b11) * cbv + out_rnd) >> out_sh)), 10);
#undef AVG
        }
    }
}

#define LOAD_TONEMAP_PARAMS     TonemapxContext *s = ctx->priv; \
ThreadData *td = arg;                                           \
AVFrame *in = td->in;                                           \
AVFrame *out = td->out;                                         \
const AVPixFmtDescriptor *desc  = td->desc;                     \
const AVPixFmtDescriptor *odesc = td->odesc;                    \
const int ss = 1 << FFMAX(desc->log2_chroma_h, odesc->log2_chroma_h); \
const int slice_start = (in->height / ss *  jobnr     ) / nb_jobs * ss; \
const int slice_end   = (in->height / ss * (jobnr + 1)) / nb_jobs * ss; \
TonemapIntParams params = {                                     \
.lut_peak            = s->lut_peak,                             \
.lin_lut             = s->lin_lut,                              \
.tonemap_lut         = s->tonemap_lut,                          \
.delin_lut           = s->delin_lut,                            \
.in_yuv_off          = s->in_yuv_off,                           \
.out_yuv_off         = s->out_yuv_off,                          \
.yuv2rgb_coeffs      = &s->yuv2rgb_coeffs,                      \
.rgb2yuv_coeffs      = &s->rgb2yuv_coeffs,                      \
.rgb2rgb_coeffs      = &s->rgb2rgb_coeffs,                      \
.rgb2rgb_passthrough = in->color_primaries == out->color_primaries,   \
.coeffs              = s->coeffs,                               \
.ocoeffs             = s->ocoeffs,                              \
.desat               = s->desat,                                \
.dovi = s->dovi,                                                \
.dovi_pbuf = s->dovi_pbuf,                                      \
.lms2rgb_matrix = &s->lms2rgb_matrix,                           \
.ycc_offset = &s->ycc_offset                                    \
};

static int filter_slice_planar8(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    LOAD_TONEMAP_PARAMS
    av_log(s, AV_LOG_DEBUG, "planar dst depth: %d, src depth: %d\n", odesc->comp[0].depth, desc->comp[0].depth);

    s->tonemap_func_planar8(out->data[0] + out->linesize[0] * slice_start,
                            out->data[1] + out->linesize[1] * AV_CEIL_RSHIFT(slice_start, desc->log2_chroma_h),
                            out->data[2] + out->linesize[2] * AV_CEIL_RSHIFT(slice_start, desc->log2_chroma_h),
                            (void*)(in->data[0] + in->linesize[0] * slice_start),
                            (void*)(in->data[1] + in->linesize[1] * AV_CEIL_RSHIFT(slice_start, odesc->log2_chroma_h)),
                            (void*)(in->data[2] + in->linesize[2] * AV_CEIL_RSHIFT(slice_start, odesc->log2_chroma_h)),
                            out->linesize, in->linesize,
                            odesc->comp[0].depth, desc->comp[0].depth,
                            out->width, slice_end - slice_start,
                            &params);

    return 0;
}

static int filter_slice_biplanar8(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    LOAD_TONEMAP_PARAMS
    av_log(s, AV_LOG_DEBUG, "biplanar dst depth: %d, src depth: %d\n", odesc->comp[0].depth, desc->comp[0].depth);

    s->tonemap_func_biplanar8(out->data[0] + out->linesize[0] * slice_start,
                              out->data[1] + out->linesize[1] * AV_CEIL_RSHIFT(slice_start, desc->log2_chroma_h),
                              (void*)(in->data[0] + in->linesize[0] * slice_start),
                              (void*)(in->data[1] + in->linesize[1] * AV_CEIL_RSHIFT(slice_start, odesc->log2_chroma_h)),
                              out->linesize, in->linesize,
                              odesc->comp[0].depth, desc->comp[0].depth,
                              out->width, slice_end - slice_start,
                              &params);

    return 0;
}

static int filter_slice_planar10(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    LOAD_TONEMAP_PARAMS
    av_log(s, AV_LOG_DEBUG, "planar dst depth: %d, src depth: %d\n", odesc->comp[0].depth, desc->comp[0].depth);

    s->tonemap_func_planar10((uint16_t *) (out->data[0] + out->linesize[0] * slice_start),
                             (uint16_t *) (out->data[1] +
                                           out->linesize[1] * AV_CEIL_RSHIFT(slice_start, desc->log2_chroma_h)),
                             (uint16_t *) (out->data[2] +
                                           out->linesize[2] * AV_CEIL_RSHIFT(slice_start, desc->log2_chroma_h)),
                             (void*)(in->data[0] + in->linesize[0] * slice_start),
                             (void*)(in->data[1] + in->linesize[1] * AV_CEIL_RSHIFT(slice_start, odesc->log2_chroma_h)),
                             (void*)(in->data[2] + in->linesize[2] * AV_CEIL_RSHIFT(slice_start, odesc->log2_chroma_h)),
                             out->linesize, in->linesize,
                             odesc->comp[0].depth, desc->comp[0].depth,
                             out->width, slice_end - slice_start,
                             &params);

    return 0;
}

static int filter_slice_biplanar10(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    LOAD_TONEMAP_PARAMS
    av_log(s, AV_LOG_DEBUG, "biplanar dst depth: %d, src depth: %d\n", odesc->comp[0].depth, desc->comp[0].depth);

    s->tonemap_func_biplanar10((uint16_t *) (out->data[0] + out->linesize[0] * slice_start),
                               (uint16_t *) (out->data[1] +
                                             out->linesize[1] * AV_CEIL_RSHIFT(slice_start, desc->log2_chroma_h)),
                               (void*)(in->data[0] + in->linesize[0] * slice_start),
                               (void*)(in->data[1] + in->linesize[1] * AV_CEIL_RSHIFT(slice_start, odesc->log2_chroma_h)),
                               out->linesize, in->linesize,
                               odesc->comp[0].depth, desc->comp[0].depth,
                               out->width, slice_end - slice_start,
                               &params);

    return 0;
}

static int filter_frame(AVFilterLink *link, AVFrame *in)
{
    AVFilterContext *ctx = link->dst;
    TonemapxContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *out;
    AVFrameSideData *dovi_sd = NULL;
    const AVPixFmtDescriptor *desc;
    const AVPixFmtDescriptor *odesc;
    int ret;
    double peak = s->peak;
    const AVLumaCoefficients *coeffs;
    ThreadData td;

    desc = av_pix_fmt_desc_get(link->format);
    odesc = av_pix_fmt_desc_get(outlink->format);
    if (!desc || !odesc) {
        av_frame_free(&in);
        return AVERROR_BUG;
    }

    if (s->trc == AVCOL_TRC_SMPTE2084 && odesc->comp[0].depth == 8) {
        av_log(s, AV_LOG_ERROR, "HDR passthrough requires 10 bit output format depth\n");
        av_assert0(0);
    }

    switch (odesc->comp[2].plane) {
        case 1: // biplanar
            if (odesc->comp[0].depth == 8) {
                s->filter_slice = filter_slice_biplanar8;
            } else {
                s->filter_slice = filter_slice_biplanar10;
            }
            break;
        default:
        case 2: // planar
            if (odesc->comp[0].depth == 8) {
                s->filter_slice = filter_slice_planar8;
            } else {
                s->filter_slice = filter_slice_planar10;
            }
            break;
    }

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }

    if ((ret = av_frame_copy_props(out, in)) < 0)
        goto fail;

    if (s->apply_dovi) {
        dovi_sd = av_frame_get_side_data(in, AV_FRAME_DATA_DOVI_METADATA);
    }

    /* read peak from side data if not passed in */
    if (!peak) {
        if (dovi_sd) {
            const AVDOVIMetadata *metadata = (AVDOVIMetadata *) dovi_sd->data;
            s->peak = ff_determine_dovi_signal_peak(metadata);
        } else {
            s->peak = ff_determine_signal_peak(in);
        }
    }

    if (dovi_sd) {
        const AVDOVIMetadata *metadata = (AVDOVIMetadata *) dovi_sd->data;
        const AVDOVIRpuDataHeader *rpu = av_dovi_get_header(metadata);
        // only map dovi rpus that don't require an EL and has rpu profile == 0
        // for performance reason we only want to do reshaping when absolutely needed
        // such videos usually have vdr_rpu_profile == 0, for example profile 5 videos
        // this could be wrong as there is no public documentation on this field
        if (rpu->disable_residual_flag && rpu->vdr_rpu_profile == 0) {
            struct DoviMetadata *dovi = av_malloc(sizeof(*dovi));
            s->dovi = dovi;
            if (!s->dovi)
                goto fail;

            ff_map_dovi_metadata(s->dovi, metadata);
            in->color_trc = AVCOL_TRC_SMPTE2084;
            in->colorspace = AVCOL_SPC_BT2020_NCL;
            in->color_primaries = AVCOL_PRI_BT2020;
            if (rpu->bl_video_full_range_flag)
                in->color_range = AVCOL_RANGE_JPEG;
        }
    }

    if (s->dovi) {
        if (desc->comp[2].plane == 1) {
            av_log(s, AV_LOG_ERROR, "Input pixel format has to be yuv420p10 for Dolby Vision reshaping\n");
            av_assert0(0);
        }
        update_dovi_buf(ctx);
        ff_matrix_mul_3x3(s->lms2rgb_matrix, dovi_lms2rgb_matrix, s->dovi->linear);
        s->ycc_offset[0] = s->dovi->nonlinear_offset[0] * (float)s->dovi->nonlinear[0][0] + s->dovi->nonlinear_offset[1] * (float)s->dovi->nonlinear[0][1] + s->dovi->nonlinear_offset[2] * (float)s->dovi->nonlinear[0][2];
        s->ycc_offset[1] = s->dovi->nonlinear_offset[0] * (float)s->dovi->nonlinear[1][0] + s->dovi->nonlinear_offset[1] * (float)s->dovi->nonlinear[1][1] + s->dovi->nonlinear_offset[2] * (float)s->dovi->nonlinear[1][2];
        s->ycc_offset[2] = s->dovi->nonlinear_offset[0] * (float)s->dovi->nonlinear[2][0] + s->dovi->nonlinear_offset[1] * (float)s->dovi->nonlinear[2][1] + s->dovi->nonlinear_offset[2] * (float)s->dovi->nonlinear[2][2];
        s->tonemap_func_planar8 = s->tonemap_func_dovi8;
        s->tonemap_func_planar10 = s->tonemap_func_dovi10;
    }

    out->color_trc = s->trc == -1 ? AVCOL_TRC_UNSPECIFIED : s->trc;
    out->colorspace = outlink->colorspace;
    out->color_primaries = s->pri == -1 ? AVCOL_PRI_UNSPECIFIED : s->pri;
    out->color_range = outlink->color_range;

    if (in->color_trc == AVCOL_TRC_UNSPECIFIED)
        in->color_trc = AVCOL_TRC_SMPTE2084;
    if (out->color_trc == AVCOL_TRC_UNSPECIFIED)
        out->color_trc = AVCOL_TRC_BT709;

    if (in->colorspace == AVCOL_SPC_UNSPECIFIED)
        in->colorspace = AVCOL_SPC_BT2020_NCL;
    if (out->colorspace == AVCOL_SPC_UNSPECIFIED)
        out->colorspace = AVCOL_SPC_BT709;

    if (in->color_primaries == AVCOL_PRI_UNSPECIFIED)
        in->color_primaries = AVCOL_PRI_BT2020;
    if (out->color_primaries == AVCOL_PRI_UNSPECIFIED)
        out->color_primaries = AVCOL_PRI_BT709;

    if (in->color_range == AVCOL_RANGE_UNSPECIFIED)
        in->color_range = AVCOL_RANGE_MPEG;
    if (out->color_range == AVCOL_RANGE_UNSPECIFIED)
        out->color_range = AVCOL_RANGE_MPEG;

    if (!s->lin_lut || !s->delin_lut) {
        if ((ret = compute_trc_luts(s, in->color_trc, out->color_trc)) < 0)
            goto fail;
    }

    if (!s->tonemap_lut || s->lut_peak != peak) {
        s->lut_peak = peak;
        if ((ret = compute_tonemap_lut(s, in->color_trc)) < 0)
            goto fail;
    }

    coeffs = av_csp_luma_coeffs_from_avcsp(in->colorspace);
    if (s->coeffs != coeffs) {
        s->coeffs = coeffs;
        s->ocoeffs = av_csp_luma_coeffs_from_avcsp(out->colorspace);
        if ((ret = compute_yuv_coeffs(s, coeffs, s->ocoeffs, desc, odesc,
             in->color_range, out->color_range)) < 0)
            goto fail;
        if ((ret = compute_rgb_coeffs(s, in->color_primaries, out->color_primaries)) < 0)
            goto fail;
    }

    /* do the tonemap */
    td.in    = in;
    td.out   = out;
    td.desc  = desc;
    td.odesc = odesc;
    td.peak  = peak;
    ff_filter_execute(ctx, s->filter_slice, &td, NULL,
                      FFMIN(outlink->h >> FFMAX(desc->log2_chroma_h, odesc->log2_chroma_h), ff_filter_get_nb_threads(ctx)));

    av_frame_free(&in);

    if (s->trc !=AVCOL_TRC_SMPTE2084) {
        av_frame_remove_side_data(out, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
        av_frame_remove_side_data(out, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    }

    av_frame_remove_side_data(out, AV_FRAME_DATA_DOVI_RPU_BUFFER);
    av_frame_remove_side_data(out, AV_FRAME_DATA_DOVI_METADATA);

    return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

static void uninit(AVFilterContext *ctx)
{
    TonemapxContext *s = ctx->priv;

    av_freep(&s->lin_lut);
    av_freep(&s->delin_lut);
    av_freep(&s->tonemap_lut);

    if (s->dovi)
        av_freep(&s->dovi);
}

static int query_formats(AVFilterContext *ctx)
{
    enum AVPixelFormat valid_in_pix_fmts[4];
    AVFilterFormats *formats;
    const AVPixFmtDescriptor *desc;
    TonemapxContext *s = ctx->priv;
    int res;

    if (!strcmp(s->format_str, "same")) {
        formats = ff_make_format_list(in_pix_fmts);
        res = ff_formats_ref(formats, &ctx->inputs[0]->outcfg.formats);
        if (res < 0)
            return res;
        s->format = AV_PIX_FMT_NONE;
    } else {
        int i, j = 0;
        formats = ff_make_format_list(in_pix_fmts);
        res = ff_formats_ref(formats, &ctx->inputs[0]->outcfg.formats);
        if (res < 0)
            return res;
        if (s->format == AV_PIX_FMT_NONE) {
            av_log(ctx, AV_LOG_ERROR, "Unrecognized pixel format: %s\n", s->format_str);
            return AVERROR(EINVAL);
        }
        s->format = av_get_pix_fmt(s->format_str);
        // Check again in case of the string is invalid
        if (s->format == AV_PIX_FMT_NONE) {
            av_log(ctx, AV_LOG_ERROR, "Unrecognized pixel format: %s\n", s->format_str);
            return AVERROR(EINVAL);
        }
        desc = av_pix_fmt_desc_get(s->format);
        // Filter out the input formats for requested output formats
        // The input and output must have the same planar format, either planar or bi-planar packed
        for (i = 0; in_pix_fmts[i] != AV_PIX_FMT_NONE; i++) {
            const AVPixFmtDescriptor *tdesc = av_pix_fmt_desc_get(in_pix_fmts[i]);
            if (tdesc->comp[2].plane == desc->comp[2].plane) {
                valid_in_pix_fmts[j] = in_pix_fmts[i];
                j++;
            }
        }
        valid_in_pix_fmts[j] = AV_PIX_FMT_NONE;
        formats = ff_make_format_list(valid_in_pix_fmts);
        res = ff_formats_ref(formats, &ctx->inputs[0]->outcfg.formats);
        if (res < 0)
            return res;
        if (out_format_is_supported(s->format)) {
            formats = NULL;
            res = ff_add_format(&formats, s->format);
            if (res < 0)
                return res;
        } else {
            av_log(ctx, AV_LOG_ERROR, "Unsupported output format: %s\n",
                   av_get_pix_fmt_name(s->format));
            return AVERROR(ENOSYS);
        }
    }

    res = ff_formats_ref(formats, &ctx->outputs[0]->incfg.formats);
    if (res < 0)
        return res;

    // colorspaces and ranges
    if ((res = ff_formats_ref(ff_all_color_spaces(),
                              &ctx->inputs[0]->outcfg.color_spaces)) < 0)
        return res;

    if ((res = ff_formats_ref(ff_all_color_ranges(),
                              &ctx->inputs[0]->outcfg.color_ranges)) < 0)
        return res;

    formats = s->spc != -1
        ? ff_make_formats_list_singleton(s->spc)
        : ff_make_format_list(colorspaces_out);
    if ((res = ff_formats_ref(formats, &ctx->outputs[0]->incfg.color_spaces)) < 0)
        return res;

    formats = s->range != -1
        ? ff_make_formats_list_singleton(s->range)
        : ff_all_color_ranges();
    if ((res = ff_formats_ref(formats, &ctx->outputs[0]->incfg.color_ranges)) < 0)
        return res;

    return 0;
}

static av_cold int init(AVFilterContext *ctx)
{
    TonemapxContext *s = ctx->priv;
    enum SIMDVariant active_simd = SIMD_NONE;
    av_log(s, AV_LOG_DEBUG, "Requested output format: %s\n",
           s->format_str);

    if (s->trc == AVCOL_TRC_SMPTE2084) {
        if (s->spc != AVCOL_SPC_BT2020_NCL) {
            av_log(s, AV_LOG_ERROR, "HDR passthrough requires BT2020 Non-constant luminance matrix\n");
            return AVERROR(EINVAL);
        }

        if (s->pri != AVCOL_PRI_BT2020) {
            av_log(s, AV_LOG_ERROR, "HDR passthrough requires BT2020 primaries\n");
            return AVERROR(EINVAL);
        }

        if (!s->apply_dovi) {
            av_log(s, AV_LOG_ERROR, "HDR passthrough only works for Dolby Vision inputs at the moment\n");
            return AVERROR(EINVAL);
        }
    }

#if ARCH_AARCH64
#ifdef ENABLE_TONEMAPX_NEON_INTRINSICS
    {
        int cpu_flags = av_get_cpu_flags();
        if (have_neon(cpu_flags)) {
            s->tonemap_func_biplanar8 = tonemap_frame_p010_2_nv12_neon;
            s->tonemap_func_biplanar10 = tonemap_frame_p010_2_p010_neon;
            s->tonemap_func_planar8 = tonemap_frame_420p10_2_420p_neon;
            s->tonemap_func_planar10 = tonemap_frame_420p10_2_420p10_neon;
            s->tonemap_func_dovi8 = tonemap_frame_dovi_2_420p_neon;
            s->tonemap_func_dovi10 = s->trc == AVCOL_TRC_SMPTE2084 ? tonemap_frame_dovi_2_420hdr_neon : tonemap_frame_dovi_2_420p10_neon;
            active_simd = SIMD_NEON;
        }
    }
#else
    av_log(s, AV_LOG_WARNING, "NEON optimization disabled at compile time\n");
#endif // ENABLE_TONEMAPX_NEON_INTRINSICS
#elif ARCH_X86
#ifdef ENABLE_TONEMAPX_SSE_INTRINSICS
    {
        int cpu_flags = av_get_cpu_flags();
        if (X86_SSE42(cpu_flags)) {
            s->tonemap_func_biplanar8 = tonemap_frame_p010_2_nv12_sse;
            s->tonemap_func_biplanar10 = tonemap_frame_p010_2_p010_sse;
            s->tonemap_func_planar8 = tonemap_frame_420p10_2_420p_sse;
            s->tonemap_func_planar10 = tonemap_frame_420p10_2_420p10_sse;
            s->tonemap_func_dovi8 = tonemap_frame_dovi_2_420p_sse;
            s->tonemap_func_dovi10 = s->trc == AVCOL_TRC_SMPTE2084 ? tonemap_frame_dovi_2_420hdr_sse : tonemap_frame_dovi_2_420p10_sse;
            active_simd = SIMD_SSE;
        }
    }
#else
    av_log(s, AV_LOG_WARNING, "SSE optimization disabled at compile time\n");
#endif // ENABLE_TONEMAPX_SSE_INTRINSICS
#ifdef ENABLE_TONEMAPX_AVX_INTRINSICS
    {
        int cpu_flags = av_get_cpu_flags();
        if (X86_AVX2(cpu_flags) && X86_FMA3(cpu_flags)) {
            s->tonemap_func_biplanar8 = tonemap_frame_p010_2_nv12_avx;
            s->tonemap_func_biplanar10 = tonemap_frame_p010_2_p010_avx;
            s->tonemap_func_planar8 = tonemap_frame_420p10_2_420p_avx;
            s->tonemap_func_planar10 = tonemap_frame_420p10_2_420p10_avx;
            s->tonemap_func_dovi8 = tonemap_frame_dovi_2_420p_avx;
            s->tonemap_func_dovi10 = s->trc == AVCOL_TRC_SMPTE2084 ? tonemap_frame_dovi_2_420hdr_avx : tonemap_frame_dovi_2_420p10_avx;
            active_simd = SIMD_AVX;
        }
    }
#else
    av_log(s, AV_LOG_WARNING, "AVX optimization disabled at compile time\n");
#endif // ENABLE_TONEMAPX_AVX_INTRINSICS
#endif // ARCH_X86/ARCH_AARCH64

#if !defined(ENABLE_TONEMAPX_NEON_INTRINSICS) && \
    !defined(ENABLE_TONEMAPX_SSE_INTRINSICS) && \
    !defined(ENABLE_TONEMAPX_AVX_INTRINSICS)
    av_log(s, AV_LOG_WARNING, "SIMD optimization disabled at compile time\n");
#endif

    if (!s->tonemap_func_biplanar8) {
        s->tonemap_func_biplanar8 = tonemap_frame_p010_2_nv12;
    }

    if (!s->tonemap_func_biplanar10) {
        s->tonemap_func_biplanar10 = tonemap_frame_p010_2_p010;
    }

    if (!s->tonemap_func_planar8) {
        s->tonemap_func_planar8 = tonemap_frame_420p10_2_420p;
    }

    if (!s->tonemap_func_planar10) {
        s->tonemap_func_planar10 = tonemap_frame_420p10_2_420p10;
    }

    if (!s->tonemap_func_dovi8) {
        s->tonemap_func_dovi8 = tonemap_frame_dovi_2_420p;
    }

    if (!s->tonemap_func_dovi10) {
        s->tonemap_func_dovi10 = s->trc == AVCOL_TRC_SMPTE2084 ? tonemap_frame_dovi_2_420hdr : tonemap_frame_dovi_2_420p10;
    }

    switch (active_simd) {
        case SIMD_NEON:
            av_log(s, AV_LOG_INFO, "Using CPU capability: NEON\n");
            break;
        case SIMD_SSE:
            av_log(s, AV_LOG_INFO, "Using CPU capability: SSE4.2\n");
            break;
        case SIMD_AVX:
            av_log(s, AV_LOG_INFO, "Using CPU capabilities: AVX2 FMA3\n");
            break;
        default:
        case SIMD_NONE:
            av_log(s, AV_LOG_INFO, "No CPU SIMD extension available\n");
            break;
    }

    switch (s->tonemap) {
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
        case TONEMAP_BT2390:
            if (isnan(s->param))
                s->param = 1.0f; // diff from the spec-defined 0.5f
            else
                s->param = FFMIN(FFMAX(s->param, 0.5f), 2.0f);
            break;
    }

    if (isnan(s->param))
        s->param = 1.0f;

    if (s->peak)
        s->peak = FFMAX(s->peak / 10.0f, 1.1f);

    return 0;
}

#define OFFSET(x) offsetof(TonemapxContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM
static const AVOption tonemapx_options[] = {
    { "tonemap",      "tonemap algorithm selection", OFFSET(tonemap), AV_OPT_TYPE_INT, {.i64 = TONEMAP_BT2390}, TONEMAP_NONE, TONEMAP_MAX - 1, FLAGS, .unit = "tonemap" },
    {     "none",     0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_NONE},              0, 0, FLAGS, .unit = "tonemap" },
    {     "linear",   0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_LINEAR},            0, 0, FLAGS, .unit = "tonemap" },
    {     "gamma",    0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_GAMMA},             0, 0, FLAGS, .unit = "tonemap" },
    {     "clip",     0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_CLIP},              0, 0, FLAGS, .unit = "tonemap" },
    {     "reinhard", 0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_REINHARD},          0, 0, FLAGS, .unit = "tonemap" },
    {     "hable",    0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_HABLE},             0, 0, FLAGS, .unit = "tonemap" },
    {     "mobius",   0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_MOBIUS},            0, 0, FLAGS, .unit = "tonemap" },
    {     "bt2390",   0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_BT2390},            0, 0, FLAGS, .unit = "tonemap" },
    { "transfer",     "set transfer characteristic", OFFSET(trc), AV_OPT_TYPE_INT, {.i64 = AVCOL_TRC_BT709}, -1, INT_MAX, FLAGS, .unit = "transfer" },
    { "t",            "set transfer characteristic", OFFSET(trc), AV_OPT_TYPE_INT, {.i64 = AVCOL_TRC_BT709}, -1, INT_MAX, FLAGS, .unit = "transfer" },
    {     "bt709",    0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_TRC_BT709},           0, 0, FLAGS, .unit = "transfer" },
    {     "bt2020",   0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_TRC_BT2020_10},       0, 0, FLAGS, .unit = "transfer" },
    {     "smpte2084",0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_TRC_SMPTE2084},       0, 0, FLAGS, .unit = "transfer" },
    { "matrix",       "set colorspace matrix", OFFSET(spc), AV_OPT_TYPE_INT, {.i64 = AVCOL_SPC_BT709}, -1, INT_MAX, FLAGS, .unit = "matrix" },
    { "m",            "set colorspace matrix", OFFSET(spc), AV_OPT_TYPE_INT, {.i64 = AVCOL_SPC_BT709}, -1, INT_MAX, FLAGS, .unit = "matrix" },
    {     "bt709",    0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_SPC_BT709},           0, 0, FLAGS, .unit = "matrix" },
    {     "bt2020",   0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_SPC_BT2020_NCL},      0, 0, FLAGS, .unit = "matrix" },
    { "primaries",    "set color primaries", OFFSET(pri), AV_OPT_TYPE_INT, {.i64 = AVCOL_PRI_BT709}, -1, INT_MAX, FLAGS, .unit = "primaries" },
    { "p",            "set color primaries", OFFSET(pri), AV_OPT_TYPE_INT, {.i64 = AVCOL_PRI_BT709}, -1, INT_MAX, FLAGS, .unit = "primaries" },
    {     "bt709",    0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_PRI_BT709},           0, 0, FLAGS, .unit = "primaries" },
    {     "bt2020",   0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_PRI_BT2020},          0, 0, FLAGS, .unit = "primaries" },
    { "range",        "set color range", OFFSET(range), AV_OPT_TYPE_INT, {.i64 = -1}, -1, INT_MAX, FLAGS, .unit = "range" },
    { "r",            "set color range", OFFSET(range), AV_OPT_TYPE_INT, {.i64 = -1}, -1, INT_MAX, FLAGS, .unit = "range" },
    {     "tv",       0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_RANGE_MPEG},          0, 0, FLAGS, .unit = "range" },
    {     "pc",       0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_RANGE_JPEG},          0, 0, FLAGS, .unit = "range" },
    {     "limited",  0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_RANGE_MPEG},          0, 0, FLAGS, .unit = "range" },
    {     "full",     0, 0, AV_OPT_TYPE_CONST, {.i64 = AVCOL_RANGE_JPEG},          0, 0, FLAGS, .unit = "range" },
    { "format",       "output format",       OFFSET(format_str), AV_OPT_TYPE_STRING, { .str = "same" }, .flags = FLAGS },
    { "param",        "tonemap parameter", OFFSET(param), AV_OPT_TYPE_DOUBLE, {.dbl = NAN}, DBL_MIN, DBL_MAX, FLAGS },
    { "desat",        "desaturation strength", OFFSET(desat), AV_OPT_TYPE_DOUBLE, {.dbl = 0}, 0, DBL_MAX, FLAGS },
    { "peak",         "signal peak override", OFFSET(peak), AV_OPT_TYPE_DOUBLE, {.dbl = 0}, 0, DBL_MAX, FLAGS },
    { "apply_dovi",  "Apply Dolby Vision metadata if possible", OFFSET(apply_dovi), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(tonemapx);

static const AVFilterPad tonemapx_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
};

AVFilter ff_vf_tonemapx = {
    .name            = "tonemapx",
    .description     = NULL_IF_CONFIG_SMALL("SIMD optimized HDR to SDR tonemapping"),
    .init            = init,
    .uninit          = uninit,
    .priv_size       = sizeof(TonemapxContext),
    .priv_class      = &tonemapx_class,
    FILTER_INPUTS(tonemapx_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_QUERY_FUNC(query_formats),
    .flags           = AVFILTER_FLAG_SLICE_THREADS,
};
