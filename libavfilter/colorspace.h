/*
 * Copyright (c) 2016 Ronald S. Bultje <rsbultje@gmail.com>
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

#ifndef AVFILTER_COLORSPACE_H
#define AVFILTER_COLORSPACE_H

#include "libavutil/csp.h"
#include "libavutil/frame.h"
#include "libavutil/pixfmt.h"
#include "libavutil/dovi_meta.h"

#define REFERENCE_WHITE 100.0f
#define REFERENCE_WHITE_ALT 203.0f
#define ST2084_MAX_LUMINANCE 10000.0f
#define ST2084_M1 0.1593017578125f
#define ST2084_M2 78.84375f
#define ST2084_C1 0.8359375f
#define ST2084_C2 18.8515625f
#define ST2084_C3 18.6875f
#define ARIB_B67_A 0.17883277f
#define ARIB_B67_B 0.28466892f
#define ARIB_B67_C 0.55991073f
#define FLOAT_EPS 1e-6f

// Parsed metadata from the Dolby Vision RPU
struct DoviMetadata {
    float nonlinear_offset[3];      // input offset ("ycc_to_rgb_offset")
    double nonlinear[3][3];  // before PQ, also called "ycc_to_rgb"
    double linear[3][3];     // after PQ, also called "rgb_to_lms"

    // Reshape data, grouped by component
    struct ReshapeData {
        uint8_t num_pivots;
        float pivots[9]; // normalized to [0.0, 1.0] based on BL bit depth
        uint8_t method[8]; // 0 = polynomial, 1 = MMR
        // Note: these must be normalized (divide by coefficient_log2_denom)
        float poly_coeffs[8][3]; // x^0, x^1, x^2, unused must be 0
        uint8_t mmr_order[8]; // 1, 2 or 3
        float mmr_constant[8];
        float mmr_coeffs[8][3 /* order */][7];
    } comp[3];
};

void ff_matrix_invert_3x3(const double in[3][3], double out[3][3]);
void ff_matrix_transpose_3x3(const double in[3][3], double out[3][3]);
void ff_matrix_mul_3x3(double dst[3][3],
               const double src1[3][3], const double src2[3][3]);
void ff_matrix_mul_3x3_vec(double dst[3], const double vec[3], const double mat[3][3]);
void ff_fill_rgb2xyz_table(const AVPrimaryCoefficients *coeffs,
                           const AVWhitepointCoefficients *wp,
                           double rgb2xyz[3][3]);
void ff_fill_rgb2yuv_table(const AVLumaCoefficients *coeffs,
                           double rgb2yuv[3][3]);
double ff_determine_signal_peak(AVFrame *in);
void ff_update_hdr_metadata(AVFrame *in, double peak);

double ff_determine_dovi_signal_peak(const AVDOVIMetadata *data);
void ff_map_dovi_metadata(struct DoviMetadata *out, const AVDOVIMetadata *data);

float eotf_st2084_common(float x);
float eotf_st2084(float x, float ref_white);
float inverse_eotf_st2084_common(float x);
float inverse_eotf_st2084(float x, float ref_white);
float ootf_1_2(float x);
float inverse_ootf_1_2(float x);
float oetf_arib_b67(float x);
float inverse_oetf_arib_b67(float x);
float eotf_arib_b67(float x);
float inverse_eotf_arib_b67(float x);
float inverse_eotf_bt1886(float x);

#endif
