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

#ifndef AVCODEC_AMFENC_H
#define AVCODEC_AMFENC_H

#include <AMF/components/VideoEncoderVCE.h>
#include <AMF/components/VideoEncoderHEVC.h>
#include <AMF/components/VideoEncoderAV1.h>

#include "libavutil/fifo.h"

#include "amf.h"
#include "hwconfig.h"

/**
* AMF encoder context
*/
typedef struct AMFEncContext {
    void               *avclass;
    void               *amfctx;

    // encoder
    AMFComponent                          *encoder; ///< AMF encoder object
    amf_bool                               eof;     ///< flag indicating EOF happened
    AMF_SURFACE_FORMAT                     format;  ///< AMF surface format
    AMF_VIDEO_CONVERTER_COLOR_PROFILE_ENUM out_color_profile;
    AMF_COLOR_TRANSFER_CHARACTERISTIC_ENUM out_color_trc;
    AMF_COLOR_PRIMARIES_ENUM               out_color_prm;

    AVBufferRef        *hw_device_ctx; ///< pointer to HW accelerator (decoder)
    AVBufferRef        *hw_frames_ctx; ///< pointer to HW accelerator (frame allocator)

    int                 hwsurfaces_in_queue;
    int                 hwsurfaces_in_queue_max;

    // helpers to handle async calls
    int                 delayed_drain;
    AMFSurface         *delayed_surface;
    AVFrame            *delayed_frame;

    // shift dts back by max_b_frames in timing
    AVFifo             *timestamp_list;
    int64_t             dts_delay;

    // common encoder option options
    int                 log_to_dbg;

    // Static options, have to be set before Init() call
    int                 usage;
    int                 profile;
    int                 level;
    int                 pre_encode;
    int                 quality;
    int                 bit_depth;
    int                 qvbr_level;
    int                 b_frame_delta_qp;
    int                 ref_b_frame_delta_qp;

    // Dynamic options, can be set after Init() call
    int                 rate_control_mode;
    int                 enforce_hrd;
    int                 filler_data;
    int                 enable_vbaq;
    int                 enable_hmqb;
    int                 skip_frame;
    int                 qp_i;
    int                 qp_p;
    int                 qp_b;
    int                 max_au_size;
    int                 header_spacing;
    int                 b_frame_ref;
    int                 intra_refresh_mb;
    int                 coding_mode;
    int                 me_half_pel;
    int                 me_quarter_pel;
    int                 aud;

    // HEVC - specific options
    int                 gops_per_idr;
    int                 header_insertion_mode;
    int                 min_qp_i;
    int                 max_qp_i;
    int                 min_qp_p;
    int                 max_qp_p;
    int                 tier;

    // AV1 - specific options

    enum AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE_ENUM                 align;

} AMFEncContext;

extern const AVCodecHWConfigInternal *const ff_amfenc_hw_configs[];

/**
* Common encoder initization function
*/
int ff_amf_encode_init(AVCodecContext *avctx);
/**
* Common encoder termination function
*/
int ff_amf_encode_close(AVCodecContext *avctx);

/**
* Ecoding one frame - common function for all AMF encoders
*/
int ff_amf_receive_packet(AVCodecContext *avctx, AVPacket *avpkt);

#endif /* AVCODEC_AMFENC_H */
