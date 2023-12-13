/*
 * Copyright (c) 2023 Huseyin BIYIK
 * Copyright (c) 2023 NyanMisaka
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

/**
 * @file
 * Rockchip MPP (Media Process Platform) video encoder
 */

#include "config_components.h"
#include "rkmppenc.h"

static MppCodingType rkmpp_get_coding_type(AVCodecContext *avctx)
{
    switch (avctx->codec_id) {
    case AV_CODEC_ID_H264: return MPP_VIDEO_CodingAVC;
    case AV_CODEC_ID_HEVC: return MPP_VIDEO_CodingHEVC;
    default:               return MPP_VIDEO_CodingUnused;
    }
}

static MppFrameFormat rkmpp_get_mpp_fmt(enum AVPixelFormat pix_fmt)
{
    switch (pix_fmt) {
    case AV_PIX_FMT_YUV420P: return MPP_FMT_YUV420P;
    case AV_PIX_FMT_YUV422P: return MPP_FMT_YUV422P;
    case AV_PIX_FMT_YUV444P: return MPP_FMT_YUV444P;
    case AV_PIX_FMT_NV12:    return MPP_FMT_YUV420SP;
    case AV_PIX_FMT_NV21:    return MPP_FMT_YUV420SP_VU;
    case AV_PIX_FMT_NV16:    return MPP_FMT_YUV422SP;
    case AV_PIX_FMT_NV24:    return MPP_FMT_YUV444SP;
    case AV_PIX_FMT_YUYV422: return MPP_FMT_YUV422_YUYV;
    case AV_PIX_FMT_YVYU422: return MPP_FMT_YUV422_YVYU;
    case AV_PIX_FMT_UYVY422: return MPP_FMT_YUV422_UYVY;
    case AV_PIX_FMT_RGB24:   return MPP_FMT_RGB888;
    case AV_PIX_FMT_BGR24:   return MPP_FMT_BGR888;
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_RGB0:    return MPP_FMT_RGBA8888;
    case AV_PIX_FMT_BGRA:
    case AV_PIX_FMT_BGR0:    return MPP_FMT_BGRA8888;
    case AV_PIX_FMT_ARGB:
    case AV_PIX_FMT_0RGB:    return MPP_FMT_ARGB8888;
    case AV_PIX_FMT_ABGR:
    case AV_PIX_FMT_0BGR:    return MPP_FMT_ABGR8888;
    default:                 return MPP_FMT_BUTT;
    }
}

static uint32_t rkmpp_get_drm_afbc_format(MppFrameFormat mpp_fmt)
{
    switch (mpp_fmt & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP: return DRM_FORMAT_YUV420_8BIT;
    case MPP_FMT_YUV422SP: return DRM_FORMAT_YUYV;
    default:               return DRM_FORMAT_INVALID;
    }
}

static int get_byte_stride(const AVDRMObjectDescriptor *object,
                           const AVDRMLayerDescriptor *layer,
                           int is_rgb, int is_planar,
                           int *hs, int *vs)
{
    const AVDRMPlaneDescriptor *plane0, *plane1;
    const int is_packed_fmt = is_rgb || (!is_rgb && !is_planar);

    if (!object || !layer || !hs || !vs)
        return AVERROR(EINVAL);

    plane0 = &layer->planes[0];
    plane1 = &layer->planes[1];

    *hs = plane0->pitch;
    *vs = is_packed_fmt ?
        ALIGN_DOWN(object->size / plane0->pitch, is_rgb ? 1 : 2) :
        (plane1->offset / plane0->pitch);

    return (*hs > 0 && *vs > 0) ? 0 : AVERROR(EINVAL);
}

static int rkmpp_set_enc_cfg_dynamic(AVCodecContext *avctx, AVFrame *frame)
{
    RKMPPEncContext *r = avctx->priv_data;
    MppEncCfg cfg = r->mcfg;
    MppFrameFormat mpp_fmt = r->mpp_fmt;
    int ret;
    int hor_stride = 0, ver_stride = 0;
    const AVPixFmtDescriptor *pix_desc;
    const AVDRMFrameDescriptor *drm_desc;

    if (r->cfg_initialised)
        return 0;

    if (!frame)
        return AVERROR(EINVAL);

    drm_desc = (AVDRMFrameDescriptor *)frame->data[0];
    if (drm_desc->objects[0].fd < 0)
        return AVERROR(ENOMEM);

    pix_desc = av_pix_fmt_desc_get(r->pix_fmt);
    ret = get_byte_stride(&drm_desc->objects[0],
                          &drm_desc->layers[0],
                          (pix_desc->flags & AV_PIX_FMT_FLAG_RGB),
                          (pix_desc->flags & AV_PIX_FMT_FLAG_PLANAR),
                          &hor_stride, &ver_stride);
    if (ret < 0 || !hor_stride || !ver_stride)
        return AVERROR(EINVAL);

    if (frame->time_base.num && frame->time_base.den) {
        avctx->time_base.num = frame->time_base.num;
        avctx->time_base.den = frame->time_base.den;
    } else {
        avctx->time_base.num = avctx->framerate.den;
        avctx->time_base.den = avctx->framerate.num;
    }

    mpp_enc_cfg_set_s32(cfg, "prep:width", avctx->width);
    mpp_enc_cfg_set_s32(cfg, "prep:height", avctx->height);
    mpp_enc_cfg_set_s32(cfg, "prep:hor_stride", hor_stride);
    mpp_enc_cfg_set_s32(cfg, "prep:ver_stride", ver_stride);

    mpp_enc_cfg_set_s32(cfg, "prep:colorspace", avctx->colorspace);
    mpp_enc_cfg_set_s32(cfg, "prep:colorprim", avctx->color_primaries);
    mpp_enc_cfg_set_s32(cfg, "prep:colortrc", avctx->color_trc);
    mpp_enc_cfg_set_s32(cfg, "prep:colorrange", avctx->color_range);

    if (drm_is_afbc(drm_desc->objects[0].format_modifier)) {
        const AVDRMLayerDescriptor *layer = &drm_desc->layers[0];
        uint32_t drm_afbc_fmt = rkmpp_get_drm_afbc_format(mpp_fmt);

        if (drm_afbc_fmt != layer->format) {
            av_log(avctx, AV_LOG_ERROR, "Input format '%s' with AFBC modifier is not supported\n",
                   av_get_pix_fmt_name(r->pix_fmt));
            return AVERROR(ENOSYS);
        }
        mpp_fmt |= MPP_FRAME_FBC_AFBC_V2;
    }
    mpp_enc_cfg_set_s32(cfg, "prep:format", mpp_fmt);

    if ((ret = r->mapi->control(r->mctx, MPP_ENC_SET_CFG, cfg)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to set config with frame: %d\n", ret);
        return AVERROR_EXTERNAL;
    }

    r->cfg_initialised = 1;
    av_log(avctx, AV_LOG_VERBOSE, "Re-configured with w=%d, h=%d, format=%s\n", avctx->width,
           avctx->height, av_get_pix_fmt_name(r->pix_fmt));

    return 0;
}

static int rkmpp_set_enc_cfg(AVCodecContext *avctx)
{
    RKMPPEncContext *r = avctx->priv_data;
    MppEncCfg cfg = r->mcfg;

    RK_U32 rc_mode, fps_num, fps_den;
    MppEncHeaderMode header_mode;
    MppEncSeiMode sei_mode;
    int ret, max_bps, min_bps, qmin, qmax;

    mpp_enc_cfg_set_s32(cfg, "prep:width", avctx->width);
    mpp_enc_cfg_set_s32(cfg, "prep:height", avctx->height);
    mpp_enc_cfg_set_s32(cfg, "prep:hor_stride", FFALIGN(avctx->width, 64));
    mpp_enc_cfg_set_s32(cfg, "prep:ver_stride", FFALIGN(avctx->height, 64));
    mpp_enc_cfg_set_s32(cfg, "prep:format", MPP_FMT_YUV420SP);
    mpp_enc_cfg_set_s32(cfg, "prep:mirroring", 0);
    mpp_enc_cfg_set_s32(cfg, "prep:rotation", 0);
    mpp_enc_cfg_set_s32(cfg, "prep:flip", 0);

    if (avctx->framerate.den > 0 && avctx->framerate.num > 0)
        av_reduce(&fps_num, &fps_den, avctx->framerate.num, avctx->framerate.den, 65535);
    else
        av_reduce(&fps_num, &fps_den, avctx->time_base.den, avctx->time_base.num, 65535);

    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_flex", 0);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_num", fps_num);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_denorm", fps_den);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_flex", 0);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_num",fps_num);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_denorm", fps_den);

    mpp_enc_cfg_set_s32(cfg, "rc:gop", FFMAX(avctx->gop_size, 1));

    rc_mode = r->rc_mode;
    if (rc_mode == MPP_ENC_RC_MODE_BUTT)
        rc_mode = MPP_ENC_RC_MODE_CBR;

    switch (rc_mode) {
    case MPP_ENC_RC_MODE_VBR:
        av_log(avctx, AV_LOG_VERBOSE, "Rate Control mode is set to VBR\n"); break;
    case MPP_ENC_RC_MODE_CBR:
        av_log(avctx, AV_LOG_VERBOSE, "Rate Control mode is set to CBR\n"); break;
    case MPP_ENC_RC_MODE_FIXQP:
        av_log(avctx, AV_LOG_VERBOSE, "Rate Control mode is set to CQP\n"); break;
    case MPP_ENC_RC_MODE_AVBR:
        av_log(avctx, AV_LOG_VERBOSE, "Rate Control mode is set to AVBR\n"); break;
    }
    mpp_enc_cfg_set_u32(cfg, "rc:mode", rc_mode);

    switch (rc_mode) {
    case MPP_ENC_RC_MODE_FIXQP:
        /* do not setup bitrate on FIXQP mode */
        min_bps = max_bps = avctx->bit_rate;
        break;
    case MPP_ENC_RC_MODE_VBR:
    case MPP_ENC_RC_MODE_AVBR:
        /* VBR mode has wide bound */
        max_bps = avctx->bit_rate * 17 / 16;
        min_bps = avctx->bit_rate * 1 / 16;
        break;
    case MPP_ENC_RC_MODE_CBR:
    default:
        /* CBR mode has narrow bound */
        max_bps = avctx->bit_rate * 17 / 16;
        min_bps = avctx->bit_rate * 15 / 16;
        break;
    }
    mpp_enc_cfg_set_u32(cfg, "rc:bps_target", avctx->bit_rate);
    mpp_enc_cfg_set_s32(cfg, "rc:bps_max", max_bps);
    mpp_enc_cfg_set_s32(cfg, "rc:bps_min", min_bps);

    av_log(avctx, AV_LOG_VERBOSE, "Bitrate Target/Min/Max is set to %ld/%d/%d\n", avctx->bit_rate, min_bps, max_bps);

    mpp_enc_cfg_set_u32(cfg, "rc:drop_mode", MPP_ENC_RC_DROP_FRM_DISABLED);
    mpp_enc_cfg_set_u32(cfg, "rc:drop_thd", 20); // 20% of max bps
    mpp_enc_cfg_set_u32(cfg, "rc:drop_gap", 1); // Do not continuous drop frame

    switch (avctx->codec_id) {
    case AV_CODEC_ID_H264:
    case AV_CODEC_ID_HEVC:
        {
            qmax = QMIN_H26x + (100 - r->qmin) * (QMAX_H26x - QMIN_H26x) / 100;
            qmin = QMIN_H26x + (100 - r->qmax) * (QMAX_H26x - QMIN_H26x) / 100;
            switch (rc_mode) {
            case MPP_ENC_RC_MODE_FIXQP:
                mpp_enc_cfg_set_s32(cfg, "rc:qp_init", qmin);
                mpp_enc_cfg_set_s32(cfg, "rc:qp_max", qmin);
                mpp_enc_cfg_set_s32(cfg, "rc:qp_min", qmin);
                mpp_enc_cfg_set_s32(cfg, "rc:qp_max_i", qmin);
                mpp_enc_cfg_set_s32(cfg, "rc:qp_min_i", qmin);
                mpp_enc_cfg_set_s32(cfg, "rc:qp_ip", 0);
                break;
            case MPP_ENC_RC_MODE_CBR:
            case MPP_ENC_RC_MODE_VBR:
            case MPP_ENC_RC_MODE_AVBR:
                mpp_enc_cfg_set_s32(cfg, "rc:qp_init", qmin);
                mpp_enc_cfg_set_s32(cfg, "rc:qp_max", qmax);
                mpp_enc_cfg_set_s32(cfg, "rc:qp_min", qmin);
                mpp_enc_cfg_set_s32(cfg, "rc:qp_max_i",qmax);
                mpp_enc_cfg_set_s32(cfg, "rc:qp_min_i", qmin);
                mpp_enc_cfg_set_s32(cfg, "rc:qp_ip", 2);
                break;
            default:
                return AVERROR(EINVAL);
            }
        }
        break;
    default:
        return AVERROR(EINVAL);
    }

    switch (avctx->codec_id) {
    case AV_CODEC_ID_H264:
        {
            avctx->profile = r->profile;
            avctx->level = r->level;
            mpp_enc_cfg_set_s32(cfg, "h264:profile", avctx->profile);
            mpp_enc_cfg_set_s32(cfg, "h264:level", avctx->level);
            mpp_enc_cfg_set_s32(cfg, "h264:cabac_en", r->coder);
            mpp_enc_cfg_set_s32(cfg, "h264:cabac_idc", 0);
            mpp_enc_cfg_set_s32(cfg, "h264:trans8x8", r->dct8x8 && avctx->profile == FF_PROFILE_H264_HIGH ? 1 : 0);

            switch (avctx->profile) {
            case FF_PROFILE_H264_BASELINE:
                av_log(avctx, AV_LOG_VERBOSE, "Profile is set to BASELINE\n"); break;
            case FF_PROFILE_H264_MAIN:
                av_log(avctx, AV_LOG_VERBOSE, "Profile is set to MAIN\n"); break;
            case FF_PROFILE_H264_HIGH:
                av_log(avctx, AV_LOG_VERBOSE, "Profile is set to HIGH\n");
                if (r->dct8x8)
                    av_log(avctx, AV_LOG_VERBOSE, "8x8 Transform is enabled\n");
                break;
            }
            av_log(avctx, AV_LOG_VERBOSE, "Level is set to %d\n", avctx->level);
            av_log(avctx, AV_LOG_VERBOSE, "Coder is set to %s\n", r->coder ? "CABAC" : "CAVLC");
        }
        break;
    case AV_CODEC_ID_HEVC:
        {
            avctx->profile = FF_PROFILE_HEVC_MAIN;
            avctx->level = r->level;
            mpp_enc_cfg_set_s32(cfg, "h265:auto_tile", 1);
            mpp_enc_cfg_set_s32(cfg, "h265:profile", avctx->profile);
            mpp_enc_cfg_set_s32(cfg, "h265:level", avctx->level);

            switch (avctx->profile) {
            case FF_PROFILE_HEVC_MAIN:
                av_log(avctx, AV_LOG_VERBOSE, "Profile is set to MAIN\n"); break;
            case FF_PROFILE_HEVC_MAIN_10:
                av_log(avctx, AV_LOG_VERBOSE, "Profile is set to MAIN 10\n"); break;
            }
            av_log(avctx, AV_LOG_VERBOSE, "Level is set to %d\n", avctx->level / 3);
        }
        break;
    default:
        return AVERROR(EINVAL);
    }

    av_log(avctx, AV_LOG_VERBOSE, "Quality Min/Max is set to %d%%(Quant=%d) / %d%%(Quant=%d)\n",
           r->qmin, qmax, r->qmax, qmin);

    if ((ret = r->mapi->control(r->mctx, MPP_ENC_SET_CFG, cfg)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to set config: %d\n", ret);
        return AVERROR_EXTERNAL;
    }

    sei_mode = MPP_ENC_SEI_MODE_DISABLE;
    if ((ret = r->mapi->control(r->mctx, MPP_ENC_SET_SEI_CFG, &sei_mode)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to set SEI config: %d\n", ret);
        return AVERROR_EXTERNAL;
    }

    header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
    if (avctx->codec_id == AV_CODEC_ID_H264 ||
        avctx->codec_id == AV_CODEC_ID_HEVC) {
        if ((ret = r->mapi->control(r->mctx, MPP_ENC_SET_HEADER_MODE, &header_mode)) != MPP_OK) {
            av_log(avctx, AV_LOG_ERROR, "Failed to set header mode: %d\n", ret);
            return AVERROR_EXTERNAL;
        }
    }

    return 0;
}

static int rkmpp_send_frame(AVCodecContext *avctx, AVFrame *frame)
{
    RKMPPEncContext *r = avctx->priv_data;
    MppFrame mpp_frame = NULL;
    MppBuffer mpp_buf = NULL;
    AVFrame *drm_frame = NULL;
    int ret;

    if ((ret = mpp_frame_init(&mpp_frame)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to init MPP frame: %d\n", ret);
        ret = AVERROR_EXTERNAL;
        goto exit;
    }

    if (frame) {
        const AVDRMFrameDescriptor *drm_desc;
        const AVPixFmtDescriptor *pix_desc;
        int hor_stride = 0, ver_stride = 0;
        MppBufferInfo buf_info = {0};
        MppFrameFormat mpp_fmt = r->mpp_fmt;

        if (avctx->pix_fmt == AV_PIX_FMT_DRM_PRIME)
            drm_frame = frame;
        else {
            drm_frame = av_frame_alloc();
            if (!drm_frame) {
                ret = AVERROR(ENOMEM);
                goto exit;
            }
            if ((ret = av_hwframe_get_buffer(r->hwframe, drm_frame, 0)) < 0) {
                av_log(avctx, AV_LOG_ERROR, "Cannot allocate an internal frame: %d\n", ret);
                goto exit;
            }
            if ((ret = av_hwframe_transfer_data(drm_frame, frame, 0)) < 0) {
                av_log(avctx, AV_LOG_ERROR, "av_hwframe_transfer_data failed: %d\n", ret);
                goto exit;
            }
            if ((ret = av_frame_copy_props(drm_frame, frame)) < 0) {
                av_log(avctx, AV_LOG_ERROR, "av_frame_copy_props failed: %d\n", ret);
                goto exit;
            }
        }

        drm_desc = (AVDRMFrameDescriptor *)drm_frame->data[0];
        if (drm_desc->objects[0].fd < 0)
            return AVERROR(ENOMEM);

        pix_desc = av_pix_fmt_desc_get(r->pix_fmt);
        ret = get_byte_stride(&drm_desc->objects[0],
                              &drm_desc->layers[0],
                              (pix_desc->flags & AV_PIX_FMT_FLAG_RGB),
                              (pix_desc->flags & AV_PIX_FMT_FLAG_PLANAR),
                              &hor_stride, &ver_stride);
        if (ret < 0 || !hor_stride || !ver_stride)
            return AVERROR(EINVAL);

        buf_info.type = MPP_BUFFER_TYPE_DRM;
        buf_info.fd   = drm_desc->objects[0].fd;
        buf_info.size = drm_desc->objects[0].size;
        if ((ret = mpp_buffer_import(&mpp_buf, &buf_info)) != MPP_OK) {
            av_log(avctx, AV_LOG_ERROR, "Failed to import MPP buffer: %d\n", ret);
            ret = AVERROR_EXTERNAL;
            goto exit;
        }

        if (drm_is_afbc(drm_desc->objects[0].format_modifier)) {
            const AVDRMLayerDescriptor *layer = &drm_desc->layers[0];
            uint32_t drm_afbc_fmt = rkmpp_get_drm_afbc_format(mpp_fmt);
            int afbc_offset_y = 0;

            if (drm_afbc_fmt != layer->format) {
                av_log(avctx, AV_LOG_ERROR, "Input format '%s' with AFBC modifier is not supported\n",
                       av_get_pix_fmt_name(r->pix_fmt));
                ret = AVERROR(ENOSYS);
                goto exit;
            }
            mpp_fmt |= MPP_FRAME_FBC_AFBC_V2;

            if (layer->planes[0].offset > 0) {
                afbc_offset_y = layer->planes[0].offset / hor_stride;
                mpp_frame_set_offset_y(mpp_frame, afbc_offset_y);
            }
        }

        if (drm_frame->pict_type == AV_PICTURE_TYPE_I) {
            if ((ret = r->mapi->control(r->mctx, MPP_ENC_SET_IDR_FRAME, NULL)) != MPP_OK) {
                av_log(avctx, AV_LOG_ERROR, "Failed to set IDR frame: %d\n", ret);
                ret = AVERROR_EXTERNAL;
                goto exit;
            }
        }

        mpp_frame_set_fmt(mpp_frame, mpp_fmt);
        mpp_frame_set_width(mpp_frame, avctx->width);
        mpp_frame_set_height(mpp_frame, avctx->height);
        mpp_frame_set_hor_stride(mpp_frame, hor_stride);
        mpp_frame_set_ver_stride(mpp_frame, ver_stride);

        mpp_frame_set_colorspace(mpp_frame, avctx->colorspace);
        mpp_frame_set_color_primaries(mpp_frame, avctx->color_primaries);
        mpp_frame_set_color_trc(mpp_frame, avctx->color_trc);
        mpp_frame_set_color_range(mpp_frame, avctx->color_range);

        mpp_frame_set_buffer(mpp_frame, mpp_buf);
        mpp_frame_set_buf_size(mpp_frame, drm_desc->objects[0].size);

        mpp_frame_set_pts(mpp_frame, drm_frame->pts);
    } else {
        av_log(avctx, AV_LOG_DEBUG, "End of stream\n");
        mpp_frame_set_eos(mpp_frame, 1);
    }

    if ((ret = rkmpp_set_enc_cfg_dynamic(avctx, drm_frame)) < 0)
        goto exit;

    if ((ret = r->mapi->encode_put_frame(r->mctx, mpp_frame)) != MPP_OK) {
        av_log(avctx, AV_LOG_DEBUG, "Encoder buffer is full\n");
        ret = AVERROR(EAGAIN);
    } else
        av_log(avctx, AV_LOG_DEBUG, "Wrote %ld bytes to encoder\n", mpp_frame_get_buf_size(mpp_frame));

exit:
    if (mpp_buf)
        mpp_buffer_put(mpp_buf);
    if (mpp_frame)
        mpp_frame_deinit(&mpp_frame);

    if (drm_frame &&
        avctx->pix_fmt != AV_PIX_FMT_DRM_PRIME)
        av_frame_free(&drm_frame);

    return ret;
}

static void rkmpp_free_packet_buf(void *opaque, uint8_t *data)
{
    MppPacket mpp_pkt = opaque;
    mpp_packet_deinit(&mpp_pkt);
}

static int rkmpp_get_packet(AVCodecContext *avctx, AVPacket *packet)
{
    RKMPPEncContext *r = avctx->priv_data;
    MppPacket mpp_pkt = NULL;
    MppMeta mpp_meta = NULL;
    int ret, key_frame = 0;

    ret = r->mapi->encode_get_packet(r->mctx, &mpp_pkt);
    if (ret != MPP_OK && ret != MPP_ERR_TIMEOUT) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get packet: %d\n", ret);
        return AVERROR(EAGAIN);
    }
    if (!mpp_pkt) {
        av_log(avctx, AV_LOG_DEBUG, "Timeout getting encoded packet\n");
        return AVERROR(EAGAIN);
    }

    if (mpp_packet_get_eos(mpp_pkt)) {
        av_log(avctx, AV_LOG_DEBUG, "Received an EOS packet\n");
        ret = AVERROR_EOF;
        goto exit;
    }
    av_log(avctx, AV_LOG_DEBUG, "Received a packet\n");

    packet->data = mpp_packet_get_data(mpp_pkt);
    packet->size = mpp_packet_get_length(mpp_pkt);
    packet->buf = av_buffer_create(packet->data, packet->size, rkmpp_free_packet_buf,
                                   mpp_pkt, AV_BUFFER_FLAG_READONLY);
    if (!packet->buf) {
        ret = AVERROR(ENOMEM);
        goto exit;
    }

    packet->time_base.num = avctx->time_base.num;
    packet->time_base.den = avctx->time_base.den;
    packet->pts = mpp_packet_get_pts(mpp_pkt);
    packet->dts = mpp_packet_get_pts(mpp_pkt);

    mpp_meta = mpp_packet_get_meta(mpp_pkt);
    if (mpp_meta)
        mpp_meta_get_s32(mpp_meta, KEY_OUTPUT_INTRA, &key_frame);
    if (key_frame)
        packet->flags |= AV_PKT_FLAG_KEY;

    return 0;

exit:
    if (mpp_pkt)
        mpp_packet_deinit(&mpp_pkt);

    return ret;
}

static int rkmpp_encode_frame(AVCodecContext *avctx, AVPacket *packet, const AVFrame *frame, int *got_packet)
{
    int ret;

    ret = rkmpp_send_frame(avctx, (AVFrame *)frame);
    if (ret)
        return ret;

    ret = rkmpp_get_packet(avctx, packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        *got_packet = 0;
    } else if (ret) {
        return ret;
    } else {
        *got_packet = 1;
    }

    return 0;
}

static int rkmpp_encode_close(AVCodecContext *avctx)
{
    RKMPPEncContext *r = avctx->priv_data;

    r->cfg_initialised = 0;

    if (r->mapi) {
        r->mapi->reset(r->mctx);
        mpp_destroy(r->mctx);
        r->mctx = NULL;
    }

    if (r->hwframe)
        av_buffer_unref(&r->hwframe);
    if (r->hwdevice)
        av_buffer_unref(&r->hwdevice);

    return 0;
}

static av_cold int init_hwframes_ctx(AVCodecContext *avctx)
{
    RKMPPEncContext *r = avctx->priv_data;
    AVHWFramesContext *hwfc;
    int ret;

    av_buffer_unref(&r->hwframe);
    r->hwframe = av_hwframe_ctx_alloc(r->hwdevice);
    if (!r->hwframe)
        return AVERROR(ENOMEM);

    hwfc            = (AVHWFramesContext *)r->hwframe->data;
    hwfc->format    = AV_PIX_FMT_DRM_PRIME;
    hwfc->sw_format = avctx->pix_fmt;
    hwfc->width     = FFALIGN(avctx->width,  16);
    hwfc->height    = FFALIGN(avctx->height, 16);

    ret = av_hwframe_ctx_init(r->hwframe);
    if (ret < 0) {
        av_buffer_unref(&r->hwframe);
        av_log(avctx, AV_LOG_ERROR, "Error creating internal frames_ctx: %d\n", ret);
        return ret;
    }

    return 0;
}

static int rkmpp_encode_init(AVCodecContext *avctx)
{
    RKMPPEncContext *r = avctx->priv_data;
    enum AVPixelFormat pix_fmt = AV_PIX_FMT_NONE;
    MppFrameFormat mpp_fmt = MPP_FMT_BUTT;
    MppCodingType coding_type = MPP_VIDEO_CodingUnused;
    MppPacket mpp_pkt = NULL;
    int input_timeout = 100;
    int output_timeout = MPP_TIMEOUT_BLOCK;
    int ret;

    if ((coding_type = rkmpp_get_coding_type(avctx)) == MPP_VIDEO_CodingUnused) {
        av_log(avctx, AV_LOG_ERROR, "Unknown codec id: %d\n", avctx->codec_id);
        return AVERROR(ENOSYS);
    }

    if ((ret = mpp_check_support_format(MPP_CTX_ENC, coding_type)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "MPP doesn't support encoding codec '%s' (%d)\n",
               avcodec_get_name(avctx->codec_id), avctx->codec_id);
        return AVERROR(ENOSYS);
    }

    if ((ret = mpp_create(&r->mctx, &r->mapi)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create MPP context and api: %d\n", ret);
        ret = AVERROR_EXTERNAL;
        goto fail;
    }

    if ((ret = r->mapi->control(r->mctx, MPP_SET_INPUT_TIMEOUT, (MppParam)&input_timeout)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to set input timeout: %d\n", ret);
        return AVERROR_EXTERNAL;
    }

    if ((ret = r->mapi->control(r->mctx, MPP_SET_OUTPUT_TIMEOUT, (MppParam)&output_timeout)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to set output timeout: %d\n", ret);
        return AVERROR_EXTERNAL;
    }

    if ((ret = mpp_init(r->mctx, MPP_CTX_ENC, coding_type)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to init MPP context: %d\n", ret);
        ret = AVERROR_EXTERNAL;
        goto fail;
    }

    if ((ret = mpp_enc_cfg_init(&r->mcfg)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to init encoder config: %d\n", ret);
        ret = AVERROR_EXTERNAL;
        goto fail;
    }

    if ((ret = r->mapi->control(r->mctx, MPP_ENC_GET_CFG, r->mcfg)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get encoder config: %d\n", ret);
        ret = AVERROR_EXTERNAL;
        goto fail;
    }

    if ((ret = rkmpp_set_enc_cfg(avctx)) < 0)
        goto fail;

    if (avctx->codec_id == AV_CODEC_ID_H264 ||
        avctx->codec_id == AV_CODEC_ID_HEVC) {
        RK_U8 enc_hdr_buf[HDR_SIZE];
        size_t pkt_len;
        void *pkt_pos;

        memset(enc_hdr_buf, 0, HDR_SIZE);

        if ((ret = mpp_packet_init(&mpp_pkt, (void *)enc_hdr_buf, HDR_SIZE)) != MPP_OK || !mpp_pkt) {
            av_log(avctx, AV_LOG_ERROR, "Failed to init extra info packet: %d\n", ret);
            ret = AVERROR_EXTERNAL;
            goto fail;
        }

        mpp_packet_set_length(mpp_pkt, 0);
        if ((ret = r->mapi->control(r->mctx, MPP_ENC_GET_HDR_SYNC, mpp_pkt)) != MPP_OK) {
            av_log(avctx, AV_LOG_ERROR, "Failed to get header sync: %d\n", ret);
            ret = AVERROR_EXTERNAL;
            goto fail;
        }

        pkt_pos = mpp_packet_get_pos(mpp_pkt);
        pkt_len = mpp_packet_get_length(mpp_pkt);

        if (avctx->extradata != NULL) {
            av_free(avctx->extradata);
            avctx->extradata = NULL;
        }
        avctx->extradata = av_malloc(pkt_len + AV_INPUT_BUFFER_PADDING_SIZE);
        if (avctx->extradata == NULL) {
            ret = AVERROR(ENOMEM);
            goto fail;
        }
        avctx->extradata_size = pkt_len + AV_INPUT_BUFFER_PADDING_SIZE;
        memcpy(avctx->extradata, pkt_pos, pkt_len);
        memset(avctx->extradata + pkt_len, 0, AV_INPUT_BUFFER_PADDING_SIZE);
        mpp_packet_deinit(&mpp_pkt);
    }

    pix_fmt = avctx->pix_fmt == AV_PIX_FMT_DRM_PRIME ? avctx->sw_pix_fmt : avctx->pix_fmt;
    mpp_fmt = rkmpp_get_mpp_fmt(pix_fmt) & MPP_FRAME_FMT_MASK;

    if (mpp_fmt == MPP_FMT_BUTT) {
        av_log(avctx, AV_LOG_ERROR, "Unsupported input pixel format '%s'\n",
               av_get_pix_fmt_name(pix_fmt));
        return AVERROR(ENOSYS);
    }
    r->pix_fmt = pix_fmt;
    r->mpp_fmt = mpp_fmt;

    if (avctx->pix_fmt == AV_PIX_FMT_DRM_PRIME)
        return 0;

    if (avctx->hw_frames_ctx || avctx->hw_device_ctx) {
        AVBufferRef *device_ref = avctx->hw_device_ctx;
        AVHWFramesContext *hwfc = NULL;

        if (avctx->hw_frames_ctx) {
            hwfc = (AVHWFramesContext *)avctx->hw_frames_ctx->data;
            device_ref = hwfc->device_ref;
        }

        r->hwdevice = av_buffer_ref(device_ref);
        if (!r->hwdevice) {
            ret = AVERROR(ENOMEM);
            goto fail;
        }
        av_log(avctx, AV_LOG_VERBOSE, "Picked up an existing RKMPP hardware device\n");
    } else {
        if ((ret = av_hwdevice_ctx_create(&r->hwdevice, AV_HWDEVICE_TYPE_RKMPP, NULL, NULL, 0)) < 0) {
            av_log(avctx, AV_LOG_ERROR, "Failed to create a RKMPP hardware device: %d\n", ret);
            goto fail;
        }
        av_log(avctx, AV_LOG_VERBOSE, "Created a RKMPP hardware device\n");
    }

    ret = init_hwframes_ctx(avctx);
    if (ret < 0)
        goto fail;

    return 0;

fail:
    if (mpp_pkt)
        mpp_packet_deinit(&mpp_pkt);

    return ret;
}

static const enum AVPixelFormat rkmpp_enc_pix_fmts[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NV21,
    AV_PIX_FMT_NV16,
    AV_PIX_FMT_NV24,
    AV_PIX_FMT_YUYV422,
    AV_PIX_FMT_YVYU422,
    AV_PIX_FMT_UYVY422,
    AV_PIX_FMT_RGB24,
    AV_PIX_FMT_BGR24,
    AV_PIX_FMT_RGBA,
    AV_PIX_FMT_RGB0,
    AV_PIX_FMT_BGRA,
    AV_PIX_FMT_BGR0,
    AV_PIX_FMT_ARGB,
    AV_PIX_FMT_0RGB,
    AV_PIX_FMT_ABGR,
    AV_PIX_FMT_0BGR,
    AV_PIX_FMT_DRM_PRIME,
    AV_PIX_FMT_NONE,
};

static const AVCodecHWConfigInternal *const rkmpp_enc_hw_configs[] = {
    HW_CONFIG_ENCODER_DEVICE(NONE,      RKMPP),
    HW_CONFIG_ENCODER_FRAMES(DRM_PRIME, RKMPP),
    HW_CONFIG_ENCODER_FRAMES(DRM_PRIME, DRM),
    NULL,
};

#if CONFIG_H264_RKMPP_ENCODER
DEFINE_RKMPP_ENCODER(h264, H264)
#endif
#if CONFIG_HEVC_RKMPP_ENCODER
DEFINE_RKMPP_ENCODER(hevc, HEVC)
#endif
