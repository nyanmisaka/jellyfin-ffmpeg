/*
 * Copyright (c) 2017 Lionel CHAZALLON
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
 * Rockchip MPP (Media Process Platform) video decoder
 */

#include "config_components.h"

#include <drm_fourcc.h>
#include <rockchip/rk_mpi.h>

#include "codec_internal.h"
#include "decode.h"
#include "hwconfig.h"
#include "internal.h"

#include "libavutil/hwcontext_rkmpp.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

typedef struct RKMPPDecodeContext {
    AVClass       *class;

    MppApi        *mapi;
    MppCtx         mctx;
    MppBufferGroup buf_group;

    AVBufferRef   *hwdevice;
    AVBufferRef   *hwframe;

    int64_t        pts_step;
    int64_t        pts;
    AVPacket       last_pkt;
    AVFrame        last_frame;

    int            fast_mode;
    int            afbc_mode;
} RKMPPDecodeContext;

static MppCodingType rkmpp_get_coding_type(AVCodecContext *avctx)
{
    switch (avctx->codec_id) {
    case AV_CODEC_ID_H263:          return MPP_VIDEO_CodingH263;
    case AV_CODEC_ID_H264:          return MPP_VIDEO_CodingAVC;
    case AV_CODEC_ID_HEVC:          return MPP_VIDEO_CodingHEVC;
    case AV_CODEC_ID_AV1:           return MPP_VIDEO_CodingAV1;
    case AV_CODEC_ID_VP8:           return MPP_VIDEO_CodingVP8;
    case AV_CODEC_ID_VP9:           return MPP_VIDEO_CodingVP9;
    case AV_CODEC_ID_MPEG1VIDEO:    /* fallthrough */
    case AV_CODEC_ID_MPEG2VIDEO:    return MPP_VIDEO_CodingMPEG2;
    case AV_CODEC_ID_MPEG4:         return MPP_VIDEO_CodingMPEG4;
    default:                        return MPP_VIDEO_CodingUnused;
    }
}

static uint32_t rkmpp_get_drm_format(MppFrameFormat mpp_format)
{
    switch (mpp_format & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP:          return DRM_FORMAT_NV12;
    case MPP_FMT_YUV420SP_10BIT:    return DRM_FORMAT_NV15;
    case MPP_FMT_YUV422SP:          return DRM_FORMAT_NV16;
    case MPP_FMT_YUV422SP_10BIT:    return DRM_FORMAT_NV20;
    default:                        return DRM_FORMAT_INVALID;
    }
}

static uint32_t rkmpp_get_drm_afbc_format(MppFrameFormat mpp_format)
{
    switch (mpp_format & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP:          return DRM_FORMAT_YUV420_8BIT;
    case MPP_FMT_YUV420SP_10BIT:    return DRM_FORMAT_YUV420_10BIT;
    case MPP_FMT_YUV422SP:          return DRM_FORMAT_YUYV;
    case MPP_FMT_YUV422SP_10BIT:    return DRM_FORMAT_Y210;
    default:                        return DRM_FORMAT_INVALID;
    }
}

static uint32_t rkmpp_get_av_format(MppFrameFormat mpp_format)
{
    switch (mpp_format & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP:          return AV_PIX_FMT_NV12;
    case MPP_FMT_YUV420SP_10BIT:    return AV_PIX_FMT_NV15;
    case MPP_FMT_YUV422SP:          return AV_PIX_FMT_NV16;
    case MPP_FMT_YUV422SP_10BIT:    return AV_PIX_FMT_NV20;
    default:                        return AV_PIX_FMT_NONE;
    }
}

static av_cold int rkmpp_decode_close(AVCodecContext *avctx)
{
    RKMPPDecodeContext *r = avctx->priv_data;

    if (r->mapi) {
        r->mapi->reset(r->mctx);
        mpp_destroy(r->mctx);
        r->mctx = NULL;
    }
    if (r->buf_group) {
        mpp_buffer_group_put(r->buf_group);
        r->buf_group = NULL;
    }

    if (r->hwframe)
        av_buffer_unref(&r->hwframe);
    if (r->hwdevice)
        av_buffer_unref(&r->hwdevice);

    return 0;
}

static av_cold int rkmpp_decode_init(AVCodecContext *avctx)
{
    RKMPPDecodeContext *r = avctx->priv_data;
    MppCodingType coding_type = MPP_VIDEO_CodingUnused;
    int ret;
    enum AVPixelFormat pix_fmts[3] = { AV_PIX_FMT_DRM_PRIME,
                                       AV_PIX_FMT_NV12,
                                       AV_PIX_FMT_NONE };

    if ((ret = ff_get_format(avctx, pix_fmts)) < 0) {
        av_log(avctx, AV_LOG_ERROR, "ff_get_format failed: %d\n", ret);
        return ret;
    }
    avctx->pix_fmt = ret;

    if ((coding_type = rkmpp_get_coding_type(avctx)) == MPP_VIDEO_CodingUnused) {
        av_log(avctx, AV_LOG_ERROR, "Unknown codec id: %d\n", avctx->codec_id);
        return AVERROR(ENOSYS);
    }

    if ((ret = mpp_check_support_format(MPP_CTX_DEC, coding_type)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "MPP doesn't support codec '%s' (%d)\n",
               avcodec_get_name(avctx->codec_id), avctx->codec_id);
        return AVERROR(ENOSYS);
    }

    if ((ret = mpp_create(&r->mctx, &r->mapi)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create MPP context and api: %d\n", ret);
        ret = AVERROR_EXTERNAL;
        goto fail;
    }

    if ((ret = mpp_init(r->mctx, MPP_CTX_DEC, coding_type)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to init MPP context: %d\n", ret);
        ret = AVERROR_EXTERNAL;
        goto fail;
    }

    if (r->afbc_mode) {
        MppFrameFormat afbc_fmt = MPP_FRAME_FBC_AFBC_V2;
        if ((ret = r->mapi->control(r->mctx, MPP_DEC_SET_OUTPUT_FORMAT, &afbc_fmt)) != MPP_OK) {
            av_log(avctx, AV_LOG_ERROR, "Failed to set AFBC mode: %d\n", ret);
            ret = AVERROR_EXTERNAL;
            goto fail;
        }
    }

    if ((ret = r->mapi->control(r->mctx, MPP_DEC_SET_PARSER_FAST_MODE, &r->fast_mode)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to set parser fast mode: %d\n", ret);
        ret = AVERROR_EXTERNAL;
        goto fail;
    }
#if 0
    if ((ret = r->mapi->control(r->mctx, MPP_DEC_SET_DISABLE_ERROR, NULL)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to set disable error: %d\n", ret);
        ret = AVERROR_EXTERNAL;
        goto fail;
    }
#endif
    if (avctx->hw_device_ctx) {
        r->hwdevice = av_buffer_ref(avctx->hw_device_ctx);
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

    return 0;

fail:
    rkmpp_decode_close(avctx);
    return ret;
}

static int rkmpp_set_buffer_group(AVCodecContext *avctx,
                                  enum AVPixelFormat pix_fmt,
                                  int width, int height)
{
    RKMPPDecodeContext *r = avctx->priv_data;
    AVHWFramesContext *hwfc = NULL;
    int i, ret;

    if (!r->hwdevice)
        return AVERROR(ENOMEM);

    av_buffer_unref(&r->hwframe);

    r->hwframe = av_hwframe_ctx_alloc(r->hwdevice);
    if (!r->hwframe)
        return AVERROR(ENOMEM);

    hwfc = (AVHWFramesContext *)r->hwframe->data;
    hwfc->format    = AV_PIX_FMT_DRM_PRIME;
    hwfc->sw_format = pix_fmt;
    hwfc->width     = FFALIGN(width,  16);
    hwfc->height    = FFALIGN(height, 16);

    switch (avctx->codec_id) {
    case AV_CODEC_ID_H264:
    case AV_CODEC_ID_HEVC:
        hwfc->initial_pool_size = 20 + 3;
        break;
    default:
        hwfc->initial_pool_size = 10 + 3;
        break;
    }

    if (avctx->extra_hw_frames > 0)
        hwfc->initial_pool_size += avctx->extra_hw_frames;

    if ((ret = av_hwframe_ctx_init(r->hwframe)) < 0) {
        av_log(avctx, AV_LOG_ERROR, "Failed to init RKMPP frame pool\n");
        goto fail;
    }

    if (r->buf_group) {
        if ((ret = mpp_buffer_group_clear(r->buf_group)) != MPP_OK) {
            av_log(avctx, AV_LOG_ERROR, "Failed to clear external buffer group: %d\n", ret);
            ret = AVERROR_EXTERNAL;
            goto fail;
        }
    } else {
        if ((ret = mpp_buffer_group_get_external(&r->buf_group, MPP_BUFFER_TYPE_DRM)) != MPP_OK) {
            av_log(avctx, AV_LOG_ERROR, "Failed to get external buffer group: %d\n", ret);
            ret = AVERROR_EXTERNAL;
            goto fail;
        }
    }

    for (i = 0; i < hwfc->initial_pool_size; i++) {
        AVRKMPPFramesContext *rkmpp_fc = hwfc->hwctx;
        MppBufferInfo buf_info = {
            .index = i,
            .type  = MPP_BUFFER_TYPE_DRM,
            .fd    = rkmpp_fc->frames[i].objects[0].fd,
            .size  = rkmpp_fc->frames[i].objects[0].size,
        };

        if ((ret = mpp_buffer_commit(r->buf_group, &buf_info)) != MPP_OK) {
            av_log(avctx, AV_LOG_ERROR, "Failed to commit external buffer group: %d\n", ret);
            ret = AVERROR_EXTERNAL;
            goto fail;
        }
    }

    if ((ret = r->mapi->control(r->mctx, MPP_DEC_SET_EXT_BUF_GROUP, r->buf_group)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to assign external buffer group: %d\n", ret);
        ret = AVERROR_EXTERNAL;
        goto fail;
    }

    return 0;

fail:
    av_buffer_unref(&r->hwframe);
    return ret;
}

static void rkmpp_free_mpp_frame(void *opaque, uint8_t *data)
{
    MppFrame mpp_frame = (MppFrame)opaque;
    mpp_frame_deinit(&mpp_frame);
}

static void rkmpp_free_drm_desc(void *opaque, uint8_t *data)
{
    AVDRMFrameDescriptor *drm_desc = (AVDRMFrameDescriptor *)opaque;
    av_free(drm_desc);
}

static int frame_create_buf(AVFrame *frame,
                            uint8_t* data, int size,
                            void (*free)(void *opaque, uint8_t *data),
                            void *opaque, int flags)
{
    int i;

    for (i = 0; i < AV_NUM_DATA_POINTERS; i++) {
        if (!frame->buf[i]) {
            frame->buf[i] = av_buffer_create(data, size, free, opaque, flags);
            return frame->buf[i] ? 0 : AVERROR(ENOMEM);
        }
    }
    return AVERROR(EINVAL);
}

static int rkmpp_export_frame(AVCodecContext *avctx, AVFrame *frame, MppFrame mpp_frame)
{
    RKMPPDecodeContext *r = avctx->priv_data;
    AVDRMFrameDescriptor *desc = NULL;
    AVDRMLayerDescriptor *layer = NULL;
    MppBuffer mpp_buf = NULL;
    int ret;

    if (!frame || !mpp_frame)
        return AVERROR(ENOMEM);

    mpp_buf = mpp_frame_get_buffer(mpp_frame);
    if (!mpp_buf)
        return AVERROR(EAGAIN);

    desc = av_mallocz(sizeof(*desc));
    if (!desc)
        return AVERROR(ENOMEM);

    desc->nb_objects = 1;
    desc->objects[0].fd   = mpp_buffer_get_fd(mpp_buf);
    desc->objects[0].ptr  = mpp_buffer_get_ptr(mpp_buf);
    desc->objects[0].size = mpp_buffer_get_size(mpp_buf);

    if (r->afbc_mode)
        desc->objects[0].format_modifier =
            DRM_FORMAT_MOD_ARM_AFBC(AFBC_FORMAT_MOD_SPARSE | AFBC_FORMAT_MOD_BLOCK_SIZE_16x16);

    desc->nb_layers = 1;
    layer = &desc->layers[0];
    layer->format = r->afbc_mode ? rkmpp_get_drm_afbc_format(mpp_frame_get_fmt(mpp_frame))
                                 : rkmpp_get_drm_format(mpp_frame_get_fmt(mpp_frame));

    layer->nb_planes = r->afbc_mode ? 1 : 2;
    layer->planes[0].object_index = 0;
    layer->planes[0].offset =
        r->afbc_mode ? mpp_frame_get_offset_y(mpp_frame) * mpp_frame_get_hor_stride(mpp_frame) : 0;
    layer->planes[0].pitch = mpp_frame_get_hor_stride(mpp_frame);

    layer->planes[1].object_index = 0;
    layer->planes[1].offset = layer->planes[0].pitch * mpp_frame_get_ver_stride(mpp_frame);
    layer->planes[1].pitch = layer->planes[0].pitch;

    if ((ret = frame_create_buf(frame, mpp_frame, mpp_frame_get_buf_size(mpp_frame),
                                rkmpp_free_mpp_frame, mpp_frame, AV_BUFFER_FLAG_READONLY)) < 0)
        return ret;

    if ((ret = frame_create_buf(frame, (uint8_t *)desc, sizeof(*desc),
                                rkmpp_free_drm_desc, desc, AV_BUFFER_FLAG_READONLY)) < 0)
        return ret;

    frame->data[0] = (uint8_t *)desc;

    frame->hw_frames_ctx = av_buffer_ref(r->hwframe);
    if (!frame->hw_frames_ctx)
        return AVERROR(ENOMEM);

    if ((ret = ff_decode_frame_props(avctx, frame)) < 0)
        return ret;

    frame->width  = avctx->width;
    frame->height = avctx->height;
    frame->pts    = mpp_frame_get_pts(mpp_frame);
    {
        int val = mpp_frame_get_mode(mpp_frame);
        frame->interlaced_frame = (val & MPP_FRAME_FLAG_FIELD_ORDER_MASK) == MPP_FRAME_FLAG_DEINTERLACED;
        frame->top_field_first  = (val & MPP_FRAME_FLAG_FIELD_ORDER_MASK) == MPP_FRAME_FLAG_TOP_FIRST;
    }

    return 0;
}

static void rkmpp_export_frame_props(AVCodecContext *avctx, MppFrame mpp_frame)
{
    int val;

    if (!avctx || !mpp_frame)
        return;

    avctx->color_primaries = AVCOL_PRI_UNSPECIFIED;
    if ((val = mpp_frame_get_color_primaries(mpp_frame)) != MPP_FRAME_PRI_RESERVED0)
        avctx->color_primaries = val;

    avctx->color_trc = AVCOL_TRC_UNSPECIFIED;
    if ((val = mpp_frame_get_color_trc(mpp_frame)) != MPP_FRAME_TRC_RESERVED0)
        avctx->color_trc = val;

    avctx->colorspace = AVCOL_SPC_UNSPECIFIED;
    if ((val = mpp_frame_get_colorspace(mpp_frame)) != MPP_FRAME_SPC_RESERVED)
        avctx->colorspace = val;

    avctx->color_range = AVCOL_RANGE_UNSPECIFIED;
    if ((val = mpp_frame_get_color_range(mpp_frame)) > 0)
        avctx->color_range = val;

    avctx->chroma_sample_location = AVCHROMA_LOC_UNSPECIFIED;
    if ((val = mpp_frame_get_chroma_location(mpp_frame)) > 0)
        avctx->chroma_sample_location = val;
}

static int rkmpp_get_frame(AVCodecContext *avctx, AVFrame *frame, int timeout)
{
    RKMPPDecodeContext *r = avctx->priv_data;
    MppFrame mpp_frame = NULL;
    int ret;

    if ((ret = r->mapi->control(r->mctx, MPP_SET_OUTPUT_TIMEOUT, (MppParam)&timeout)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to set output timeout: %d\n", ret);
        return AVERROR_EXTERNAL;
    }

    ret = r->mapi->decode_get_frame(r->mctx, &mpp_frame);
    if (ret != MPP_OK && ret != MPP_ERR_TIMEOUT) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get frame: %d\n", ret);
        return AVERROR_EXTERNAL;
    }
    if (!mpp_frame) {
        av_log(avctx, AV_LOG_DEBUG, "Timeout getting decoded frame\n");
        return AVERROR(EAGAIN);
    }
    if (mpp_frame_get_eos(mpp_frame)) {
        av_log(avctx, AV_LOG_DEBUG, "Received a 'EOS' frame\n");
        ret = AVERROR_EOF;
        goto exit;
    }
    if (mpp_frame_get_discard(mpp_frame)) {
        av_log(avctx, AV_LOG_DEBUG, "Received a 'discard' frame\n");
        ret = AVERROR(EAGAIN);
        goto exit;
    }
    if (mpp_frame_get_errinfo(mpp_frame)) {
        av_log(avctx, AV_LOG_DEBUG, "Received a 'errinfo' frame\n");
        ret = AVERROR(EAGAIN);
        goto exit;
    }

    if (mpp_frame_get_info_change(mpp_frame)) {
        const MppFrameFormat mpp_fmt = mpp_frame_get_fmt(mpp_frame) & MPP_FRAME_FMT_MASK;
        enum AVPixelFormat pix_fmts[3] = { AV_PIX_FMT_DRM_PRIME,
                                           AV_PIX_FMT_NONE,
                                           AV_PIX_FMT_NONE };

        av_log(avctx, AV_LOG_VERBOSE, "Noticed an info change\n");

        pix_fmts[1] = rkmpp_get_av_format(mpp_fmt);
        if ((ret = ff_get_format(avctx, pix_fmts)) < 0)
            goto exit;

        avctx->pix_fmt      = ret;
        avctx->width        = mpp_frame_get_width(mpp_frame);
        avctx->height       = mpp_frame_get_height(mpp_frame);
        avctx->coded_width  = FFALIGN(avctx->width,  64);
        avctx->coded_height = FFALIGN(avctx->height, 64);
        rkmpp_export_frame_props(avctx, mpp_frame);

        av_log(avctx, AV_LOG_VERBOSE, "size: %dx%d | pix_fmt: %s | sw_pix_fmt: %s\n",
               avctx->width, avctx->height,
               av_get_pix_fmt_name(avctx->pix_fmt),
               av_get_pix_fmt_name(avctx->sw_pix_fmt));

        if ((ret = rkmpp_set_buffer_group(avctx, pix_fmts[1], avctx->width, avctx->height)) < 0)
            goto exit;

        if ((ret = r->mapi->control(r->mctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL)) != MPP_OK) {
            av_log(avctx, AV_LOG_ERROR, "Failed to set info change ready: %d\n", ret);
            ret = AVERROR_EXTERNAL;
            goto exit;
        }
        goto exit;
    } else {
        av_log(avctx, AV_LOG_DEBUG, "Received a frame\n");

        switch (avctx->pix_fmt) {
        case AV_PIX_FMT_DRM_PRIME:
            {
                if ((ret = rkmpp_export_frame(avctx, frame, mpp_frame)) < 0)
                    goto exit;
                return 0;
            }
            break;
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_NV16:
        case AV_PIX_FMT_NV15:
        case AV_PIX_FMT_NV20:
            {
                AVFrame *tmp_frame = av_frame_alloc();
                if (!tmp_frame) {
                    ret = AVERROR(ENOMEM);
                    goto exit;
                }
                if ((ret = rkmpp_export_frame(avctx, tmp_frame, mpp_frame)) < 0)
                    goto exit;

                if ((ret = ff_get_buffer(avctx, frame, 0)) < 0) {
                    av_log(avctx, AV_LOG_ERROR, "ff_get_buffer failed: %d\n", ret);
                    av_frame_free(&tmp_frame);
                    goto exit;
                }
                if ((ret = av_hwframe_transfer_data(frame, tmp_frame, 0)) < 0) {
                    av_log(avctx, AV_LOG_ERROR, "av_hwframe_transfer_data failed: %d\n", ret);
                    av_frame_free(&tmp_frame);
                    goto exit;
                }
                if ((ret = av_frame_copy_props(frame, tmp_frame)) < 0) {
                    av_log(avctx, AV_LOG_ERROR, "av_frame_copy_props failed: %d\n", ret);
                    av_frame_free(&tmp_frame);
                    goto exit;
                }
                av_frame_free(&tmp_frame);
                return 0;
            }
            break;
        default:
            {
                ret = AVERROR_BUG;
                goto exit;
            }
            break;
        }
    }

exit:
    if (mpp_frame)
        mpp_frame_deinit(&mpp_frame);
    return ret;
}

static int rkmpp_send_eos(AVCodecContext *avctx)
{
    RKMPPDecodeContext *r = avctx->priv_data;
    MppPacket mpkt;
    int ret;

    if ((ret = mpp_packet_init(&mpkt, NULL, 0)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to init 'EOS' packet: %d\n", ret);
        return AVERROR_EXTERNAL;
    }
    mpp_packet_set_eos(mpkt);

    do {
        ret = r->mapi->decode_put_packet(r->mctx, mpkt);
    } while (ret != MPP_OK);

    mpp_packet_deinit(&mpkt);
    return 0;
}

static int rkmpp_send_packet(AVCodecContext *avctx, AVPacket *pkt)
{
    RKMPPDecodeContext *r = avctx->priv_data;
    MppPacket mpkt;
    int64_t pts = pkt->pts;
    int ret;
#if 0
    /* generate alternative pts */
    if (pts == AV_NOPTS_VALUE || pts < 0) {
        if (!r->pts_step && avctx->framerate.den && avctx->framerate.num) {
            int64_t x = avctx->pkt_timebase.den * (int64_t)avctx->framerate.den;
            int64_t y = avctx->pkt_timebase.num * (int64_t)avctx->framerate.num;
            r->pts_step = x / y;
        }
        if (r->pts_step && (pkt->dts == AV_NOPTS_VALUE || pkt->dts < 0)) {
            pts = r->pts;
            r->pts += r->pts_step;
        } else {
            r->pts = pkt->dts;
            pts = pkt->dts;
        }
    }
#endif
    if ((ret = mpp_packet_init(&mpkt, pkt->data, pkt->size)) != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to init packet: %d\n", ret);
        return AVERROR_EXTERNAL;
    }
    mpp_packet_set_pts(mpkt, pts);

    if ((ret = r->mapi->decode_put_packet(r->mctx, mpkt)) != MPP_OK) {
        av_log(avctx, AV_LOG_TRACE, "Decoder buffer is full\n");
        mpp_packet_deinit(&mpkt);
        return AVERROR(EAGAIN);
    }
    av_log(avctx, AV_LOG_DEBUG, "Wrote %d bytes to decoder\n", pkt->size);

    mpp_packet_deinit(&mpkt);
    return 0;
}

static int rkmpp_decode_receive_frame(AVCodecContext *avctx, AVFrame *frame)
{
    AVCodecInternal *avci = avctx->internal;
    RKMPPDecodeContext *r = avctx->priv_data;
    AVPacket *pkt = &r->last_pkt;
    int ret_send, ret_get;

    if (!avci->draining) {
        if (!pkt->size) {
            switch (ff_decode_get_packet(avctx, pkt)) {
            case AVERROR_EOF:
                av_log(avctx, AV_LOG_DEBUG, "Decoder draining\n");
                return rkmpp_send_eos(avctx);
            case AVERROR(EAGAIN):
                av_log(avctx, AV_LOG_TRACE, "Decoder could not get packet, retrying\n");
                return AVERROR(EAGAIN);
            }
        }
send_pkt:
        /* there is definitely a packet to send to decoder */
        ret_send = rkmpp_send_packet(avctx, pkt);
        if (ret_send == 0) {
            /* send successful, continue until decoder input buffer is full */
            av_packet_unref(pkt);
            return AVERROR(EAGAIN);
        } else if (ret_send < 0 && ret_send != AVERROR(EAGAIN)) {
            /* something went wrong, raise error */
            av_log(avctx, AV_LOG_ERROR, "Decoder failed to send data: %d", ret_send);
            return ret_send;
        }
    }

    /* were here only when draining and buffer is full */
    ret_get = rkmpp_get_frame(avctx, frame, 100);
    if (ret_get == AVERROR_EOF)
        av_log(avctx, AV_LOG_DEBUG, "Decoder is at EOF\n");
    /* this is not likely but lets handle it in case synchronization issues of MPP */
    else if (ret_get == AVERROR(EAGAIN) && ret_send == AVERROR(EAGAIN))
        goto send_pkt;
    else if (ret_get < 0 && ret_get != AVERROR(EAGAIN)) /* FIXME */
        av_log(avctx, AV_LOG_ERROR, "Decoder failed to get frame: %d\n", ret_get);

    return ret_get;
}

static void rkmpp_decode_flush(AVCodecContext *avctx)
{
    RKMPPDecodeContext *r = avctx->priv_data;
    int ret;

    if ((ret = r->mapi->reset(r->mctx)) != MPP_OK)
        av_log(avctx, AV_LOG_ERROR, "Failed to reset MPP context: %d\n", ret);

    av_packet_unref(&r->last_pkt);
    av_frame_unref(&r->last_frame);
}

static const AVCodecHWConfigInternal *const rkmpp_decoder_hw_configs[] = {
    &(const AVCodecHWConfigInternal) {
        .public = {
            .pix_fmt     = AV_PIX_FMT_DRM_PRIME,
            .methods     = AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX |
                           AV_CODEC_HW_CONFIG_METHOD_INTERNAL,
            .device_type = AV_HWDEVICE_TYPE_RKMPP,
        },
        .hwaccel = NULL,
    },
    NULL
};

#define OFFSET(x) offsetof(RKMPPDecodeContext, x)
#define VD (AV_OPT_FLAG_DECODING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)

static const AVOption options[] = {
    { "fast_mode", "Enable fast parsing to improve decoding parallelism", OFFSET(fast_mode), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, VD },
    { "afbc_mode", "Enable AFBC (Arm Frame Buffer Compression) to save bandwidth", OFFSET(afbc_mode), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, VD },
    { NULL }
};

#define DEFINE_RKMPP_DECODER(x, X, bsf_name) \
static const AVClass x##_rkmpp_decoder_class = { \
    .class_name = #x "_rkmpp_decoder", \
    .item_name  = av_default_item_name, \
    .option     = options, \
    .version    = LIBAVUTIL_VERSION_INT, \
}; \
const FFCodec ff_##x##_rkmpp_decoder = { \
    .p.name         = #x "_rkmpp", \
    CODEC_LONG_NAME("Rockchip MPP (Media Process Platform) " #X " decoder"), \
    .p.type         = AVMEDIA_TYPE_VIDEO, \
    .p.id           = AV_CODEC_ID_##X, \
    .priv_data_size = sizeof(RKMPPDecodeContext), \
    .p.priv_class   = &x##_rkmpp_decoder_class, \
    .init           = rkmpp_decode_init, \
    .close          = rkmpp_decode_close, \
    FF_CODEC_RECEIVE_FRAME_CB(rkmpp_decode_receive_frame), \
    .flush          = rkmpp_decode_flush, \
    .bsfs           = bsf_name, \
    .p.capabilities = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_AVOID_PROBING | AV_CODEC_CAP_HARDWARE, \
    .caps_internal  = FF_CODEC_CAP_NOT_INIT_THREADSAFE | \
                      FF_CODEC_CAP_SETS_FRAME_PROPS, \
    .p.pix_fmts     = (const enum AVPixelFormat[]){ AV_PIX_FMT_DRM_PRIME, \
                                                    AV_PIX_FMT_NV12, \
                                                    AV_PIX_FMT_NV16, \
                                                    AV_PIX_FMT_NV15, \
                                                    AV_PIX_FMT_NV20, \
                                                    AV_PIX_FMT_NONE }, \
    .hw_configs     = rkmpp_decoder_hw_configs, \
    .p.wrapper_name = "rkmpp", \
};

#if CONFIG_H263_RKMPP_DECODER
DEFINE_RKMPP_DECODER(h263, H263, NULL)
#endif
#if CONFIG_H264_RKMPP_DECODER
DEFINE_RKMPP_DECODER(h264, H264, "h264_mp4toannexb")
#endif
#if CONFIG_HEVC_RKMPP_DECODER
DEFINE_RKMPP_DECODER(hevc, HEVC, "hevc_mp4toannexb")
#endif
#if CONFIG_VP8_RKMPP_DECODER
DEFINE_RKMPP_DECODER(vp8, VP8, NULL)
#endif
#if CONFIG_VP9_RKMPP_DECODER
DEFINE_RKMPP_DECODER(vp9, VP9, NULL)
#endif
#if CONFIG_AV1_RKMPP_DECODER
DEFINE_RKMPP_DECODER(av1, AV1, NULL)
#endif
#if CONFIG_MPEG1_RKMPP_DECODER
DEFINE_RKMPP_DECODER(mpeg1, MPEG1VIDEO, NULL)
#endif
#if CONFIG_MPEG2_RKMPP_DECODER
DEFINE_RKMPP_DECODER(mpeg2, MPEG2VIDEO, NULL)
#endif
#if CONFIG_MPEG4_RKMPP_DECODER
DEFINE_RKMPP_DECODER(mpeg4, MPEG4, "mpeg4_unpack_bframes")
#endif
