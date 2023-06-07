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

#include "libavutil/avassert.h"
#include "libavutil/hwcontext.h"
#include "libavutil/imgutils.h"
#include "libavutil/mastering_display_metadata.h"

#include "libavutil/mem.h"
#include "libavutil/time.h"

#include "amfenc.h"
#include "encode.h"
#include "internal.h"

#define PTS_PROP L"PtsProp"

static int amf_init_context(AVCodecContext *avctx)
{
    AMFEncContext *ctx = avctx->priv_data;
    AVAMFContext *amfctx = NULL;
    AMF_RESULT  res;
    int ret;

    ctx->dts_delay = 0;
    ctx->hwsurfaces_in_queue = 0;
    ctx->hwsurfaces_in_queue_max = 16;

    ctx->delayed_frame = av_frame_alloc();
    if (!ctx->delayed_frame)
        return AVERROR(ENOMEM);

    // hardcoded to current HW queue size - will auto-realloc if too small
    ctx->timestamp_list = av_fifo_alloc2(avctx->max_b_frames + 16, sizeof(int64_t),
                                         AV_FIFO_FLAG_AUTO_GROW);
    if (!ctx->timestamp_list)
        return AVERROR(ENOMEM);

    amfctx = av_mallocz(sizeof(AVAMFContext));
    if (!amfctx)
        return AVERROR(ENOMEM);

    ctx->amfctx = amfctx;
    amfctx->avclass = avctx;
    amfctx->log_to_dbg = ctx->log_to_dbg;

    ret = amf_load_library(amfctx);
    if (ret < 0)
        return ret;

    ret = amf_create_context(amfctx);
    if (ret < 0)
        return ret;

    // If a device was passed to the encoder, try to initialise from that.
    if (avctx->hw_frames_ctx) {
        AVHWFramesContext *frames_ctx = (AVHWFramesContext*)avctx->hw_frames_ctx->data;

        if (amf_av_to_amf_format(frames_ctx->sw_format) == AMF_SURFACE_UNKNOWN) {
            av_log(avctx, AV_LOG_ERROR, "Format of input frames context (%s) is not supported by AMF.\n",
                   av_get_pix_fmt_name(frames_ctx->sw_format));
            return AVERROR(EINVAL);
        }

        switch (frames_ctx->device_ctx->type) {
#if CONFIG_D3D11VA
        case AV_HWDEVICE_TYPE_D3D11VA:
            res = amf_context_derive_dx11(amfctx, frames_ctx->device_ctx->hwctx);
            if (res != AMF_OK)
                return res;
            break;
#endif
#if CONFIG_DXVA2
        case AV_HWDEVICE_TYPE_DXVA2:
            res = amf_context_derive_dx9(amfctx, frames_ctx->device_ctx->hwctx);
            if (res != AMF_OK)
                return res;
            break;
#endif
        default:
            av_log(avctx, AV_LOG_ERROR, "AMF initialisation from a %s frames context is not supported.\n",
                   av_hwdevice_get_type_name(frames_ctx->device_ctx->type));
            return AVERROR(ENOSYS);
        }

        ctx->hw_frames_ctx = av_buffer_ref(avctx->hw_frames_ctx);
        if (!ctx->hw_frames_ctx)
            return AVERROR(ENOMEM);

        if (frames_ctx->initial_pool_size > 0)
            ctx->hwsurfaces_in_queue_max = frames_ctx->initial_pool_size - 1;

    } else if (avctx->hw_device_ctx) {
        AVHWDeviceContext *device_ctx = (AVHWDeviceContext*)avctx->hw_device_ctx->data;

        switch (device_ctx->type) {
#if CONFIG_D3D11VA
        case AV_HWDEVICE_TYPE_D3D11VA:
            res = amf_context_derive_dx11(amfctx, device_ctx->hwctx);
            if (res != AMF_OK)
                return res;
            break;
#endif
#if CONFIG_DXVA2
        case AV_HWDEVICE_TYPE_DXVA2:
            res = amf_context_derive_dx9(amfctx, device_ctx->hwctx);
            if (res != AMF_OK)
                return res;
            break;
#endif
        default:
            av_log(avctx, AV_LOG_ERROR, "AMF initialisation from a %s device is not supported.\n",
                   av_hwdevice_get_type_name(device_ctx->type));
            return AVERROR(ENOSYS);
        }

        ctx->hw_device_ctx = av_buffer_ref(avctx->hw_device_ctx);
        if (!ctx->hw_device_ctx)
            return AVERROR(ENOMEM);

    } else {
#ifdef _WIN32
        res = amf_context_init_dx11(amfctx);
        if (res != AMF_OK) {
            res = amf_context_init_dx9(amfctx);
            if (res != AMF_OK) {
#endif
                res = amf_context_init_vulkan(amfctx);
                if (res != AMF_OK) {
                    av_log(avctx, AV_LOG_ERROR, "AMF initialisation is not supported.\n");
                    return AVERROR(ENOSYS);
                }
#ifdef _WIN32
            }
        }
#endif
    }

    return 0;
}

static int amf_check_hevc_encoder_10bit_support(AVCodecContext *avctx)
{
    AMFEncContext *ctx = avctx->priv_data;
    AVAMFContext  *amfctx = ctx->amfctx;
    const wchar_t *codec_id = AMFVideoEncoder_HEVC;
    AMF_RESULT     res;

    res = amfctx->factory->pVtbl->CreateComponent(amfctx->factory, amfctx->context, codec_id, &ctx->encoder);
    AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR_ENCODER_NOT_FOUND, "CreateComponent(%ls) failed with error %d\n", codec_id, res);

    AMF_ASSIGN_PROPERTY_INT64(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_COLOR_BIT_DEPTH, AMF_COLOR_BIT_DEPTH_10);
    AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR(EINVAL), "Assigning 10-bit property failed with error %d\n", res);

    res = ctx->encoder->pVtbl->Init(ctx->encoder, AMF_SURFACE_P010, avctx->width, avctx->height);
    if (res == AMF_OK) {
        ctx->encoder->pVtbl->Terminate(ctx->encoder);
        ctx->encoder->pVtbl->Release(ctx->encoder);
        ctx->encoder = NULL;
    } else {
        return AVERROR(EINVAL);
    }
    return res;
}

static int amf_init_encoder(AVCodecContext *avctx)
{
    AMFEncContext     *ctx = avctx->priv_data;
    AVAMFContext      *amfctx = ctx->amfctx;
    const wchar_t     *codec_id = NULL;
    enum AVPixelFormat pix_fmt;
    AMF_RESULT         res;

    switch (avctx->codec->id) {
        case AV_CODEC_ID_H264:
            codec_id = AMFVideoEncoderVCE_AVC;
            break;
        case AV_CODEC_ID_HEVC:
            codec_id = AMFVideoEncoder_HEVC;
            break;
        case AV_CODEC_ID_AV1 :
            codec_id = AMFVideoEncoder_AV1;
            break;
        default:
            break;
    }
    AMF_RETURN_IF_FALSE(avctx, codec_id != NULL,
        AVERROR(EINVAL), "Codec %d is not supported\n", avctx->codec->id);

    pix_fmt = avctx->hw_frames_ctx ? ((AVHWFramesContext*)avctx->hw_frames_ctx->data)->sw_format
                                   : avctx->pix_fmt;

    ctx->format = amf_av_to_amf_format(pix_fmt);
    AMF_RETURN_IF_FALSE(avctx, ctx->format != AMF_SURFACE_UNKNOWN,
        AVERROR(EINVAL), "Format %s is not supported\n", av_get_pix_fmt_name(pix_fmt));

    ctx->bit_depth = 8;
    if (pix_fmt == AV_PIX_FMT_P010) {
        switch (avctx->codec->id) {
        case AV_CODEC_ID_HEVC:
            // GPU >= Navi or APU >= Renoir is required.
            res = amf_check_hevc_encoder_10bit_support(avctx);
            if (res == AMF_OK) {
                ctx->bit_depth = 10;
            } else {
                av_log(avctx, AV_LOG_ERROR, "HEVC 10-bit encoding is not supported by the given AMF device\n");
                return res;
            }
            break;
        default:
            av_log(avctx, AV_LOG_ERROR, "10-bit encoding is not supported by AMF %s encoder\n", avctx->codec->name);
            return AVERROR(EINVAL);
        }
    }

    ctx->out_color_trc = amf_av_to_amf_color_trc(avctx->color_trc);
    ctx->out_color_prm = amf_av_to_amf_color_prm(avctx->color_primaries);

    switch (avctx->colorspace) {
        case AVCOL_SPC_BT470BG:
        case AVCOL_SPC_SMPTE170M:
        case AVCOL_SPC_SMPTE240M:
            ctx->out_color_profile = AMF_VIDEO_CONVERTER_COLOR_PROFILE_601;
            break;
        case AVCOL_SPC_BT709:
            ctx->out_color_profile = AMF_VIDEO_CONVERTER_COLOR_PROFILE_709;
            break;
        case AVCOL_SPC_BT2020_NCL:
        case AVCOL_SPC_BT2020_CL:
            ctx->out_color_profile = AMF_VIDEO_CONVERTER_COLOR_PROFILE_2020;
            break;
        case AVCOL_SPC_RGB:
            ctx->out_color_profile = AMF_VIDEO_CONVERTER_COLOR_PROFILE_JPEG;
            break;
        default:
            ctx->out_color_profile = AMF_VIDEO_CONVERTER_COLOR_PROFILE_UNKNOWN;
            break;
    }

    res = amfctx->factory->pVtbl->CreateComponent(amfctx->factory, amfctx->context, codec_id, &ctx->encoder);
    AMF_RETURN_IF_FALSE(avctx, res == AMF_OK,
        AVERROR_ENCODER_NOT_FOUND, "CreateComponent(%ls) failed with error %d\n", codec_id, res);

    return 0;
}

av_cold int ff_amf_encode_close(AVCodecContext *avctx)
{
    AMFEncContext *ctx = avctx->priv_data;
    AVAMFContext  *amfctx = ctx->amfctx;

    if (ctx->delayed_surface) {
        ctx->delayed_surface->pVtbl->Release(ctx->delayed_surface);
        ctx->delayed_surface = NULL;
    }

    if (ctx->encoder) {
        ctx->encoder->pVtbl->Terminate(ctx->encoder);
        ctx->encoder->pVtbl->Release(ctx->encoder);
        ctx->encoder = NULL;
    }

    amf_unload_library(amfctx);
    if (amfctx)
        av_freep(&amfctx);

    ctx->delayed_drain = 0;
    av_buffer_unref(&ctx->hw_device_ctx);
    av_buffer_unref(&ctx->hw_frames_ctx);

    av_frame_free(&ctx->delayed_frame);
    av_fifo_freep2(&ctx->timestamp_list);
    return 0;
}

av_cold int ff_amf_encode_init(AVCodecContext *avctx)
{
    int ret;

    if ((ret = amf_init_context(avctx)) == 0)
        if ((ret = amf_init_encoder(avctx)) == 0)
            return 0;

    ff_amf_encode_close(avctx);
    return ret;
}

static int amf_copy_surface(AVCodecContext *avctx,
                            const AVFrame *frame,
                            AMFSurface* surface)
{
    AMFPlane *plane;
    uint8_t  *dst_data[4];
    int       dst_linesize[4];
    int       planes;
    int       i;

    planes = surface->pVtbl->GetPlanesCount(surface);
    av_assert0(planes < FF_ARRAY_ELEMS(dst_data));

    for (i = 0; i < planes; i++) {
        plane = surface->pVtbl->GetPlaneAt(surface, i);
        dst_data[i] = plane->pVtbl->GetNative(plane);
        dst_linesize[i] = plane->pVtbl->GetHPitch(plane);
    }

    av_image_copy(dst_data, dst_linesize,
        (const uint8_t**)frame->data, frame->linesize, frame->format,
        avctx->width, avctx->height);
    return 0;
}

static int amf_copy_buffer(AVCodecContext *avctx,
                           AVPacket *pkt,
                           AMFBuffer *buffer)
{
    AMFEncContext   *ctx = avctx->priv_data;
    int              ret;
    AMFVariantStruct var = { 0 };
    int64_t          timestamp = AV_NOPTS_VALUE;
    int64_t          size = buffer->pVtbl->GetSize(buffer);

    if ((ret = ff_get_encode_buffer(avctx, pkt, size, 0)) < 0)
        return ret;
    memcpy(pkt->data, buffer->pVtbl->GetNative(buffer), size);

    switch (avctx->codec->id) {
        case AV_CODEC_ID_H264:
            buffer->pVtbl->GetProperty(buffer, AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE, &var);
            if (var.int64Value == AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE_IDR)
                pkt->flags = AV_PKT_FLAG_KEY;
            break;
        case AV_CODEC_ID_HEVC:
            buffer->pVtbl->GetProperty(buffer, AMF_VIDEO_ENCODER_HEVC_OUTPUT_DATA_TYPE, &var);
            if (var.int64Value == AMF_VIDEO_ENCODER_HEVC_OUTPUT_DATA_TYPE_IDR)
                pkt->flags = AV_PKT_FLAG_KEY;
            break;
        case AV_CODEC_ID_AV1:
            buffer->pVtbl->GetProperty(buffer, AMF_VIDEO_ENCODER_AV1_OUTPUT_FRAME_TYPE, &var);
            if (var.int64Value == AMF_VIDEO_ENCODER_AV1_OUTPUT_FRAME_TYPE_KEY) {
                pkt->flags = AV_PKT_FLAG_KEY;
            }
        default:
            break;
    }

    buffer->pVtbl->GetProperty(buffer, PTS_PROP, &var);

    pkt->pts = var.int64Value; // original pts

    AMF_RETURN_IF_FALSE(ctx, av_fifo_read(ctx->timestamp_list, &timestamp, 1) >= 0, AVERROR_UNKNOWN, "timestamp_list is empty\n");

    // calc dts shift if max_b_frames > 0
    if (avctx->max_b_frames > 0 && ctx->dts_delay == 0) {
        int64_t timestamp_last = AV_NOPTS_VALUE;
        size_t can_read = av_fifo_can_read(ctx->timestamp_list);

        AMF_RETURN_IF_FALSE(ctx, can_read > 0, AVERROR_UNKNOWN, AVERROR_UNKNOWN, "timestamp_list is empty while max_b_frames = %d\n", avctx->max_b_frames);

        av_fifo_peek(ctx->timestamp_list, &timestamp_last, 1, can_read - 1);
        if (timestamp < 0 || timestamp_last < AV_NOPTS_VALUE)
            return AVERROR(ERANGE);
        ctx->dts_delay = timestamp_last - timestamp;
    }
    pkt->dts = timestamp - ctx->dts_delay;
    return 0;
}

static AMF_RESULT amf_set_property_buffer(AMFSurface *object,
                                          const wchar_t *name,
                                          AMFBuffer *val)
{
    AMF_RESULT res;
    AMFVariantStruct var;
    res = AMFVariantInit(&var);
    if (res == AMF_OK) {
        AMFGuid guid_AMFInterface = IID_AMFInterface();
        AMFInterface *amf_interface;
        res = val->pVtbl->QueryInterface(val, &guid_AMFInterface, (void**)&amf_interface);

        if (res == AMF_OK) {
            res = AMFVariantAssignInterface(&var, amf_interface);
            amf_interface->pVtbl->Release(amf_interface);
        }
        if (res == AMF_OK)
            res = object->pVtbl->SetProperty(object, name, var);
        AMFVariantClear(&var);
    }
    return res;
}

static AMF_RESULT amf_get_property_buffer(AMFData *object,
                                          const wchar_t *name,
                                          AMFBuffer **val)
{
    AMF_RESULT res;
    AMFVariantStruct var;
    res = AMFVariantInit(&var);
    if (res == AMF_OK) {
        res = object->pVtbl->GetProperty(object, name, &var);
        if (res == AMF_OK) {
            if (var.type == AMF_VARIANT_INTERFACE) {
                AMFGuid guid_AMFBuffer = IID_AMFBuffer();
                AMFInterface *amf_interface = AMFVariantInterface(&var);
                res = amf_interface->pVtbl->QueryInterface(amf_interface, &guid_AMFBuffer, (void**)val);
            } else {
                res = AMF_INVALID_DATA_TYPE;
            }
        }
        AMFVariantClear(&var);
    }
    return res;
}

static AMFBuffer *amf_create_buffer_with_frame_ref(const AVFrame *frame, AMFContext *context)
{
    AVFrame *frame_ref;
    AMFBuffer *frame_ref_storage_buffer = NULL;
    AMF_RESULT res;

    res = context->pVtbl->AllocBuffer(context, AMF_MEMORY_HOST, sizeof(frame_ref), &frame_ref_storage_buffer);
    if (res == AMF_OK) {
        frame_ref = av_frame_clone(frame);
        if (frame_ref) {
            memcpy(frame_ref_storage_buffer->pVtbl->GetNative(frame_ref_storage_buffer), &frame_ref, sizeof(frame_ref));
        } else {
            frame_ref_storage_buffer->pVtbl->Release(frame_ref_storage_buffer);
            frame_ref_storage_buffer = NULL;
        }
    }
    return frame_ref_storage_buffer;
}

static void amf_release_buffer_with_frame_ref(AMFBuffer *frame_ref_storage_buffer)
{
    AVFrame *frame_ref;
    memcpy(&frame_ref, frame_ref_storage_buffer->pVtbl->GetNative(frame_ref_storage_buffer), sizeof(frame_ref));
    av_frame_free(&frame_ref);
    frame_ref_storage_buffer->pVtbl->Release(frame_ref_storage_buffer);
}

static int amf_save_hdr_metadata(AVCodecContext *avctx, const AVFrame *frame, AMFHDRMetadata *hdrmeta)
{
    AVFrameSideData            *sd_display;
    AVFrameSideData            *sd_light;
    AVMasteringDisplayMetadata *display_meta;
    AVContentLightMetadata     *light_meta;

    sd_display = av_frame_get_side_data(frame, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    if (sd_display) {
        display_meta = (AVMasteringDisplayMetadata *)sd_display->data;
        if (display_meta->has_luminance) {
            const unsigned int luma_den = 10000;
            hdrmeta->maxMasteringLuminance =
                (amf_uint32)(luma_den * av_q2d(display_meta->max_luminance));
            hdrmeta->minMasteringLuminance =
                FFMIN((amf_uint32)(luma_den * av_q2d(display_meta->min_luminance)), hdrmeta->maxMasteringLuminance);
        }
        if (display_meta->has_primaries) {
            const unsigned int chroma_den = 50000;
            hdrmeta->redPrimary[0] =
                FFMIN((amf_uint16)(chroma_den * av_q2d(display_meta->display_primaries[0][0])), chroma_den);
            hdrmeta->redPrimary[1] =
                FFMIN((amf_uint16)(chroma_den * av_q2d(display_meta->display_primaries[0][1])), chroma_den);
            hdrmeta->greenPrimary[0] =
                FFMIN((amf_uint16)(chroma_den * av_q2d(display_meta->display_primaries[1][0])), chroma_den);
            hdrmeta->greenPrimary[1] =
                FFMIN((amf_uint16)(chroma_den * av_q2d(display_meta->display_primaries[1][1])), chroma_den);
            hdrmeta->bluePrimary[0] =
                FFMIN((amf_uint16)(chroma_den * av_q2d(display_meta->display_primaries[2][0])), chroma_den);
            hdrmeta->bluePrimary[1] =
                FFMIN((amf_uint16)(chroma_den * av_q2d(display_meta->display_primaries[2][1])), chroma_den);
            hdrmeta->whitePoint[0] =
                FFMIN((amf_uint16)(chroma_den * av_q2d(display_meta->white_point[0])), chroma_den);
            hdrmeta->whitePoint[1] =
                FFMIN((amf_uint16)(chroma_den * av_q2d(display_meta->white_point[1])), chroma_den);
        }

        sd_light = av_frame_get_side_data(frame, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
        if (sd_light) {
            light_meta = (AVContentLightMetadata *)sd_light->data;
            if (light_meta) {
                hdrmeta->maxContentLightLevel = (amf_uint16)light_meta->MaxCLL;
                hdrmeta->maxFrameAverageLightLevel = (amf_uint16)light_meta->MaxFALL;
            }
        }
        return 0;
    }
    return 1;
}

int ff_amf_receive_packet(AVCodecContext *avctx, AVPacket *avpkt)
{
    AMFEncContext *ctx = avctx->priv_data;
    AVAMFContext *amfctx = ctx->amfctx;
    AMFSurface *surface;
    AMF_RESULT  res;
    int         ret;
    AMF_RESULT  res_query;
    AMFData    *data = NULL;
    AVFrame    *frame = ctx->delayed_frame;
    int         block_and_wait;

    if (!ctx->encoder)
        return AVERROR(EINVAL);

    if (!frame->buf[0]) {
        ret = ff_encode_get_frame(avctx, frame);
        if (ret < 0 && ret != AVERROR_EOF)
            return ret;
    }

    if (!frame->buf[0]) { // submit drain
        if (!ctx->eof) { // submit drain one time only
            if (ctx->delayed_surface != NULL) {
                ctx->delayed_drain = 1; // input queue is full: resubmit Drain() in ff_amf_receive_packet
            } else if(!ctx->delayed_drain) {
                res = ctx->encoder->pVtbl->Drain(ctx->encoder);
                if (res == AMF_INPUT_FULL) {
                    ctx->delayed_drain = 1; // input queue is full: resubmit Drain() in ff_amf_receive_packet
                } else {
                    if (res == AMF_OK)
                        ctx->eof = 1; // drain started
                    AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR_UNKNOWN, "Drain() failed with error %d\n", res);
                }
            }
        }
    } else if (!ctx->delayed_surface) { // submit frame
        int hw_surface = 0;

        // prepare surface from frame
        switch (frame->format) {
#if CONFIG_D3D11VA
        case AV_PIX_FMT_D3D11:
            {
                static const GUID AMFTextureArrayIndexGUID = { 0x28115527, 0xe7c3, 0x4b66, { 0x99, 0xd3, 0x4f, 0x2a, 0xe6, 0xb4, 0x7f, 0xaf } };
                ID3D11Texture2D *texture = (ID3D11Texture2D*)frame->data[0]; // actual texture
                int index = (intptr_t)frame->data[1]; // index is a slice in texture array is - set to tell AMF which slice to use

                texture->lpVtbl->SetPrivateData(texture, &AMFTextureArrayIndexGUID, sizeof(index), &index);

                res = amfctx->context->pVtbl->CreateSurfaceFromDX11Native(amfctx->context, texture, &surface, NULL); // wrap to AMF surface
                AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR(ENOMEM), "CreateSurfaceFromDX11Native() failed with error %d\n", res);

                hw_surface = 1;
            }
            break;
#endif
#if CONFIG_DXVA2
        case AV_PIX_FMT_DXVA2_VLD:
            {
                IDirect3DSurface9 *texture = (IDirect3DSurface9 *)frame->data[3]; // actual texture

                res = amfctx->context->pVtbl->CreateSurfaceFromDX9Native(amfctx->context, texture, &surface, NULL); // wrap to AMF surface
                AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR(ENOMEM), "CreateSurfaceFromDX9Native() failed with error %d\n", res);

                hw_surface = 1;
            }
            break;
#endif
        default:
            {
                res = amfctx->context->pVtbl->AllocSurface(amfctx->context, AMF_MEMORY_HOST, ctx->format, avctx->width, avctx->height, &surface);
                AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR(ENOMEM), "AllocSurface() failed with error %d\n", res);
                amf_copy_surface(avctx, frame, surface);
            }
            break;
        }

        if (hw_surface) {
            AMFBuffer *frame_ref_storage_buffer;

            // input HW surfaces can be vertically aligned by 16; tell AMF the real size
            surface->pVtbl->SetCrop(surface, 0, 0, frame->width, frame->height);

            frame_ref_storage_buffer = amf_create_buffer_with_frame_ref(frame, amfctx->context);
            AMF_RETURN_IF_FALSE(avctx, frame_ref_storage_buffer != NULL, AVERROR(ENOMEM), "create_buffer_with_frame_ref() returned NULL\n");

            res = amf_set_property_buffer(surface, L"av_frame_ref", frame_ref_storage_buffer);
            AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR_UNKNOWN, "SetProperty failed for \"av_frame_ref\" with error %d\n", res);

            ctx->hwsurfaces_in_queue++;
            frame_ref_storage_buffer->pVtbl->Release(frame_ref_storage_buffer);
        }

        // HDR10 metadata
        if (frame->color_trc == AVCOL_TRC_SMPTE2084) {
            AMFBuffer *hdrmeta_buffer = NULL;
            res = amfctx->context->pVtbl->AllocBuffer(amfctx->context, AMF_MEMORY_HOST, sizeof(AMFHDRMetadata), &hdrmeta_buffer);
            if (res == AMF_OK) {
                AMFHDRMetadata *hdrmeta = (AMFHDRMetadata*)hdrmeta_buffer->pVtbl->GetNative(hdrmeta_buffer);
                if (amf_save_hdr_metadata(avctx, frame, hdrmeta) == 0) {
                    switch (avctx->codec->id) {
                    case AV_CODEC_ID_H264:
                        AMF_ASSIGN_PROPERTY_INTERFACE(res, ctx->encoder, AMF_VIDEO_ENCODER_INPUT_HDR_METADATA, hdrmeta_buffer); break;
                    case AV_CODEC_ID_HEVC:
                        AMF_ASSIGN_PROPERTY_INTERFACE(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_INPUT_HDR_METADATA, hdrmeta_buffer); break;
                    case AV_CODEC_ID_AV1:
                        AMF_ASSIGN_PROPERTY_INTERFACE(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_INPUT_HDR_METADATA, hdrmeta_buffer); break;
                    }
                    res = amf_set_property_buffer(surface, L"av_frame_hdrmeta", hdrmeta_buffer);
                    AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR_UNKNOWN, "SetProperty failed for \"av_frame_hdrmeta\" with error %d\n", res);
                }
                hdrmeta_buffer->pVtbl->Release(hdrmeta_buffer);
            }
        }

        surface->pVtbl->SetPts(surface, frame->pts);
        AMF_ASSIGN_PROPERTY_INT64(res, surface, PTS_PROP, frame->pts);

        switch (avctx->codec->id) {
        case AV_CODEC_ID_H264:
            AMF_ASSIGN_PROPERTY_INT64(res, surface, AMF_VIDEO_ENCODER_INSERT_SPS, 1);
            AMF_ASSIGN_PROPERTY_INT64(res, surface, AMF_VIDEO_ENCODER_INSERT_PPS, 1);
            AMF_ASSIGN_PROPERTY_INT64(res, surface, AMF_VIDEO_ENCODER_INSERT_AUD, !!ctx->aud); break;
        case AV_CODEC_ID_HEVC:
            AMF_ASSIGN_PROPERTY_INT64(res, surface, AMF_VIDEO_ENCODER_HEVC_INSERT_AUD, !!ctx->aud); break;
        //case AV_CODEC_ID_AV1 not supported
        }

        // submit surface
        res = ctx->encoder->pVtbl->SubmitInput(ctx->encoder, (AMFData*)surface);
        if (res == AMF_INPUT_FULL) { // handle full queue
            //store surface for later submission
            ctx->delayed_surface = surface;
        } else {
            int64_t pts = frame->pts;
            surface->pVtbl->Release(surface);
            AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR_UNKNOWN, "SubmitInput() failed with error %d\n", res);

            av_frame_unref(frame);
            ret = av_fifo_write(ctx->timestamp_list, &pts, 1);
            if (ret < 0)
                return ret;
        }
    }

    do {
        block_and_wait = 0;
        // poll data
        res_query = ctx->encoder->pVtbl->QueryOutput(ctx->encoder, &data);
        if (data) {
            // copy data to packet
            AMFBuffer* buffer;
            AMFGuid guid = IID_AMFBuffer();
            data->pVtbl->QueryInterface(data, &guid, (void**)&buffer); // query for buffer interface
            ret = amf_copy_buffer(avctx, avpkt, buffer);

            buffer->pVtbl->Release(buffer);

            if (data->pVtbl->HasProperty(data, L"av_frame_ref")) {
                AMFBuffer *frame_ref_storage_buffer;
                res = amf_get_property_buffer(data, L"av_frame_ref", &frame_ref_storage_buffer);
                AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR_UNKNOWN, "GetProperty failed for \"av_frame_ref\" with error %d\n", res);
                amf_release_buffer_with_frame_ref(frame_ref_storage_buffer);
                ctx->hwsurfaces_in_queue--;
            }

            data->pVtbl->Release(data);

            AMF_RETURN_IF_FALSE(avctx, ret >= 0, ret, "amf_copy_buffer() failed with error %d\n", ret);

            if (ctx->delayed_surface != NULL) { // try to resubmit frame
                if (ctx->delayed_surface->pVtbl->HasProperty(ctx->delayed_surface, L"av_frame_hdrmeta")) {
                    AMFBuffer *hdrmeta_buffer = NULL;
                    res = amf_get_property_buffer((AMFData *)ctx->delayed_surface, L"av_frame_hdrmeta", &hdrmeta_buffer);
                    AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR_UNKNOWN, "GetProperty failed for \"av_frame_hdrmeta\" with error %d\n", res);
                    switch (avctx->codec->id) {
                    case AV_CODEC_ID_H264:
                        AMF_ASSIGN_PROPERTY_INTERFACE(res, ctx->encoder, AMF_VIDEO_ENCODER_INPUT_HDR_METADATA, hdrmeta_buffer); break;
                    case AV_CODEC_ID_HEVC:
                        AMF_ASSIGN_PROPERTY_INTERFACE(res, ctx->encoder, AMF_VIDEO_ENCODER_HEVC_INPUT_HDR_METADATA, hdrmeta_buffer); break;
                    case AV_CODEC_ID_AV1:
                        AMF_ASSIGN_PROPERTY_INTERFACE(res, ctx->encoder, AMF_VIDEO_ENCODER_AV1_INPUT_HDR_METADATA, hdrmeta_buffer); break;
                    }
                    hdrmeta_buffer->pVtbl->Release(hdrmeta_buffer);
                }
                res = ctx->encoder->pVtbl->SubmitInput(ctx->encoder, (AMFData*)ctx->delayed_surface);
                if (res != AMF_INPUT_FULL) {
                    int64_t pts = ctx->delayed_surface->pVtbl->GetPts(ctx->delayed_surface);
                    ctx->delayed_surface->pVtbl->Release(ctx->delayed_surface);
                    ctx->delayed_surface = NULL;
                    av_frame_unref(ctx->delayed_frame);
                    AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR_UNKNOWN, "Repeated SubmitInput() failed with error %d\n", res);

                    ret = av_fifo_write(ctx->timestamp_list, &pts, 1);
                    if (ret < 0)
                        return ret;
                } else {
                    av_log(avctx, AV_LOG_WARNING, "Data acquired but delayed frame submission got AMF_INPUT_FULL- should not happen\n");
                }
            } else if (ctx->delayed_drain) { // try to resubmit drain
                res = ctx->encoder->pVtbl->Drain(ctx->encoder);
                if (res != AMF_INPUT_FULL) {
                    ctx->delayed_drain = 0;
                    ctx->eof = 1; // drain started
                    AMF_RETURN_IF_FALSE(avctx, res == AMF_OK, AVERROR_UNKNOWN, "Repeated Drain() failed with error %d\n", res);
                } else {
                    av_log(avctx, AV_LOG_WARNING, "Data acquired but delayed drain submission got AMF_INPUT_FULL- should not happen\n");
                }
            }
        } else if (ctx->delayed_surface != NULL ||
                   ctx->delayed_drain ||
                   (ctx->eof && res_query != AMF_EOF) ||
                   (ctx->hwsurfaces_in_queue >= ctx->hwsurfaces_in_queue_max)) {
            block_and_wait = 1;
            av_usleep(1000); // wait and poll again
        }
    } while (block_and_wait);

    if (res_query == AMF_EOF)
        ret = AVERROR_EOF;
    else if (data == NULL)
        ret = AVERROR(EAGAIN);
    else
        ret = 0;
    return ret;
}

const AVCodecHWConfigInternal *const ff_amfenc_hw_configs[] = {
#if CONFIG_D3D11VA
    HW_CONFIG_ENCODER_FRAMES(D3D11, D3D11VA),
    HW_CONFIG_ENCODER_DEVICE(NONE,  D3D11VA),
#endif
#if CONFIG_DXVA2
    HW_CONFIG_ENCODER_FRAMES(DXVA2_VLD, DXVA2),
    HW_CONFIG_ENCODER_DEVICE(NONE,      DXVA2),
#endif
    NULL,
};
