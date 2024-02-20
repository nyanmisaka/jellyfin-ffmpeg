/*
 * Copyright (C) 2024 Gnattu OC <gnattuoc@me.com>
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

#include <CoreImage/CoreImage.h>
#include <VideoToolbox/VideoToolbox.h>
#include "internal.h"
#include "metal/utils.h"
#include "framesync.h"
#include "libavutil/hwcontext.h"
#include "libavutil/pixdesc.h"
#include "libavutil/opt.h"
#include "libavutil/objc.h"
#include "video.h"

#include <assert.h>

extern char ff_vf_overlay_videotoolbox_metallib_data[];
extern unsigned int ff_vf_overlay_videotoolbox_metallib_len;

// Although iOS 8.0 introduced basic Metal support, its feature set is not complete and does not have CoreImage compatability.
// We have to set the minimum iOS version to 9.0.
typedef struct API_AVAILABLE(macos(10.11), ios(9.0)) OverlayVideoToolboxContext {
    AVBufferRef *device_ref;
    FFFrameSync fs;
    CVMetalTextureCacheRef textureCache;
    CVPixelBufferRef inputMainPixelBufferCache;
    CVPixelBufferRef outputPixelBufferCache;
    CVPixelBufferRef inputOverlayPixelBufferCache;
    CIContext *coreImageCtx;
    VTPixelTransferSessionRef vtSession;

    id<MTLDevice> mtlDevice;
    id<MTLLibrary> mtlLibrary;
    id<MTLCommandQueue> mtlQueue;
    id<MTLComputePipelineState> mtlPipeline;
    id<MTLFunction> mtlFunction;
    id<MTLBuffer> mtlParamsBuffer;

    int output_configured;
    uint x_position;
    uint y_position;
    enum AVPixelFormat output_format;

} OverlayVideoToolboxContext API_AVAILABLE(macos(10.11), ios(9.0));

struct mtlBlendParams {
    uint x_position;
    uint y_position;
};

// Using sizeof(OverlayVideoToolboxContext) without an availability check will error
// if we're targeting an older OS version, so we need to calculate the size ourselves
// (we'll statically verify it's correct in overlay_videotoolbox_init behind a check)
#define OVERLAY_VT_CTX_SIZE (sizeof(FFFrameSync) + sizeof(int) * 1 + sizeof(uint) * 2 + sizeof(void*) * 13 + sizeof(enum AVPixelFormat))

static void call_kernel(AVFilterContext *avctx,
                        id<MTLTexture> dst,
                        id<MTLTexture> main,
                        id<MTLTexture> overlay,
                        uint x_position,
                        uint y_position) API_AVAILABLE(macos(10.11), ios(9.0))
{
    OverlayVideoToolboxContext *ctx = avctx->priv;
    // Both the command buffer and encoder are auto-released by objc on default.
    // Use CFBridgingRetain to get a more C-like behavior.
    id<MTLCommandBuffer> buffer = CFBridgingRetain(ctx->mtlQueue.commandBuffer);
    id<MTLComputeCommandEncoder> encoder = CFBridgingRetain((__bridge id<MTLCommandBuffer>)buffer.computeCommandEncoder);

    struct mtlBlendParams *params = (struct mtlBlendParams *)ctx->mtlParamsBuffer.contents;
    *params = (struct mtlBlendParams){
        .x_position = x_position,
        .y_position = y_position,
    };
    [(__bridge id<MTLComputeCommandEncoder>)encoder setTexture:main atIndex:0];
    [(__bridge id<MTLComputeCommandEncoder>)encoder setTexture:overlay atIndex:1];
    [(__bridge id<MTLComputeCommandEncoder>)encoder setTexture:dst atIndex:2];
    [(__bridge id<MTLComputeCommandEncoder>)encoder setBuffer:ctx->mtlParamsBuffer offset:0 atIndex:3];
    ff_metal_compute_encoder_dispatch(ctx->mtlDevice, ctx->mtlPipeline, (__bridge id<MTLComputeCommandEncoder>)encoder, dst.width, dst.height);
    [(__bridge id<MTLComputeCommandEncoder>)encoder endEncoding];
    [(__bridge id<MTLCommandBuffer>)buffer commit];
    [(__bridge id<MTLCommandBuffer>)buffer waitUntilCompleted];
    ff_objc_release(&encoder);
    ff_objc_release(&buffer);
}

static int transfer_pixel_buffer(OverlayVideoToolboxContext *ctx, CVPixelBufferRef source, CVPixelBufferRef destination) {
    if (@available(macOS 10.8, iOS 16.0, *)) {
        int ret = 0;
        ret = VTPixelTransferSessionTransferImage(ctx->vtSession,source ,destination);
        if (ret < 0)
            return ret;
    } else {
        CIImage *temp_image = NULL;
        temp_image = CFBridgingRetain([CIImage imageWithCVPixelBuffer: source]);
        [(__bridge CIContext*)ctx->coreImageCtx render: (__bridge CIImage*)temp_image toCVPixelBuffer: destination];
        CFRelease(temp_image);
        CVBufferPropagateAttachments(source, destination);
    }
    return 0;
}

static int overlay_vt_blend(FFFrameSync *fs) API_AVAILABLE(macos(10.11), ios(9.0))
{
    AVFilterContext *avctx = fs->parent;
    OverlayVideoToolboxContext *ctx = avctx->priv;
    AVFilterLink *outlink = avctx->outputs[0];
    AVFilterLink *inlink = avctx->inputs[0];
    AVFilterLink *inlink_overlay = avctx->inputs[1];
    AVFrame *input_main, *input_overlay;
    AVFrame *output;
    AVHWFramesContext *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVHWFramesContext *frames_ctx_overlay = (AVHWFramesContext*)inlink_overlay->hw_frames_ctx->data;
    const AVPixFmtDescriptor *in_main_desc, *in_overlay_desc;

    CVMetalTextureRef main, dst, overlay;
    id<MTLTexture> tex_main, tex_overlay, tex_dst;

    MTLPixelFormat mtl_format = MTLPixelFormatBGRA8Unorm;
    OSType cv_format = kCVPixelFormatType_32BGRA;
    int ret;
    int i, overlay_planes = 0;
    in_main_desc = av_pix_fmt_desc_get(frames_ctx->sw_format);
    in_overlay_desc = av_pix_fmt_desc_get(frames_ctx_overlay->sw_format);
    if (in_main_desc->comp[0].depth >= 10) {
        if (@available(macOS 11.3, iOS 14.2, *)) {
            mtl_format = MTLPixelFormatRGBA16Unorm;
            cv_format = kCVPixelFormatType_64RGBALE;
        } else {
            mtl_format = MTLPixelFormatRGBA16Float;
            cv_format = kCVPixelFormatType_64RGBAHalf;
        }
    }
    // read main and overlay frames from inputs
    ret = ff_framesync_get_frame(fs, 0, &input_main, 0);
    if (ret < 0)
        return ret;
    ret = ff_framesync_get_frame(fs, 1, &input_overlay, 0);
    if (ret < 0)
        return ret;
    if (!input_main)
        return AVERROR_BUG;
    output = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    ret = av_frame_copy_props(output, input_main);
    if (ret < 0)
        return ret;
    if (!input_overlay) {
        transfer_pixel_buffer(ctx, (CVPixelBufferRef)input_main->data[3], (CVPixelBufferRef)output->data[3]);
        return ff_filter_frame(outlink, output);
    }
    for (i = 0; i < in_overlay_desc->nb_components; i++)
        overlay_planes = FFMAX(overlay_planes,
                               in_overlay_desc->comp[i].plane + 1);
    // We need to convert input overlay when it is planer or the color depth does not match
    if (overlay_planes > 1 || in_main_desc->comp[0].depth != in_overlay_desc->comp[0].depth) {
        if (!ctx->inputOverlayPixelBufferCache) {
            ret = CVPixelBufferCreate(kCFAllocatorDefault,
                                      CVPixelBufferGetWidthOfPlane((CVPixelBufferRef)input_overlay->data[3], 0),
                                      CVPixelBufferGetHeightOfPlane((CVPixelBufferRef)input_overlay->data[3], 0),
                                      cv_format,
                                      (__bridge CFDictionaryRef)@{
                                          (NSString *)kCVPixelBufferCGImageCompatibilityKey: @(YES),
                                          (NSString *)kCVPixelBufferMetalCompatibilityKey: @(YES)
                                      },
                                      &ctx->inputOverlayPixelBufferCache);
            if (ret < 0)
                return ret;
        }
        transfer_pixel_buffer(ctx, (CVPixelBufferRef)input_overlay->data[3], ctx->inputOverlayPixelBufferCache);
        overlay = ff_metal_texture_from_pixbuf(avctx, ctx->textureCache, ctx->inputOverlayPixelBufferCache, 0, mtl_format);
    } else {
        overlay = ff_metal_texture_from_pixbuf(avctx, ctx->textureCache, (CVPixelBufferRef)input_overlay->data[3], 0, mtl_format);
    }
    if (!ctx->inputMainPixelBufferCache) {
        ret = CVPixelBufferCreate(kCFAllocatorDefault,
                                  CVPixelBufferGetWidthOfPlane((CVPixelBufferRef)input_main->data[3], 0),
                                  CVPixelBufferGetHeightOfPlane((CVPixelBufferRef)input_main->data[3], 0),
                                  cv_format,
                                  (__bridge CFDictionaryRef)@{
                                      (NSString *)kCVPixelBufferCGImageCompatibilityKey: @(YES),
                                      (NSString *)kCVPixelBufferMetalCompatibilityKey: @(YES)
                                  },
                                  &ctx->inputMainPixelBufferCache);
        if (ret < 0)
            return ret;
    }
    if (!ctx->outputPixelBufferCache) {
        ret = CVPixelBufferCreate(kCFAllocatorDefault,
                                  CVPixelBufferGetWidthOfPlane((CVPixelBufferRef)input_main->data[3], 0),
                                  CVPixelBufferGetHeightOfPlane((CVPixelBufferRef)input_main->data[3], 0),
                                  cv_format,
                                  (__bridge CFDictionaryRef)@{
                                      (NSString *)kCVPixelBufferCGImageCompatibilityKey: @(YES),
                                      (NSString *)kCVPixelBufferMetalCompatibilityKey: @(YES)
                                  },
                                  &ctx->outputPixelBufferCache);
        if (ret < 0)
            return ret;
    }
    transfer_pixel_buffer(ctx, (CVPixelBufferRef)input_main->data[3], ctx->inputMainPixelBufferCache);
    main = ff_metal_texture_from_pixbuf(avctx, ctx->textureCache, ctx->inputMainPixelBufferCache, 0, mtl_format);
    dst = ff_metal_texture_from_pixbuf(avctx, ctx->textureCache, ctx->outputPixelBufferCache, 0, mtl_format);
    tex_main = CVMetalTextureGetTexture(main);
    tex_overlay  = CVMetalTextureGetTexture(overlay);
    tex_dst = CVMetalTextureGetTexture(dst);
    call_kernel(avctx, tex_dst, tex_main, tex_overlay, ctx->x_position, ctx->y_position);
    transfer_pixel_buffer(ctx, ctx->outputPixelBufferCache, (CVPixelBufferRef)output->data[3]);
    CFRelease(main);
    CFRelease(overlay);
    CFRelease(dst);

    return ff_filter_frame(outlink, output);
}

static av_cold void do_uninit(AVFilterContext *avctx) API_AVAILABLE(macos(10.11), ios(9.0))
{
    OverlayVideoToolboxContext *ctx = avctx->priv;
    if(ctx->coreImageCtx) {
        CFRelease(ctx->coreImageCtx);
        ctx->coreImageCtx = NULL;
    }
    if (ctx->output_configured) {
        av_buffer_unref(&ctx->device_ref);
    }

    ff_objc_release(&ctx->mtlParamsBuffer);
    ff_objc_release(&ctx->mtlFunction);
    ff_objc_release(&ctx->mtlPipeline);
    ff_objc_release(&ctx->mtlQueue);
    ff_objc_release(&ctx->mtlLibrary);
    ff_objc_release(&ctx->mtlDevice);

    if (ctx->textureCache) {
        CFRelease(ctx->textureCache);
        ctx->textureCache = NULL;
    }
    if (ctx->inputMainPixelBufferCache) {
        CFRelease(ctx->inputMainPixelBufferCache);
        ctx->inputMainPixelBufferCache = NULL;
    }
    if (ctx->inputOverlayPixelBufferCache) {
        CFRelease(ctx->inputOverlayPixelBufferCache);
        ctx->inputOverlayPixelBufferCache = NULL;
    }
    if (ctx->outputPixelBufferCache) {
        CFRelease(ctx->outputPixelBufferCache);
        ctx->outputPixelBufferCache = NULL;
    }
    if(ctx->vtSession) {
        VTPixelTransferSessionInvalidate(ctx->vtSession);
        CFRelease(ctx->vtSession);
        ctx->vtSession = NULL;
    }
    ff_framesync_uninit(&ctx->fs);
}

static av_cold void overlay_videotoolbox_uninit(AVFilterContext *ctx)
{
    if (@available(macOS 10.11, iOS 9.0, *)) {
        do_uninit(ctx);
    }
}

static av_cold int do_init(AVFilterContext *ctx) API_AVAILABLE(macos(10.11), ios(9.0))
{
    OverlayVideoToolboxContext *s = ctx->priv;
    NSError *err = nil;
    CVReturn ret;
    dispatch_data_t libData;

    s->mtlDevice = MTLCreateSystemDefaultDevice();
    if (!s->mtlDevice) {
        av_log(ctx, AV_LOG_ERROR, "Unable to find Metal device\n");
        goto fail;
    }

    av_log(ctx, AV_LOG_INFO, "Using Metal device: %s\n", s->mtlDevice.name.UTF8String);

    libData = dispatch_data_create(
        ff_vf_overlay_videotoolbox_metallib_data,
        ff_vf_overlay_videotoolbox_metallib_len,
        nil,
        nil);

    s->mtlLibrary = [s->mtlDevice newLibraryWithData:libData error:&err];
    dispatch_release(libData);
    libData = nil;
    s->mtlFunction = [s->mtlLibrary newFunctionWithName:@"blend_shader"];
    if (!s->mtlFunction) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create Metal function!\n");
        goto fail;
    }

    s->mtlQueue = s->mtlDevice.newCommandQueue;
    if (!s->mtlQueue) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create Metal command queue!\n");
        goto fail;
    }

    s->mtlPipeline = [s->mtlDevice
        newComputePipelineStateWithFunction:s->mtlFunction
        error:&err];
    if (err) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create Metal compute pipeline: %s\n", err.description.UTF8String);
        goto fail;
    }

    s->mtlParamsBuffer = [s->mtlDevice
        newBufferWithLength:sizeof(struct mtlBlendParams)
        options:MTLResourceStorageModeShared];
    if (!s->mtlParamsBuffer) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create Metal buffer for parameters\n");
        goto fail;
    }

    ret = CVMetalTextureCacheCreate(
        NULL,
        NULL,
        s->mtlDevice,
        NULL,
        &s->textureCache
    );
    if (ret != kCVReturnSuccess) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create CVMetalTextureCache: %d\n", ret);
        goto fail;
    }

    if (@available(macOS 10.8, iOS 16.0, *)) {
        ret = VTPixelTransferSessionCreate(NULL, &s->vtSession);
        if (ret < 0)
            return ret;
    } else {
        // Use CoreImage as fallback for old OS.
        // CoreImage has comparable performance to VTPixelTransferSession, but it supports less pixel formats than VTPixelTransferSession.
        // Warn user about possible incorrect results.
        av_log(ctx, AV_LOG_WARNING, "VTPixelTransferSessionTransferImage is not available on this OS version, fallback using CoreImage\n");
        av_log(ctx, AV_LOG_WARNING, "Try an overlay with BGRA format if you see no overlay\n");
        if (@available(macOS 10.15, iOS 13.0, *)) {
            s->coreImageCtx = CFBridgingRetain([CIContext contextWithMTLCommandQueue: s->mtlQueue]);
        } else {
            s->coreImageCtx = CFBridgingRetain([CIContext contextWithMTLDevice: s->mtlDevice]);
        }
    }
    s->fs.on_event = &overlay_vt_blend;
    s->output_format = AV_PIX_FMT_NONE;

    return 0;
fail:
    overlay_videotoolbox_uninit(ctx);
    return AVERROR_EXTERNAL;
}

static av_cold int overlay_videotoolbox_init(AVFilterContext *ctx)
{
    if (@available(macOS 10.11, iOS 9.0, *)) {
        // Ensure we calculated OVERLAY_VT_CTX_SIZE correctly
        static_assert(OVERLAY_VT_CTX_SIZE == sizeof(OverlayVideoToolboxContext), "Incorrect OVERLAY_VT_CTX_SIZE value!");
        return do_init(ctx);
    } else {
        av_log(ctx, AV_LOG_ERROR, "Metal is not available on this OS version\n");
        return AVERROR(ENOSYS);
    }
}

static int do_config_input(AVFilterLink *inlink) API_AVAILABLE(macos(10.11), ios(9.0))
{
    AVFilterContext *avctx = inlink->dst;
    OverlayVideoToolboxContext *ctx = avctx->priv;
    AVBufferRef *input_ref;
    AVHWFramesContext *input_frames;

    if (!inlink->hw_frames_ctx) {
        av_log(avctx, AV_LOG_ERROR, "A hardware frames reference is "
               "required to associate the processing device.\n");
        return AVERROR(EINVAL);
    }
    input_ref = av_buffer_ref(inlink->hw_frames_ctx);
    input_frames = (AVHWFramesContext*)input_ref->data;
    av_assert0(input_frames);
    ctx->device_ref = av_buffer_ref(input_frames->device_ref);

    if (!ctx->device_ref) {
        av_log(ctx, AV_LOG_ERROR, "A device reference create "
                                  "failed.\n");
        return AVERROR(ENOMEM);
    }
    if (ctx->output_format == AV_PIX_FMT_NONE)
        ctx->output_format = input_frames->sw_format;
    ctx->output_configured = 1;

    return 0;
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    if (@available(macOS 10.11, iOS 9.0, *)) {
        return do_config_input(inlink);
    } else {
        av_log(ctx, AV_LOG_ERROR, "Metal is not available on this OS version\n");
        return AVERROR(ENOSYS);
    }
}

static int do_config_output(AVFilterLink *link) API_AVAILABLE(macos(10.11), ios(9.0))
{
    AVHWFramesContext *output_frames;
    AVFilterContext *avctx = link->src;
    OverlayVideoToolboxContext *ctx = avctx->priv;
    int ret = 0;

    link->hw_frames_ctx = av_hwframe_ctx_alloc(ctx->device_ref);
    link->time_base = avctx->inputs[0]->time_base;
    ctx->fs.time_base = link->time_base;
    if (!link->hw_frames_ctx) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create HW frame context "
               "for output.\n");
        ret = AVERROR(ENOMEM);
        return ret;
    }

    output_frames = (AVHWFramesContext*)link->hw_frames_ctx->data;

    output_frames->format    = AV_PIX_FMT_VIDEOTOOLBOX;
    output_frames->sw_format = ctx->output_format;
    output_frames->width     = avctx->inputs[0]->w;
    output_frames->height    = avctx->inputs[0]->h;

    ret = ff_filter_init_hw_frames(avctx, link, 10);
    if (ret < 0)
        return ret;

    ret = av_hwframe_ctx_init(link->hw_frames_ctx);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "Failed to initialise VideoToolbox frame "
               "context for output: %d\n", ret);
        return ret;
    }

    ret = ff_framesync_init_dualinput(&ctx->fs, avctx);
    if (ret < 0)
        return ret;

    ret = ff_framesync_configure(&ctx->fs);
    return ret;
}

static int config_output(AVFilterLink *link)
{
    AVFilterContext *ctx = link->src;
    if (@available(macOS 10.11, iOS 9.0, *)) {
        return do_config_output(link);
    } else {
        av_log(ctx, AV_LOG_ERROR, "Metal is not available on this OS version\n");
        return AVERROR(ENOSYS);
    }
}

static int overlay_videotoolbox_activate(AVFilterContext *avctx) {
    OverlayVideoToolboxContext *ctx = avctx->priv;
    return ff_framesync_activate(&ctx->fs);
}

#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
#define CONST(name, help, val, unit) { name, help, 0, AV_OPT_TYPE_CONST, {.i64=val}, INT_MIN, INT_MAX, FLAGS, unit }
#define OFFSET(x) offsetof(OverlayVideoToolboxContext, x)

static const AVOption overlay_videotoolbox_options[] = {
    { "x", "Overlay x position",
        OFFSET(x_position), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, .flags = FLAGS },
    { "y", "Overlay y position",
        OFFSET(y_position), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, .flags = FLAGS },
    { "eof_action", "Action to take when encountering EOF from secondary input ",
        OFFSET(fs.opt_eof_action), AV_OPT_TYPE_INT, { .i64 = EOF_ACTION_REPEAT },
        EOF_ACTION_REPEAT, EOF_ACTION_PASS, .flags = FLAGS, .unit = "eof_action" },
        { "repeat", "Repeat the previous frame.",   0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_REPEAT }, .flags = FLAGS, .unit = "eof_action" },
        { "endall", "End both streams.",            0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_ENDALL }, .flags = FLAGS, .unit = "eof_action" },
        { "pass",   "Pass through the main input.", 0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_PASS },   .flags = FLAGS, .unit = "eof_action" },
    { "shortest", "force termination when the shortest input terminates", OFFSET(fs.opt_shortest), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { "repeatlast", "repeat overlay of the last overlay frame", OFFSET(fs.opt_repeatlast), AV_OPT_TYPE_BOOL, {.i64=1}, 0, 1, FLAGS },
    { NULL },
};

FRAMESYNC_DEFINE_CLASS(overlay_videotoolbox, OverlayVideoToolboxContext, fs);

static const AVFilterPad overlay_videotoolbox_inputs[] = {
    {
        .name         = "main",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
    },
    {
        .name         = "overlay",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
    },
};

static const AVFilterPad overlay_videotoolbox_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
};

const AVFilter ff_vf_overlay_videotoolbox = {
    .name           = "overlay_videotoolbox",
    .description    = NULL_IF_CONFIG_SMALL("Overlay filter for VideoToolbox frames using Metal compute"),
    .priv_size      = OVERLAY_VT_CTX_SIZE,
    .priv_class     = &overlay_videotoolbox_class,
    .init           = overlay_videotoolbox_init,
    .uninit         = overlay_videotoolbox_uninit,
    .activate       = overlay_videotoolbox_activate,
    .preinit        = overlay_videotoolbox_framesync_preinit,
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_VIDEOTOOLBOX),
    FILTER_INPUTS(overlay_videotoolbox_inputs),
    FILTER_OUTPUTS(overlay_videotoolbox_outputs),
    .flags          = AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
