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

#include "libavutil/buffer.h"
#include "libavutil/hwcontext.h"
#include "libavutil/log.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#if HAVE_OPENCL_D3D11
#include "libavutil/hwcontext_d3d11va.h"
#endif

#if CONFIG_VULKAN
#include "libavutil/hwcontext_vulkan.h"
#include "libavutil/vulkan_loader.h"
#include "libavutil/vulkan.h"
#endif

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "video.h"

typedef struct HWMapContext {
    const AVClass *class;

    AVBufferRef   *hwframes_ref;

    int            mode;
    char          *derive_device_type;
    int            reverse;

#if CONFIG_VULKAN
    FFVulkanFunctions vkfn;
    int               vkfn_loaded;
#endif
} HWMapContext;

static int hwmap_query_formats(const AVFilterContext *avctx,
                               AVFilterFormatsConfig **cfg_in,
                               AVFilterFormatsConfig **cfg_out)
{
    int ret;

    if ((ret = ff_formats_ref(ff_all_formats(AVMEDIA_TYPE_VIDEO),
                              &cfg_in[0]->formats)) < 0 ||
        (ret = ff_formats_ref(ff_all_formats(AVMEDIA_TYPE_VIDEO),
                              &cfg_out[0]->formats)) < 0)
        return ret;

    return 0;
}

static int hwmap_config_output(AVFilterLink *outlink)
{
    FilterLink       *outl = ff_filter_link(outlink);
    AVFilterContext *avctx = outlink->src;
    HWMapContext      *ctx = avctx->priv;
    AVFilterLink   *inlink = avctx->inputs[0];
    FilterLink        *inl = ff_filter_link(inlink);
    AVHWFramesContext *hwfc;
    AVBufferRef *device;
    const AVPixFmtDescriptor *desc;
    int err, device_is_derived;

    av_log(avctx, AV_LOG_DEBUG, "Configure hwmap %s -> %s.\n",
           av_get_pix_fmt_name(inlink->format),
           av_get_pix_fmt_name(outlink->format));

    av_buffer_unref(&ctx->hwframes_ref);

    device = avctx->hw_device_ctx;
    device_is_derived = 0;

    if (inl->hw_frames_ctx) {
        hwfc = (AVHWFramesContext*)inl->hw_frames_ctx->data;

        if (ctx->derive_device_type) {
            enum AVHWDeviceType type;

            type = av_hwdevice_find_type_by_name(ctx->derive_device_type);
            if (type == AV_HWDEVICE_TYPE_NONE) {
                av_log(avctx, AV_LOG_ERROR, "Invalid device type.\n");
                err = AVERROR(EINVAL);
                goto fail;
            }

            err = av_hwdevice_ctx_create_derived(&device, type,
                                                 hwfc->device_ref, 0);
            if (err < 0) {
                av_log(avctx, AV_LOG_ERROR, "Failed to created derived "
                       "device context: %d.\n", err);
                goto fail;
            }
            device_is_derived = 1;
        }

        desc = av_pix_fmt_desc_get(outlink->format);
        if (!desc) {
            err = AVERROR(EINVAL);
            goto fail;
        }

        if (inlink->format == hwfc->format &&
            (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) &&
            !ctx->reverse) {
            // Map between two hardware formats (including the case of
            // undoing an existing mapping).

            if (!device) {
                av_log(avctx, AV_LOG_ERROR, "A device reference is "
                       "required to map to a hardware format.\n");
                err = AVERROR(EINVAL);
                goto fail;
            }

            err = av_hwframe_ctx_create_derived(&ctx->hwframes_ref,
                                                outlink->format,
                                                device,
                                                inl->hw_frames_ctx,
                                                ctx->mode);
            if (err < 0) {
                av_log(avctx, AV_LOG_ERROR, "Failed to create derived "
                       "frames context: %d.\n", err);
                goto fail;
            }

            if (hwfc->initial_pool_size) {
                outlink->fixed_pool_size = hwfc->initial_pool_size;
                av_log(avctx, AV_LOG_DEBUG, "Saved the fixed_pool_size from "
                       "initial_pool_size: %d\n", outlink->fixed_pool_size);
            }

        } else if (inlink->format == hwfc->format &&
                   (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) &&
                   ctx->reverse) {
            // Map between two hardware formats, but do it in reverse.
            // Make a new hwframe context for the target type, and then
            // overwrite the input hwframe context with a derived context
            // mapped from that back to the source type.
            AVBufferRef *source;
            AVHWFramesContext *frames;
#if HAVE_OPENCL_D3D11
            D3D11_TEXTURE2D_DESC texDesc = { .BindFlags = D3D11_BIND_DECODER, };
#endif

            ctx->hwframes_ref = av_hwframe_ctx_alloc(device);
            if (!ctx->hwframes_ref) {
                err = AVERROR(ENOMEM);
                goto fail;
            }
            frames = (AVHWFramesContext*)ctx->hwframes_ref->data;

            frames->format    = outlink->format;
            frames->sw_format = hwfc->sw_format;
            frames->width     = hwfc->width;
            frames->height    = hwfc->height;

            if (inlink->fixed_pool_size)
                frames->initial_pool_size = inlink->fixed_pool_size;

            if (frames->initial_pool_size == 0) {
                // Dynamic allocation.
            } else {
                frames->initial_pool_size += (avctx->extra_hw_frames > 2 ? avctx->extra_hw_frames : 2);
            }

#if HAVE_OPENCL_D3D11
            if (frames->format == AV_PIX_FMT_D3D11)
                frames->user_opaque = &texDesc;
#endif

            err = av_hwframe_ctx_init(ctx->hwframes_ref);
            if (err < 0) {
                av_log(avctx, AV_LOG_ERROR, "Failed to initialise "
                       "target frames context: %d.\n", err);
                goto fail;
            }

            err = av_hwframe_ctx_create_derived(&source,
                                                inlink->format,
                                                hwfc->device_ref,
                                                ctx->hwframes_ref,
                                                ctx->mode);
            if (err < 0) {
                av_log(avctx, AV_LOG_ERROR, "Failed to create "
                       "derived source frames context: %d.\n", err);
                goto fail;
            }

#if CONFIG_VULKAN
            if (inl->hw_frames_ctx && inlink->format == AV_PIX_FMT_VULKAN) {
                AVHWFramesContext     *hwfc_src = (AVHWFramesContext *)inl->hw_frames_ctx->data;
                AVVulkanFramesContext *vkfc_src = hwfc_src->hwctx;
                AVHWFramesContext     *hwfc_dst = (AVHWFramesContext *)source->data;
                AVVulkanFramesContext *vkfc_dst = hwfc_dst->hwctx;

                // Passthrough the VK_IMAGE_USAGE_*_BIT
                vkfc_dst->usage = vkfc_src->usage;
            }
#endif

            // Here is the naughty bit.  This overwriting changes what
            // ff_get_video_buffer() in the previous filter returns -
            // it will now give a frame allocated here mapped back to
            // the format it expects.  If there were any additional
            // constraints on the output frames there then this may
            // break nastily.
            av_buffer_unref(&inl->hw_frames_ctx);
            inl->hw_frames_ctx = source;

        } else if ((outlink->format == hwfc->format &&
                    inlink->format  == hwfc->sw_format) ||
                   inlink->format == hwfc->format) {
            // Map from a hardware format to a software format, or
            // undo an existing such mapping.

            ctx->hwframes_ref = av_buffer_ref(inl->hw_frames_ctx);
            if (!ctx->hwframes_ref) {
                err = AVERROR(ENOMEM);
                goto fail;
            }

        } else {
            // Non-matching formats - not supported.

            av_log(avctx, AV_LOG_ERROR, "Unsupported formats for "
                   "hwmap: from %s (%s) to %s.\n",
                   av_get_pix_fmt_name(inlink->format),
                   av_get_pix_fmt_name(hwfc->format),
                   av_get_pix_fmt_name(outlink->format));
            err = AVERROR(EINVAL);
            goto fail;
        }
    } else if (avctx->hw_device_ctx) {
        // Map from a software format to a hardware format.  This
        // creates a new hwframe context like hwupload, but then
        // returns frames mapped from that to the previous link in
        // order to fill them without an additional copy.

        if (!device) {
            av_log(avctx, AV_LOG_ERROR, "A device reference is "
                   "required to create new frames with reverse "
                   "mapping.\n");
            err = AVERROR(EINVAL);
            goto fail;
        }

        ctx->reverse = 1;

        ctx->hwframes_ref = av_hwframe_ctx_alloc(device);
        if (!ctx->hwframes_ref) {
            err = AVERROR(ENOMEM);
            goto fail;
        }
        hwfc = (AVHWFramesContext*)ctx->hwframes_ref->data;

        hwfc->format    = outlink->format;
        hwfc->sw_format = inlink->format;
        hwfc->width     = inlink->w;
        hwfc->height    = inlink->h;

        if (avctx->extra_hw_frames >= 0)
            hwfc->initial_pool_size = 2 + avctx->extra_hw_frames;

        err = av_hwframe_ctx_init(ctx->hwframes_ref);
        if (err < 0) {
            av_log(avctx, AV_LOG_ERROR, "Failed to create frame "
                   "context for reverse mapping: %d.\n", err);
            goto fail;
        }

    } else {
        av_log(avctx, AV_LOG_ERROR, "Mapping requires a hardware "
               "context (a device, or frames on input).\n");
        return AVERROR(EINVAL);
    }

    outl->hw_frames_ctx = av_buffer_ref(ctx->hwframes_ref);
    if (!outl->hw_frames_ctx) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    outlink->w = inlink->w;
    outlink->h = inlink->h;

    if (device_is_derived)
        av_buffer_unref(&device);
    return 0;

fail:
    if (device_is_derived)
        av_buffer_unref(&device);
    av_buffer_unref(&ctx->hwframes_ref);
    return err;
}

static AVFrame *hwmap_get_buffer(AVFilterLink *inlink, int w, int h)
{
    FilterLink          *l = ff_filter_link(inlink);
    AVFilterContext *avctx = inlink->dst;
    AVFilterLink  *outlink = avctx->outputs[0];
    HWMapContext      *ctx = avctx->priv;

    if (ctx->reverse && !l->hw_frames_ctx) {
        AVFrame *src, *dst;
        int err;

        src = ff_get_video_buffer(outlink, w, h);
        if (!src) {
            av_log(avctx, AV_LOG_ERROR, "Failed to allocate source "
                   "frame for software mapping.\n");
            return NULL;
        }

        dst = av_frame_alloc();
        if (!dst) {
            av_frame_free(&src);
            return NULL;
        }

        err = av_hwframe_map(dst, src, ctx->mode);
        if (err) {
            av_log(avctx, AV_LOG_ERROR, "Failed to map frame to "
                   "software: %d.\n", err);
            av_frame_free(&src);
            av_frame_free(&dst);
            return NULL;
        }

        av_frame_free(&src);
        return dst;
    } else {
        return ff_default_get_video_buffer(inlink, w, h);
    }
}

static int hwmap_filter_frame(AVFilterLink *link, AVFrame *input)
{
    AVFilterContext *avctx = link->dst;
    AVFilterLink  *outlink = avctx->outputs[0];
    HWMapContext      *ctx = avctx->priv;
    AVFrame *map = NULL;
    int err;

    av_log(ctx, AV_LOG_DEBUG, "Filter input: %s, %ux%u (%"PRId64").\n",
           av_get_pix_fmt_name(input->format),
           input->width, input->height, input->pts);

    map = av_frame_alloc();
    if (!map) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    map->format = outlink->format;
    map->hw_frames_ctx = av_buffer_ref(ctx->hwframes_ref);
    if (!map->hw_frames_ctx) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    if (ctx->reverse && !input->hw_frames_ctx) {
        // If we mapped backwards from hardware to software, we need
        // to attach the hardware frame context to the input frame to
        // make the mapping visible to av_hwframe_map().
        input->hw_frames_ctx = av_buffer_ref(ctx->hwframes_ref);
        if (!input->hw_frames_ctx) {
            err = AVERROR(ENOMEM);
            goto fail;
        }
    }

    err = av_hwframe_map(map, input, ctx->mode);
    if (err < 0) {
        av_log(avctx, AV_LOG_ERROR, "Failed to map frame: %d.\n", err);
        goto fail;
    }

    err = av_frame_copy_props(map, input);
    if (err < 0)
        goto fail;

#if CONFIG_VULKAN
    if (ctx->reverse &&
        input->hw_frames_ctx && map->hw_frames_ctx &&
        input->format == AV_PIX_FMT_VULKAN &&
        (map->format == AV_PIX_FMT_VAAPI ||
         map->format == AV_PIX_FMT_DRM_PRIME)) {
        // If we mapped backwards from vulkan to drm_prime, we need
        // to wait for the AVVkFrame semaphores to be signaled.
        AVHWFramesContext     *hwfc    = (AVHWFramesContext *)input->hw_frames_ctx->data;
        AVVulkanDeviceContext *vk_dev  = hwfc->device_ctx->hwctx;
        AVVkFrame             *vkf     = (AVVkFrame *)input->data[0];
        const int              nb_sems = vkf ? ff_vk_count_images(vkf) : 0;

        if (hwfc->device_ctx->type != AV_HWDEVICE_TYPE_VULKAN || !vkf || !nb_sems)
            goto exit;

        if (!ctx->vkfn_loaded) {
            uint64_t exts = ff_vk_extensions_to_mask(vk_dev->enabled_dev_extensions,
                                                     vk_dev->nb_enabled_dev_extensions);
            err = ff_vk_load_functions(hwfc->device_ctx, &ctx->vkfn, exts, 1, 1);
            if (err < 0)
                goto fail;
            ctx->vkfn_loaded = 1;
        }
        if (ctx->vkfn.WaitSemaphores) {
            VkSemaphoreWaitInfo wait_info = {
                .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .flags          = 0x0,
                .semaphoreCount = nb_sems,
                .pSemaphores    = vkf->sem,
                .pValues        = vkf->sem_value,
            };
            ctx->vkfn.WaitSemaphores(vk_dev->act_dev, &wait_info, UINT64_MAX);
            av_log(avctx, AV_LOG_DEBUG, "Vulkan sems for reverse-mapped DRM objects signaled!\n");
        }
    }
exit:
#endif

    av_frame_free(&input);

    av_log(ctx, AV_LOG_DEBUG, "Filter output: %s, %ux%u (%"PRId64").\n",
           av_get_pix_fmt_name(map->format),
           map->width, map->height, map->pts);

    return ff_filter_frame(outlink, map);

fail:
    av_frame_free(&input);
    av_frame_free(&map);
    return err;
}

static av_cold void hwmap_uninit(AVFilterContext *avctx)
{
    HWMapContext *ctx = avctx->priv;

    av_buffer_unref(&ctx->hwframes_ref);
}

#define OFFSET(x) offsetof(HWMapContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption hwmap_options[] = {
    { "mode", "Frame mapping mode",
      OFFSET(mode), AV_OPT_TYPE_FLAGS,
      { .i64 = AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_WRITE },
      0, INT_MAX, FLAGS, .unit = "mode" },

    { "read", "Mapping should be readable",
      0, AV_OPT_TYPE_CONST, { .i64 = AV_HWFRAME_MAP_READ },
      INT_MIN, INT_MAX, FLAGS, .unit = "mode" },
    { "write", "Mapping should be writeable",
      0, AV_OPT_TYPE_CONST, { .i64 = AV_HWFRAME_MAP_WRITE },
      INT_MIN, INT_MAX, FLAGS, .unit = "mode" },
    { "overwrite", "Mapping will always overwrite the entire frame",
      0, AV_OPT_TYPE_CONST, { .i64 = AV_HWFRAME_MAP_OVERWRITE },
      INT_MIN, INT_MAX, FLAGS, .unit = "mode" },
    { "direct", "Mapping should not involve any copying",
      0, AV_OPT_TYPE_CONST, { .i64 = AV_HWFRAME_MAP_DIRECT },
      INT_MIN, INT_MAX, FLAGS, .unit = "mode" },

    { "derive_device", "Derive a new device of this type",
      OFFSET(derive_device_type), AV_OPT_TYPE_STRING,
      { .str = NULL }, 0, 0, FLAGS },
    { "reverse", "Map in reverse (create and allocate in the sink)",
      OFFSET(reverse), AV_OPT_TYPE_INT,
      { .i64 = 0 }, 0, 1, FLAGS },

    { NULL }
};

AVFILTER_DEFINE_CLASS(hwmap);

static const AVFilterPad hwmap_inputs[] = {
    {
        .name             = "default",
        .type             = AVMEDIA_TYPE_VIDEO,
        .get_buffer.video = hwmap_get_buffer,
        .filter_frame     = hwmap_filter_frame,
    },
};

static const AVFilterPad hwmap_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = hwmap_config_output,
    },
};

const FFFilter ff_vf_hwmap = {
    .p.name         = "hwmap",
    .p.description  = NULL_IF_CONFIG_SMALL("Map hardware frames"),
    .p.priv_class   = &hwmap_class,
    .p.flags        = AVFILTER_FLAG_HWDEVICE,
    .uninit         = hwmap_uninit,
    .priv_size      = sizeof(HWMapContext),
    FILTER_INPUTS(hwmap_inputs),
    FILTER_OUTPUTS(hwmap_outputs),
    FILTER_QUERY_FUNC2(hwmap_query_formats),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
