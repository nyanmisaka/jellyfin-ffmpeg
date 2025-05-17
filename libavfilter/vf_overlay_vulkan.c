/*
 * Copyright (c) Lynne
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

#include "libavutil/random_seed.h"
#include "libavutil/opt.h"
#include "vulkan_filter.h"
#include "vulkan_spirv.h"

#include "filters.h"
#include "framesync.h"
#include "video.h"

typedef struct OverlayVulkanContext {
    FFVulkanContext vkctx;
    FFFrameSync fs;

    int initialized;
    FFVulkanPipeline pl;
    FFVulkanPipeline pl_pass;
    FFVkExecPool e;
    FFVkQueueFamilyCtx qf;
    FFVkSPIRVShader shd;
    FFVkSPIRVShader shd_pass;
    VkSampler sampler;

    /* Push constants / options */
    struct {
        int32_t o_offset[2*3];
        int32_t o_size[2*3];
    } opts;

    int overlay_x;
    int overlay_y;
    int overlay_w;
    int overlay_h;

    int opt_repeatlast;
    int opt_shortest;
    int opt_eof_action;
} OverlayVulkanContext;

static const char overlay_noalpha[] = {
    C(0, void overlay_noalpha(int i, ivec2 pos)                                )
    C(0, {                                                                     )
    C(1,     if ((o_offset[i].x <= pos.x) && (o_offset[i].y <= pos.y) &&
                 (pos.x < (o_offset[i].x + o_size[i].x)) &&
                 (pos.y < (o_offset[i].y + o_size[i].y))) {                    )
    C(2,         vec4 res = texture(overlay_img[i], pos - o_offset[i]);        )
    C(2,         imageStore(output_img[i], pos, res);                          )
    C(1,     } else {                                                          )
    C(2,         vec4 res = texture(main_img[i], pos);                         )
    C(2,         imageStore(output_img[i], pos, res);                          )
    C(1,     }                                                                 )
    C(0, }                                                                     )
};

static const char overlay_alpha[] = {
    C(0, void overlay_alpha_opaque(int i, ivec2 pos)                           )
    C(0, {                                                                     )
    C(1,     vec4 res = texture(main_img[i], pos);                             )
    C(1,     if ((o_offset[i].x <= pos.x) && (o_offset[i].y <= pos.y) &&
                 (pos.x < (o_offset[i].x + o_size[i].x)) &&
                 (pos.y < (o_offset[i].y + o_size[i].y))) {                    )
    C(2,         vec4 ovr = texture(overlay_img[i], pos - o_offset[i]);        )
    C(2,         res = ovr * ovr.a + res * (1.0f - ovr.a);                     )
    C(2,         res.a = 1.0f;                                                 )
    C(2,         imageStore(output_img[i], pos, res);                          )
    C(1,     }                                                                 )
    C(1,     imageStore(output_img[i], pos, res);                              )
    C(0, }                                                                     )
};

static av_cold int init_filter(AVFilterContext *ctx)
{
    int err;
    uint8_t *spv_data, *spv_data_pass;
    size_t spv_len;
    void *spv_opaque = NULL;
    void *spv_opaque_pass = NULL;
    OverlayVulkanContext *s = ctx->priv;
    FFVulkanContext *vkctx = &s->vkctx;
    const int planes = av_pix_fmt_count_planes(s->vkctx.output_format);
    const int ialpha = av_pix_fmt_desc_get(s->vkctx.input_format)->flags & AV_PIX_FMT_FLAG_ALPHA;
    const AVPixFmtDescriptor *pix_desc = av_pix_fmt_desc_get(s->vkctx.output_format);
    FFVkSPIRVShader *shd;
    FFVkSPIRVCompiler *spv;
    FFVulkanDescriptorSetBinding *desc;

    spv = ff_vk_spirv_init();
    if (!spv) {
        av_log(ctx, AV_LOG_ERROR, "Unable to initialize SPIR-V compiler!\n");
        return AVERROR_EXTERNAL;
    }

    ff_vk_qf_init(vkctx, &s->qf, VK_QUEUE_COMPUTE_BIT);
    RET(ff_vk_exec_pool_init(vkctx, &s->qf, &s->e, s->qf.nb_queues*4, 0, 0, 0, NULL));
    RET(ff_vk_init_sampler(vkctx, &s->sampler, 1, VK_FILTER_NEAREST));

    /* overlay_compute */
    RET(ff_vk_shader_init(&s->pl, &s->shd, "overlay_compute",
                          VK_SHADER_STAGE_COMPUTE_BIT, 0));
    {
        shd = &s->shd;
        ff_vk_shader_set_compute_sizes(shd, 32, 32, 1);

        GLSLC(0, layout(push_constant, std430) uniform pushConstants {        );
        GLSLC(1,    ivec2 o_offset[3];                                        );
        GLSLC(1,    ivec2 o_size[3];                                          );
        GLSLC(0, };                                                           );
        GLSLC(0,                                                              );

        ff_vk_add_push_constant(&s->pl, 0, sizeof(s->opts),
                                VK_SHADER_STAGE_COMPUTE_BIT);

        desc = (FFVulkanDescriptorSetBinding []) {
            {
                .name       = "main_img",
                .type       = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .dimensions = 2,
                .elems      = planes,
                .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
                .samplers   = DUP_SAMPLER(s->sampler),
            },
            {
                .name       = "overlay_img",
                .type       = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .dimensions = 2,
                .elems      = planes,
                .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
                .samplers   = DUP_SAMPLER(s->sampler),
            },
            {
                .name       = "output_img",
                .type       = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .mem_layout = ff_vk_shader_rep_fmt(s->vkctx.output_format),
                .mem_quali  = "writeonly",
                .dimensions = 2,
                .elems      = planes,
                .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
            },
        };

        RET(ff_vk_pipeline_descriptor_set_add(vkctx, &s->pl, shd, desc, 3, 0, 0));

        GLSLD(   overlay_noalpha                                              );
        GLSLD(   overlay_alpha                                                );
        GLSLC(0, void main()                                                  );
        GLSLC(0, {                                                            );
        GLSLC(1,     ivec2 pos = ivec2(gl_GlobalInvocationID.xy);             );
        GLSLF(1,     int planes = %i;                                  ,planes);
        GLSLC(1,     for (int i = 0; i < planes; i++) {                       );
        if (ialpha)
            GLSLC(2,         overlay_alpha_opaque(i, pos);                    );
        else
            GLSLC(2,         overlay_noalpha(i, pos);                         );
        GLSLC(1,     }                                                        );
        GLSLC(0, }                                                            );

        RET(spv->compile_shader(spv, ctx, shd, &spv_data, &spv_len, "main",
                                &spv_opaque));
        RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));

        RET(ff_vk_init_compute_pipeline(vkctx, &s->pl, shd));
        RET(ff_vk_exec_pipeline_register(vkctx, &s->e, &s->pl));

        s->opts.o_offset[0] = s->overlay_x;
        s->opts.o_offset[1] = s->overlay_y;
        s->opts.o_offset[2] = s->opts.o_offset[0] >> pix_desc->log2_chroma_w;
        s->opts.o_offset[3] = s->opts.o_offset[1] >> pix_desc->log2_chroma_h;
        s->opts.o_offset[4] = s->opts.o_offset[0] >> pix_desc->log2_chroma_w;
        s->opts.o_offset[5] = s->opts.o_offset[1] >> pix_desc->log2_chroma_h;

        s->opts.o_size[0] = s->overlay_w;
        s->opts.o_size[1] = s->overlay_h;
        s->opts.o_size[2] = s->opts.o_size[0] >> pix_desc->log2_chroma_w;
        s->opts.o_size[3] = s->opts.o_size[1] >> pix_desc->log2_chroma_h;
        s->opts.o_size[4] = s->opts.o_size[0] >> pix_desc->log2_chroma_w;
        s->opts.o_size[5] = s->opts.o_size[1] >> pix_desc->log2_chroma_h;
    }

    /* overlay_compute_pass */
    RET(ff_vk_shader_init(&s->pl_pass, &s->shd_pass, "overlay_compute_pass",
                          VK_SHADER_STAGE_COMPUTE_BIT, 0));
    {
        shd = &s->shd_pass;
        ff_vk_shader_set_compute_sizes(shd, 32, 32, 1);

        desc = (FFVulkanDescriptorSetBinding []) {
            {
                .name       = "main_img",
                .type       = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .dimensions = 2,
                .elems      = planes,
                .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
                .samplers   = DUP_SAMPLER(s->sampler),
            },
            {
                .name       = "output_img",
                .type       = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .mem_layout = ff_vk_shader_rep_fmt(s->vkctx.output_format),
                .mem_quali  = "writeonly",
                .dimensions = 2,
                .elems      = planes,
                .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
            },
        };

        RET(ff_vk_pipeline_descriptor_set_add(vkctx, &s->pl_pass, shd, desc, 2, 0, 0));

        GLSLC(0, void main()                                                  );
        GLSLC(0, {                                                            );
        GLSLC(1,     ivec2 pos = ivec2(gl_GlobalInvocationID.xy);             );
        GLSLF(1,     int planes = %i;                                  ,planes);
        GLSLC(1,     for (int i = 0; i < planes; i++) {                       );
        GLSLC(2,         vec4 res = texture(main_img[i], pos);                );
        GLSLC(2,         imageStore(output_img[i], pos, res);                 );
        GLSLC(1,     }                                                        );
        GLSLC(0, }                                                            );

        RET(spv->compile_shader(spv, ctx, shd, &spv_data_pass, &spv_len, "main",
                                &spv_opaque));
        RET(ff_vk_shader_create(vkctx, shd, spv_data_pass, spv_len, "main"));

        RET(ff_vk_init_compute_pipeline(vkctx, &s->pl_pass, shd));
        RET(ff_vk_exec_pipeline_register(vkctx, &s->e, &s->pl_pass));
    }

    s->initialized = 1;

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);
    if (spv_opaque_pass)
        spv->free_shader(spv, &spv_opaque_pass);
    if (spv)
        spv->uninit(&spv);

    return err;
}

static int overlay_vulkan_blend(FFFrameSync *fs)
{
    int err;
    AVFilterContext *ctx = fs->parent;
    OverlayVulkanContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *input_main, *input_overlay, *out;

    err = ff_framesync_get_frame(fs, 0, &input_main, 0);
    if (err < 0)
        goto fail;
    err = ff_framesync_get_frame(fs, 1, &input_overlay, 0);
    if (err < 0)
        goto fail;

    if (!input_main)
        return AVERROR_BUG;

    if (!s->initialized)
        RET(init_filter(ctx));

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    if (input_overlay)
        RET(ff_vk_filter_process_Nin(&s->vkctx, &s->e, &s->pl,
                                     out, (AVFrame *[]){ input_main, input_overlay }, 2,
                                     s->sampler, &s->opts, sizeof(s->opts)));
    else /* passthrough */
        RET(ff_vk_filter_process_simple(&s->vkctx, &s->e, &s->pl_pass, out, input_main,
                                        s->sampler, NULL, 0));

    err = av_frame_copy_props(out, input_main);
    if (err < 0)
        goto fail;

    return ff_filter_frame(outlink, out);

fail:
    av_frame_free(&out);
    return err;
}

static int overlay_vulkan_config_output(AVFilterLink *outlink)
{
    int err;
    AVFilterContext *avctx = outlink->src;
    OverlayVulkanContext *s = avctx->priv;
    AVFilterLink *inlink = avctx->inputs[0];
    AVFilterLink *inlink_overlay = avctx->inputs[1];
    FilterLink *inl = ff_filter_link(inlink);
    FilterLink *inl_overlay = ff_filter_link(inlink_overlay);
    AVHWFramesContext *main_fc = (AVHWFramesContext*)inl->hw_frames_ctx->data;
    AVHWFramesContext *overlay_fc = (AVHWFramesContext*)inl_overlay->hw_frames_ctx->data;

    if (main_fc->sw_format != overlay_fc->sw_format) {
        av_log(avctx, AV_LOG_ERROR, "Mismatching sw formats!\n");
        return AVERROR(EINVAL);
    }

    s->overlay_w = inlink_overlay->w;
    s->overlay_h = inlink_overlay->h;

    err = ff_vk_filter_config_output(outlink);
    if (err < 0)
        return err;

    err = ff_framesync_init_dualinput(&s->fs, avctx);
    if (err < 0)
        return err;

    s->fs.opt_repeatlast = s->opt_repeatlast;
    s->fs.opt_shortest = s->opt_shortest;
    s->fs.opt_eof_action = s->opt_eof_action;
    s->fs.time_base = outlink->time_base = inlink->time_base;

    return ff_framesync_configure(&s->fs);
}

static int overlay_vulkan_activate(AVFilterContext *avctx)
{
    OverlayVulkanContext *s = avctx->priv;

    return ff_framesync_activate(&s->fs);
}

static av_cold int overlay_vulkan_init(AVFilterContext *avctx)
{
    OverlayVulkanContext *s = avctx->priv;

    s->fs.on_event = &overlay_vulkan_blend;

    return ff_vk_filter_init(avctx);
}

static void overlay_vulkan_uninit(AVFilterContext *avctx)
{
    OverlayVulkanContext *s = avctx->priv;
    FFVulkanContext *vkctx = &s->vkctx;
    FFVulkanFunctions *vk = &vkctx->vkfn;

    ff_vk_exec_pool_free(vkctx, &s->e);
    ff_vk_pipeline_free(vkctx, &s->pl);
    ff_vk_pipeline_free(vkctx, &s->pl_pass);
    ff_vk_shader_free(vkctx, &s->shd);
    ff_vk_shader_free(vkctx, &s->shd_pass);

    if (s->sampler)
        vk->DestroySampler(vkctx->hwctx->act_dev, s->sampler,
                           vkctx->hwctx->alloc);

    ff_vk_uninit(&s->vkctx);
    ff_framesync_uninit(&s->fs);

    s->initialized = 0;
}

#define OFFSET(x) offsetof(OverlayVulkanContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption overlay_vulkan_options[] = {
    { "x", "Set horizontal offset", OFFSET(overlay_x), AV_OPT_TYPE_INT, {.i64 = 0}, 0, INT_MAX, .flags = FLAGS },
    { "y", "Set vertical offset",   OFFSET(overlay_y), AV_OPT_TYPE_INT, {.i64 = 0}, 0, INT_MAX, .flags = FLAGS },
    { "eof_action", "Action to take when encountering EOF from secondary input ",
        OFFSET(opt_eof_action), AV_OPT_TYPE_INT, { .i64 = EOF_ACTION_REPEAT },
        EOF_ACTION_REPEAT, EOF_ACTION_PASS, .flags = FLAGS, .unit = "eof_action" },
        { "repeat", "Repeat the previous frame.",   0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_REPEAT }, .flags = FLAGS, .unit = "eof_action" },
        { "endall", "End both streams.",            0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_ENDALL }, .flags = FLAGS, .unit = "eof_action" },
        { "pass",   "Pass through the main input.", 0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_PASS },   .flags = FLAGS, .unit = "eof_action" },
    { "shortest", "force termination when the shortest input terminates", OFFSET(opt_shortest), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { "repeatlast", "repeat overlay of the last overlay frame", OFFSET(opt_repeatlast), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { NULL },
};

AVFILTER_DEFINE_CLASS(overlay_vulkan);

static const AVFilterPad overlay_vulkan_inputs[] = {
    {
        .name         = "main",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &ff_vk_filter_config_input,
    },
    {
        .name         = "overlay",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &ff_vk_filter_config_input,
    },
};

static const AVFilterPad overlay_vulkan_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = &overlay_vulkan_config_output,
    },
};

const AVFilter ff_vf_overlay_vulkan = {
    .name           = "overlay_vulkan",
    .description    = NULL_IF_CONFIG_SMALL("Overlay a source on top of another"),
    .priv_size      = sizeof(OverlayVulkanContext),
    .init           = &overlay_vulkan_init,
    .uninit         = &overlay_vulkan_uninit,
    .activate       = &overlay_vulkan_activate,
    FILTER_INPUTS(overlay_vulkan_inputs),
    FILTER_OUTPUTS(overlay_vulkan_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_VULKAN),
    .priv_class     = &overlay_vulkan_class,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
    .flags          = AVFILTER_FLAG_HWDEVICE,
};
