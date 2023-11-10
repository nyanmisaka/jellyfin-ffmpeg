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

#ifndef AVUTIL_HWCONTEXT_RKMPP_H
#define AVUTIL_HWCONTEXT_RKMPP_H

#include <stddef.h>
#include <stdint.h>
#include <libdrm/drm_fourcc.h>

#include "hwcontext_drm.h"

#ifndef DRM_FORMAT_NV15
#define DRM_FORMAT_NV15 fourcc_code('N', 'A', '1', '2')
#endif
#ifndef DRM_FORMAT_NV20
#define DRM_FORMAT_NV20 fourcc_code('N', 'V', '2', '0')
#endif

#ifndef DRM_FORMAT_MOD_VENDOR_ARM
#define DRM_FORMAT_MOD_VENDOR_ARM 0x08
#endif
#ifndef DRM_FORMAT_MOD_ARM_TYPE_AFBC
#define DRM_FORMAT_MOD_ARM_TYPE_AFBC 0x00
#endif

#define drm_is_afbc(mod) \
        ((mod >> 52) == (DRM_FORMAT_MOD_ARM_TYPE_AFBC | \
                (DRM_FORMAT_MOD_VENDOR_ARM << 4)))

/**
 * @file
 * API-specific header for AV_HWDEVICE_TYPE_RKMPP.
 */

enum {
    /**
     * rockchip_drm::ROCKCHIP_BO_CACHABLE
     * cachable mapping.
     */
    AV_RKMPP_FLAG_BO_CACHABLE = (1 << 1),
    /**
     * rockchip_drm::ROCKCHIP_BO_DMA32
     * alloc page with gfp_dma32.
     */
    AV_RKMPP_FLAG_BO_DMA32    = (1 << 5),
};

/**
 * RKMPP-specific data associated with a frame pool.
 *
 * Allocated as AVHWFramesContext.hwctx.
 */
typedef struct AVRKMPPFramesContext {
    /**
     * The descriptors of all frames in the pool after creation.
     * Only valid if AVHWFramesContext.initial_pool_size was positive.
     * These are intended to be used as the buffer of RKMPP decoder.
     */
    AVDRMFrameDescriptor *frames;
    int                nb_frames;
} AVRKMPPFramesContext;

/**
 * RKMPP device details.
 *
 * Allocated as AVHWDeviceContext.hwctx
 */
typedef struct AVRKMPPDeviceContext {
    /**
     * Rockchip DRM device fd.
     */
    int fd;

    /**
     * Rockchip frame allocation flags.
     */
    int flags;
} AVRKMPPDeviceContext;

#endif /* AVUTIL_HWCONTEXT_RKMPP_H */
