/*
 * Copyright (C) 2026 NyanMisaka
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

#include "ffprobe_hw_internal.h"

#if CONFIG_LIBDRM
#   include <drm.h>
#   include <xf86drm.h>
#   include <dlfcn.h>
#   include <fcntl.h>
#   if HAVE_UNISTD_H
#       include <unistd.h>
#   endif
#   include "libavutil/hwcontext_drm.h"
#endif

#if (CONFIG_VAAPI && HAVE_VAAPI_DRM)
#   include <va/va_drm.h>
#   include <va/va_drmcommon.h>
#   include "libavutil/hwcontext_vaapi.h"
#endif

#if CONFIG_LIBDRM
struct udev;
struct udev_hwdb;
struct udev_list_entry;

typedef struct udev* (*udev_new_t)(void);
typedef struct udev* (*udev_unref_t)(struct udev *udev);
typedef struct udev_hwdb* (*udev_hwdb_new_t)(struct udev *udev);
typedef struct udev_hwdb* (*udev_hwdb_unref_t)(struct udev_hwdb *hwdb);
typedef struct udev_list_entry* (*udev_hwdb_get_properties_list_entry_t)(struct udev_hwdb *hwdb, const char *modalias, unsigned int flags);
typedef const char* (*udev_list_entry_get_name_t)(struct udev_list_entry *list_entry);
typedef const char* (*udev_list_entry_get_value_t)(struct udev_list_entry *list_entry);
typedef struct udev_list_entry* (*udev_list_entry_get_next_t)(struct udev_list_entry *list_entry);

typedef struct UdevFunctions {
    void *handle;
    udev_new_t udev_new;
    udev_unref_t udev_unref;
    udev_hwdb_new_t udev_hwdb_new;
    udev_hwdb_unref_t udev_hwdb_unref;
    udev_hwdb_get_properties_list_entry_t udev_hwdb_get_properties_list_entry;
    udev_list_entry_get_name_t udev_list_entry_get_name;
    udev_list_entry_get_value_t udev_list_entry_get_value;
    udev_list_entry_get_next_t udev_list_entry_get_next;
} UdevFunctions;

static UdevFunctions udev_functions = { NULL };
#endif

int init_udev_functions(void)
{
#if CONFIG_LIBDRM
    if (udev_functions.handle)
        return 0;

    udev_functions.handle = dlopen("libudev.so.1", RTLD_LAZY);
    if (!udev_functions.handle)
        udev_functions.handle = dlopen("libudev.so", RTLD_LAZY);

    if (!udev_functions.handle) {
        av_log(NULL, AV_LOG_DEBUG, "Failed to load libudev.so\n");
        return AVERROR(ENOSYS);
    }

#   define LOAD_UDEV_FUNCTION(type, name) do { \
    udev_functions.name = (type)dlsym(udev_functions.handle, #name); \
    if (!udev_functions.name) { \
        av_log(NULL, AV_LOG_DEBUG, "Missing symbol lookup reference: %s\n", #name); \
        goto fail; \
    } \
} while (0)

    LOAD_UDEV_FUNCTION(udev_new_t, udev_new);
    LOAD_UDEV_FUNCTION(udev_unref_t, udev_unref);
    LOAD_UDEV_FUNCTION(udev_hwdb_new_t, udev_hwdb_new);
    LOAD_UDEV_FUNCTION(udev_hwdb_unref_t, udev_hwdb_unref);
    LOAD_UDEV_FUNCTION(udev_hwdb_get_properties_list_entry_t, udev_hwdb_get_properties_list_entry);
    LOAD_UDEV_FUNCTION(udev_list_entry_get_name_t, udev_list_entry_get_name);
    LOAD_UDEV_FUNCTION(udev_list_entry_get_value_t, udev_list_entry_get_value);
    LOAD_UDEV_FUNCTION(udev_list_entry_get_next_t, udev_list_entry_get_next);
#   undef LOAD_UDEV_FUNCTION

    return 0;

fail:
    uninit_udev_functions();
    return AVERROR_UNKNOWN;
#else
    return AVERROR(ENOSYS);
#endif
}

void uninit_udev_functions(void)
{
#if CONFIG_LIBDRM
    if (udev_functions.handle) {
        dlclose(udev_functions.handle);
        memset(&udev_functions, 0, sizeof(udev_functions));
    }
#endif
}

#if CONFIG_LIBDRM
static int print_drm_pci_device_name(AVTextFormatContext *tfc, drmDevice *drm_dev)
{
    struct udev *udev_ctx = NULL;
    struct udev_hwdb *hwdb_ctx = NULL;
    struct udev_list_entry *entries = NULL;
    struct udev_list_entry *entry = NULL;
    char modalias[128];
    const char *model = NULL;
    int ret = 0;

    if (!tfc || !drm_dev || drm_dev->bustype != DRM_BUS_PCI || !drm_dev->deviceinfo.pci)
        return AVERROR(EINVAL);

    if ((ret = init_udev_functions()) < 0)
        return ret;

    if (!udev_functions.handle)
        return AVERROR(ENOSYS);

    udev_ctx = udev_functions.udev_new();
    if (!udev_ctx)
        return AVERROR(ENOMEM);

    hwdb_ctx = udev_functions.udev_hwdb_new(udev_ctx);
    if (!hwdb_ctx) {
        ret = AVERROR(ENOENT);
        goto fail;
    }

    snprintf(modalias, sizeof(modalias), "pci:v0000%04Xd0000%04Xsv0000%04Xsd0000%04X*",
             drm_dev->deviceinfo.pci->vendor_id,
             drm_dev->deviceinfo.pci->device_id,
             drm_dev->deviceinfo.pci->subvendor_id,
             drm_dev->deviceinfo.pci->subdevice_id);
    entries = udev_functions.udev_hwdb_get_properties_list_entry(hwdb_ctx, modalias, 0);

    if (!entries) {
        snprintf(modalias, sizeof(modalias), "pci:v0000%04Xd0000%04X*",
                 drm_dev->deviceinfo.pci->vendor_id,
                 drm_dev->deviceinfo.pci->device_id);
        entries = udev_functions.udev_hwdb_get_properties_list_entry(hwdb_ctx, modalias, 0);
    }
    if (!entries) {
        ret = AVERROR(ENOENT);
        goto fail;
    }

    for (entry = entries; entry; entry = udev_functions.udev_list_entry_get_next(entry)) {
        const char *name  = udev_functions.udev_list_entry_get_name(entry);
        const char *value = udev_functions.udev_list_entry_get_value(entry);

        if (!name || !value)
            continue;

        if (!strcmp(name, "ID_MODEL_FROM_DATABASE") ||
            (!strcmp(name, "ID_MODEL_TEXT") && !model)) {
            model = value;
        }
    }
    if (!model) {
        ret = AVERROR(ENODEV);
        goto fail;
    }

    print_str("device_name", model);

fail:
    if (hwdb_ctx)
        udev_functions.udev_hwdb_unref(hwdb_ctx);
    if (udev_ctx)
        udev_functions.udev_unref(udev_ctx);

    return ret;
}

static int ff_drmDevicesEqual(drmDevice *a, drmDevice *b)
{
    if (!a || !b || a->bustype != b->bustype)
        return 0;

    switch (a->bustype) {
    case DRM_BUS_PCI:
        return !memcmp(a->businfo.pci, b->businfo.pci, sizeof(*a->businfo.pci));
    case DRM_BUS_USB:
        return !memcmp(a->businfo.usb, b->businfo.usb, sizeof(*a->businfo.usb));
    case DRM_BUS_PLATFORM:
        return !memcmp(a->businfo.platform, b->businfo.platform, sizeof(*a->businfo.platform));
#   ifdef DRM_BUS_HOST1X
    case DRM_BUS_HOST1X:
        return !memcmp(a->businfo.host1x, b->businfo.host1x, sizeof(*a->businfo.host1x));
#   endif
#   ifdef DRM_BUS_FAUX
    case DRM_BUS_FAUX:
        return !memcmp(a->businfo.faux, b->businfo.faux, sizeof(*a->businfo.faux));
#   endif
    }

    return 0;
}
#endif

int create_drm_devices(HwDeviceRefs *refs, const char *device_path)
{
    return create_drm_devices_with_filter(refs, -1, device_path);
}

int create_drm_devices_with_filter(HwDeviceRefs *refs, int vendor_id, const char *device_path)
{
#if CONFIG_LIBDRM
    int i, j, n = 0, ret = 0;
    drmDevice *drm_devs[FFPROBE_HW_MAX_DEV_NUM];
    drmDevice *in_drm_dev = NULL;
    int in_path_fd = -1;

#   ifdef DRM_DEVICE_GET_PCI_REVISION
    n = drmGetDevices2(0, drm_devs, FF_ARRAY_ELEMS(drm_devs));
#   else
    n = drmGetDevices(drm_devs, FF_ARRAY_ELEMS(drm_devs));
#   endif
    if (n <= 0)
        return AVERROR(ENOSYS);

    if (device_path) {
        in_path_fd = open(device_path, O_RDWR);
        if (in_path_fd < 0) {
            ret = AVERROR(EINVAL);
            goto exit;
        }
#   ifdef DRM_DEVICE_GET_PCI_REVISION
        ret = drmGetDevice2(in_path_fd, 0, &in_drm_dev);
#   else
        ret = drmGetDevice(in_path_fd, &in_drm_dev);
#   endif
        if (ret || !in_drm_dev) {
            ret = AVERROR(EINVAL);
            goto exit;
        }
    }

    for (i = 0, j = 0; i < FFMIN(n, FFPROBE_HW_MAX_DEV_NUM) && refs; i++) {
        drmDevice *drm_dev = drm_devs[i];
        const char *node_path = NULL;
        char *device_path_drm = refs[j].device_path_drm;
        const size_t device_path_drm_sz = FF_ARRAY_ELEMS(refs[j].device_path_drm);

        if (drm_dev->available_nodes & (1 << DRM_NODE_RENDER))
            node_path = drm_dev->nodes[DRM_NODE_RENDER];
        else if (drm_dev->available_nodes & (1 << DRM_NODE_PRIMARY))
            node_path = drm_dev->nodes[DRM_NODE_PRIMARY];

        if (!node_path)
            continue;

        /* Filter out by the requested device path */
        if (in_drm_dev && !ff_drmDevicesEqual(in_drm_dev, drm_dev))
            continue;

        /* Filter out by the requested vendor id */
        if (vendor_id > 0 &&
            (drm_dev->bustype != DRM_BUS_PCI ||
             vendor_id != drm_dev->deviceinfo.pci->vendor_id))
            continue;

        ret = av_hwdevice_ctx_create(&refs[j].drm_ref, AV_HWDEVICE_TYPE_DRM,
                                     node_path, NULL, 0);
        if (ret < 0)
            continue;

        av_strlcpy(device_path_drm, node_path, device_path_drm_sz);

        refs[j].device_vendor_id = drm_dev->bustype == DRM_BUS_PCI
                                   ? drm_dev->deviceinfo.pci->vendor_id : 0;
        j++;
    }

exit:
    if (in_drm_dev)
        drmFreeDevice(&in_drm_dev);
#   if HAVE_UNISTD_H
    if (in_path_fd >= 0)
        close(in_path_fd);
#   endif
    if (n > 0)
        drmFreeDevices(drm_devs, n);
    return ret;
#else
    return AVERROR(ENOSYS);
#endif
}

int print_drm_device_info(AVTextFormatContext *tfc, AVBufferRef *drm_ref)
{
#if CONFIG_LIBDRM
    AVHWDeviceContext *dev_ctx = NULL;
    AVDRMDeviceContext  *hwctx = NULL;
    drmVersion *drm_ver = NULL;
    drmDevice *drm_dev = NULL;
    int ret = 0;

    if (!tfc || !drm_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)drm_ref->data;
    if (!dev_ctx)
        return AVERROR(ENOSYS);

    hwctx = dev_ctx->hwctx;
    if (hwctx->fd < 0)
        return AVERROR(ENOSYS);

    drm_ver = drmGetVersion(hwctx->fd);
    if (!drm_ver)
        return AVERROR(ENOSYS);

#   ifdef DRM_DEVICE_GET_PCI_REVISION
    ret = drmGetDevice2(hwctx->fd, 0, &drm_dev);
#   else
    ret = drmGetDevice(hwctx->fd, &drm_dev);
#   endif
    if (ret || !drm_dev) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    init_udev_functions();

    mark_section_show_entries(SECTION_ID_DEVICE_INFO_DRM, 1, NULL);
    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_INFO_DRM);

    print_drm_pci_device_name(tfc, drm_dev);

    print_str("device_driver_name", drm_ver->name);
    print_int("device_driver_version_major", drm_ver->version_major);
    print_int("device_driver_version_minor", drm_ver->version_minor);
    print_int("device_driver_version_patchlevel", drm_ver->version_patchlevel);

    if (drm_dev->bustype == DRM_BUS_PCI) {
        print_int("device_vid", drm_dev->deviceinfo.pci->vendor_id);
        print_int("device_did", drm_dev->deviceinfo.pci->device_id);
        print_int("device_sub_vid", drm_dev->deviceinfo.pci->subvendor_id);
        print_int("device_sub_did", drm_dev->deviceinfo.pci->subdevice_id);
        print_int("device_pci_domain", drm_dev->businfo.pci->domain);
        print_int("device_pci_bus", drm_dev->businfo.pci->bus);
        print_int("device_pci_dev", drm_dev->businfo.pci->dev);
        print_int("device_pci_func", drm_dev->businfo.pci->func);
    }

    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_INFO_DRM

    uninit_udev_functions();

exit:
    if (drm_ver)
        drmFreeVersion(drm_ver);
    if (drm_dev)
        drmFreeDevice(&drm_dev);
    return ret;
#endif
    return AVERROR(ENOSYS);
}

/* DRM -> VAAPI */
void create_derive_vaapi_devices_from_drm(HwDeviceRefs *refs)
{
    for (unsigned i = 0; i < FFPROBE_HW_MAX_DEV_NUM && refs && refs[i].drm_ref; i++) {
        av_hwdevice_ctx_create_derived(&refs[i].vaapi_ref, AV_HWDEVICE_TYPE_VAAPI,
                                       refs[i].drm_ref, 0);
    }
}

/* DRM -> VULKAN */
void create_derive_vulkan_devices_from_drm(HwDeviceRefs *refs)
{
    for (unsigned i = 0; i < FFPROBE_HW_MAX_DEV_NUM && refs && refs[i].drm_ref; i++) {
        av_hwdevice_ctx_create_derived(&refs[i].vulkan_ref, AV_HWDEVICE_TYPE_VULKAN,
                                       refs[i].drm_ref, 0);
    }
}

#if (CONFIG_VAAPI && HAVE_VAAPI_DRM)
typedef struct VaapiCodecMode {
    const char          *name;
    const enum AVCodecID codec;
    const int            profile;
    const VAProfile      va_profile;
} VaapiCodecMode;

static const VaapiCodecMode vaapidec_modes[] = {
#   define MAP(n, c, p, v) { n, AV_CODEC_ID_ ## c, AV_PROFILE_ ## p, v }
    MAP("VAAPI MPEG-2 Decoder, Simple",             MPEG2VIDEO, MPEG2_SIMPLE,               0),  /* VAProfileMPEG2Simple */
    MAP("VAAPI MPEG-2 Decoder, Main",               MPEG2VIDEO, MPEG2_MAIN,                 1),  /* VAProfileMPEG2Main */
    MAP("VAAPI H.264 Decoder, ConstrainedBaseline", H264,       H264_CONSTRAINED_BASELINE,  13), /* VAProfileH264ConstrainedBaseline */
    MAP("VAAPI H.264 Decoder, Main",                H264,       H264_MAIN,                  6),  /* VAProfileH264Main */
    MAP("VAAPI H.264 Decoder, High",                H264,       H264_HIGH,                  7),  /* VAProfileH264High */
    MAP("VAAPI H.264 Decoder, High10",              H264,       H264_HIGH_10,               36), /* VAProfileH264High10 */
    MAP("VAAPI H.264 Decoder, High422",             H264,       H264_HIGH_422,              40), /* VAProfileH264High422 */
    MAP("VAAPI VC-1 Decoder, Simple",               VC1,        VC1_SIMPLE,                 8),  /* VAProfileVC1Simple */
    MAP("VAAPI VC-1 Decoder, Main",                 VC1,        VC1_MAIN,                   9),  /* VAProfileVC1Main */
    MAP("VAAPI VC-1 Decoder, Advanced",             VC1,        VC1_ADVANCED,               10), /* VAProfileVC1Advanced */
    MAP("VAAPI MJPEG Decoder, Baseline",            MJPEG,      MJPEG_HUFFMAN_BASELINE_DCT, 12), /* VAProfileJPEGBaseline */
    MAP("VAAPI VP8 Decoder, Profile0_3",            VP8,        UNKNOWN,                    14), /* VAProfileVP8Version0_3 */
    MAP("VAAPI HEVC Decoder, Main",                 HEVC,       HEVC_MAIN,                  17), /* VAProfileHEVCMain */
    MAP("VAAPI HEVC Decoder, Main10",               HEVC,       HEVC_MAIN_10,               18), /* VAProfileHEVCMain10 */
    MAP("VAAPI HEVC Decoder, Main12",               HEVC,       HEVC_REXT,                  23), /* VAProfileHEVCMain12 */
    MAP("VAAPI HEVC Decoder, Main422_10",           HEVC,       HEVC_REXT,                  24), /* VAProfileHEVCMain422_10 */
    MAP("VAAPI HEVC Decoder, Main422_12",           HEVC,       HEVC_REXT,                  25), /* VAProfileHEVCMain422_12 */
    MAP("VAAPI HEVC Decoder, Main444",              HEVC,       HEVC_REXT,                  26), /* VAProfileHEVCMain444 */
    MAP("VAAPI HEVC Decoder, Main444_10",           HEVC,       HEVC_REXT,                  27), /* VAProfileHEVCMain444_10 */
    MAP("VAAPI HEVC Decoder, Main444_12",           HEVC,       HEVC_REXT,                  28), /* VAProfileHEVCMain444_12 */
    MAP("VAAPI VP9 Decoder, Profile0",              VP9,        VP9_0,                      19), /* VAProfileVP9Profile0 */
    MAP("VAAPI VP9 Decoder, Profile1",              VP9,        VP9_1,                      20), /* VAProfileVP9Profile1 */
    MAP("VAAPI VP9 Decoder, Profile2",              VP9,        VP9_2,                      21), /* VAProfileVP9Profile2 */
    MAP("VAAPI VP9 Decoder, Profile3",              VP9,        VP9_3,                      22), /* VAProfileVP9Profile3 */
    MAP("VAAPI AV1 Decoder, Profile0",              AV1,        AV1_MAIN,                   32), /* VAProfileAV1Profile0 */
    MAP("VAAPI VVC Decoder, Main10",                VVC,        VVC_MAIN_10,                37), /* VAProfileVVCMain10 */
    MAP(NULL, NONE, UNKNOWN, -1), /* VAProfileNone */
};

static const VaapiCodecMode vaapienc_modes[] = {
    MAP("VAAPI H.264 Encoder, ConstrainedBaseline", H264,       H264_CONSTRAINED_BASELINE,  13), /* VAProfileH264ConstrainedBaseline */
    MAP("VAAPI H.264 Encoder, Main",                H264,       H264_MAIN,                  6),  /* VAProfileH264Main */
    MAP("VAAPI H.264 Encoder, High",                H264,       H264_HIGH,                  7),  /* VAProfileH264High */
    MAP("VAAPI MJPEG Encoder, Baseline",            MJPEG,      MJPEG_HUFFMAN_BASELINE_DCT, 12), /* VAProfileJPEGBaseline */
    MAP("VAAPI HEVC Encoder, Main",                 HEVC,       HEVC_MAIN,                  17), /* VAProfileHEVCMain */
    MAP("VAAPI HEVC Encoder, Main10",               HEVC,       HEVC_MAIN_10,               18), /* VAProfileHEVCMain10 */
    MAP("VAAPI HEVC Encoder, Main422_10",           HEVC,       HEVC_REXT,                  24), /* VAProfileHEVCMain422_10 */
    MAP("VAAPI HEVC Encoder, Main444",              HEVC,       HEVC_REXT,                  26), /* VAProfileHEVCMain444 */
    MAP("VAAPI HEVC Encoder, Main444_10",           HEVC,       HEVC_REXT,                  27), /* VAProfileHEVCMain444_10 */
    MAP("VAAPI VP9 Encoder, Profile0",              VP9,        VP9_0,                      19), /* VAProfileVP9Profile0 */
    MAP("VAAPI VP9 Encoder, Profile1",              VP9,        VP9_1,                      20), /* VAProfileVP9Profile1 */
    MAP("VAAPI VP9 Encoder, Profile2",              VP9,        VP9_2,                      21), /* VAProfileVP9Profile2 */
    MAP("VAAPI VP9 Encoder, Profile3",              VP9,        VP9_3,                      22), /* VAProfileVP9Profile3 */
    MAP("VAAPI AV1 Encoder, Profile0",              AV1,        AV1_MAIN,                   32), /* VAProfileAV1Profile0 */
    MAP(NULL, NONE, UNKNOWN, -1), /* VAProfileNone */
#   undef MAP
};

enum VaapiVppType {
    VAAPI_VPP_SCALE,
    VAAPI_VPP_DEINT,
    VAAPI_VPP_OVERLAY,
    VAAPI_VPP_ROTATE,
    VAAPI_VPP_FLIP,
    VAAPI_VPP_DENOISE,
    VAAPI_VPP_DETAIL,
    VAAPI_VPP_PROCAMP,
    VAAPI_VPP_TONEMAP,
    VAAPI_VPP_NONE,
};

typedef struct VaapiVppMode {
    const char               *name;
    const enum VaapiVppType   vpp;
    const VAProcFilterType    va_proc;
} VaapiVppMode;

static const VaapiVppMode vaapivpp_modes[] = {
    { "VAAPI VPP Scale Filter",       VAAPI_VPP_SCALE,   0 }, /* VAProcFilterNone */
    { "VAAPI VPP Deinterlace Filter", VAAPI_VPP_DEINT,   2 }, /* VAProcFilterDeinterlacing */
#   if VA_CHECK_VERSION(1, 1, 0)
    { "VAAPI VPP Overlay Filter",     VAAPI_VPP_OVERLAY, 0 }, /* VAProcFilterNone */
    { "VAAPI VPP Rotate Filter",      VAAPI_VPP_ROTATE,  0 }, /* VAProcFilterNone */
    { "VAAPI VPP Flip Filter",        VAAPI_VPP_FLIP,    0 }, /* VAProcFilterNone */
#   endif
    { "VAAPI VPP Denoise Filter",     VAAPI_VPP_DENOISE, 1 }, /* VAProcFilterNoiseReduction */
    { "VAAPI VPP Detail Filter",      VAAPI_VPP_DETAIL,  3 }, /* VAProcFilterSharpening */
    { "VAAPI VPP Procamp Filter",     VAAPI_VPP_PROCAMP, 4 }, /* VAProcFilterColorBalance */
#   if VA_CHECK_VERSION(1, 4, 0)
    { "VAAPI VPP Tonemap Filter",     VAAPI_VPP_TONEMAP, 8 }, /* VAProcFilterHighDynamicRangeToneMapping */
#   endif
    { NULL, VAAPI_VPP_NONE, 0 },
};
#endif

int print_vaapi_device_info(AVTextFormatContext *tfc, AVBufferRef *vaapi_ref, AVBufferRef *drm_ref)
{
#if (CONFIG_VAAPI && HAVE_VAAPI_DRM)
    AVHWDeviceContext  *dev_ctx = NULL;
    AVVAAPIDeviceContext *hwctx = NULL;
    unsigned ver_major = 0;
    unsigned ver_minor = 0;
    const char *vendor_str = NULL;
    unsigned short vendor_id = 0;
    unsigned short device_id = 0;

    if (!tfc || !vaapi_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)vaapi_ref->data;
    if (!dev_ctx)
        return AVERROR(ENOSYS);

    hwctx = dev_ctx->hwctx;

    if (!vaDisplayIsValid(hwctx->display))
        return AVERROR(ENOSYS);

#   if CONFIG_LIBDRM
    if (drm_ref) {
        AVHWDeviceContext *dev_ctx_drm = (AVHWDeviceContext*)drm_ref->data;
        AVDRMDeviceContext *hwctx_drm = dev_ctx_drm ? dev_ctx_drm->hwctx : NULL;
        VADisplay display = hwctx_drm ? vaGetDisplayDRM(hwctx_drm->fd) : NULL;

        if (display) {
#       if VA_CHECK_VERSION(1, 0, 0)
            vaSetErrorCallback(display, NULL, NULL);
            vaSetInfoCallback(display, NULL, NULL);
#       endif
            vaInitialize(display, &ver_major, &ver_minor);
            vaTerminate(display);
        }
    }
#   endif

    vendor_str = vaQueryVendorString(hwctx->display);

#   if VA_CHECK_VERSION(1, 15, 0)
    {
        VADisplayAttribute pci_attr = { .type = VADisplayPCIID };
        VAStatus sts = vaGetDisplayAttributes(hwctx->display, &pci_attr, 1);

        if (sts == VA_STATUS_SUCCESS &&
            pci_attr.flags != VA_DISPLAY_ATTRIB_NOT_SUPPORTED) {
            vendor_id = (pci_attr.value >> 16) & 0xFFFF;
            device_id = pci_attr.value & 0xFFFF;
        }
    }
#   endif

    mark_section_show_entries(SECTION_ID_DEVICE_INFO_VAAPI, 1, NULL);
    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_INFO_VAAPI);

    if (vendor_str) {
        print_str("device_vendor_string", vendor_str);
    }
    if (vendor_id && device_id) {
        print_int("device_vid", vendor_id);
        print_int("device_did", device_id);
    }
    if (ver_major || ver_minor) {
        print_int("device_vaapi_impl_version_major", ver_major);
        print_int("device_vaapi_impl_version_minor", ver_minor);
    }
    print_int("device_vaapi_api_version_major", VA_MAJOR_VERSION);
    print_int("device_vaapi_api_version_minor", VA_MINOR_VERSION);
    print_int("device_vaapi_api_version_micro", VA_MICRO_VERSION);

    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_INFO_VAAPI

    return 0;
#else
    return AVERROR(ENOSYS);
#endif
}

#if (CONFIG_VAAPI && HAVE_VAAPI_DRM)
static int vaapi_map_va_fourcc_to_av_pix_fmt(unsigned fourcc) {
    switch (fourcc) {
    case VA_FOURCC('N','V','1','2'): return AV_PIX_FMT_NV12;
    case VA_FOURCC('P','0','1','0'): return AV_PIX_FMT_P010;
    case VA_FOURCC('P','0','1','2'): return AV_PIX_FMT_P012;

    case VA_FOURCC('Y','U','Y','2'): return AV_PIX_FMT_YUYV422;
    case VA_FOURCC('U','Y','V','Y'): return AV_PIX_FMT_UYVY422;
    case VA_FOURCC('Y','2','1','0'): return AV_PIX_FMT_Y210;
    case VA_FOURCC('Y','2','1','2'): return AV_PIX_FMT_Y212;

    case VA_FOURCC('X','Y','U','V'): return AV_PIX_FMT_VUYX;
    case VA_FOURCC('Y','4','1','0'): return AV_PIX_FMT_XV30;
    case VA_FOURCC('Y','4','1','2'): return AV_PIX_FMT_XV36;

    case VA_FOURCC('Y','8','0','0'): return AV_PIX_FMT_GRAY8;
    case VA_FOURCC('4','1','1','P'): return AV_PIX_FMT_YUV411P;
    case VA_FOURCC('I','4','2','0'): return AV_PIX_FMT_YUV420P;
    case VA_FOURCC('I','M','C','3'): return AV_PIX_FMT_YUV420P;
    case VA_FOURCC('4','2','2','V'): return AV_PIX_FMT_YUV440P;
    case VA_FOURCC('4','2','2','H'): return AV_PIX_FMT_YUV422P;
    case VA_FOURCC('4','4','4','P'): return AV_PIX_FMT_YUV444P;

    case VA_FOURCC('B','G','R','A'): return AV_PIX_FMT_BGRA;
    case VA_FOURCC('B','G','R','X'): return AV_PIX_FMT_BGR0;
    case VA_FOURCC('R','G','B','A'): return AV_PIX_FMT_RGBA;
    case VA_FOURCC('R','G','B','X'): return AV_PIX_FMT_RGB0;
    case VA_FOURCC('A','B','G','R'): return AV_PIX_FMT_ABGR;
    case VA_FOURCC('X','B','G','R'): return AV_PIX_FMT_0BGR;
    case VA_FOURCC('A','R','G','B'): return AV_PIX_FMT_ARGB;
    case VA_FOURCC('X','R','G','B'): return AV_PIX_FMT_0RGB;

    case VA_FOURCC('X','R','3','0'): return AV_PIX_FMT_X2RGB10;
    case VA_FOURCC('X','B','3','0'): return AV_PIX_FMT_X2BGR10;
    default:                         return AV_PIX_FMT_NONE;
    }
}
#endif

static int print_vaapi_codec_info(AVTextFormatContext *tfc, AVBufferRef *vaapi_ref, int print_encoder)
{
#if (CONFIG_VAAPI && HAVE_VAAPI_DRM)
    AVHWDeviceContext  *dev_ctx = NULL;
    AVVAAPIDeviceContext *hwctx = NULL;
    VAStatus sts;
    VAEntrypoint *va_entries = NULL;
    unsigned nb_va_entries = 0;
    VAImageFormat *va_images = NULL;
    unsigned nb_va_images = 0;
    const char *vendor_str = NULL;
    int header_printed = 0;
    unsigned i, j, k;
    int ret = 0;
    const unsigned section_id  = print_encoder ? SECTION_ID_DEVICE_ENCODERS_VAAPI
                                               : SECTION_ID_DEVICE_DECODERS_VAAPI;
    const unsigned section_id2 = print_encoder ? SECTION_ID_DEVICE_ENCODER
                                               : SECTION_ID_DEVICE_DECODER;

    if (!tfc || !vaapi_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)vaapi_ref->data;
    if (!dev_ctx)
        return AVERROR(ENOSYS);

    hwctx = dev_ctx->hwctx;

    if (!vaDisplayIsValid(hwctx->display))
        return AVERROR(ENOSYS);

    nb_va_entries = vaMaxNumEntrypoints(hwctx->display);
    if (!nb_va_entries)
        return AVERROR(ENOSYS);

    va_entries = av_malloc_array(nb_va_entries, sizeof(*va_entries));
    if (!va_entries)
        return AVERROR(ENOMEM);

    nb_va_images = vaMaxNumImageFormats(hwctx->display);
    if (!nb_va_images) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    va_images = av_malloc_array(nb_va_images, sizeof(*va_images));
    if (!va_images) {
        ret = AVERROR(ENOMEM);
        goto exit;
    }

    sts = vaQueryImageFormats(hwctx->display, va_images, &nb_va_images);
    if (sts != VA_STATUS_SUCCESS) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    vendor_str = vaQueryVendorString(hwctx->display);

    for (i = 0; (print_encoder ? vaapienc_modes[i].name : vaapidec_modes[i].name); i++) {
        const VaapiCodecMode *mode = print_encoder ? &vaapienc_modes[i] : &vaapidec_modes[i];

        sts = vaQueryConfigEntrypoints(hwctx->display, mode->va_profile,
                                       va_entries, &nb_va_entries);
        if (sts != VA_STATUS_SUCCESS)
            continue;

        for (j = 0; j < nb_va_entries; j++) {
            const VAEntrypoint e = va_entries[j];
            const int is_dec_vid_entry = e == VAEntrypointVLD;
            const int is_enc_pic_entry = e == VAEntrypointEncPicture;
            const int is_enc_vid_entry_lp = e == 8; /* VAEntrypointEncSliceLP */
            const int is_enc_vid_entry = is_enc_vid_entry_lp ||
                                         e == VAEntrypointEncSlice;
            VAConfigID config_id = VA_INVALID_ID;
            VASurfaceAttrib *surf_attrs = NULL;
            unsigned nb_surf_attrs = 0;
            VAConfigAttrib max_sz_conf_attrs[] = {
                { .type = 18 /* VAConfigAttribMaxPictureWidth  */ },
                { .type = 19 /* VAConfigAttribMaxPictureHeight */ },
            };
            VAConfigAttrib vid_enc_conf_attrs[] = {
                { .type =       VAConfigAttribRateControl         },
                { .type = 21 /* VAConfigAttribEncQualityRange */  },
                { .type =       VAConfigAttribEncMaxRefFrames     },
            };
            unsigned min_width = 0, min_height = 0;
            unsigned max_width = 0, max_height = 0;
            int header2_printed = 0;
            char full_str[128] = { 0 };

            if (print_encoder) {
                if (!(is_enc_vid_entry || is_enc_pic_entry))
                    continue;
                if (mode->codec != AV_CODEC_ID_MJPEG && !is_enc_vid_entry)
                    continue;
                if (mode->codec == AV_CODEC_ID_MJPEG && !is_enc_pic_entry)
                    continue;
            } else if (!is_dec_vid_entry)
                continue;

            sts = vaCreateConfig(hwctx->display, mode->va_profile, e, NULL, 0, &config_id);
            if (sts != VA_STATUS_SUCCESS)
                continue;

            sts = vaQuerySurfaceAttributes(hwctx->display, config_id, NULL, &nb_surf_attrs);
            if (sts != VA_STATUS_SUCCESS)
                goto next;

            surf_attrs = av_malloc_array(nb_surf_attrs, sizeof(*surf_attrs));
            if (!surf_attrs)
                goto next;

            sts = vaQuerySurfaceAttributes(hwctx->display, config_id, surf_attrs, &nb_surf_attrs);
            if (sts != VA_STATUS_SUCCESS)
                goto next;

            for (k = 0; k < nb_surf_attrs; k++) {
                switch (surf_attrs[k].type) {
                case VASurfaceAttribMinWidth:  min_width  = surf_attrs[k].value.value.i; break;
                case VASurfaceAttribMinHeight: min_height = surf_attrs[k].value.value.i; break;
                case VASurfaceAttribMaxWidth:  max_width  = surf_attrs[k].value.value.i; break;
                case VASurfaceAttribMaxHeight: max_height = surf_attrs[k].value.value.i; break;
                }
            }
            sts = vaGetConfigAttributes(hwctx->display, mode->va_profile,
                                        e, max_sz_conf_attrs,
                                        FF_ARRAY_ELEMS(max_sz_conf_attrs));
            if (sts == VA_STATUS_SUCCESS) {
                const unsigned max_pic_width  = max_sz_conf_attrs[0].value;
                const unsigned max_pic_height = max_sz_conf_attrs[1].value;

                if (max_pic_width && max_pic_height &&
                    max_pic_width  != VA_ATTRIB_NOT_SUPPORTED &&
                    max_pic_height != VA_ATTRIB_NOT_SUPPORTED) {
                    max_width  = FFMIN(max_width,  max_pic_width);
                    max_height = FFMIN(max_height, max_pic_height);
                }
            }

            snprintf(full_str, sizeof(full_str), "%s%s",
                     mode->name, is_enc_vid_entry_lp ? " (Low-Power)" : "");

            if (!header_printed) {
                mark_section_show_entries(section_id, 1, NULL);
                avtext_print_section_header(tfc, NULL, section_id);
                header_printed = 1;
            }

            mark_section_show_entries(section_id2, 1, NULL);
            avtext_print_section_header(tfc, NULL, section_id2);
            print_str("codec_name", avcodec_get_name(mode->codec));
            print_int("codec_id", mode->codec);
            print_str("codec_desc", full_str);
            if (min_width && min_height) {
                print_int("min_width",  min_width);
                print_int("min_height", min_height);
            }
            if (max_width && max_height) {
                print_int("max_width",  max_width);
                print_int("max_height", max_height);
            }

            if (print_encoder && mode->codec != AV_CODEC_ID_MJPEG) {
                print_int("low_power", is_enc_vid_entry_lp);

                sts = vaGetConfigAttributes(hwctx->display, mode->va_profile,
                                            e, vid_enc_conf_attrs,
                                            FF_ARRAY_ELEMS(vid_enc_conf_attrs));
                if (sts == VA_STATUS_SUCCESS) {
                    const unsigned rc   = vid_enc_conf_attrs[0].value;
                    const unsigned q_lv = vid_enc_conf_attrs[1].value;
                    const unsigned refs = vid_enc_conf_attrs[2].value;

                    if (rc != VA_ATTRIB_NOT_SUPPORTED) {
                        static const unsigned rc_cqp  = VA_RC_CQP;
                        static const unsigned rc_cbr  = VA_RC_CBR;
                        static const unsigned rc_vbr  = VA_RC_VBR;
                        static const unsigned rc_qvbr = 0x00000400; /* VA_RC_QVBR */
                        static const unsigned rc_icq  = 0x00000040; /* VA_RC_ICQ  */
                        static const unsigned rc_mb   = 0x00000080; /* VA_RC_MB   */
#   define CHECK_RC(x) ((rc & (x)) == (x))
                        print_int("cqp_ratecontrol",  CHECK_RC(rc_cqp));
                        print_int("cbr_ratecontrol",  CHECK_RC(rc_cbr));
                        print_int("cbr_mbbrc",        CHECK_RC(rc_cbr  | rc_mb));
                        print_int("vbr_ratecontrol",  CHECK_RC(rc_vbr));
                        print_int("vbr_mbbrc",        CHECK_RC(rc_vbr  | rc_mb));
                        print_int("qvbr_ratecontrol", CHECK_RC(rc_qvbr));
                        print_int("qvbr_mbbrc",       CHECK_RC(rc_qvbr | rc_mb));
                        print_int("icq_ratecontrol",  CHECK_RC(rc_icq));
                        print_int("icq_mbbrc",        CHECK_RC(rc_icq  | rc_mb));
#   undef CHECK_RC
                    }
                    if (q_lv != VA_ATTRIB_NOT_SUPPORTED) {
                        print_int("max_quality_level", q_lv);
                    }
                    if (refs != VA_ATTRIB_NOT_SUPPORTED) {
                        print_int("max_ref_frames_l0", refs & 0xFFFF);
                        print_int("max_ref_frames_l1", (refs >> 16) & 0xFFFF);
                    }
                }
            }

            /* Profile */
            if (mode->profile != AV_PROFILE_UNKNOWN) {
                mark_section_show_entries(SECTION_ID_DEVICE_PROFILES, 1, NULL);
                avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PROFILES);
                mark_section_show_entries(SECTION_ID_DEVICE_PROFILE, 1, NULL);
                avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PROFILE);
                print_str("profile_name", avcodec_profile_name(mode->codec, mode->profile));
                print_int("profile_id", mode->profile);
                avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PROFILE
                avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PROFILES
            }

            /* Formats */
            for (k = 0; k < nb_surf_attrs; k++) {
                int format = AV_PIX_FMT_NONE;
                unsigned p = 0, surf_fourcc = 0;

                if (surf_attrs[k].type != VASurfaceAttribPixelFormat)
                    continue;

                surf_fourcc = surf_attrs[k].value.value.i;
                format = vaapi_map_va_fourcc_to_av_pix_fmt(surf_fourcc);
                if (format == AV_PIX_FMT_NONE)
                    continue;

                /* Fixup Intel i965 driver false reporting */
                if (e == VAEntrypointEncSlice &&
                    format != AV_PIX_FMT_NV12 &&
                    format != AV_PIX_FMT_P010 &&
                    strstr(vendor_str, "Intel i965 driver"))
                    continue;

                for (p = 0; p < nb_va_images && format != vaapi_map_va_fourcc_to_av_pix_fmt(va_images[p].fourcc); p++);
                if (p >= nb_va_images)
                    continue;

                if (!header2_printed) {
                    mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMTS, 1, NULL);
                    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMTS);
                    header2_printed = 1;
                }
                mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMT, 1, NULL);
                avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMT);
                print_str("format_name", av_get_pix_fmt_name(format));
                print_int("format_id", format);
                print_str("format_fourcc", av_fourcc2str(surf_fourcc));
                avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMT
            }
            if (header2_printed)
                avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMTS

            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_{DEC,ENC}ODER
next:
            if (config_id != VA_INVALID_ID)
                vaDestroyConfig(hwctx->display, config_id);

            av_freep(&surf_attrs);
        }
    }
    if (header_printed)
        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_{DEC,ENC}ODERS_VAAPI

exit:
    av_freep(&va_images);
    av_freep(&va_entries);

    return ret;
#else
    return AVERROR(ENOSYS);
#endif
}

int print_vaapi_decoder_info(AVTextFormatContext *tfc, AVBufferRef *vaapi_ref)
{
    return print_vaapi_codec_info(tfc, vaapi_ref, 0);
}

int print_vaapi_encoder_info(AVTextFormatContext *tfc, AVBufferRef *vaapi_ref)
{
    return print_vaapi_codec_info(tfc, vaapi_ref, 1);
}

#if (CONFIG_VAAPI && HAVE_VAAPI_DRM)
static const char *vaapi_map_vpp_type_to_str(enum VaapiVppType vpp)
{
    switch (vpp) {
    case VAAPI_VPP_SCALE:   return "scale";
    case VAAPI_VPP_DEINT:   return "deint";
    case VAAPI_VPP_OVERLAY: return "overlay";
    case VAAPI_VPP_ROTATE:  return "rotate";
    case VAAPI_VPP_FLIP:    return "flip";
    case VAAPI_VPP_DENOISE: return "denoise";
    case VAAPI_VPP_DETAIL:  return "detail";
    case VAAPI_VPP_PROCAMP: return "procamp";
    case VAAPI_VPP_TONEMAP: return "tonemap";
    default:                return "";
    }
}
#endif

int print_vaapi_vpp_info(AVTextFormatContext *tfc, AVBufferRef *vaapi_ref)
{
#if (CONFIG_VAAPI && HAVE_VAAPI_DRM)
    AVHWDeviceContext  *dev_ctx = NULL;
    AVVAAPIDeviceContext *hwctx = NULL;
    VAStatus sts;
    VAEntrypoint *va_entries = NULL;
    unsigned nb_va_entries = 0;
    VAImageFormat *va_images = NULL;
    unsigned nb_va_images = 0;
    int header_printed = 0;
    unsigned i, j, k;
    int ret = 0;

    if (!tfc || !vaapi_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)vaapi_ref->data;
    if (!dev_ctx)
        return AVERROR(ENOSYS);

    hwctx = dev_ctx->hwctx;

    if (!vaDisplayIsValid(hwctx->display))
        return AVERROR(ENOSYS);

    nb_va_entries = vaMaxNumEntrypoints(hwctx->display);
    if (!nb_va_entries)
        return AVERROR(ENOSYS);

    va_entries = av_malloc_array(nb_va_entries, sizeof(*va_entries));
    if (!va_entries)
        return AVERROR(ENOMEM);

    sts = vaQueryConfigEntrypoints(hwctx->display, VAProfileNone,
                                   va_entries, &nb_va_entries);
    if (sts != VA_STATUS_SUCCESS) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    nb_va_images = vaMaxNumImageFormats(hwctx->display);
    if (!nb_va_images) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    va_images = av_malloc_array(nb_va_images, sizeof(*va_images));
    if (!va_images) {
        ret = AVERROR(ENOMEM);
        goto exit;
    }

    sts = vaQueryImageFormats(hwctx->display, va_images, &nb_va_images);
    if (sts != VA_STATUS_SUCCESS) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    for (i = 0; i < nb_va_entries; i++) {
        VAConfigID config_id = VA_INVALID_ID;
        VAContextID vpp_ctx = VA_INVALID_ID;
        VAProcFilterType va_filters[10] = { 0 }; /* VAProcFilterCount */
        unsigned nb_va_filters = FF_ARRAY_ELEMS(va_filters);
        VASurfaceAttrib *surf_attrs = NULL;
        unsigned nb_surf_attrs = 0;
        unsigned min_width = 0, min_height = 0;
        unsigned max_width = 0, max_height = 0;

        if (va_entries[i] != VAEntrypointVideoProc)
            continue;

        sts = vaCreateConfig(hwctx->display, VAProfileNone, VAEntrypointVideoProc, NULL, 0, &config_id);
        if (sts != VA_STATUS_SUCCESS)
            break;

        sts = vaQuerySurfaceAttributes(hwctx->display, config_id, NULL, &nb_surf_attrs);
        if (sts != VA_STATUS_SUCCESS)
            goto next;

        surf_attrs = av_malloc_array(nb_surf_attrs, sizeof(*surf_attrs));
        if (!surf_attrs)
            goto next;

        sts = vaQuerySurfaceAttributes(hwctx->display, config_id, surf_attrs, &nb_surf_attrs);
        if (sts != VA_STATUS_SUCCESS)
            goto next;

        for (j = 0; j < nb_surf_attrs; j++) {
            switch (surf_attrs[j].type) {
            case VASurfaceAttribMinWidth:  min_width  = surf_attrs[j].value.value.i; break;
            case VASurfaceAttribMinHeight: min_height = surf_attrs[j].value.value.i; break;
            case VASurfaceAttribMaxWidth:  max_width  = surf_attrs[j].value.value.i; break;
            case VASurfaceAttribMaxHeight: max_height = surf_attrs[j].value.value.i; break;
            }
        }

        sts = vaCreateContext(hwctx->display, config_id,
                              FFMIN(FFMAX(min_width, 1280), max_width),
                              FFMIN(FFMAX(min_height, 720), max_height),
                              VA_PROGRESSIVE, NULL, 0, &vpp_ctx);
        if (sts != VA_STATUS_SUCCESS)
            goto next;

        sts = vaQueryVideoProcFilters(hwctx->display, vpp_ctx, va_filters, &nb_va_filters);
        if (sts != VA_STATUS_SUCCESS)
            goto next;

        for (j = 0; vaapivpp_modes[j].name; j++) {
            const VaapiVppMode *mode = &vaapivpp_modes[j];

            for (k = 0; k < nb_va_filters; k++) {
                VABufferID filter_buf = VA_INVALID_ID;
                VABufferID *p_filter_buf = NULL;
                unsigned nb_filter_buf = 0;
                VAProcFilterParameterBuffer               denoise_param_buf = { 0 };
                VAProcFilterParameterBuffer               detail_param_buf = { 0 };
                VAProcFilterParameterBufferDeinterlacing  deint_param_buf = { 0 };
                VAProcFilterParameterBufferColorBalance   procamp_param_buf = { 0 };
#   if VA_CHECK_VERSION(1, 4, 0)
                VAProcFilterParameterBufferHDRToneMapping tonemap_param_buf = { 0 };
#   endif
                void *p_param_buf = NULL;
                size_t sz_param_buf = 0;
                unsigned nb_param_buf = 0;
                VAProcPipelineCaps caps = { 0 };
                unsigned mask_deint_caps = 0;
                unsigned mask_procamp_caps = 0;
                av_unused unsigned mask_tonemap_caps = 0;
                int header2_printed = 0;

                if (mode->va_proc != va_filters[k] &&
                    mode->va_proc != VAProcFilterNone)
                    continue;

                switch (mode->vpp) {
#   define VPP_PARAM_BUF(vpp, param_buf) \
                case (vpp): {                             \
                    (param_buf).type = mode->va_proc;     \
                    p_param_buf      = &(param_buf);      \
                    sz_param_buf     = sizeof(param_buf); \
                    nb_param_buf++;                       \
                    break;                                \
                }
                VPP_PARAM_BUF(VAAPI_VPP_DEINT,   deint_param_buf)
                VPP_PARAM_BUF(VAAPI_VPP_DENOISE, denoise_param_buf)
                VPP_PARAM_BUF(VAAPI_VPP_DETAIL,  detail_param_buf)
                VPP_PARAM_BUF(VAAPI_VPP_PROCAMP, procamp_param_buf)
#   if VA_CHECK_VERSION(1, 4, 0)
                VPP_PARAM_BUF(VAAPI_VPP_TONEMAP, tonemap_param_buf)
#   endif
#   undef VPP_PARAM_BUF
                }

                if (p_filter_buf && sz_param_buf && nb_filter_buf) {
                    sts = vaCreateBuffer(hwctx->display, vpp_ctx,
                                         VAProcFilterParameterBufferType,
                                         sz_param_buf, nb_param_buf, p_param_buf, &filter_buf);
                    if (sts != VA_STATUS_SUCCESS)
                        continue;

                    p_filter_buf = &filter_buf;
                    nb_filter_buf++;
                }

                sts = vaQueryVideoProcPipelineCaps(hwctx->display, vpp_ctx, p_filter_buf, nb_filter_buf, &caps);
                if (sts != VA_STATUS_SUCCESS) {
                    if (filter_buf != VA_INVALID_ID)
                        vaDestroyBuffer(hwctx->display, filter_buf);
                    continue;
                }

#   if VA_CHECK_VERSION(1, 1, 0)
                {
                    unsigned src_flags = 0, dst_flags = 0;

                    switch (mode->vpp) {
                    case VAAPI_VPP_OVERLAY:
                        src_flags = caps.blend_flags;
                        dst_flags = VA_BLEND_GLOBAL_ALPHA;
                        break;
                    case VAAPI_VPP_ROTATE:
                        src_flags = caps.rotation_flags;
                        dst_flags = (1 << VA_ROTATION_90) | (1 << VA_ROTATION_180) | (1 << VA_ROTATION_270);
                        break;
                    case VAAPI_VPP_FLIP:
                        src_flags = caps.mirror_flags;
                        dst_flags = VA_MIRROR_HORIZONTAL | VA_MIRROR_VERTICAL;
                        break;
                    }
                    if (dst_flags && ((src_flags & dst_flags) != dst_flags)) {
                        if (filter_buf != VA_INVALID_ID)
                            vaDestroyBuffer(hwctx->display, filter_buf);
                        continue;
                    }
                }
#   endif

                {
                    VAProcFilterCapDeinterlacing    deint_caps[VAProcDeinterlacingCount] = { 0 };
                    VAProcFilterCap                 denoise_caps = { 0 };
                    VAProcFilterCap                 detail_caps = { 0 };
                    VAProcFilterCapColorBalance     procamp_caps[VAProcColorBalanceCount] = { 0 };
#   if VA_CHECK_VERSION(1, 4, 0)
                    VAProcFilterCapHighDynamicRange tonemap_caps[VAProcHighDynamicRangeMetadataTypeCount] = { 0 };
#   endif
                    void *p_caps = NULL;
                    unsigned nb_caps = 0;

                    switch (mode->vpp) {
                    case VAAPI_VPP_DEINT:
                        p_caps = &deint_caps; nb_caps = FF_ARRAY_ELEMS(deint_caps); break;
                    case VAAPI_VPP_DENOISE:
                        p_caps = &denoise_caps; nb_caps = 1; break;
                    case VAAPI_VPP_DETAIL:
                        p_caps = &detail_caps; nb_caps = 1; break;
                    case VAAPI_VPP_PROCAMP:
                        p_caps = &procamp_caps; nb_caps = FF_ARRAY_ELEMS(procamp_caps); break;
#   if VA_CHECK_VERSION(1, 4, 0)
                    case VAAPI_VPP_TONEMAP:
                        p_caps = &tonemap_caps; nb_caps = FF_ARRAY_ELEMS(tonemap_caps); break;
#   endif
                    }
                    if (p_caps && nb_caps) {
                        unsigned c = 0, mask_caps = 0;
                        const unsigned is_caps_arr = nb_caps > 1;

                        sts = vaQueryVideoProcFilterCaps(hwctx->display, vpp_ctx, mode->va_proc, p_caps, &nb_caps);
                        if (sts == VA_STATUS_SUCCESS && is_caps_arr) {
                            switch (mode->vpp) {
                            case VAAPI_VPP_DEINT:
                                for (c = 0; c < nb_caps; c++)
                                    mask_caps |= (deint_caps[c].type > 0) ? (1 << deint_caps[c].type) : 0;
                                mask_deint_caps = mask_caps;
                                break;
                            case VAAPI_VPP_PROCAMP:
                                for (c = 0; c < nb_caps; c++)
                                    mask_caps |= (procamp_caps[c].type > 0) ? (1 << procamp_caps[c].type) : 0;
                                mask_procamp_caps = mask_caps;
                                break;
#   if VA_CHECK_VERSION(1, 4, 0)
                            case VAAPI_VPP_TONEMAP:
                                for (c = 0; c < nb_caps; c++)
                                    mask_caps |= (tonemap_caps[c].metadata_type == 1) ? tonemap_caps[c].caps_flag : 0;
                                mask_tonemap_caps = mask_caps;
                                break;
#   endif
                            }
                        }
                        if (sts != VA_STATUS_SUCCESS || !nb_caps || (is_caps_arr && !mask_caps)) {
                            if (filter_buf != VA_INVALID_ID)
                                vaDestroyBuffer(hwctx->display, filter_buf);
                            continue;
                        }
                    }
                }

#   if VA_CHECK_VERSION(1, 1, 0)
                {
                    unsigned min_vpp_width = 0, min_vpp_height = 0;
                    unsigned max_vpp_width = 0, max_vpp_height = 0;
#       define INTERSECT_MIN(a, b) (((a) && (b)) ? ((a) > (b) ? (a) : (b)) : ((a) ? (a) : (b)))
#       define INTERSECT_MAX(a, b) (((a) && (b)) ? ((a) < (b) ? (a) : (b)) : ((a) ? (a) : (b)))
                    min_vpp_width  = INTERSECT_MIN(caps.min_input_width,  caps.min_output_width);
                    min_vpp_height = INTERSECT_MIN(caps.min_input_height, caps.min_output_height);
                    max_vpp_width  = INTERSECT_MAX(caps.max_input_width,  caps.max_output_width);
                    max_vpp_height = INTERSECT_MAX(caps.max_input_height, caps.max_output_height);
                    min_width      = INTERSECT_MIN(min_width,  min_vpp_width);
                    min_height     = INTERSECT_MIN(min_height, min_vpp_height);
                    max_width      = INTERSECT_MAX(max_width,  max_vpp_width);
                    max_height     = INTERSECT_MAX(max_height, max_vpp_height);
#       undef INTERSECT_MIN
#       undef INTERSECT_MAX
                }
#   endif

                if (!header_printed) {
                    mark_section_show_entries(SECTION_ID_DEVICE_FILTERS_VAAPI, 1, NULL);
                    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_FILTERS_VAAPI);
                    header_printed = 1;
                }

                mark_section_show_entries(SECTION_ID_DEVICE_FILTER, 1, NULL);
                avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_FILTER);
                print_str("filter_name", vaapi_map_vpp_type_to_str(mode->vpp));
                print_str("filter_desc", mode->name);
                if (min_width && min_height) {
                    print_int("min_width",  min_width);
                    print_int("min_height", min_height);
                }
                if (max_width && max_height) {
                    print_int("max_width",  max_width);
                    print_int("max_height", max_height);
                }

                switch (mode->vpp) {
                case VAAPI_VPP_DEINT:
                    print_int("deint_mode_bob",                !!(mask_deint_caps & (1 << VAProcDeinterlacingBob)));
                    print_int("deint_mode_weave",              !!(mask_deint_caps & (1 << VAProcDeinterlacingWeave)));
                    print_int("deint_mode_motion_adaptive",    !!(mask_deint_caps & (1 << VAProcDeinterlacingMotionAdaptive)));
                    print_int("deint_mode_motion_compensated", !!(mask_deint_caps & (1 << VAProcDeinterlacingMotionCompensated)));
                    break;
                case VAAPI_VPP_PROCAMP:
                    print_int("procamp_mode_hue",              !!(mask_procamp_caps & (1 << VAProcColorBalanceHue)));
                    print_int("procamp_mode_saturation",       !!(mask_procamp_caps & (1 << VAProcColorBalanceSaturation)));
                    print_int("procamp_mode_brightness",       !!(mask_procamp_caps & (1 << VAProcColorBalanceBrightness)));
                    print_int("procamp_mode_contrast",         !!(mask_procamp_caps & (1 << VAProcColorBalanceContrast)));
                    break;
#   if VA_CHECK_VERSION(1, 4, 0)
                case VAAPI_VPP_TONEMAP:
                    print_int("tonemap_mode_hdr10_to_sdr",     !!(mask_tonemap_caps & VA_TONE_MAPPING_HDR_TO_SDR));
                    print_int("tonemap_mode_hdr10_to_hdr10",   !!(mask_tonemap_caps & VA_TONE_MAPPING_HDR_TO_HDR));
                    break;
#   endif
                }

                /* Formats */
                for (k = 0; k < nb_surf_attrs; k++) {
                    int format = AV_PIX_FMT_NONE;
                    unsigned p = 0, surf_fourcc = 0;

                    if (surf_attrs[k].type != VASurfaceAttribPixelFormat)
                        continue;

                    surf_fourcc = surf_attrs[k].value.value.i;
                    format = vaapi_map_va_fourcc_to_av_pix_fmt(surf_fourcc);
                    if (format == AV_PIX_FMT_NONE)
                        continue;

                    for (p = 0; p < nb_va_images && format != vaapi_map_va_fourcc_to_av_pix_fmt(va_images[p].fourcc); p++);
                    if (p >= nb_va_images)
                        continue;

                    if (!header2_printed) {
                        mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMTS, 1, NULL);
                        avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMTS);
                        header2_printed = 1;
                    }
                    mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMT, 1, NULL);
                    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMT);
                    print_str("format_name", av_get_pix_fmt_name(format));
                    print_int("format_id", format);
                    print_str("format_fourcc", av_fourcc2str(surf_fourcc));
                    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMT
                }
                if (header2_printed)
                    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMTS

                avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_FILTER

                if (filter_buf != VA_INVALID_ID)
                    vaDestroyBuffer(hwctx->display, filter_buf);

                break;
            }
        }
next:
        if (vpp_ctx != VA_INVALID_ID)
            vaDestroyContext(hwctx->display, vpp_ctx);

        if (config_id != VA_INVALID_ID)
            vaDestroyConfig(hwctx->display, config_id);

        av_freep(&surf_attrs);

        break;
    }
    if (header_printed)
        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_FILTERS_VAAPI

exit:
    av_freep(&va_images);
    av_freep(&va_entries);

    return ret;
#else
    return AVERROR(ENOSYS);
#endif
}

/* VAAPI -> QSV */
void create_derive_qsv_devices_from_vaapi(HwDeviceRefs *refs)
{
    for (unsigned i = 0; i < FFPROBE_HW_MAX_DEV_NUM && refs && refs[i].vaapi_ref; i++) {
        if (refs[i].device_vendor_id != FFPROBE_HW_VENDOR_ID_INTEL)
            continue;
        av_hwdevice_ctx_create_derived(&refs[i].qsv_ref, AV_HWDEVICE_TYPE_QSV,
                                       refs[i].vaapi_ref, 0);
    }
}

/* VAAPI -> OPENCL */
void create_derive_opencl_devices_from_vaapi(HwDeviceRefs *refs)
{
    for (unsigned i = 0; i < FFPROBE_HW_MAX_DEV_NUM && refs && refs[i].vaapi_ref; i++) {
        if (refs[i].device_vendor_id != FFPROBE_HW_VENDOR_ID_INTEL)
            continue;
        av_hwdevice_ctx_create_derived(&refs[i].opencl_ref, AV_HWDEVICE_TYPE_OPENCL,
                                       refs[i].vaapi_ref, 0);
    }
}
