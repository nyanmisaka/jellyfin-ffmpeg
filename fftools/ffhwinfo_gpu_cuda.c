/*
 * Copyright (c) 2023 NyanMisaka
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ffhwinfo_gpu.h"
#include "libavutil/hwcontext.h"

#if CONFIG_CUDA
#   define CHECK_CU(x) FF_CUDA_CHECK_DL(NULL, cu, x)
#   define CHECK_ML(x) FF_NVML_CHECK_DL(NULL, nvml_ext, x)
#   include "libavutil/cuda_check.h"
#   include "libavutil/hwcontext_cuda_internal.h"
#endif

#if (CONFIG_CUDA && (CONFIG_CUVID || CONFIG_NVDEC))
static CuvidFunctions *cuvid = NULL;
typedef struct CuvidMode {
    const char    *name;
    enum AVCodecID codec;
    const int     *formats;
} CuvidMode;

static const int formats_8_420[]        = { AV_PIX_FMT_NV12, AV_PIX_FMT_NONE };
static const int formats_8_10_420[]     = { AV_PIX_FMT_NV12, AV_PIX_FMT_P010, AV_PIX_FMT_NONE };
static const int formats_8_12_420[]     = { AV_PIX_FMT_NV12, AV_PIX_FMT_P010, AV_PIX_FMT_P016, AV_PIX_FMT_NONE };
static const int formats_8_12_420_444[] = { AV_PIX_FMT_NV12, AV_PIX_FMT_P010, AV_PIX_FMT_P016,
                                            AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV444P16, AV_PIX_FMT_NONE };
static const CuvidMode cuvid_modes[] = {
    { "NVDEC / CUVID MPEG1 decoder", AV_CODEC_ID_MPEG1VIDEO, formats_8_420 },
    { "NVDEC / CUVID MPEG2 decoder", AV_CODEC_ID_MPEG2VIDEO, formats_8_420 },
    { "NVDEC / CUVID MPEG4 decoder", AV_CODEC_ID_MPEG4,      formats_8_420 },
    { "NVDEC / CUVID VC1 decoder",   AV_CODEC_ID_VC1,        formats_8_420 },
    { "NVDEC / CUVID VC1 decoder",   AV_CODEC_ID_WMV3,       formats_8_420 },
    { "NVDEC / CUVID H.264 decoder", AV_CODEC_ID_H264,       formats_8_420 },
    { "NVDEC / CUVID JPEG decoder",  AV_CODEC_ID_MJPEG,      formats_8_420 },
    { "NVDEC / CUVID HEVC decoder",  AV_CODEC_ID_HEVC,       formats_8_12_420_444 },
    { "NVDEC / CUVID VP8 decoder",   AV_CODEC_ID_VP8,        formats_8_420 },
    { "NVDEC / CUVID VP9 decoder",   AV_CODEC_ID_VP9,        formats_8_12_420 },
    { "NVDEC / CUVID AV1 decoder",   AV_CODEC_ID_AV1,        formats_8_10_420 },
    { NULL, 0, NULL },
};
#endif

#if CONFIG_CUDA
static CudaFunctions *cu = NULL;
static CudaFunctionsExt *cu_ext = NULL;
static NvmlFunctionsExt *nvml_ext = NULL;
static char drv_ver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE+1] = {0};
static char nvml_ver[NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE+1] = {0};
static const struct {
    int attr_val;
    const char *attr_str;
} cuda_device_attrs[] = {
    { CU_DEVICE_ATTRIBUTE_CLOCK_RATE,               "ClockRate"              },
    { CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,        "TextureAlignment"       },
    { CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,     "MultiprocessorCount"    },
    { CU_DEVICE_ATTRIBUTE_INTEGRATED,               "Integrated"             },
    { CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,      "CanMapHostMemory"       },
    { CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,             "ComputeMode"            },
    { CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,       "ConcurrentKernels"      },
    { CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,               "PciBusId"               },
    { CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,            "PciDeviceId"            },
    { CU_DEVICE_ATTRIBUTE_TCC_DRIVER,               "TccDriver"              },
    { CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,        "MemoryClockRate"        },
    { CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,  "GlobalMemoryBusWidth"   },
    { CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,       "AsyncEngineCount"       },
    { CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,       "UnifiedAddressing"      },
    { CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,            "PciDomainId"            },
    { CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,  "TexturePitchAlignment"  },
    { CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "ComputeCapabilityMajor" },
    { CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, "ComputeCapabilityMinor" },
    { CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,           "ManagedMemory"          },
    { CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,          "MultiGpuBoard"          },
    { CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, "MultiGpuBoardGroupId"   },
};
#endif

int init_cuda_functions(void)
{
#if CONFIG_CUDA
    int ret = 0;
    if (!cu) {
        ret = cuda_load_functions(&cu, NULL);
        if (ret < 0)
            goto exit;

        ret = CHECK_CU(cu->cuInit(0));
        if (ret < 0)
            goto exit;
    }
    if (!cu_ext) {
        ret = cuda_ext_load_functions(&cu_ext, NULL);
        if (ret < 0)
            goto exit;
    }
    return 0;
exit:
    if (cu)
        cuda_free_functions(&cu);
    if (cu_ext)
        cuda_ext_free_functions(&cu_ext);
    return ret;
#else
    return AVERROR(ENOSYS);
#endif
}

void uninit_cuda_functions(void)
{
#if CONFIG_CUDA
    if (cu)
        cuda_free_functions(&cu);
    if (cu_ext)
        cuda_ext_free_functions(&cu_ext);
#endif
}

int init_nvml_functions(void)
{
#if CONFIG_CUDA
    int ret = 0;
    if (!nvml_ext) {
        ret = nvml_ext_load_functions(&nvml_ext, NULL);
        if (ret < 0)
            goto exit;

        ret = CHECK_ML(nvml_ext->nvmlInit());
        if (ret < 0)
            goto exit;
    }
    return 0;
exit:
    if (nvml_ext) {
        CHECK_ML(nvml_ext->nvmlShutdown());
        nvml_ext_free_functions(&nvml_ext);
    }
    return ret;
#else
    return AVERROR(ENOSYS);
#endif
}

void uninit_nvml_functions(void)
{
#if CONFIG_CUDA
    if (nvml_ext) {
        CHECK_ML(nvml_ext->nvmlShutdown());
        nvml_ext_free_functions(&nvml_ext);
    }
#endif
}

int init_cuvid_functions(void)
{
#if (CONFIG_CUDA && (CONFIG_CUVID || CONFIG_NVDEC))
    int ret = 0;
    if (!cuvid) {
        ret = cuvid_load_functions(&cuvid, NULL);
        if (ret < 0)
            goto exit;
    }
    return 0;
exit:
    if (cuvid)
        cuvid_free_functions(&cuvid);
    return ret;
#else
    return AVERROR(ENOSYS);
#endif
}

void uninit_cuvid_functions(void)
{
#if (CONFIG_CUDA && (CONFIG_CUVID || CONFIG_NVDEC))
    if (cuvid)
        cuvid_free_functions(&cuvid);
#endif
}

/* CUDA */
int create_cuda_devices(HwDeviceRefs *refs)
{
#if CONFIG_CUDA
    unsigned i, j;
    int n = 0, ret = 0;
    char ibuf[4];

    if ((ret = init_cuda_functions()) < 0)
        goto exit;

    ret = CHECK_CU(cu->cuDeviceGetCount(&n));
    if (ret < 0)
        goto exit;

    if (n <= 0) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    n = FFMIN(n, HWINFO_MAX_DEV_NUM);
    for (i = 0, j = 0; i < n && refs; i++) {
        snprintf(ibuf, sizeof(ibuf), "%d", i);
        ret = av_hwdevice_ctx_create(&refs[j].cuda_ref, AV_HWDEVICE_TYPE_CUDA,
                                     ibuf, NULL, 0);
        if (ret < 0)
            continue;

        refs[j].device_index_cuda = i;
        refs[j].device_vendor_id  = HWINFO_VENDOR_ID_NVIDIA;
        ++j;
    }

    ret = 0;

exit:
    return ret;
#else
    return AVERROR(ENOSYS);
#endif
}

// win32 cudaDeviceProp::luid -> DXGI_ADAPTER_DESC::AdapterLuid
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#matching-device-luids

/* CUDA -> D3D11VA */
void create_derive_d3d11va_devices_from_cuda(HwDeviceRefs *refs)
{
#if CONFIG_CUDA
    int ret = 0;

    if (!refs)
        return;
    if ((ret = init_cuda_functions()) < 0)
        return;
    if (!cu_ext->cuDeviceGetLuid)
        return;

    for (unsigned i = 0; i < HWINFO_MAX_DEV_NUM && refs[i].cuda_ref; i++) {
        char cuda_luid[8];
        unsigned int node_mask;
        AVHWDeviceContext *dev_ctx = (AVHWDeviceContext*)refs[i].cuda_ref->data;
        AVCUDADeviceContext *hwctx = dev_ctx->hwctx;

        /* Values are undefined on TCC and non-Windows platforms */
        ret = CHECK_CU(cu_ext->cuDeviceGetLuid(cuda_luid, &node_mask,
                                               hwctx->internal->cuda_device));
        if (ret < 0)
            continue;

        create_d3d11va_devices_with_filter(refs, -1, i, cuda_luid);
    }
#endif
}

int init_nvml_driver_version(void)
{
#if CONFIG_CUDA
    int ret = 0;

    if ((ret = init_nvml_functions()) < 0)
        return ret;

    ret = CHECK_ML(nvml_ext->nvmlSystemGetDriverVersion(drv_ver,
                                                        NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE));
    if (ret < 0)
        return ret;

    ret = CHECK_ML(nvml_ext->nvmlSystemGetNVMLVersion(nvml_ver,
                                                      NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE));
    if (ret < 0)
        return ret;

    return 0;
#else
    return AVERROR(ENOSYS);
#endif
}

int print_cuda_device_info(WriterContext *wctx, AVBufferRef *cuda_ref, int nvml_ret)
{
#if CONFIG_CUDA
    AVHWDeviceContext *dev_ctx = NULL;
    AVCUDADeviceContext *hwctx = NULL;
    CUdevice dev;
    int val, cuda_ver = 0, ret = 0;
    char device_name[256] = {0};

    if (!wctx || !cuda_ref)
        return AVERROR(EINVAL);

    if ((ret = init_cuda_functions()) < 0)
        return AVERROR(ENOSYS);

    dev_ctx = (AVHWDeviceContext*)cuda_ref->data;
    hwctx = dev_ctx->hwctx;
    dev = hwctx->internal->cuda_device;

    ret = CHECK_CU(cu->cuDeviceGetName(device_name, sizeof(device_name), dev));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu_ext->cuDriverGetVersion(&cuda_ver));
    if (ret < 0)
        return ret;

    mark_section_show_entries(SECTION_ID_DEVICE_INFO_CUDA, 1, NULL);
    writer_print_section_header(wctx, SECTION_ID_DEVICE_INFO_CUDA);

    print_str("DeviceName", device_name);
    if (nvml_ret == 0) {
        print_str("DriverVersion", drv_ver);
        print_str("NvmlVersion", nvml_ver);
    }
    print_int("CudaVersion", cuda_ver);

    for (unsigned i = 0; i < FF_ARRAY_ELEMS(cuda_device_attrs); i++) {
        val = 0;
        ret = CHECK_CU(cu->cuDeviceGetAttribute(&val, cuda_device_attrs[i].attr_val, dev));
        if (ret == 0)
            print_int(cuda_device_attrs[i].attr_str, val);
    }

    writer_print_section_footer(wctx);

    return ret;
#else
    return 0;
#endif
}

#if (CONFIG_CUDA && (CONFIG_CUVID || CONFIG_NVDEC))
static int cuda_map_av_to_cuvid_codec(enum AVCodecID codec)
{
    switch (codec) {
    case AV_CODEC_ID_MPEG1VIDEO: return cudaVideoCodec_MPEG1;
    case AV_CODEC_ID_MPEG2VIDEO: return cudaVideoCodec_MPEG2;
    case AV_CODEC_ID_MPEG4:      return cudaVideoCodec_MPEG4;
    case AV_CODEC_ID_WMV3:       /* fallthrough */
    case AV_CODEC_ID_VC1:        return cudaVideoCodec_VC1;
    case AV_CODEC_ID_H264:       return cudaVideoCodec_H264;
    case AV_CODEC_ID_MJPEG:      return cudaVideoCodec_JPEG;
    case AV_CODEC_ID_HEVC:       return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_VP8:        return cudaVideoCodec_VP8;
    case AV_CODEC_ID_VP9:        return cudaVideoCodec_VP9;
    case AV_CODEC_ID_AV1:        return cudaVideoCodec_AV1;
    default:                     return -1;
    }
}

static int cuda_map_av_to_cuvid_chroma(enum AVPixelFormat pix_fmt)
{
    switch (pix_fmt) {
    case AV_PIX_FMT_NV12:      /* fallthrough */
    case AV_PIX_FMT_P010:      /* fallthrough */
    case AV_PIX_FMT_P016:      return cudaVideoChromaFormat_420;
    case AV_PIX_FMT_YUV444P:   /* fallthrough */
    case AV_PIX_FMT_YUV444P16: return cudaVideoChromaFormat_444;
    default:                   return -1;
    }
}

static int cuda_map_av_to_cuvid_surface(enum AVPixelFormat pix_fmt)
{
    switch (pix_fmt) {
    case AV_PIX_FMT_NV12:      return cudaVideoSurfaceFormat_NV12;
    case AV_PIX_FMT_P010:      /* fallthrough */
    case AV_PIX_FMT_P016:      return cudaVideoSurfaceFormat_P016;
    case AV_PIX_FMT_YUV444P:   return cudaVideoSurfaceFormat_YUV444;
    case AV_PIX_FMT_YUV444P16: return cudaVideoSurfaceFormat_YUV444_16Bit;
    default:                   return -1;
    }
}
#endif

int print_cuda_decoder_info(WriterContext *wctx, AVBufferRef *cuda_ref)
{
#if (CONFIG_CUDA && (CONFIG_CUVID || CONFIG_NVDEC))
    AVHWDeviceContext *dev_ctx = NULL;
    AVCUDADeviceContext *hwctx = NULL;
    CUVIDDECODECAPS caps = {0};
    CUcontext dummy;
    int header_printed = 0;
    int ret = 0;
    unsigned i, j;

    if (!wctx || !cuda_ref)
        return AVERROR(EINVAL);

    if ((ret = init_cuda_functions()) < 0)
        return AVERROR(ENOSYS);

    if ((ret = init_cuvid_functions()) < 0)
        return AVERROR(ENOSYS);

    if (!cuvid->cuvidGetDecoderCaps)
        return AVERROR(ENOSYS);

    dev_ctx = (AVHWDeviceContext*)cuda_ref->data;
    hwctx = dev_ctx->hwctx;

    ret = CHECK_CU(cu->cuCtxPushCurrent(hwctx->cuda_ctx));
    if (ret < 0)
        return ret;

    for (i = 0; cuvid_modes[i].name; i++) {
        int header2_printed = 0;
        const CuvidMode *mode = &cuvid_modes[i];
        if (!mode->formats)
            continue;

        caps.eCodecType = cuda_map_av_to_cuvid_codec(mode->codec);
        if (caps.eCodecType < 0)
            continue;

        for (j = 0; mode->formats[j] != AV_PIX_FMT_NONE; j++) {
            int surface = -1;
            const int format = mode->formats[j];
            const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(format);

            caps.nBitDepthMinus8 = FFMIN(desc->comp[0].depth, 12) - 8;
            caps.eChromaFormat = cuda_map_av_to_cuvid_chroma(format);
            if (caps.nBitDepthMinus8 < 0 || caps.eChromaFormat < 0)
                continue;

            ret = CHECK_CU(cuvid->cuvidGetDecoderCaps(&caps));
            if (ret < 0)
                continue;

            if (!caps.bIsSupported)
                continue;

            surface = cuda_map_av_to_cuvid_surface(format);
            if (surface < 0 || !(caps.nOutputFormatMask & (1 << surface)))
                continue;

            if (!header_printed) {
                mark_section_show_entries(SECTION_ID_DECODERS_CUDA, 1, NULL);
                writer_print_section_header(wctx, SECTION_ID_DECODERS_CUDA);
                header_printed = 1;
            }

            if (!header2_printed) {
                mark_section_show_entries(SECTION_ID_DECODER, 1, NULL);
                writer_print_section_header(wctx, SECTION_ID_DECODER);
                print_str("CodecName", avcodec_get_name(mode->codec));
                print_int("CodecId", mode->codec);
                print_str("CodecDesc", mode->name);
                print_int("MinWidth", caps.nMinWidth);
                print_int("MinHeight", caps.nMinHeight);
                print_int("MaxWidth", caps.nMaxWidth);
                print_int("MaxHeight", caps.nMaxHeight);
                print_int("MaxMBCount", caps.nMaxMBCount);
                mark_section_show_entries(SECTION_ID_PIXEL_FORMATS, 1, NULL);
                writer_print_section_header(wctx, SECTION_ID_PIXEL_FORMATS);
                header2_printed = 1;
            }

            mark_section_show_entries(SECTION_ID_PIXEL_FORMAT, 1, NULL);
            writer_print_section_header(wctx, SECTION_ID_PIXEL_FORMAT);
            print_str("FormatName", av_get_pix_fmt_name(format));
            print_int("FormatId", format);
            writer_print_section_footer(wctx);
        }

        if (header2_printed) {
            writer_print_section_footer(wctx);
            writer_print_section_footer(wctx);
        }
    }

    if (header_printed)
        writer_print_section_footer(wctx);

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return 0;
#else
    return 0;
#endif
}

int print_cuda_encoder_info(WriterContext *wctx, AVBufferRef *cuda_ref)
{
    return 0;
}
