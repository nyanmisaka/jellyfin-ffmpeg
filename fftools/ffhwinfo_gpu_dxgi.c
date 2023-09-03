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

#if CONFIG_D3D11VA
#   define COBJMACROS
#   include <windows.h>
#   include <initguid.h>
#   include <d3d11.h>
#   include <dxgi1_2.h>
#   include "libavutil/hwcontext_d3d11va.h"
#   include "compat/w32dlfcn.h"
#endif

#if CONFIG_D3D11VA
#   include <winapifamily.h>
#   if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#       define BUILD_FOR_UAP 0
#   else
#       define BUILD_FOR_UAP 1
#   endif
#   if defined(WINAPI_FAMILY)
#       undef WINAPI_FAMILY
#   endif
#   define WINAPI_FAMILY WINAPI_PARTITION_DESKTOP
#   include <wbemidl.h>
#endif

#if CONFIG_D3D11VA
typedef struct DxvaRes {
    const char    *name;
    const unsigned width;
    const unsigned height;
} DxvaRes;

typedef struct DxvaMode {
    const char    *name;
    const char    *guid_name;
    const GUID    *guid;
    enum AVCodecID codec;
    const int      legacy;
    const int     *profiles;
    const int     *formats;
} DxvaMode;

static const DxvaRes dxva_res_ascend[] = {
    { "64x64",     64,   64   },
    { "128x128",   128,  128  },
    { "144x144",   144,  144  },
    { "256x256",   256,  256  },
    { "720x480",   720,  480  },
    { "1280x720",  1280, 720  },
    { "2048x1024", 2048, 1024 },
    { "1920x1080", 1920, 1080 },
    { "1920x1088", 1920, 1088 },
    { "2560x1440", 2560, 1440 },
    { "2048x2048", 2048, 2048 },
    { "3840x2160", 3840, 2160 },
    { "4096x2160", 4096, 2160 },
    { "4096x2304", 4096, 2304 },
    { "4096x2318", 4096, 2318 },
    { "3840x3840", 3840, 3840 },
    { "4080x4080", 4080, 4080 },
    { "4096x4096", 4096, 4096 },
    { "7680x4320", 7680, 4320 },
    { "8192x4320", 8192, 4320 },
    { "8192x4352", 8192, 4352 },
    { "8192x8192", 8192, 8192 },
    { NULL,        0,    0    },
};

static const int prof_mpeg2_main[] =   { FF_PROFILE_MPEG2_SIMPLE,
                                         FF_PROFILE_MPEG2_MAIN,
                                         FF_PROFILE_UNKNOWN };
static const int prof_h264_high[] =    { FF_PROFILE_H264_CONSTRAINED_BASELINE,
                                         FF_PROFILE_H264_MAIN,
                                         FF_PROFILE_H264_HIGH,
                                         FF_PROFILE_UNKNOWN };
static const int prof_hevc_main[] =    { FF_PROFILE_HEVC_MAIN,
                                         FF_PROFILE_UNKNOWN };
static const int prof_hevc_main10[] =  { FF_PROFILE_HEVC_MAIN_10,
                                         FF_PROFILE_UNKNOWN };
static const int prof_hevc_rext[] =    { FF_PROFILE_HEVC_REXT,
                                         FF_PROFILE_UNKNOWN };
static const int prof_vp9_profile0[] = { FF_PROFILE_VP9_0,
                                         FF_PROFILE_UNKNOWN };
static const int prof_vp9_profile2[] = { FF_PROFILE_VP9_2,
                                         FF_PROFILE_UNKNOWN };
static const int prof_av1_profile0[] = { FF_PROFILE_AV1_MAIN,
                                         FF_PROFILE_UNKNOWN };

static const int format_nv12[] =         { AV_PIX_FMT_NV12, AV_PIX_FMT_NONE };
static const int format_p010[] =         { AV_PIX_FMT_P010, AV_PIX_FMT_NONE };
static const int format_p010_nv12[] =    { AV_PIX_FMT_P010, AV_PIX_FMT_NV12, AV_PIX_FMT_NONE };
static const int format_p012[] =         { AV_PIX_FMT_P012, AV_PIX_FMT_NONE };
static const int format_y210_yuyv422[] = { AV_PIX_FMT_Y210, AV_PIX_FMT_YUYV422, AV_PIX_FMT_NONE };
static const int format_y212[] =         { AV_PIX_FMT_Y212, AV_PIX_FMT_NONE };
static const int format_vuyx[] =         { AV_PIX_FMT_VUYX, AV_PIX_FMT_NONE };
static const int format_xv30[] =         { AV_PIX_FMT_XV30, AV_PIX_FMT_NONE };
static const int format_xv36[] =         { AV_PIX_FMT_XV36, AV_PIX_FMT_NONE };

DEFINE_GUID(_DXVA_ModeMPEG2_VLD,                 0xee27417f,0x5e28,0x4e65,0xbe,0xea,0x1d,0x26,0xb5,0x08,0xad,0xc9);
DEFINE_GUID(_DXVA_ModeMPEG2and1_VLD,             0x86695f12,0x340e,0x4f04,0x9f,0xd3,0x92,0x53,0xdd,0x32,0x74,0x60);
DEFINE_GUID(_DXVA_ModeH264_E,                    0x1b81be68,0xa0c7,0x11d3,0xb9,0x84,0x00,0xc0,0x4f,0x2e,0x73,0xc5);
DEFINE_GUID(_DXVA_ModeH264_F,                    0x1b81be69,0xa0c7,0x11d3,0xb9,0x84,0x00,0xc0,0x4f,0x2e,0x73,0xc5);
DEFINE_GUID(_DXVA_ModeH264_E_Intel,              0x604F8E68,0x4951,0x4C54,0x88,0xFE,0xAB,0xD2,0x5C,0x15,0xB3,0xD6);
DEFINE_GUID(_DXVA_ModeVC1_D,                     0x1b81beA3,0xa0c7,0x11d3,0xb9,0x84,0x00,0xc0,0x4f,0x2e,0x73,0xc5);
DEFINE_GUID(_DXVA_ModeVC1_D2010,                 0x1b81beA4,0xa0c7,0x11d3,0xb9,0x84,0x00,0xc0,0x4f,0x2e,0x73,0xc5);
DEFINE_GUID(_DXVA_ModeHEVC_VLD_Main,             0x5b11d51b,0x2f4c,0x4452,0xbc,0xc3,0x09,0xf2,0xa1,0x16,0x0c,0xc0);
DEFINE_GUID(_DXVA_ModeHEVC_VLD_Main10,           0x107af0e0,0xef1a,0x4d19,0xab,0xa8,0x67,0xa1,0x63,0x07,0x3d,0x13);
DEFINE_GUID(_DXVA_ModeHEVC_VLD_Main12_Intel,     0x8ff8a3aa,0xc456,0x4132,0xb6,0xef,0x69,0xd9,0xdd,0x72,0x57,0x1d);
DEFINE_GUID(_DXVA_ModeHEVC_VLD_Main422_10_Intel, 0xe484dcb8,0xcac9,0x4859,0x99,0xf5,0x5c,0x0d,0x45,0x06,0x90,0x89);
DEFINE_GUID(_DXVA_ModeHEVC_VLD_Main422_12_Intel, 0xc23dd857,0x874b,0x423c,0xb6,0xe0,0x82,0xce,0xaa,0x9b,0x11,0x8a);
DEFINE_GUID(_DXVA_ModeHEVC_VLD_Main444_Intel,    0x41a5af96,0xe415,0x4b0c,0x9d,0x03,0x90,0x78,0x58,0xe2,0x3e,0x78);
DEFINE_GUID(_DXVA_ModeHEVC_VLD_Main444_10_Intel, 0x6a6a81ba,0x912a,0x485d,0xb5,0x7f,0xcc,0xd2,0xd3,0x7b,0x8d,0x94);
DEFINE_GUID(_DXVA_ModeHEVC_VLD_Main444_12_Intel, 0x5b08e35d,0x0c66,0x4c51,0xa6,0xf1,0x89,0xd0,0x0c,0xb2,0xc1,0x97);
DEFINE_GUID(_DXVA_ModeVP9_VLD_Profile0,          0x463707f8,0xa1d0,0x4585,0x87,0x6d,0x83,0xaa,0x6d,0x60,0xb8,0x9e);
DEFINE_GUID(_DXVA_ModeVP9_VLD_10bit_Profile2,    0xa4c749ef,0x6ecf,0x48aa,0x84,0x48,0x50,0xa7,0xa1,0x16,0x5f,0xf7);
DEFINE_GUID(_DXVA_ModeAV1_VLD_Profile0,          0xb8be4ccb,0xcf53,0x46ba,0x8d,0x59,0xd6,0xb8,0xa6,0xda,0x5d,0x2a);

static const DxvaMode dxva_modes[] = {
    { "MPEG-2 variable-length decoder", "DXVA_ModeMPEG2_VLD", &_DXVA_ModeMPEG2_VLD,
        AV_CODEC_ID_MPEG2VIDEO, 1, prof_mpeg2_main, format_nv12 },
    { "MPEG-2 & MPEG-1 variable-length decoder", "DXVA_ModeMPEG2and1_VLD", &_DXVA_ModeMPEG2and1_VLD,
        AV_CODEC_ID_MPEG2VIDEO, 1, prof_mpeg2_main, format_nv12 },
    { "H.264 variable-length decoder, no film grain technology", "DXVA_ModeH264_E", &_DXVA_ModeH264_E,
        AV_CODEC_ID_H264, 1, prof_h264_high, format_nv12 },
    { "H.264 variable-length decoder, film grain technology", "DXVA_ModeH264_F", &_DXVA_ModeH264_F,
        AV_CODEC_ID_H264, 1, prof_h264_high, format_nv12 },
    { "H.264 variable-length decoder, no film grain technology (Intel)", "DXVA_ModeH264_E_Intel", &_DXVA_ModeH264_E_Intel,
        AV_CODEC_ID_H264, 1, prof_h264_high, format_nv12 },
    { "VC-1 variable-length decoder", "DXVA_ModeVC1_D", &_DXVA_ModeVC1_D,
        AV_CODEC_ID_VC1, 1, NULL, format_nv12 },
    { "VC-1 variable-length decoder (2010)", "DXVA_ModeVC1_D2010", &_DXVA_ModeVC1_D2010,
        AV_CODEC_ID_VC1, 1, NULL, format_nv12 },
    { "VC-1 variable-length decoder", "DXVA_ModeVC1_D", &_DXVA_ModeVC1_D,
        AV_CODEC_ID_WMV3, 1, NULL, format_nv12 },
    { "VC-1 variable-length decoder (2010)", "DXVA_ModeVC1_D2010", &_DXVA_ModeVC1_D2010,
        AV_CODEC_ID_WMV3, 1, NULL, format_nv12 },
    { "HEVC / H.265 variable-length decoder, main", "DXVA_ModeHEVC_VLD_Main", &_DXVA_ModeHEVC_VLD_Main,
        AV_CODEC_ID_HEVC, 0, prof_hevc_main, format_nv12 },
    { "HEVC / H.265 variable-length decoder, main10", "DXVA_ModeHEVC_VLD_Main10", &_DXVA_ModeHEVC_VLD_Main10,
        AV_CODEC_ID_HEVC, 0, prof_hevc_main10, format_p010 },
    { "HEVC / H.265 variable-length decoder, main12 (Intel)", "DXVA_ModeHEVC_VLD_Main12_Intel", &_DXVA_ModeHEVC_VLD_Main12_Intel,
        AV_CODEC_ID_HEVC, 0, prof_hevc_rext, format_p012 },
    { "HEVC / H.265 variable-length decoder, main422_10 (Intel)", "DXVA_ModeHEVC_VLD_Main422_10_Intel", &_DXVA_ModeHEVC_VLD_Main422_10_Intel,
        AV_CODEC_ID_HEVC, 0, prof_hevc_rext, format_y210_yuyv422 },
    { "HEVC / H.265 variable-length decoder, main422_12 (Intel)", "DXVA_ModeHEVC_VLD_Main422_12_Intel", &_DXVA_ModeHEVC_VLD_Main422_12_Intel,
        AV_CODEC_ID_HEVC, 0, prof_hevc_rext, format_y212 },
    { "HEVC / H.265 variable-length decoder, main444 (Intel)", "DXVA_ModeHEVC_VLD_Main444_Intel", &_DXVA_ModeHEVC_VLD_Main444_Intel,
        AV_CODEC_ID_HEVC, 0, prof_hevc_rext, format_vuyx },
    { "HEVC / H.265 variable-length decoder, main444_10 (Intel)", "DXVA_ModeHEVC_VLD_Main444_10_Intel", &_DXVA_ModeHEVC_VLD_Main444_10_Intel,
        AV_CODEC_ID_HEVC, 0, prof_hevc_rext, format_xv30 },
    { "HEVC / H.265 variable-length decoder, main444_12 (Intel)", "DXVA_ModeHEVC_VLD_Main444_12_Intel", &_DXVA_ModeHEVC_VLD_Main444_12_Intel,
        AV_CODEC_ID_HEVC, 0, prof_hevc_rext, format_xv36 },
    { "VP9 variable-length decoder, profile 0", "DXVA_ModeVP9_VLD_Profile0", &_DXVA_ModeVP9_VLD_Profile0,
        AV_CODEC_ID_VP9, 0, prof_vp9_profile0, format_nv12 },
    { "VP9 variable-length decoder, profile 0", "DXVA_ModeVP9_VLD_10bit_Profile2", &_DXVA_ModeVP9_VLD_10bit_Profile2,
        AV_CODEC_ID_VP9, 0, prof_vp9_profile2, format_p010 },
    { "AV1 variable-length decoder, profile 0", "DXVA_ModeAV1_VLD_Profile0", &_DXVA_ModeAV1_VLD_Profile0,
        AV_CODEC_ID_AV1, 0, prof_av1_profile0, format_p010_nv12 },
    { NULL, NULL, NULL,
        0, 0, NULL, NULL },
};

static DXGI_FORMAT d3d11va_map_av_to_dxgi_format(enum AVPixelFormat pix_fmt)
{
    switch (pix_fmt) {
    case AV_PIX_FMT_NV12:    return DXGI_FORMAT_NV12;
    case AV_PIX_FMT_P010:    return DXGI_FORMAT_P010;
    case AV_PIX_FMT_P012:    return DXGI_FORMAT_P016;
    case AV_PIX_FMT_YUYV422: return DXGI_FORMAT_YUY2;
    case AV_PIX_FMT_Y210:    return DXGI_FORMAT_Y210;
    case AV_PIX_FMT_Y212:    return DXGI_FORMAT_Y216;
    case AV_PIX_FMT_VUYX:    return DXGI_FORMAT_AYUV;
    case AV_PIX_FMT_XV30:    return DXGI_FORMAT_Y410;
    case AV_PIX_FMT_XV36:    return DXGI_FORMAT_Y416;
    case AV_PIX_FMT_YUV420P: return DXGI_FORMAT_420_OPAQUE;
    default:                 return DXGI_FORMAT_UNKNOWN;
    }
}
#endif

/* D3D11VA */
#if CONFIG_D3D11VA
typedef HRESULT(WINAPI *PFN_CREATE_DXGI_FACTORY)(REFIID riid, void **ppFactory);
static PFN_CREATE_DXGI_FACTORY mCreateDXGIFactory = NULL;
#endif

int create_d3d11va_devices(HwDeviceRefs *refs)
{
    return create_d3d11va_devices_with_filter(refs, -1, -1, NULL);
}

int create_d3d11va_devices_with_filter(HwDeviceRefs *refs, int vendor_id, int idx_luid, char *luid)
{
#if CONFIG_D3D11VA
    int ret = 0;
    int single = (idx_luid >= 0) && luid;
    unsigned i, j;
    char ibuf[4];
    HRESULT hr;
    DXGI_ADAPTER_DESC desc;
    IDXGIFactory2 *pDXGIFactory = NULL;
    IDXGIAdapter *pDXGIAdapter = NULL;
#if !HAVE_UWP
    HANDLE dxgilib;

    if (!mCreateDXGIFactory) {
        dxgilib = dlopen("dxgi.dll", 0);
        if (!dxgilib)
            return AVERROR(ENOSYS);

        mCreateDXGIFactory = (PFN_CREATE_DXGI_FACTORY) GetProcAddress(dxgilib, "CreateDXGIFactory");
    }
#else
    if (!mCreateDXGIFactory)
        mCreateDXGIFactory = (PFN_CREATE_DXGI_FACTORY) CreateDXGIFactory1;
#endif
    if (!mCreateDXGIFactory)
        return AVERROR(ENOSYS);

    hr = mCreateDXGIFactory(&IID_IDXGIFactory2, (void **)&pDXGIFactory);
    if (FAILED(hr))
        return AVERROR(ENOSYS);

    for (i = 0, j = 0; i < HWINFO_MAX_DEV_NUM && refs; i++) {
        hr = IDXGIFactory2_EnumAdapters(pDXGIFactory, i, &pDXGIAdapter);
        if (FAILED(hr))
            continue;

        hr = IDXGIAdapter2_GetDesc(pDXGIAdapter, &desc);
        if (pDXGIAdapter)
            IDXGIAdapter_Release(pDXGIAdapter);

        /* Filter out 'Microsoft Basic Render Driver' */
        if (SUCCEEDED(hr) && desc.VendorId == 0x1414)
            continue;

        /* Filter out by the requested vendor id */
        if (SUCCEEDED(hr) && vendor_id > 0 && desc.VendorId != vendor_id)
            continue;

        /* Filter by the requested LUID on Windows */
        if (single && SUCCEEDED(hr)) {
            LUID dxgiLuid = desc.AdapterLuid;

            if (!memcmp(&dxgiLuid.LowPart, luid, sizeof(dxgiLuid.LowPart)) &&
                !memcmp(&dxgiLuid.HighPart, luid + sizeof(dxgiLuid.LowPart), sizeof(dxgiLuid.HighPart))) {
                snprintf(ibuf, sizeof(ibuf), "%d", i);
                av_hwdevice_ctx_create(&refs[idx_luid].d3d11va_ref,
                                       AV_HWDEVICE_TYPE_D3D11VA,
                                       ibuf, NULL, 0);
                if (ret >= 0) {
                    refs[idx_luid].device_index_dxgi = i;
                    refs[idx_luid].device_vendor_id  = desc.VendorId;
                }
                break;
            } else
                continue;
        }

        snprintf(ibuf, sizeof(ibuf), "%d", i);
        ret = av_hwdevice_ctx_create(&refs[j].d3d11va_ref,
                                     AV_HWDEVICE_TYPE_D3D11VA,
                                     ibuf, NULL, 0);
        if (ret < 0)
            continue;

        refs[j].device_index_dxgi = i;
        refs[j].device_vendor_id  = desc.VendorId;
        ++j;
    }

    IDXGIFactory2_Release(pDXGIFactory);
    return 0;
#else
    return AVERROR(ENOSYS);
#endif
}

/* D3D11VA -> QSV */
void create_derive_qsv_devices_from_d3d11va(HwDeviceRefs *refs)
{
    for (unsigned i = 0; i < HWINFO_MAX_DEV_NUM && refs && refs[i].d3d11va_ref; i++) {
        if (refs[i].device_vendor_id != HWINFO_VENDOR_ID_INTEL)
            continue;
        av_hwdevice_ctx_create_derived(&refs[i].qsv_ref, AV_HWDEVICE_TYPE_QSV,
                                       refs[i].d3d11va_ref, 0);
    }
}

/* D3D11VA -> OPENCL */
void create_derive_opencl_devices_from_d3d11va(HwDeviceRefs *refs)
{
    for (unsigned i = 0; i < HWINFO_MAX_DEV_NUM && refs && refs[i].d3d11va_ref; i++) {
        if (!(refs[i].device_vendor_id == HWINFO_VENDOR_ID_INTEL ||
              refs[i].device_vendor_id == HWINFO_VENDOR_ID_AMD))
            continue;
        av_hwdevice_ctx_create_derived(&refs[i].opencl_ref, AV_HWDEVICE_TYPE_OPENCL,
                                       refs[i].d3d11va_ref, 0);
    }
}

/* D3D11VA -> CUDA */
void create_derive_cuda_devices_from_d3d11va(HwDeviceRefs *refs)
{
    for (unsigned i = 0; i < HWINFO_MAX_DEV_NUM && refs && refs[i].d3d11va_ref; i++) {
        if (refs[i].device_vendor_id != HWINFO_VENDOR_ID_NVIDIA)
            continue;
        av_hwdevice_ctx_create_derived(&refs[i].cuda_ref, AV_HWDEVICE_TYPE_CUDA,
                                       refs[i].d3d11va_ref, 0);
    }
}

#if CONFIG_D3D11VA
static int print_d3d11va_driver_version(WriterContext *wctx, DXGI_ADAPTER_DESC desc)
{
    int ret = 0;
    HRESULT hr;
    IWbemLocator *pLoc = NULL;
    IWbemServices *pSvc = NULL;
    IEnumWbemClassObject* pEnumerator = NULL;
    BSTR bRootNamespace;
    BSTR bWQL;
    BSTR bVideoController;

    if (!wctx)
        return AVERROR(EINVAL);

    bRootNamespace = SysAllocString(L"ROOT\\CIMV2");
    bWQL = SysAllocString(L"WQL");
    {
        WCHAR lookup[256];
        _snwprintf(lookup, FF_ARRAY_ELEMS(lookup),
                   L"SELECT * FROM Win32_VideoController WHERE PNPDeviceID LIKE 'PCI\\\\VEN_%04X&DEV_%04X&SUBSYS_%08X&REV_%02X%%'",
                   desc.VendorId, desc.DeviceId,
                   desc.SubSysId, desc.Revision);
        bVideoController = SysAllocString(lookup);
    }

    hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr))
    {
        av_log(NULL, AV_LOG_ERROR, "Unable to initialize COM library!\n");
        return AVERROR(ENOSYS);
    }

    {
        MULTI_QI res = { 0 };
        res.pIID = &IID_IWbemLocator;
#if !BUILD_FOR_UAP
        hr = CoCreateInstanceEx(&CLSID_WbemLocator, NULL, CLSCTX_INPROC_SERVER, 0,
                                1, &res);
#else // BUILD_FOR_UAP
        hr = CoCreateInstanceFromApp(&CLSID_WbemLocator, NULL, CLSCTX_INPROC_SERVER, 0,
                                     1, &res);
#endif // BUILD_FOR_UAP
        if (FAILED(hr) || FAILED(res.hr))
        {
            ret = AVERROR(ENOSYS);
            av_log(NULL, AV_LOG_ERROR, "Failed to create IWbemLocator object!\n");
            goto exit;
        }
        pLoc = (IWbemLocator *)res.pItf;
    }

    hr = IWbemLocator_ConnectServer(pLoc, bRootNamespace,
                                    NULL, NULL, NULL, 0, NULL, NULL, &pSvc);
    if (FAILED(hr))
    {
        ret = AVERROR(ENOSYS);
        av_log(NULL, AV_LOG_ERROR, "Could not connect to namespace!\n");
        goto exit;
    }

#if !BUILD_FOR_UAP
    hr = CoSetProxyBlanket(
        (IUnknown*)pSvc,
        RPC_C_AUTHN_WINNT,
        RPC_C_AUTHZ_NONE,
        NULL,
        RPC_C_AUTHN_LEVEL_CALL,
        RPC_C_IMP_LEVEL_IMPERSONATE,
        NULL,
        EOAC_NONE
    );
    if (FAILED(hr))
    {
        ret = AVERROR(ENOSYS);
        av_log(NULL, AV_LOG_ERROR, "Could not set proxy blanket!\n");
        goto exit;
    }
#endif // !BUILD_FOR_UAP

    hr = IWbemServices_ExecQuery(pSvc, bWQL, bVideoController,
        WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY, NULL, &pEnumerator);
    if (FAILED(hr) || !pEnumerator)
    {
        ret = AVERROR(ENOSYS);
        av_log(NULL, AV_LOG_ERROR, "Query for Win32_VideoController failed!\n");
        goto exit;
    }

    {
        IWbemClassObject *pclsObj = NULL;
        ULONG uReturn = 0;

        hr = IEnumWbemClassObject_Next(pEnumerator, WBEM_INFINITE, 1, &pclsObj, &uReturn);
        if (!uReturn)
        {
            ret = AVERROR(ENOSYS);
            av_log(NULL, AV_LOG_ERROR, "Failed to find the device!\n");
            goto exit;
        }

        {
            VARIANT vtProp;
            const WCHAR *szData;
            int model, feature, revision, build;

            VariantInit(&vtProp);

            hr = IWbemClassObject_Get(pclsObj, L"DriverVersion", 0, &vtProp, 0, 0);
            if (FAILED(hr))
            {
                ret = AVERROR(ENOSYS);
                av_log(NULL, AV_LOG_ERROR, "Failed to read the driver version!\n");
                goto exit;
            }

            /* see https://docs.microsoft.com/en-us/windows-hardware/drivers/display/wddm-2-1-features#driver-versioning */
            szData = (WCHAR *)vtProp.bstrVal;
            if (swscanf(szData, L"%d.%d.%d.%d", &model, &feature, &revision, &build) != 4)
            {
                ret = AVERROR(ENOSYS);
                av_log(NULL, AV_LOG_ERROR, "The adapter DriverVersion '%ls' doesn't match the expected format!\n", szData);
                goto exit;
            }
#if 0
            if (desc.VendorId == HWINFO_VENDOR_ID_INTEL && revision >= 100)
            {
                /* new Intel driver format */
                build += (revision - 100) * 1000;
            }
#endif
            VariantClear(&vtProp);

            print_int("WddmModelVersion", model);
            print_int("WddmD3dFeatureLevel", feature);
            print_int("WddmVendorRevision", revision);
            print_int("WddmVendorBuild", build);
        }
        IWbemClassObject_Release(pclsObj);
    }

exit:
    SysFreeString(bRootNamespace);
    SysFreeString(bWQL);
    SysFreeString(bVideoController);
    if (pEnumerator)
        IEnumWbemClassObject_Release(pEnumerator);
    if (pSvc)
        IWbemServices_Release(pSvc);
    if (pLoc)
        IWbemLocator_Release(pLoc);
    CoUninitialize();
    return ret;
}
#endif

int print_d3d11va_device_info(WriterContext *wctx, AVBufferRef *d3d11va_ref)
{
#if CONFIG_D3D11VA
    AVHWDeviceContext    *dev_ctx = NULL;
    AVD3D11VADeviceContext *hwctx = NULL;
    IDXGIDevice     *pDXGIDevice = NULL;
    IDXGIAdapter   *pDXGIAdapter = NULL;
    DXGI_ADAPTER_DESC desc;
    D3D_FEATURE_LEVEL level;
    HRESULT hr;
    char device_desc[256] = {0};
    int uma = -1, ext_sharing = -1, ret = 0;

    if (!wctx || !d3d11va_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)d3d11va_ref->data;
    hwctx = dev_ctx->hwctx;

    hr = ID3D11Device_QueryInterface(hwctx->device, &IID_IDXGIDevice, (void **)&pDXGIDevice);
    if (FAILED(hr)) {
        ret = AVERROR(ENOSYS);
        av_log(NULL, AV_LOG_ERROR, "ID3D11Device_QueryInterface failed!\n");
        goto exit;
    }

    hr = IDXGIDevice_GetAdapter(pDXGIDevice, &pDXGIAdapter);
    if (FAILED(hr)) {
        ret = AVERROR(ENOSYS);
        av_log(NULL, AV_LOG_ERROR, "IDXGIDevice_GetAdapter failed!\n");
        goto exit;
    }
  
    hr = IDXGIAdapter_GetDesc(pDXGIAdapter, &desc);
    if (FAILED(hr)) {
        ret = AVERROR(ENOSYS);
        av_log(NULL, AV_LOG_ERROR, "IDXGIAdapter_GetDesc failed!\n");
        goto exit;
    }

    level = ID3D11Device_GetFeatureLevel(hwctx->device);

    {
        D3D11_FEATURE_DATA_D3D11_OPTIONS data;
        D3D11_FEATURE_DATA_D3D11_OPTIONS2 data2;

        hr = ID3D11Device_CheckFeatureSupport(hwctx->device,
                                              D3D11_FEATURE_D3D11_OPTIONS,
                                              &data, sizeof(data));
        ext_sharing = SUCCEEDED(hr) && data.ExtendedResourceSharing;

        hr = ID3D11Device_CheckFeatureSupport(hwctx->device,
                                              D3D11_FEATURE_D3D11_OPTIONS2,
                                              &data2, sizeof(data2));
        uma = SUCCEEDED(hr) && data2.UnifiedMemoryArchitecture;
    }

    wcstombs(device_desc, desc.Description, 128);

    mark_section_show_entries(SECTION_ID_DEVICE_INFO_D3D11VA, 1, NULL);
    writer_print_section_header(wctx, SECTION_ID_DEVICE_INFO_D3D11VA);

    print_str("Description", device_desc);
    print_int("VendorId", desc.VendorId);
    print_int("DeviceId", desc.DeviceId);
    print_int("SubSysId", desc.SubSysId);
    print_int("Revision", desc.Revision);
    print_int("DedicatedVideoMemory", desc.DedicatedVideoMemory);
    print_int("DedicatedSystemMemory", desc.DedicatedSystemMemory);
    print_int("SharedSystemMemory", desc.SharedSystemMemory);
    print_int("AdapterLuidLowPart", desc.AdapterLuid.LowPart);
    print_int("AdapterLuidHighPart", desc.AdapterLuid.HighPart);
    print_int("FeatureLevel", level);
    print_int("ExtendedResourceSharing", ext_sharing);
    print_int("UnifiedMemoryArchitecture", uma);

    print_d3d11va_driver_version(wctx, desc);

    writer_print_section_footer(wctx);

exit:
    if (pDXGIAdapter)
        IDXGIAdapter_Release(pDXGIAdapter);
    if (pDXGIDevice)
        IDXGIDevice_Release(pDXGIDevice);
    return ret;
#else
    return 0;
#endif
}

#if CONFIG_D3D11VA
static int check_d3d11va_decoder_config(ID3D11VideoDevice *video_device,
                                        D3D11_VIDEO_DECODER_DESC desc,
                                        enum AVCodecID codec)
{
    unsigned i, cfg_cnt = 0;
    D3D11_VIDEO_DECODER_CONFIG *cfg_list = NULL;
    HRESULT hr;

    hr = ID3D11VideoDevice_GetVideoDecoderConfigCount(video_device, &desc, &cfg_cnt);
    if (FAILED(hr)) {
        av_log(NULL, AV_LOG_ERROR, "Unable to retrieve decoder configurations count\n");
        return AVERROR(EINVAL);
    }

    cfg_list = av_malloc_array(cfg_cnt, sizeof(D3D11_VIDEO_DECODER_CONFIG));
    if (!cfg_list)
        return AVERROR(ENOMEM);

    for (i = 0; i < cfg_cnt; i++) {
        D3D11_VIDEO_DECODER_CONFIG *cfg = &cfg_list[i];

        hr = ID3D11VideoDevice_GetVideoDecoderConfig(video_device, &desc, i, cfg);
        if (FAILED(hr)) {
            av_log(NULL, AV_LOG_ERROR, "Unable to retrieve decoder configurations. (hr=0x%lX)\n", hr);
            av_free(cfg_list);
            return AVERROR(EINVAL);
        }

        if (cfg->ConfigBitstreamRaw == 1 ||
            (codec == AV_CODEC_ID_H264 && cfg->ConfigBitstreamRaw == 2)) {
            av_free(cfg_list);
            return 1;
        }
    }

    av_free(cfg_list);
    return 0;
}
#endif

int print_d3d11va_decoder_info(WriterContext *wctx, AVBufferRef *d3d11va_ref)
{
#if CONFIG_D3D11VA
    AVHWDeviceContext    *dev_ctx = NULL;
    AVD3D11VADeviceContext *hwctx = NULL;
    HRESULT hr;
    int header_printed = 0;
    unsigned i, j, p_cnt = 0;
    GUID *p_list = NULL;

    if (!wctx || !d3d11va_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)d3d11va_ref->data;
    hwctx = dev_ctx->hwctx;

    p_cnt = ID3D11VideoDevice_GetVideoDecoderProfileCount(hwctx->video_device);
    p_list = av_malloc_array(p_cnt, sizeof(*p_list));
    if (!p_list || p_cnt == 0) {
        av_log(NULL, AV_LOG_ERROR, "Failed to get the decoder GUIDs\n");
        av_free(p_list);
        return AVERROR(EINVAL);
    }

    for (i = 0; i < p_cnt; i++) {
        hr = ID3D11VideoDevice_GetVideoDecoderProfile(hwctx->video_device, i, &p_list[i]);
        if (FAILED(hr)) {
            av_log(NULL, AV_LOG_ERROR, "Failed to retrieve decoder GUID %d\n", i);
            av_free(p_list);
            return AVERROR(EINVAL);
        }
    }

    for (i = 0; dxva_modes[i].name; i++) {
        int supported = 0;
        unsigned min_width = 0, min_height = 0;
        unsigned max_width = 0, max_height = 0;
        const DxvaMode *mode = &dxva_modes[i];
        D3D11_VIDEO_DECODER_DESC desc = {0};
        DXGI_FORMAT dxgi_fmt = DXGI_FORMAT_UNKNOWN;

        for (const GUID *g = &p_list[0]; !supported && g < &p_list[p_cnt]; g++) {
            supported = IsEqualGUID(mode->guid, g);
        }
        if (!supported)
            continue;

        if (!mode->formats)
            continue;

        /* Use the most significant format for this profile */
        dxgi_fmt = d3d11va_map_av_to_dxgi_format(mode->formats[0]);
        if (dxgi_fmt == DXGI_FORMAT_UNKNOWN)
            continue;

        desc.Guid = *mode->guid;

        /* Check min res first */
        for (const DxvaRes *r = &dxva_res_ascend[0]; r->name; r++) {
            if (mode->legacy && (r->width > 4096 || r->height > 4096))
                break;

            desc.SampleWidth = r->width;
            desc.SampleHeight = r->height;
            desc.OutputFormat = dxgi_fmt;

            if (1 == check_d3d11va_decoder_config(hwctx->video_device, desc, mode->codec)) {
                min_width = r->width;
                min_height = r->height;
                break;
            }
        }
        if (min_width == 0 || min_height == 0)
            continue;

        /* Test rest formats for this profile, bail out directly if any fails */
        for (j = 0; mode->formats[j] != AV_PIX_FMT_NONE; j++) {
            desc.SampleWidth = min_width;
            desc.SampleHeight = min_height;
            desc.OutputFormat = d3d11va_map_av_to_dxgi_format(mode->formats[0]);

            if (1 != check_d3d11va_decoder_config(hwctx->video_device, desc, mode->codec)) {
                min_width = min_height = 0;
                break;
            }
        }
        if (min_width == 0 || min_height == 0)
            continue;

        /* Check max res */
        for (const DxvaRes *r = &dxva_res_ascend[FF_ARRAY_ELEMS(dxva_res_ascend)]; r >= &dxva_res_ascend[0]; r--) {
            if (!r->name)
                continue;
            if (r->width <= min_width && r->height <= min_width)
                break;
            if (mode->legacy && (r->width > 4096 || r->height > 4096))
                continue;

            desc.SampleWidth = r->width;
            desc.SampleHeight = r->height;
            desc.OutputFormat = dxgi_fmt;

            if (check_d3d11va_decoder_config(hwctx->video_device, desc, mode->codec) == 1) {
                max_width = r->width;
                max_height = r->height;
                break;
            }
        }
        if (max_width == 0 || max_height == 0)
            continue;

        if (!header_printed) {
            mark_section_show_entries(SECTION_ID_DECODERS_D3D11VA, 1, NULL);
            writer_print_section_header(wctx, SECTION_ID_DECODERS_D3D11VA);
            header_printed = 1;
        }

        mark_section_show_entries(SECTION_ID_DECODER, 1, NULL);
        writer_print_section_header(wctx, SECTION_ID_DECODER);
        print_str("CodecName", avcodec_get_name(mode->codec));
        print_int("CodecId", mode->codec);
        print_str("GuidDesc", mode->name);
        print_str("GuidName", mode->guid_name);
        print_int("MinWidth", min_width);
        print_int("MinHeight", min_height);
        print_int("MaxWidth", max_width);
        print_int("MaxHeight", max_height);

        /* PixelFormats */
        if (mode->formats) {
            mark_section_show_entries(SECTION_ID_PIXEL_FORMATS, 1, NULL);
            writer_print_section_header(wctx, SECTION_ID_PIXEL_FORMATS);
            for (j = 0; mode->formats[j] != AV_PIX_FMT_NONE; j++) {
                mark_section_show_entries(SECTION_ID_PIXEL_FORMAT, 1, NULL);
                writer_print_section_header(wctx, SECTION_ID_PIXEL_FORMAT);
                print_str("FormatName", av_get_pix_fmt_name(mode->formats[j]));
                print_int("FormatId", mode->formats[j]);
                writer_print_section_footer(wctx);
            }
            writer_print_section_footer(wctx);
        }

        /* Profiles */
        if (mode->profiles) {
            mark_section_show_entries(SECTION_ID_PROFILES, 1, NULL);
            writer_print_section_header(wctx, SECTION_ID_PROFILES);
            for (j = 0; mode->profiles[j] != FF_PROFILE_UNKNOWN; j++) {
                mark_section_show_entries(SECTION_ID_PROFILE, 1, NULL);
                writer_print_section_header(wctx, SECTION_ID_PROFILE);
                print_str("ProfileName", avcodec_profile_name(mode->codec, mode->profiles[j]));
                print_int("ProfileId", mode->profiles[j]);
                writer_print_section_footer(wctx);
            }
            writer_print_section_footer(wctx);
        }

        writer_print_section_footer(wctx);
    }

    if (header_printed)
        writer_print_section_footer(wctx);

    return 0;
#else
    return 0;
#endif
}
