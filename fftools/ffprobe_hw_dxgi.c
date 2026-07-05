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

#if CONFIG_D3D11VA
#   define COBJMACROS
#   include <windows.h>
#   include <wctype.h>
#   include <initguid.h>
#   include <d3d11.h>
#   include <dxgi1_2.h>
#   include "libavutil/hwcontext_d3d11va.h"
#   include "libavutil/wchar_filename.h"
#   include "compat/w32dlfcn.h"
#endif

#if CONFIG_D3D11VA
#   if HAVE_IDXGIADAPTER3 && HAVE_PDH
#       include <dxgi1_4.h>
#       include <pdh.h>
#       define FFPROBE_HW_DXGI1_4_PDH_HELPER
#   endif
#endif

#if CONFIG_D3D11VA && CONFIG_AMF && !HAVE_UWP
#   if HAVE_ADVAPI32 && HAVE_CFGMGR32 && HAVE_OLE32
#       include <cfgmgr32.h>
#       include <devguid.h>
#       define FFPROBE_HW_WDDM_VERSION_HELPER
#       define FFPROBE_HW_AMF_DLL_HELPER
#   endif
#endif

#if CONFIG_D3D11VA
typedef struct DxvaMode {
    const char               *name;
    const enum AVCodecID      codec;
    const int                *profiles;
    const enum AVPixelFormat *formats;
    const GUID               *guid;
    const unsigned            legacy;
} DxvaMode;

static const int prof_mpeg2_main[] = {
    AV_PROFILE_MPEG2_SIMPLE,
    AV_PROFILE_MPEG2_MAIN,
    AV_PROFILE_UNKNOWN
};
static const int prof_h264_high[] = {
    AV_PROFILE_H264_CONSTRAINED_BASELINE,
    AV_PROFILE_H264_MAIN,
    AV_PROFILE_H264_HIGH,
    AV_PROFILE_UNKNOWN
};
static const int prof_vc1_advanced[] = {
    AV_PROFILE_VC1_SIMPLE,
    AV_PROFILE_VC1_MAIN,
    AV_PROFILE_VC1_ADVANCED,
    AV_PROFILE_UNKNOWN
};
static const int prof_hevc_main[] = {
    AV_PROFILE_HEVC_MAIN,
    AV_PROFILE_UNKNOWN
};
static const int prof_hevc_main10[] = {
    AV_PROFILE_HEVC_MAIN_10,
    AV_PROFILE_UNKNOWN
};
static const int prof_hevc_rext[] = {
    AV_PROFILE_HEVC_REXT,
    AV_PROFILE_UNKNOWN
};
static const int prof_vp9_profile0[] = {
    AV_PROFILE_VP9_0,
    AV_PROFILE_UNKNOWN
};
static const int prof_vp9_profile2[] = {
    AV_PROFILE_VP9_2,
    AV_PROFILE_UNKNOWN
};
static const int prof_av1_profile0[] = {
    AV_PROFILE_AV1_MAIN,
    AV_PROFILE_UNKNOWN
};

static const enum AVPixelFormat formats_8_420[]    = { AV_PIX_FMT_NV12, AV_PIX_FMT_NONE };
static const enum AVPixelFormat formats_10_420[]   = { AV_PIX_FMT_P010, AV_PIX_FMT_NONE };
static const enum AVPixelFormat formats_8_10_420[] = { AV_PIX_FMT_P010, AV_PIX_FMT_NV12, AV_PIX_FMT_NONE };
static const enum AVPixelFormat formats_12_420[]   = { AV_PIX_FMT_P012, AV_PIX_FMT_NONE };
static const enum AVPixelFormat formats_8_10_422[] = { AV_PIX_FMT_Y210, AV_PIX_FMT_YUYV422, AV_PIX_FMT_NONE };
static const enum AVPixelFormat formats_12_422[]   = { AV_PIX_FMT_Y212, AV_PIX_FMT_NONE };
static const enum AVPixelFormat formats_8_444[]    = { AV_PIX_FMT_VUYX, AV_PIX_FMT_NONE };
static const enum AVPixelFormat formats_10_444[]   = { AV_PIX_FMT_XV30, AV_PIX_FMT_NONE };
static const enum AVPixelFormat formats_12_444[]   = { AV_PIX_FMT_XV36, AV_PIX_FMT_NONE };

DEFINE_GUID(ff_DXVA_ModeMPEG2_VLD,                 0xee27417f,0x5e28,0x4e65,0xbe,0xea,0x1d,0x26,0xb5,0x08,0xad,0xc9);
DEFINE_GUID(ff_DXVA_ModeMPEG2and1_VLD,             0x86695f12,0x340e,0x4f04,0x9f,0xd3,0x92,0x53,0xdd,0x32,0x74,0x60);
DEFINE_GUID(ff_DXVA_ModeH264_E,                    0x1b81be68,0xa0c7,0x11d3,0xb9,0x84,0x00,0xc0,0x4f,0x2e,0x73,0xc5);
DEFINE_GUID(ff_DXVA_ModeH264_F,                    0x1b81be69,0xa0c7,0x11d3,0xb9,0x84,0x00,0xc0,0x4f,0x2e,0x73,0xc5);
DEFINE_GUID(ff_DXVA_ModeH264_E_Intel,              0x604F8E68,0x4951,0x4C54,0x88,0xFE,0xAB,0xD2,0x5C,0x15,0xB3,0xD6);
DEFINE_GUID(ff_DXVA_ModeVC1_D,                     0x1b81beA3,0xa0c7,0x11d3,0xb9,0x84,0x00,0xc0,0x4f,0x2e,0x73,0xc5);
DEFINE_GUID(ff_DXVA_ModeVC1_D2010,                 0x1b81beA4,0xa0c7,0x11d3,0xb9,0x84,0x00,0xc0,0x4f,0x2e,0x73,0xc5);
DEFINE_GUID(ff_DXVA_ModeHEVC_VLD_Main,             0x5b11d51b,0x2f4c,0x4452,0xbc,0xc3,0x09,0xf2,0xa1,0x16,0x0c,0xc0);
DEFINE_GUID(ff_DXVA_ModeHEVC_VLD_Main10,           0x107af0e0,0xef1a,0x4d19,0xab,0xa8,0x67,0xa1,0x63,0x07,0x3d,0x13);
DEFINE_GUID(ff_DXVA_ModeHEVC_VLD_Main12_Intel,     0x8ff8a3aa,0xc456,0x4132,0xb6,0xef,0x69,0xd9,0xdd,0x72,0x57,0x1d);
DEFINE_GUID(ff_DXVA_ModeHEVC_VLD_Main422_10_Intel, 0xe484dcb8,0xcac9,0x4859,0x99,0xf5,0x5c,0x0d,0x45,0x06,0x90,0x89);
DEFINE_GUID(ff_DXVA_ModeHEVC_VLD_Main422_12_Intel, 0xc23dd857,0x874b,0x423c,0xb6,0xe0,0x82,0xce,0xaa,0x9b,0x11,0x8a);
DEFINE_GUID(ff_DXVA_ModeHEVC_VLD_Main444_Intel,    0x41a5af96,0xe415,0x4b0c,0x9d,0x03,0x90,0x78,0x58,0xe2,0x3e,0x78);
DEFINE_GUID(ff_DXVA_ModeHEVC_VLD_Main444_10_Intel, 0x6a6a81ba,0x912a,0x485d,0xb5,0x7f,0xcc,0xd2,0xd3,0x7b,0x8d,0x94);
DEFINE_GUID(ff_DXVA_ModeHEVC_VLD_Main444_12_Intel, 0x5b08e35d,0x0c66,0x4c51,0xa6,0xf1,0x89,0xd0,0x0c,0xb2,0xc1,0x97);
DEFINE_GUID(ff_DXVA_ModeVP9_VLD_Profile0,          0x463707f8,0xa1d0,0x4585,0x87,0x6d,0x83,0xaa,0x6d,0x60,0xb8,0x9e);
DEFINE_GUID(ff_DXVA_ModeVP9_VLD_10bit_Profile2,    0xa4c749ef,0x6ecf,0x48aa,0x84,0x48,0x50,0xa7,0xa1,0x16,0x5f,0xf7);
DEFINE_GUID(ff_DXVA_ModeAV1_VLD_Profile0,          0xb8be4ccb,0xcf53,0x46ba,0x8d,0x59,0xd6,0xb8,0xa6,0xda,0x5d,0x2a);

static const DxvaMode dxva_modes[] = {
#   define MAP(n, c, p, f, g, l) { n, AV_CODEC_ID_ ## c, p, f, g, l }
    MAP("D3D11VA MPEG-2 Decoder",             MPEG2VIDEO, prof_mpeg2_main,      formats_8_420, &ff_DXVA_ModeMPEG2_VLD,                 1),
    MAP("D3D11VA MPEG-2 and MPEG-1 Decoder",  MPEG2VIDEO, prof_mpeg2_main,      formats_8_420, &ff_DXVA_ModeMPEG2and1_VLD,             1),
    MAP("D3D11VA H.264 Decoder, NoFGT",             H264, prof_h264_high,       formats_8_420, &ff_DXVA_ModeH264_E,                    0),
    MAP("D3D11VA H.264 Decoder, FGT",               H264, prof_h264_high,       formats_8_420, &ff_DXVA_ModeH264_F,                    1),
    MAP("D3D11VA H.264 Decoder, NoFGT (Intel)",     H264, prof_h264_high,       formats_8_420, &ff_DXVA_ModeH264_E_Intel,              1),
    MAP("D3D11VA VC-1 Decoder",                      VC1, prof_vc1_advanced,    formats_8_420, &ff_DXVA_ModeVC1_D,                     1),
    MAP("D3D11VA VC-1 Decoder (2010)",               VC1, prof_vc1_advanced,    formats_8_420, &ff_DXVA_ModeVC1_D2010,                 1),
    MAP("D3D11VA HEVC Decoder, Main",               HEVC, prof_hevc_main,       formats_8_420, &ff_DXVA_ModeHEVC_VLD_Main,             0),
    MAP("D3D11VA HEVC Decoder, Main10",             HEVC, prof_hevc_main10,    formats_10_420, &ff_DXVA_ModeHEVC_VLD_Main10,           0),
    MAP("D3D11VA HEVC Decoder, Main12 (Intel)",     HEVC, prof_hevc_rext,      formats_12_420, &ff_DXVA_ModeHEVC_VLD_Main12_Intel,     0),
    MAP("D3D11VA HEVC Decoder, Main422_10 (Intel)", HEVC, prof_hevc_rext,    formats_8_10_422, &ff_DXVA_ModeHEVC_VLD_Main422_10_Intel, 0),
    MAP("D3D11VA HEVC Decoder, Main422_12 (Intel)", HEVC, prof_hevc_rext,      formats_12_422, &ff_DXVA_ModeHEVC_VLD_Main422_12_Intel, 0),
    MAP("D3D11VA HEVC Decoder, Main444 (Intel)",    HEVC, prof_hevc_rext,       formats_8_444, &ff_DXVA_ModeHEVC_VLD_Main444_Intel,    0),
    MAP("D3D11VA HEVC Decoder, Main444_10 (Intel)", HEVC, prof_hevc_rext,      formats_10_444, &ff_DXVA_ModeHEVC_VLD_Main444_10_Intel, 0),
    MAP("D3D11VA HEVC Decoder, Main444_12 (Intel)", HEVC, prof_hevc_rext,      formats_12_444, &ff_DXVA_ModeHEVC_VLD_Main444_12_Intel, 0),
    MAP("D3D11VA VP9 Decoder, Profile 0",            VP9, prof_vp9_profile0,    formats_8_420, &ff_DXVA_ModeVP9_VLD_Profile0,          0),
    MAP("D3D11VA VP9 Decoder, Profile 2",            VP9, prof_vp9_profile2,   formats_10_420, &ff_DXVA_ModeVP9_VLD_10bit_Profile2,    0),
    MAP("D3D11VA AV1 Decoder, Profile 0",            AV1, prof_av1_profile0, formats_8_10_420, &ff_DXVA_ModeAV1_VLD_Profile0,          0),
    MAP(NULL, NONE, NULL, NULL, NULL, 0),
#   undef MAP
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

#if CONFIG_D3D11VA
#   if !HAVE_UWP
typedef UINT ff_D3DKMT_HANDLE;

#       pragma pack(push, 8)
typedef struct _ff_D3DKMT_OPENADAPTERFROMLUID {
    LUID             AdapterLuid;
    ff_D3DKMT_HANDLE hAdapter;
} ff_D3DKMT_OPENADAPTERFROMLUID;

typedef struct _ff_D3DKMT_CLOSEADAPTER {
    ff_D3DKMT_HANDLE hAdapter;
} ff_D3DKMT_CLOSEADAPTER;

typedef struct _ff_D3DKMT_QUERYADAPTERINFO {
    ff_D3DKMT_HANDLE hAdapter;
    UINT             Type;               // KMTQUERYADAPTERINFOTYPE value
    VOID            *pPrivateDriverData;
    UINT             PrivateDriverDataSize;
} ff_D3DKMT_QUERYADAPTERINFO;

typedef struct _ff_D3DKMT_ADAPTERTYPE {
    union {
        struct {
            UINT RenderSupported            : 1;
            UINT DisplaySupported           : 1;
            UINT SoftwareDevice             : 1;
            UINT PostDevice                 : 1;
            UINT HybridDiscrete             : 1;
            UINT HybridIntegrated           : 1;
            UINT IndirectDisplayDevice      : 1;
            UINT Paravirtualized            : 1;
            UINT ACGSupported               : 1;
            UINT SupportSetTimingsFromVidPn : 1;
            UINT Detachable                 : 1;
            UINT ComputeOnly                : 1;
            UINT Prototype                  : 1;
            UINT RuntimePowerManagement     : 1;
            UINT Reserved                   : 18;
        };
        UINT Value;
    };
} ff_D3DKMT_ADAPTERTYPE;
#       pragma pack(pop)

typedef LONG (APIENTRY *PFN_D3DKMT_OPEN_ADAPTER_FROM_LUID)(ff_D3DKMT_OPENADAPTERFROMLUID *);
typedef LONG (APIENTRY *PFN_D3DKMT_CLOSE_ADAPTER)(const ff_D3DKMT_CLOSEADAPTER *);
typedef LONG (APIENTRY *PFN_D3DKMT_QUERY_ADAPTER_INFO)(const ff_D3DKMT_QUERYADAPTERINFO *);

static int load_d3dkmt_functions(PFN_D3DKMT_OPEN_ADAPTER_FROM_LUID *open_fn,
                                 PFN_D3DKMT_QUERY_ADAPTER_INFO *query_fn,
                                 PFN_D3DKMT_CLOSE_ADAPTER *close_fn)
{
    static void *gdi32 = NULL;
    static PFN_D3DKMT_OPEN_ADAPTER_FROM_LUID cached_open = NULL;
    static PFN_D3DKMT_QUERY_ADAPTER_INFO cached_query = NULL;
    static PFN_D3DKMT_CLOSE_ADAPTER cached_close = NULL;
    static int resolved = 0; // 0 = not tried, 1 = succeeded, -1 = failed

    if (!resolved) {
        gdi32 = dlopen("gdi32.dll", 0);
        if (gdi32) {
            cached_open  = (PFN_D3DKMT_OPEN_ADAPTER_FROM_LUID)dlsym(gdi32, "D3DKMTOpenAdapterFromLuid");
            cached_query = (PFN_D3DKMT_QUERY_ADAPTER_INFO)dlsym(gdi32, "D3DKMTQueryAdapterInfo");
            cached_close = (PFN_D3DKMT_CLOSE_ADAPTER)dlsym(gdi32, "D3DKMTCloseAdapter");
        }
        resolved = (cached_open && cached_query && cached_close) ? 1 : -1;
    }

    *open_fn  = cached_open;
    *query_fn = cached_query;
    *close_fn = cached_close;
    return resolved == 1;
}
#   endif // !HAVE_UWP

static int is_physical_dxgi_adapter(LUID luid)
{
#   if !HAVE_UWP
#       define ff_KMTQAITYPE_ADAPTERTYPE 15
#       define ff_STATUS_SUCCESS_LOCAL   0L
    PFN_D3DKMT_OPEN_ADAPTER_FROM_LUID d3dkmt_open;
    PFN_D3DKMT_QUERY_ADAPTER_INFO d3dkmt_query;
    PFN_D3DKMT_CLOSE_ADAPTER d3dkmt_close;
    ff_D3DKMT_OPENADAPTERFROMLUID open_adapter = { 0 };
    ff_D3DKMT_CLOSEADAPTER close_adapter = { 0 };
    ff_D3DKMT_QUERYADAPTERINFO query = { 0 };
    ff_D3DKMT_ADAPTERTYPE adapter_type = { 0 };
    int is_phy_adapter = 1; // default: assume physical on any failure

    if (!load_d3dkmt_functions(&d3dkmt_open, &d3dkmt_query, &d3dkmt_close))
        return is_phy_adapter;

    open_adapter.AdapterLuid = luid;
    if (d3dkmt_open(&open_adapter) != ff_STATUS_SUCCESS_LOCAL)
        return is_phy_adapter;

    query.hAdapter = open_adapter.hAdapter;
    query.Type = ff_KMTQAITYPE_ADAPTERTYPE;
    query.pPrivateDriverData = &adapter_type;
    query.PrivateDriverDataSize = sizeof(adapter_type);

    if (d3dkmt_query(&query) == ff_STATUS_SUCCESS_LOCAL) {
        if (adapter_type.SoftwareDevice ||
            adapter_type.IndirectDisplayDevice)
            is_phy_adapter = 0;
    }

    close_adapter.hAdapter = open_adapter.hAdapter;
    d3dkmt_close(&close_adapter);

    return is_phy_adapter;
#   else // !HAVE_UWP
    return 1;
#   endif
}
#endif

#if CONFIG_D3D11VA
static int find_amf_dll_for_dxgi_adapter(DXGI_ADAPTER_DESC desc, wchar_t *out_path, size_t out_len)
{
#   ifdef FFPROBE_HW_AMF_DLL_HELPER
#       if ARCH_X86_64 || ARCH_AARCH64
#           define FFPROBE_HW_AMF_DLL_NAME L"amfrt64.dll"
#       else
#           define FFPROBE_HW_AMF_DLL_NAME L"amfrt32.dll"
#       endif
    CONFIGRET cr;
    ULONG device_id_list_size = 0;
    WCHAR *device_id_list = NULL;
    WCHAR *begin, *end;
    WCHAR display_guid[39];
    WCHAR dev_id[32];
    int found = 0;
    int ret = 0;

    if (desc.VendorId != FFPROBE_HW_VENDOR_ID_AMD)
        return AVERROR(EINVAL);

    if (!out_path || !out_len)
        return AVERROR(EINVAL);

    swprintf(dev_id, FF_ARRAY_ELEMS(dev_id), L"DEV_%04X", desc.DeviceId);

    if (StringFromGUID2(&GUID_DEVCLASS_DISPLAY, display_guid,
                        FF_ARRAY_ELEMS(display_guid)) == 0) {
        av_log(NULL, AV_LOG_DEBUG, "AMF_DLL: StringFromGUID2 failed\n");
        return AVERROR(ENOSYS);
    }

    do {
        cr = CM_Get_Device_ID_List_SizeW(&device_id_list_size, display_guid,
                                         CM_GETIDLIST_FILTER_CLASS |
                                         CM_GETIDLIST_FILTER_PRESENT);
        if (cr != CR_SUCCESS) {
            av_log(NULL, AV_LOG_DEBUG,
                   "AMF_DLL: CM_Get_Device_ID_List_SizeW failed '%lu'\n", cr);
            ret = AVERROR(ENOSYS);
            goto done;
        }

        av_freep(&device_id_list);
        device_id_list = av_malloc(device_id_list_size * sizeof(*device_id_list));
        if (!device_id_list) {
            ret = AVERROR(ENOMEM);
            goto done;
        }

        cr = CM_Get_Device_ID_ListW(display_guid, device_id_list, device_id_list_size,
                                    CM_GETIDLIST_FILTER_CLASS |
                                    CM_GETIDLIST_FILTER_PRESENT);
    } while (cr == CR_BUFFER_SMALL);

    if (cr != CR_SUCCESS) {
        av_log(NULL, AV_LOG_DEBUG,
               "AMF_DLL: CM_Get_Device_ID_ListW failed '%lu'\n", cr);
        ret = AVERROR(ENOSYS);
        goto done;
    }

    begin = device_id_list;
    end   = begin + device_id_list_size;

    for (; (begin < end) && *begin; begin += wcslen(begin) + 1) {
        DEVINST dev_inst;
        HKEY hkey;
        WCHAR upper[MAX_DEVICE_ID_LEN];
        WCHAR um_driver[MAX_PATH * 2];
        WCHAR candidate[MAX_PATH * 2];
        WCHAR *sep;
        DWORD um_size = sizeof(um_driver);
        size_t k;

        wcsncpy(upper, begin, FF_ARRAY_ELEMS(upper) - 1);
        upper[FF_ARRAY_ELEMS(upper) - 1] = L'\0';
        for (k = 0; upper[k]; k++)
            upper[k] = towupper(upper[k]);

        if (!wcsstr(upper, dev_id))
            continue;

        cr = CM_Locate_DevNodeW(&dev_inst, begin, CM_LOCATE_DEVNODE_NORMAL);
        if (cr != CR_SUCCESS) {
            av_log(NULL, AV_LOG_DEBUG,
                   "AMF_DLL: CM_Locate_DevNodeW failed '%lu'\n", cr);
            continue;
        }

        cr = CM_Open_DevNode_Key(dev_inst, KEY_READ, 0,
                                 RegDisposition_OpenExisting,
                                 &hkey, CM_REGISTRY_SOFTWARE);
        if (cr != CR_SUCCESS) {
            av_log(NULL, AV_LOG_DEBUG,
                   "AMF_DLL: CM_Open_DevNode_Key failed '%lu'\n", cr);
            continue;
        }

        if (RegQueryValueExW(hkey, L"UserModeDriverName", NULL, NULL,
                             (LPBYTE)um_driver, &um_size) != ERROR_SUCCESS) {
            av_log(NULL, AV_LOG_DEBUG,
                   "AMF_DLL: UserModeDriverName query failed '%lu'\n", GetLastError());
            RegCloseKey(hkey);
            continue;
        }
        RegCloseKey(hkey);

        um_driver[FF_ARRAY_ELEMS(um_driver) - 1] = L'\0';

        sep = wcsrchr(um_driver, L'\\');
        if (!sep) {
            av_log(NULL, AV_LOG_DEBUG,
                   "AMF_DLL: unexpected UserModeDriverName format '%ls'\n", um_driver);
            continue;
        }
        *sep = L'\0';

        /* candidate path: <driverstore_folder>\amfrt64.dll */
        if (swprintf(candidate, FF_ARRAY_ELEMS(candidate),
                     L"%ls\\" FFPROBE_HW_AMF_DLL_NAME, um_driver) < 0) {
            av_log(NULL, AV_LOG_DEBUG, "AMF_DLL: path too long\n");
            continue;
        }

        if (GetFileAttributesW(candidate) == INVALID_FILE_ATTRIBUTES) {
            av_log(NULL, AV_LOG_DEBUG,
                   "AMF_DLL: not found at '%ls'\n", candidate);
            continue;
        }

        wcsncpy(out_path, candidate, out_len - 1);
        out_path[out_len - 1] = L'\0';
        if (wcslen(candidate) >= out_len) {
            av_log(NULL, AV_LOG_DEBUG, "AMF_DLL: output buffer too small\n");
            ret = AVERROR(ERANGE);
            goto done;
        }

        av_log(NULL, AV_LOG_DEBUG, "AMF_DLL: found at '%ls'\n", candidate);
        found = 1;
        break;
    }

    if (!found && !ret) {
        av_log(NULL, AV_LOG_DEBUG,
               "AMF_DLL: not found for device '%04X'\n", desc.DeviceId);
        ret = AVERROR(ENOSYS);
    }

done:
    av_freep(&device_id_list);
    return ret;
#       undef FFPROBE_HW_AMF_DLL_NAME
#   else // FFPROBE_HW_AMF_DLL_HELPER
    return AVERROR(ENOSYS);
#   endif
}
#endif

int create_d3d11va_devices(HwDeviceRefs *refs, int device_idx)
{
    return create_d3d11va_devices_with_filter(refs, -1, -1, NULL, device_idx);
}

#if CONFIG_D3D11VA
typedef HRESULT(WINAPI *PFN_CREATE_DXGI_FACTORY)(REFIID riid, void **ppFactory);
static PFN_CREATE_DXGI_FACTORY mCreateDXGIFactory = NULL;
#endif

int create_d3d11va_devices_with_filter(HwDeviceRefs *refs, int vendor_id,
                                       int idx_luid, const char *luid, int device_idx)
{
#if CONFIG_D3D11VA
    int ret = 0;
    const int start_idx = device_idx < 0 ? 0 : device_idx;
    unsigned i, j;
    char ibuf[4];
    HRESULT hr;
    DXGI_ADAPTER_DESC desc;
    IDXGIFactory2 *pDXGIFactory = NULL;
    IDXGIAdapter *pDXGIAdapter = NULL;
#   if !HAVE_UWP
    HANDLE dxgilib;

    if (!mCreateDXGIFactory) {
        dxgilib = dlopen("dxgi.dll", 0);
        if (!dxgilib)
            return AVERROR(ENOSYS);

        mCreateDXGIFactory = (PFN_CREATE_DXGI_FACTORY)GetProcAddress(dxgilib, "CreateDXGIFactory1");
    }
#   else
    if (!mCreateDXGIFactory)
        mCreateDXGIFactory = (PFN_CREATE_DXGI_FACTORY)CreateDXGIFactory1;
#   endif
    if (!mCreateDXGIFactory)
        return AVERROR(ENOSYS);

    hr = mCreateDXGIFactory(&IID_IDXGIFactory2, (void**)&pDXGIFactory);
    if (FAILED(hr))
        return AVERROR(ENOSYS);

    for (i = start_idx, j = 0; i < FFPROBE_HW_MAX_DEV_NUM && refs; i++) {
        hr = IDXGIFactory2_EnumAdapters(pDXGIFactory, i, &pDXGIAdapter);
        if (FAILED(hr)) {
            if (device_idx < 0) continue;
            else break;
        }

        hr = IDXGIAdapter2_GetDesc(pDXGIAdapter, &desc);
        if (pDXGIAdapter)
            IDXGIAdapter_Release(pDXGIAdapter);

        /* Filter out 'Microsoft Basic Render Driver' */
        if (SUCCEEDED(hr) &&
            desc.VendorId == FFPROBE_HW_VENDOR_ID_MICROSOFT) {
            if (device_idx < 0) continue;
            else break;
        }

        /* Filter out RDP/indirect/software adapters */
        if (SUCCEEDED(hr) && !is_physical_dxgi_adapter(desc.AdapterLuid)) {
            if (device_idx < 0) continue;
            else break;
        }

        /* Filter out by the requested vendor id */
        if (SUCCEEDED(hr) && vendor_id > 0 && desc.VendorId != vendor_id) {
            if (device_idx < 0) continue;
            else break;
        }

        /* Filter by the requested LUID on Windows from CUDA */
        if (SUCCEEDED(hr) && idx_luid >= 0 && luid && device_idx < 0) {
            const LUID dxgi_luid = desc.AdapterLuid;
            const size_t lo_size = sizeof(dxgi_luid.LowPart);
            const size_t hi_size = sizeof(dxgi_luid.HighPart);

            if (!memcmp(&dxgi_luid.LowPart, luid, lo_size) &&
                !memcmp(&dxgi_luid.HighPart, luid + lo_size, hi_size)) {
                snprintf(ibuf, sizeof(ibuf), "%d", i);
                av_hwdevice_ctx_create(&refs[idx_luid].d3d11va_ref,
                                       AV_HWDEVICE_TYPE_D3D11VA,
                                       ibuf, NULL, 0);
                break;
            } else
                continue;
        }

        snprintf(ibuf, sizeof(ibuf), "%d", i);
        ret = av_hwdevice_ctx_create(&refs[j].d3d11va_ref,
                                     AV_HWDEVICE_TYPE_D3D11VA,
                                     ibuf, NULL, 0);
        if (ret < 0) {
            if (device_idx < 0) continue;
            else break;
        }

        if (desc.VendorId == FFPROBE_HW_VENDOR_ID_AMD) {
            find_amf_dll_for_dxgi_adapter(desc, refs[j].amf_dll_path,
                                          FF_ARRAY_ELEMS(refs[j].amf_dll_path));
        }
        refs[j].device_index_dxgi = i;
        refs[j].device_luid_dxgi  = ((uint64_t)desc.AdapterLuid.HighPart << 32) |
                                     (uint64_t)desc.AdapterLuid.LowPart;
        refs[j].device_vendor_id  = desc.VendorId;
        j++;

        /* Filter by the requested device index */
        if (device_idx >= 0)
            break;
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
    for (unsigned i = 0; i < FFPROBE_HW_MAX_DEV_NUM && refs && refs[i].d3d11va_ref; i++) {
        if (refs[i].device_vendor_id != FFPROBE_HW_VENDOR_ID_INTEL)
            continue;
        av_hwdevice_ctx_create_derived(&refs[i].qsv_ref, AV_HWDEVICE_TYPE_QSV,
                                       refs[i].d3d11va_ref, 0);
    }
}

/* D3D11VA -> OPENCL */
void create_derive_opencl_devices_from_d3d11va(HwDeviceRefs *refs)
{
    for (unsigned i = 0; i < FFPROBE_HW_MAX_DEV_NUM && refs && refs[i].d3d11va_ref; i++) {
        if (!(refs[i].device_vendor_id == FFPROBE_HW_VENDOR_ID_INTEL ||
              refs[i].device_vendor_id == FFPROBE_HW_VENDOR_ID_AMD))
            continue;
        av_hwdevice_ctx_create_derived(&refs[i].opencl_ref, AV_HWDEVICE_TYPE_OPENCL,
                                       refs[i].d3d11va_ref, 0);
    }
}

#if CONFIG_D3D11VA
static int print_d3d11va_driver_version(AVTextFormatContext *tfc, DXGI_ADAPTER_DESC desc)
{
#   ifdef FFPROBE_HW_WDDM_VERSION_HELPER
    CONFIGRET cr;
    ULONG device_id_list_size = 0;
    WCHAR *device_id_list = NULL;
    WCHAR *begin, *end;
    WCHAR display_guid[39];
    WCHAR dev_id[64];
    int found = 0;
    int ret = 0;

    if (!tfc)
        return AVERROR(EINVAL);

    swprintf(dev_id, FF_ARRAY_ELEMS(dev_id),
             L"VEN_%04X&DEV_%04X&SUBSYS_%08X&REV_%02X",
             desc.VendorId, desc.DeviceId, desc.SubSysId, desc.Revision);

    if (StringFromGUID2(&GUID_DEVCLASS_DISPLAY, display_guid,
                        FF_ARRAY_ELEMS(display_guid)) == 0) {
        av_log(NULL, AV_LOG_DEBUG, "WDDM: StringFromGUID2 failed\n");
        return AVERROR(ENOSYS);
    }

    do {
        cr = CM_Get_Device_ID_List_SizeW(&device_id_list_size, display_guid,
                                         CM_GETIDLIST_FILTER_CLASS |
                                         CM_GETIDLIST_FILTER_PRESENT);
        if (cr != CR_SUCCESS) {
            av_log(NULL, AV_LOG_DEBUG,
                   "WDDM: CM_Get_Device_ID_List_SizeW failed '%lu'\n", cr);
            ret = AVERROR(ENOSYS);
            goto done;
        }

        av_freep(&device_id_list);
        device_id_list = av_malloc(device_id_list_size * sizeof(*device_id_list));
        if (!device_id_list) {
            ret = AVERROR(ENOMEM);
            goto done;
        }

        cr = CM_Get_Device_ID_ListW(display_guid, device_id_list, device_id_list_size,
                                    CM_GETIDLIST_FILTER_CLASS |
                                    CM_GETIDLIST_FILTER_PRESENT);
    } while (cr == CR_BUFFER_SMALL);

    if (cr != CR_SUCCESS) {
        av_log(NULL, AV_LOG_DEBUG,
               "WDDM: CM_Get_Device_ID_ListW failed '%lu'\n", cr);
        ret = AVERROR(ENOSYS);
        goto done;
    }

    begin = device_id_list;
    end   = begin + device_id_list_size;

    for (; (begin < end) && *begin; begin += wcslen(begin) + 1) {
        DEVINST dev_inst;
        HKEY hkey;
        WCHAR upper[MAX_DEVICE_ID_LEN];
        WCHAR ver_str[64];
        DWORD ver_size = sizeof(ver_str);
        DWORD type = 0;
        int model, feature, revision, build;
        size_t k;

        wcsncpy(upper, begin, FF_ARRAY_ELEMS(upper) - 1);
        upper[FF_ARRAY_ELEMS(upper) - 1] = L'\0';
        for (k = 0; upper[k]; k++)
            upper[k] = towupper(upper[k]);

        if (!wcsstr(upper, dev_id))
            continue;

        cr = CM_Locate_DevNodeW(&dev_inst, begin, CM_LOCATE_DEVNODE_NORMAL);
        if (cr != CR_SUCCESS) {
            av_log(NULL, AV_LOG_DEBUG,
                   "WDDM: CM_Locate_DevNodeW failed '%lu'\n", cr);
            continue;
        }

        cr = CM_Open_DevNode_Key(dev_inst, KEY_READ, 0,
                                 RegDisposition_OpenExisting,
                                 &hkey, CM_REGISTRY_SOFTWARE);
        if (cr != CR_SUCCESS) {
            av_log(NULL, AV_LOG_DEBUG,
                   "WDDM: CM_Open_DevNode_Key failed '%lu'\n", cr);
            continue;
        }

        if (RegQueryValueExW(hkey, L"DriverVersion", NULL, &type,
                             (LPBYTE)ver_str, &ver_size) != ERROR_SUCCESS ||
            type != REG_SZ) {
            av_log(NULL, AV_LOG_DEBUG,
                   "WDDM: DriverVersion query failed '%lu'\n", GetLastError());
            RegCloseKey(hkey);
            continue;
        }
        RegCloseKey(hkey);

        /* https://learn.microsoft.com/en-us/windows-hardware/drivers/display/wddm-2-1-features */
        if (swscanf(ver_str, L"%d.%d.%d.%d",
                    &model, &feature, &revision, &build) != 4) {
            av_log(NULL, AV_LOG_DEBUG,
                   "WDDM: DriverVersion '%ls' doesn't match expected format\n", ver_str);
            continue;
        }

        print_int("device_wddm_model_version", model);
        print_int("device_wddm_d3d_feature_level", feature);
        print_int("device_wddm_vendor_revision", revision);
        print_int("device_wddm_vendor_build", build);

        found = 1;
        break;
    }

    if (!found && !ret) {
        av_log(NULL, AV_LOG_DEBUG,
               "WDDM: driver version not found for device VEN_%04X&DEV_%04X\n",
               desc.VendorId, desc.DeviceId);
        ret = AVERROR(ENOSYS);
    }

done:
    av_freep(&device_id_list);
    return ret;
#   else // FFPROBE_HW_WDDM_VERSION_HELPER
    return 0;
#   endif
}
#endif

int print_d3d11va_device_info(AVTextFormatContext *tfc, AVBufferRef *d3d11va_ref)
{
#if CONFIG_D3D11VA
    AVHWDeviceContext    *dev_ctx = NULL;
    AVD3D11VADeviceContext *hwctx = NULL;
    IDXGIDevice     *pDXGIDevice = NULL;
    IDXGIAdapter   *pDXGIAdapter = NULL;
    DXGI_ADAPTER_DESC desc = { 0 };
    D3D_FEATURE_LEVEL level = 0;
    D3D11_FEATURE_DATA_D3D11_OPTIONS data = { 0 };
    D3D11_FEATURE_DATA_D3D11_OPTIONS2 data2 = { 0 };
    HRESULT hr;
    char device_desc[128*3+1] = { 0 };
    int uma = -1, ext_sharing = -1, ret = 0;

    if (!tfc || !d3d11va_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)d3d11va_ref->data;
    hwctx = dev_ctx->hwctx;

    hr = ID3D11Device_QueryInterface(hwctx->device, &IID_IDXGIDevice, (void**)&pDXGIDevice);
    if (FAILED(hr)) {
        ret = AVERROR(ENOSYS);
        av_log(NULL, AV_LOG_DEBUG, "ID3D11Device_QueryInterface failed: 0x%08lx\n", hr);
        goto exit;
    }

    hr = IDXGIDevice_GetAdapter(pDXGIDevice, &pDXGIAdapter);
    if (FAILED(hr)) {
        ret = AVERROR(ENOSYS);
        av_log(NULL, AV_LOG_DEBUG, "IDXGIDevice_GetAdapter failed: 0x%08lx\n", hr);
        goto exit;
    }

    hr = IDXGIAdapter_GetDesc(pDXGIAdapter, &desc);
    if (FAILED(hr)) {
        ret = AVERROR(ENOSYS);
        av_log(NULL, AV_LOG_DEBUG, "IDXGIAdapter_GetDesc failed: 0x%08lx\n", hr);
        goto exit;
    }

    level = ID3D11Device_GetFeatureLevel(hwctx->device);

    hr = ID3D11Device_CheckFeatureSupport(hwctx->device,
                                          D3D11_FEATURE_D3D11_OPTIONS,
                                          &data, sizeof(data));
    ext_sharing = SUCCEEDED(hr) && data.ExtendedResourceSharing;

    hr = ID3D11Device_CheckFeatureSupport(hwctx->device,
                                          D3D11_FEATURE_D3D11_OPTIONS2,
                                          &data2, sizeof(data2));
    uma = SUCCEEDED(hr) && data2.UnifiedMemoryArchitecture;

    if (WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1,
                            device_desc, FF_ARRAY_ELEMS(device_desc),
                            NULL, NULL) <= 0) {
        av_strlcpy(device_desc, "Unknown GPU", sizeof(device_desc));
    } else
        device_desc[FF_ARRAY_ELEMS(device_desc) - 1] = '\0';

    mark_section_show_entries(SECTION_ID_DEVICE_INFO_D3D11VA, 1, NULL);
    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_INFO_D3D11VA);

    print_str("device_name", device_desc);
    print_int("device_vid", desc.VendorId);
    print_int("device_did", desc.DeviceId);
    print_int("device_subsys_id", desc.SubSysId);
    print_int("device_revision", desc.Revision);
    print_int("device_dedicated_video_memory", desc.DedicatedVideoMemory);
    print_int("device_shared_system_memory", desc.SharedSystemMemory);
    print_int("device_luid_lo", desc.AdapterLuid.LowPart);
    print_int("device_luid_hi", desc.AdapterLuid.HighPart);
    print_int("device_feature_level", level);
    print_int("device_extended_resource_sharing", ext_sharing);
    print_int("device_unified_memory_architecture", uma);

    print_d3d11va_driver_version(tfc, desc);

    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_INFO_D3D11VA

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

#ifdef FFPROBE_HW_DXGI1_4_PDH_HELPER
static BOOL match_dxgi_adapter_luids(const WCHAR *inst,
                                     const WCHAR *luid_hi, const WCHAR *luid_lo)
{
    WCHAR luid_hi_upper[16], luid_lo_upper[16];
    DWORD i;

    if (wcsstr(inst, luid_hi) && wcsstr(inst, luid_lo))
        return TRUE;

    for (i = 0; i < FF_ARRAY_ELEMS(luid_hi_upper) - 1 && luid_hi[i]; i++)
        luid_hi_upper[i] = towupper(luid_hi[i]);
    luid_hi_upper[i] = L'\0';

    if (!(wcsstr(inst, luid_hi) || wcsstr(inst, luid_hi_upper)))
        return FALSE;

    for (i = 0; i < FF_ARRAY_ELEMS(luid_lo_upper) - 1 && luid_lo[i]; i++)
        luid_lo_upper[i] = towupper(luid_lo[i]);
    luid_lo_upper[i] = L'\0';

    return wcsstr(inst, luid_lo) || wcsstr(inst, luid_lo_upper);
}

static BOOL add_dxgi_adapter_memory_counter(PDH_HQUERY query, LUID luid,
                                            PDH_HCOUNTER *counter, int shared)
{
    DWORD buf_size = 0, instance_count = 0;
    WCHAR path[PDH_MAX_COUNTER_PATH];
    WCHAR luid_hi[16], luid_lo[16];
    WCHAR *counter_buf = NULL;
    WCHAR *instance_buf = NULL;
    const WCHAR *inst;
    BOOL found = FALSE;
    PDH_STATUS st;

    swprintf(luid_hi, FF_ARRAY_ELEMS(luid_hi), L"%08x", (UINT32)luid.HighPart);
    swprintf(luid_lo, FF_ARRAY_ELEMS(luid_lo), L"%08x", (UINT32)luid.LowPart);

    /* Try uppercase first (most drivers) */
    swprintf(path, FF_ARRAY_ELEMS(path), (shared ?
        L"\\GPU Adapter Memory(luid_0x%08X_0x%08X_phys_0)\\Shared Usage" :
        L"\\GPU Adapter Memory(luid_0x%08X_0x%08X_phys_0)\\Dedicated Usage"),
        (UINT32)luid.HighPart,
        (UINT32)luid.LowPart);

    st = PdhAddEnglishCounterW(query, path, 0, counter);
    if (st == ERROR_SUCCESS)
        return TRUE;

    /* Fall back: enumerate and match LUID */
    PdhEnumObjectItemsW(NULL, NULL, L"GPU Adapter Memory",
        NULL, &buf_size, NULL, &instance_count,
        PERF_DETAIL_WIZARD, 0);

    counter_buf  = av_malloc(buf_size * sizeof(*counter_buf));
    instance_buf = av_malloc(instance_count * sizeof(*instance_buf));
    if (!counter_buf || !instance_buf)
        goto cleanup;

    st = PdhEnumObjectItemsW(NULL, NULL, L"GPU Adapter Memory",
                             counter_buf, &buf_size,
                             instance_buf, &instance_count,
                             PERF_DETAIL_WIZARD, 0);
    if (st != ERROR_SUCCESS)
        goto cleanup;

    inst = instance_buf;
    while (*inst) {
        if (match_dxgi_adapter_luids(inst, luid_hi, luid_lo)) {
            swprintf(path, FF_ARRAY_ELEMS(path), (shared ?
                L"\\GPU Adapter Memory(%ls)\\Shared Usage" :
                L"\\GPU Adapter Memory(%ls)\\Dedicated Usage"), inst);
            st = PdhAddEnglishCounterW(query, path, 0, counter);
            if (st == ERROR_SUCCESS) {
                found = TRUE;
                break;
            }
        }
        inst += wcslen(inst) + 1;
    }

cleanup:
    av_free(counter_buf);
    av_free(instance_buf);
    return found;
}

typedef struct {
    PDH_HCOUNTER handle;
    WCHAR        path[PDH_MAX_COUNTER_PATH];
} EngineCounter;

static BOOL is_relevant_engine_type(const WCHAR *inst)
{
    const WCHAR engtype[] = L"engtype_";
    const DWORD engtype_len = FF_ARRAY_ELEMS(engtype) - 1;
    const WCHAR *p = wcsstr(inst, engtype);
    if (!p)
        return FALSE;
    p += engtype_len;

#   define MATCH(e) (!wcsncmp(p, (e), FF_ARRAY_ELEMS(e) - 1))
    return (MATCH(L"3D")               ||
            MATCH(L"VideoDecode")      ||
            MATCH(L"Video Decode")     ||
            MATCH(L"VideoEncode")      ||
            MATCH(L"Video Encode")     ||
            MATCH(L"VideoCodec")       ||
            MATCH(L"Video Codec")      ||
            MATCH(L"VideoProcessing")  ||
            MATCH(L"Video Processing") ||
            MATCH(L"Copy")             ||
            MATCH(L"Compute"));
#   undef MATCH
}

static BOOL add_dxgi_adapter_engine_counters(PDH_HQUERY query, LUID luid,
                                             EngineCounter **counters,
                                             DWORD *counter_count)
{
    DWORD buf_size = 0, instance_count = 0;
    WCHAR luid_hi[16] = { 0 };
    WCHAR luid_lo[16] = { 0 };
    WCHAR *counter_buf = NULL;
    WCHAR *instance_buf = NULL;
    const WCHAR *inst;
    DWORD capacity = 0;
    BOOL success = FALSE;
    PDH_STATUS st;

    *counters = NULL;
    *counter_count = 0;

    swprintf(luid_hi, FF_ARRAY_ELEMS(luid_hi), L"%08x", (UINT32)luid.HighPart);
    swprintf(luid_lo, FF_ARRAY_ELEMS(luid_lo), L"%08x", (UINT32)luid.LowPart);

    PdhEnumObjectItemsW(NULL, NULL, L"GPU Engine",
        NULL, &buf_size, NULL, &instance_count,
        PERF_DETAIL_WIZARD, 0);

    counter_buf  = av_malloc(buf_size * sizeof(*counter_buf));
    instance_buf = av_malloc(instance_count * sizeof(*instance_buf));
    if (!counter_buf || !instance_buf)
        goto cleanup;

    st = PdhEnumObjectItemsW(NULL, NULL, L"GPU Engine",
                             counter_buf, &buf_size,
                             instance_buf, &instance_count,
                             PERF_DETAIL_WIZARD, 0);
    if (st != ERROR_SUCCESS)
        goto cleanup;

    inst = instance_buf;
    while (*inst) {
        if (is_relevant_engine_type(inst) &&
            match_dxgi_adapter_luids(inst, luid_hi, luid_lo)) {
            EngineCounter *ec = NULL;

            if (*counter_count >= capacity) {
                capacity = capacity ? (capacity * 2) : 8;
                *counters = av_realloc_array(*counters, capacity,
                                             sizeof(EngineCounter));
                if (!*counters)
                    goto cleanup;
            }

            ec = &(*counters)[*counter_count];
            swprintf(ec->path, FF_ARRAY_ELEMS(ec->path),
                L"\\GPU Engine(%ls)\\Utilization Percentage", inst);

            st = PdhAddEnglishCounterW(query, ec->path, 0, &ec->handle);
            if (st == ERROR_SUCCESS)
                (*counter_count)++;
        }
        inst += wcslen(inst) + 1;
    }

    success = TRUE;

cleanup:
    av_free(counter_buf);
    av_free(instance_buf);
    return success;
}

static void get_dxgi_adapter_engine_type(const WCHAR *path, WCHAR *buf, DWORD buf_len)
{
    const WCHAR engtype[] = L"engtype_";
    const DWORD engtype_len = FF_ARRAY_ELEMS(engtype) - 1;
    const WCHAR prefix[] = L"device_";
    const DWORD prefix_len = FF_ARRAY_ELEMS(prefix) - 1;
    const WCHAR suffix[] = L"_utilization";
    const DWORD suffix_len = FF_ARRAY_ELEMS(suffix) - 1;
    const WCHAR *p;
    DWORD i;

    buf[0] = L'\0';

    p = wcsstr(path, engtype);
    if (!p)
        return;

    p += engtype_len;

    if (buf_len < prefix_len + 1)
        return;

    wcscpy(buf, prefix);
    i = prefix_len;

    /* Drop spaces and convert to lowercase */
    while (*p && *p != L')' && i < buf_len - 1) {
        WCHAR c = *p++;

        if (c == L' ')
            continue;
        buf[i++] = towlower(c);
    }
    buf[i] = L'\0';

    /* Drop engine type index (engine_N) */
    while (i > prefix_len && buf[i - 1] >= L'0' && buf[i - 1] <= L'9') i--;
    if (i > prefix_len && buf[i - 1] == L'_') i--;
    buf[i] = L'\0';

    if (i + suffix_len + 1 <= buf_len)
        wcscpy(buf + i, suffix);

    buf[i + suffix_len] = L'\0';
}

static void print_dxgi_adapter_engine_utilizations(AVTextFormatContext *tfc,
                                                   EngineCounter *counters,
                                                   DWORD counter_count)
{
    struct {
        WCHAR  engine_type[256];
        double val_total;
        int    active_nodes[32];
        int    active_node_count;
    } summary[32] = { 0 };
    DWORD summary_count = 0;
    DWORD i, j, k;
    PDH_STATUS st;

    if (!tfc || !counters || !counter_count)
        return;

    for (i = 0; i < counter_count; i++) {
        const WCHAR eng[] = L"_eng_";
        const DWORD eng_len = FF_ARRAY_ELEMS(eng) - 1;
        WCHAR engine_type[256] = { 0 };
        PDH_FMT_COUNTERVALUE val;
        int current_node_id = -1;
        const WCHAR *p;

        get_dxgi_adapter_engine_type(counters[i].path, engine_type, FF_ARRAY_ELEMS(engine_type));
        if (engine_type[0] == L'\0')
            continue;

        st = PdhGetFormattedCounterValue(counters[i].handle, PDH_FMT_DOUBLE, NULL, &val);
        if (st != ERROR_SUCCESS)
            continue;

        p = wcsstr(counters[i].path, eng);
        if (p)
            current_node_id = (int)wcstol(p + eng_len, NULL, 10);

        for (j = 0; j < summary_count; j++) {
            if (wcscmp(summary[j].engine_type, engine_type) == 0) {
                if (val.doubleValue > 0.0)
                    summary[j].val_total += val.doubleValue;
                if (current_node_id != -1) {
                    for (k = 0; k < (DWORD)summary[j].active_node_count; k++) {
                        if (summary[j].active_nodes[k] == current_node_id)
                            break;
                    }
                    if (k == (DWORD)summary[j].active_node_count && val.doubleValue > 0.0 &&
                        summary[j].active_node_count < FF_ARRAY_ELEMS(summary[j].active_nodes)) {
                        summary[j].active_nodes[summary[j].active_node_count++] = current_node_id;
                    }
                }
                break;
            }
        }
        if (j == summary_count && summary_count < FF_ARRAY_ELEMS(summary)) {
            wcscpy(summary[summary_count].engine_type, engine_type);
            summary[summary_count].val_total = val.doubleValue > 0.0 ? val.doubleValue : 0.0;
            summary[summary_count].active_node_count = 0;
            if (current_node_id != -1) {
                summary[summary_count].active_nodes[0] = current_node_id;
                if (val.doubleValue > 0.0)
                    summary[summary_count].active_node_count = 1;
            }
            summary_count++;
        }
    }

    for (j = 0; j < summary_count; j++) {
        char *engine_type_utf8 = NULL;

        if (!wchartoutf8((wchar_t*)summary[j].engine_type, &engine_type_utf8)) {
            double avg_val = summary[j].val_total;

            if (summary[j].active_node_count > 0)
                avg_val /= summary[j].active_node_count;

            print_int(engine_type_utf8, (int)FFMIN(avg_val + 0.5, 100.0));
            av_free(engine_type_utf8);
        }
    }
}
#endif

int print_d3d11va_device_util(AVTextFormatContext *tfc, AVBufferRef *d3d11va_ref)
{
#ifdef FFPROBE_HW_DXGI1_4_PDH_HELPER
    AVHWDeviceContext    *dev_ctx = NULL;
    AVD3D11VADeviceContext *hwctx = NULL;
    IDXGIDevice      *pDXGIDevice = NULL;
    IDXGIAdapter    *pDXGIAdapter = NULL;
    IDXGIAdapter3  *pDXGIAdapter3 = NULL;
    DXGI_ADAPTER_DESC        desc = { 0 };
    HRESULT hr;
    PDH_HQUERY             pQuery = NULL;
    PDH_HCOUNTER  pMemCounterDgfx = NULL;
    PDH_HCOUNTER  pMemCounterIgfx = NULL;
    EngineCounter   *pEngCounters = NULL;
    DWORD            engine_count = 0;
    char device_desc[128*3+1] = { 0 };
    int ret = 0;

    if (!tfc || !d3d11va_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)d3d11va_ref->data;
    hwctx = dev_ctx->hwctx;

    hr = ID3D11Device_QueryInterface(hwctx->device, &IID_IDXGIDevice, (void**)&pDXGIDevice);
    if (FAILED(hr)) {
        ret = AVERROR(ENOSYS);
        av_log(NULL, AV_LOG_DEBUG, "ID3D11Device_QueryInterface failed: 0x%08lx\n", hr);
        goto exit;
    }

    hr = IDXGIDevice_GetAdapter(pDXGIDevice, &pDXGIAdapter);
    if (FAILED(hr)) {
        ret = AVERROR(ENOSYS);
        av_log(NULL, AV_LOG_DEBUG, "IDXGIDevice_GetAdapter failed: 0x%08lx\n", hr);
        goto exit;
    }

    hr = IDXGIAdapter_GetDesc(pDXGIAdapter, &desc);
    if (FAILED(hr)) {
        ret = AVERROR(ENOSYS);
        av_log(NULL, AV_LOG_DEBUG, "IDXGIAdapter_GetDesc failed: 0x%08lx\n", hr);
        goto exit;
    }

    if (WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1,
                            device_desc, FF_ARRAY_ELEMS(device_desc),
                            NULL, NULL) <= 0) {
        av_strlcpy(device_desc, "Unknown GPU", sizeof(device_desc));
    } else
        device_desc[FF_ARRAY_ELEMS(device_desc) - 1] = '\0';

    if (PdhOpenQueryW(NULL, 0, &pQuery) == ERROR_SUCCESS) {
        if (add_dxgi_adapter_memory_counter(pQuery, desc.AdapterLuid, &pMemCounterDgfx, 0) |
            add_dxgi_adapter_memory_counter(pQuery, desc.AdapterLuid, &pMemCounterIgfx, 1) |
            add_dxgi_adapter_engine_counters(pQuery, desc.AdapterLuid, &pEngCounters, &engine_count)) {
            PdhCollectQueryData(pQuery);
            Sleep(100);
            PdhCollectQueryData(pQuery);
        }
    }

    mark_section_show_entries(SECTION_ID_DEVICE_UTIL_D3D11VA, 1, NULL);
    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_UTIL_D3D11VA);

    print_str("device_name", device_desc);
    print_int("device_luid_lo", desc.AdapterLuid.LowPart);
    print_int("device_luid_hi", desc.AdapterLuid.HighPart);
    print_int("device_dedicated_video_memory", desc.DedicatedVideoMemory);
    print_int("device_shared_system_memory", desc.SharedSystemMemory);

    hr = IDXGIAdapter_QueryInterface(pDXGIAdapter, &IID_IDXGIAdapter3, (void**)&pDXGIAdapter3);
    if (SUCCEEDED(hr)) {
        DXGI_QUERY_VIDEO_MEMORY_INFO memory_info;

        hr = IDXGIAdapter3_QueryVideoMemoryInfo(pDXGIAdapter3, 0,
                                                DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory_info);
        if (SUCCEEDED(hr))
            print_int("device_memory_budget", memory_info.Budget);
    }

    {
        PDH_STATUS st;
        PDH_FMT_COUNTERVALUE mem_val;

        if (pMemCounterDgfx) {
            st = PdhGetFormattedCounterValue(pMemCounterDgfx, PDH_FMT_LARGE, NULL, &mem_val);
            if (st == ERROR_SUCCESS)
                print_int("device_memory_dedicated_usage", mem_val.largeValue);
        }
        if (pMemCounterIgfx) {
            st = PdhGetFormattedCounterValue(pMemCounterIgfx, PDH_FMT_LARGE, NULL, &mem_val);
            if (st == ERROR_SUCCESS)
                print_int("device_memory_shared_usage", mem_val.largeValue);
        }
        if (pEngCounters && engine_count > 0)
            print_dxgi_adapter_engine_utilizations(tfc, pEngCounters, engine_count);
    }


    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_UTIL_D3D11VA

exit:
    av_free(pEngCounters);
    if (pQuery != NULL)
        PdhCloseQuery(pQuery); // Also free counters
    if (pDXGIAdapter3)
        IDXGIAdapter3_Release(pDXGIAdapter3);
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
        av_log(NULL, AV_LOG_DEBUG, "ID3D11VideoDevice_GetVideoDecoderConfigCount failed: 0x%08lx\n", hr);
        return AVERROR(EINVAL);
    }

    cfg_list = av_malloc_array(cfg_cnt, sizeof(*cfg_list));
    if (!cfg_list)
        return AVERROR(ENOMEM);

    for (i = 0; i < cfg_cnt; i++) {
        D3D11_VIDEO_DECODER_CONFIG *cfg = &cfg_list[i];

        hr = ID3D11VideoDevice_GetVideoDecoderConfig(video_device, &desc, i, cfg);
        if (FAILED(hr)) {
            av_log(NULL, AV_LOG_DEBUG, "ID3D11VideoDevice_GetVideoDecoderConfig failed: 0x%08lx\n", hr);
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

int print_d3d11va_decoder_info(AVTextFormatContext *tfc, AVBufferRef *d3d11va_ref)
{
#if CONFIG_D3D11VA
    AVHWDeviceContext    *dev_ctx = NULL;
    AVD3D11VADeviceContext *hwctx = NULL;
    HRESULT hr;
    int header_printed = 0;
    unsigned i, j, p_cnt = 0;
    GUID *p_list = NULL;

    if (!tfc || !d3d11va_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)d3d11va_ref->data;
    hwctx = dev_ctx->hwctx;

    p_cnt = ID3D11VideoDevice_GetVideoDecoderProfileCount(hwctx->video_device);
    p_list = av_malloc_array(p_cnt, sizeof(*p_list));
    if (!p_list || !p_cnt) {
        av_log(NULL, AV_LOG_DEBUG, "ID3D11VideoDevice_GetVideoDecoderProfileCount failed\n");
        av_free(p_list);
        return AVERROR(EINVAL);
    }

    for (i = 0; i < p_cnt; i++) {
        hr = ID3D11VideoDevice_GetVideoDecoderProfile(hwctx->video_device, i, &p_list[i]);
        if (FAILED(hr)) {
            av_log(NULL, AV_LOG_DEBUG, "ID3D11VideoDevice_GetVideoDecoderProfile %u failed\n", i);
            av_free(p_list);
            return AVERROR(EINVAL);
        }
    }

    for (i = 0; dxva_modes[i].name && dxva_modes[i].guid; i++) {
        int supported = 0;
        unsigned min_width = 0, min_height = 0;
        unsigned max_width = 0, max_height = 0;
        const DxvaMode *mode = &dxva_modes[i];
        D3D11_VIDEO_DECODER_DESC desc = { 0 };
        DXGI_FORMAT dxgi_fmt = DXGI_FORMAT_UNKNOWN;

        if (!mode->formats)
            continue;

        for (const GUID *g = &p_list[0]; !supported && g < &p_list[p_cnt]; g++) {
            supported = IsEqualGUID(mode->guid, g);
        }
        if (!supported)
            continue;

        /* Use the most significant format for this profile */
        dxgi_fmt = d3d11va_map_av_to_dxgi_format(mode->formats[0]);
        if (dxgi_fmt == DXGI_FORMAT_UNKNOWN)
            continue;

        desc.Guid = *mode->guid;

        /* Check min res first */
        for (const HwRes *r = &hw_res_ascend[0]; r->name; r++) {
            if (mode->legacy && (r->width > 4096 || r->height > 4096))
                break;

            desc.SampleWidth  = r->width;
            desc.SampleHeight = r->height;
            desc.OutputFormat = dxgi_fmt;

            if (check_d3d11va_decoder_config(hwctx->video_device, desc, mode->codec) == 1) {
                min_width  = r->width;
                min_height = r->height;
                break;
            }
        }
        if (!min_width || !min_height)
            continue;

        /* Test rest formats for this profile, bail out directly if any fails */
        for (j = 0; mode->formats[j] != AV_PIX_FMT_NONE; j++) {
            desc.SampleWidth  = min_width;
            desc.SampleHeight = min_height;
            desc.OutputFormat = d3d11va_map_av_to_dxgi_format(mode->formats[j]);

            if (check_d3d11va_decoder_config(hwctx->video_device, desc, mode->codec) != 1) {
                min_width = min_height = 0;
                break;
            }
        }
        if (!min_width || !min_height)
            continue;

        /* Check max res */
        for (const HwRes *r = &hw_res_ascend[FF_ARRAY_ELEMS(hw_res_ascend) - 1]; r >= &hw_res_ascend[0]; r--) {
            if (!r->name)
                continue;
            if (r->width <= min_width && r->height <= min_height) {
                max_width  = r->width;
                max_height = r->height;
                break;
            }
            if (mode->legacy && (r->width > 4096 || r->height > 4096))
                continue;

            desc.SampleWidth  = r->width;
            desc.SampleHeight = r->height;
            desc.OutputFormat = dxgi_fmt;

            if (check_d3d11va_decoder_config(hwctx->video_device, desc, mode->codec) == 1) {
                max_width  = r->width;
                max_height = r->height;
                break;
            }
        }
        if (!max_width || !max_height)
            continue;

        if (!header_printed) {
            mark_section_show_entries(SECTION_ID_DEVICE_DECODERS_D3D11VA, 1, NULL);
            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_DECODERS_D3D11VA);
            header_printed = 1;
        }

        mark_section_show_entries(SECTION_ID_DEVICE_DECODER, 1, NULL);
        avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_DECODER);
        print_str("codec_name", avcodec_get_name(mode->codec));
        print_int("codec_id", mode->codec);
        print_str("codec_desc", mode->name);
        print_int("min_width", min_width);
        print_int("min_height", min_height);
        print_int("max_width", max_width);
        print_int("max_height", max_height);

        /* Formats */
        if (mode->formats) {
            mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMTS, 1, NULL);
            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMTS);
            for (j = 0; mode->formats[j] != AV_PIX_FMT_NONE; j++) {
                mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMT, 1, NULL);
                avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMT);
                print_str("format_name", av_get_pix_fmt_name(mode->formats[j]));
                print_int("format_id", mode->formats[j]);
                avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMT
            }
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMTS
        }

        /* Profiles */
        if (mode->profiles) {
            mark_section_show_entries(SECTION_ID_DEVICE_PROFILES, 1, NULL);
            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PROFILES);
            for (j = 0; mode->profiles[j] != AV_PROFILE_UNKNOWN; j++) {
                mark_section_show_entries(SECTION_ID_DEVICE_PROFILE, 1, NULL);
                avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PROFILE);
                print_str("profile_name", avcodec_profile_name(mode->codec, mode->profiles[j]));
                print_int("profile_id", mode->profiles[j]);
                avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PROFILE
            }
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PROFILES
        }

        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_DECODER
    }

    if (header_printed)
        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_DECODERS_D3D11VA
#endif
    return 0;
}
