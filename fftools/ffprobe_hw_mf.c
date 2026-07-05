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

#if (CONFIG_MEDIAFOUNDATION && CONFIG_D3D11VA)
#   define COBJMACROS
#   include <windows.h>
#   include <initguid.h>
#   include <mfapi.h>
#   include <mferror.h>
#   include <mfobjects.h>
#   include <mftransform.h>
#   include "libavutil/hwcontext_d3d11va.h"
#   include "compat/w32dlfcn.h"
#endif

#if (CONFIG_MEDIAFOUNDATION && CONFIG_D3D11VA)
DEFINE_GUID(ff_MFT_ENUM_ADAPTER_LUID, 0x1d39518c, 0xe220, 0x4da8, 0xa0, 0x7f, 0xba, 0x17, 0x25, 0x52, 0xd6, 0xb1);

typedef struct MFFunctions {
    HRESULT (WINAPI *MFStartup)(ULONG Version, DWORD dwFlags);
    HRESULT (WINAPI *MFShutdown)(void);
    HRESULT (WINAPI *MFCreateMediaType)(IMFMediaType **ppMFType);
    HRESULT (WINAPI *MFCreateAttributes)(IMFAttributes **ppAttributes,
                                         UINT32 cbInitialSize);
    HRESULT (WINAPI *MFCreateDXGIDeviceManager)(UINT *pResetToken,
                                                IMFDXGIDeviceManager **ppDeviceManager);
    HRESULT (WINAPI *MFTEnum2)(GUID guidCategory, UINT32 Flags,
                               const MFT_REGISTER_TYPE_INFO *pInputType,
                               const MFT_REGISTER_TYPE_INFO *pOutputType,
                               IMFAttributes *pAttributes,
                               IMFActivate ***pppActivate,
                               UINT32 *pcCount);
} MFFunctions;

static void                 *mf_library = NULL;
static MFFunctions           mf_functions = { NULL };
static IMFDXGIDeviceManager *mf_dxgi_manager = NULL;
static UINT                  mf_reset_token  = 0;

typedef struct MfEncMode {
    const char               *name;
    const GUID               *guid;
    const enum AVCodecID      codec;
    const int                *profiles;
    const enum AVPixelFormat *formats;
    const int                 legacy;
} MfEncMode;

static const int profiles_h264[] = {
    AV_PROFILE_H264_BASELINE,
    AV_PROFILE_H264_MAIN,
    AV_PROFILE_H264_HIGH,
    AV_PROFILE_UNKNOWN
};
static const int profiles_hevc[] = {
    AV_PROFILE_HEVC_MAIN,
    AV_PROFILE_HEVC_MAIN_10,
    AV_PROFILE_UNKNOWN
};
static const int profiles_av1[] = {
    AV_PROFILE_AV1_MAIN,
    AV_PROFILE_UNKNOWN
};

static const enum AVPixelFormat formats_8_420[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NONE
};
static const enum AVPixelFormat formats_8_10_420[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_NONE
};

static const MfEncMode mfenc_modes[] = {
    { "MFT Hardware H.264 Encoder", &MFVideoFormat_H264, AV_CODEC_ID_H264, profiles_h264, formats_8_420,    1 },
    { "MFT Hardware HEVC Encoder",  &MFVideoFormat_HEVC, AV_CODEC_ID_HEVC, profiles_hevc, formats_8_10_420, 0 },
    { "MFT Hardware AV1 Encoder",   &MFVideoFormat_AV1,  AV_CODEC_ID_AV1,  profiles_av1,  formats_8_10_420, 0 },
    { NULL, NULL, AV_CODEC_ID_NONE, NULL, NULL, 0 }
};

enum MfTestType {
    MF_TEST_BREAK    = -2,
    MF_TEST_CONTINUE = -1,
    MF_TEST_SUCCESS  =  0,
};
#endif

#if !HAVE_UWP
#   define LOAD_MF_FUNCTION(func_name) \
        mf_functions.func_name = (void *)dlsym(mf_library, #func_name); \
        if (!mf_functions.func_name) { \
            av_log(NULL, AV_LOG_DEBUG, "DLL mfplat.dll failed to find function " #func_name "\n"); \
            return AVERROR_UNKNOWN; \
        }
#else
#   define LOAD_MF_FUNCTION(func_name) \
        mf_functions.func_name = func_name; \
        if (!mf_functions.func_name) { \
            av_log(NULL, AV_LOG_DEBUG, "Failed to find function " #func_name "\n"); \
            return AVERROR_UNKNOWN; \
        }
#endif

int init_mf_functions(void)
{
#if (CONFIG_MEDIAFOUNDATION && CONFIG_D3D11VA)
#   if !HAVE_UWP
    mf_library = dlopen("mfplat.dll", 0);
    if (!mf_library) {
        av_log(NULL, AV_LOG_DEBUG, "DLL mfplat.dll failed to open\n");
        return AVERROR_UNKNOWN;
    }
#   endif

    LOAD_MF_FUNCTION(MFStartup);
    LOAD_MF_FUNCTION(MFShutdown);
    LOAD_MF_FUNCTION(MFCreateMediaType);
    LOAD_MF_FUNCTION(MFCreateAttributes);
    LOAD_MF_FUNCTION(MFCreateDXGIDeviceManager);
    LOAD_MF_FUNCTION(MFTEnum2);

    if (FAILED(mf_functions.MFStartup(MF_VERSION, MFSTARTUP_LITE))) {
        av_log(NULL, AV_LOG_DEBUG, "MFStartup failed\n");
        uninit_mf_functions();
        return AVERROR_UNKNOWN;
    }
    return 0;
#else
    return AVERROR(ENOSYS);
#endif
}

void uninit_mf_functions(void)
{
#if (CONFIG_MEDIAFOUNDATION && CONFIG_D3D11VA)
    if (mf_dxgi_manager) {
        IMFDXGIDeviceManager_Release(mf_dxgi_manager);
        mf_dxgi_manager = NULL;
        mf_reset_token  = 0;
    }
    if (mf_functions.MFShutdown)
        mf_functions.MFShutdown();
#   if !HAVE_UWP
    if (mf_library) {
        dlclose(mf_library);
        mf_library = NULL;
    }
#   endif
    memset(&mf_functions, 0, sizeof(mf_functions));
#endif
}

int create_derive_mf_device_from_d3d11va(AVBufferRef *d3d11va_ref)
{
    int ret = AVERROR(ENOSYS);
#if (CONFIG_MEDIAFOUNDATION && CONFIG_D3D11VA)
    AVHWDeviceContext      *dev_ctx = NULL;
    AVD3D11VADeviceContext *hwctx   = NULL;
    HRESULT hr;

    if (!d3d11va_ref)
        return AVERROR(EINVAL);
    if ((ret = init_mf_functions()) < 0)
        goto exit;

    dev_ctx = (AVHWDeviceContext*)d3d11va_ref->data;
    hwctx = dev_ctx->hwctx;

    hr = mf_functions.MFCreateDXGIDeviceManager(&mf_reset_token, &mf_dxgi_manager);
    if (FAILED(hr)) {
        av_log(NULL, AV_LOG_DEBUG, "MFCreateDXGIDeviceManager failed: 0x%08lx\n", hr);
        ret = AVERROR_EXTERNAL;
        goto exit;
    }

    hr = IMFDXGIDeviceManager_ResetDevice(mf_dxgi_manager,
                                          (IUnknown*)hwctx->device, mf_reset_token);
    if (FAILED(hr)) {
        av_log(NULL, AV_LOG_DEBUG, "IMFDXGIDeviceManager_ResetDevice failed: 0x%08lx\n", hr);
        ret = AVERROR_EXTERNAL;
        goto exit;
    }

    return 0;

exit:
    if (mf_dxgi_manager) {
        IMFDXGIDeviceManager_Release(mf_dxgi_manager);
        mf_dxgi_manager = NULL;
        mf_reset_token = 0;
    }
#endif
    return ret;
}

#if (CONFIG_MEDIAFOUNDATION && CONFIG_D3D11VA)
enum ff_eAVEncH264VProfile {
    ff_eAVEncH264VProfile_Base   = 66,
    ff_eAVEncH264VProfile_Main   = 77,
    ff_eAVEncH264VProfile_High   = 100,
    ff_eAVEncH264VProfile_High10 = 110,
};

enum ff_eAVEncH265VProfile {
    ff_eAVEncH265VProfile_Main_420_8  = 1,
    ff_eAVEncH265VProfile_Main_420_10 = 2,
};

enum ff_eAVEncAV1VProfile {
    ff_eAVEncAV1VProfile_Main_420_8  = 1,
    ff_eAVEncAV1VProfile_Main_420_10 = 2,
};

static unsigned mf_map_av_to_default_mf_profile(enum AVCodecID codec, int profile)
{
    if (codec == AV_CODEC_ID_H264) {
        switch (profile) {
        case AV_PROFILE_H264_BASELINE: return ff_eAVEncH264VProfile_Base;
        case AV_PROFILE_H264_MAIN:     return ff_eAVEncH264VProfile_Main;
        case AV_PROFILE_H264_HIGH:     return ff_eAVEncH264VProfile_High;
        case AV_PROFILE_H264_HIGH_10:  return ff_eAVEncH264VProfile_High10;
        }
    } else if (codec == AV_CODEC_ID_HEVC) {
        switch (profile) {
        case AV_PROFILE_HEVC_MAIN:    return ff_eAVEncH265VProfile_Main_420_8;
        case AV_PROFILE_HEVC_MAIN_10: return ff_eAVEncH265VProfile_Main_420_10;
        }
    } else if (codec == AV_CODEC_ID_AV1) {
        switch (profile) {
        case AV_PROFILE_AV1_MAIN: return ff_eAVEncAV1VProfile_Main_420_8;
        }
    }

    return 0;
}

static const GUID *mf_map_mf_profile_to_mf_format_guid(const GUID *codec, unsigned profile)
{
    if (!codec)
        return NULL;

    if (IsEqualGUID(&MFVideoFormat_H264, codec)) {
        switch (profile) {
        case ff_eAVEncH264VProfile_Base:
        case ff_eAVEncH264VProfile_Main:
        case ff_eAVEncH264VProfile_High:   return &MFVideoFormat_NV12;
        case ff_eAVEncH264VProfile_High10: return &MFVideoFormat_P010;
        }
    } else if (IsEqualGUID(&MFVideoFormat_HEVC, codec)) {
        switch (profile) {
        case ff_eAVEncH265VProfile_Main_420_8:  return &MFVideoFormat_NV12;
        case ff_eAVEncH265VProfile_Main_420_10: return &MFVideoFormat_P010;
        }
    } else if (IsEqualGUID(&MFVideoFormat_AV1, codec)) {
        switch (profile) {
        case ff_eAVEncAV1VProfile_Main_420_8:  return &MFVideoFormat_NV12;
        case ff_eAVEncAV1VProfile_Main_420_10: return &MFVideoFormat_P010;
        }
    }

    return NULL;
}

static const GUID *mf_map_av_to_mf_format_guid(enum AVPixelFormat pix_fmt) {
    switch (pix_fmt) {
    case AV_PIX_FMT_NV12: return &MFVideoFormat_NV12;
    case AV_PIX_FMT_P010: return &MFVideoFormat_P010;
    default:              return NULL;
    }
}

static enum MfTestType mf_test_params(IMFActivate *pActivate,
                                      IMFTransform **ppMFT,
                                      const GUID *codec,
                                      UINT32 width, UINT32 height,
                                      UINT32 profile)
{
    IMFMediaType *out_type = NULL;
    IMFMediaType  *in_type = NULL;
    HRESULT hr;
    enum MfTestType test_ret = MF_TEST_CONTINUE;

    if (!pActivate || !ppMFT)
        return MF_TEST_BREAK;

    if (*ppMFT) {
        IMFTransform_SetOutputType(*ppMFT, 0, NULL, 0);
        IMFTransform_SetInputType(*ppMFT, 0, NULL, 0);
        IMFTransform_ProcessMessage(*ppMFT, MFT_MESSAGE_COMMAND_FLUSH, 0);
    } else {
        IMFAttributes *attrs = NULL;

        IMFActivate_ShutdownObject(pActivate);
        if (FAILED(IMFActivate_ActivateObject(pActivate, &IID_IMFTransform,
                                              (void**)ppMFT))) {
            IMFActivate_ShutdownObject(pActivate);
            return MF_TEST_BREAK;
        }

        if (SUCCEEDED(IMFTransform_GetAttributes(*ppMFT, &attrs))) {
            UINT32 is_mft_async = 0, is_d3d11_aware = 0;

            IMFAttributes_GetUINT32(attrs, &MF_TRANSFORM_ASYNC, &is_mft_async);
            IMFAttributes_GetUINT32(attrs, &MF_SA_D3D11_AWARE, &is_d3d11_aware);

            hr = IMFAttributes_SetUINT32(attrs, &MF_TRANSFORM_ASYNC_UNLOCK, TRUE);
            IMFAttributes_Release(attrs);

            if (!is_mft_async || !is_d3d11_aware || FAILED(hr)) {
                if (!is_mft_async)
                    av_log(NULL, AV_LOG_DEBUG, "MF_TRANSFORM_ASYNC unsupported\n");
                if (!is_d3d11_aware)
                    av_log(NULL, AV_LOG_DEBUG, "MF_SA_D3D11_AWARE unsupported\n");
                if (FAILED(hr))
                    av_log(NULL, AV_LOG_DEBUG, "MF_TRANSFORM_ASYNC_UNLOCK failed\n");

                test_ret = MF_TEST_BREAK;
                goto fail;
            }
        } else {
            test_ret = MF_TEST_BREAK;
            goto fail;
        }

        IMFTransform_ProcessMessage(*ppMFT, MFT_MESSAGE_COMMAND_FLUSH, 0);
        if (FAILED(IMFTransform_ProcessMessage(*ppMFT, MFT_MESSAGE_SET_D3D_MANAGER,
                                               (ULONG_PTR)mf_dxgi_manager))) {
            av_log(NULL, AV_LOG_DEBUG, "MFT_MESSAGE_SET_D3D_MANAGER failed\n");
            test_ret = MF_TEST_BREAK;
            goto fail;
        }
    }

    if (FAILED(mf_functions.MFCreateMediaType(&out_type)))
        return MF_TEST_CONTINUE;

#   define SET_MINIMAL_MEDIA_TYPES(type, subtype) do { \
        IMFMediaType_SetGUID((type), &MF_MT_MAJOR_TYPE, &MFMediaType_Video); \
        IMFMediaType_SetGUID((type), &MF_MT_SUBTYPE, (subtype)); \
        IMFMediaType_SetUINT64((type), &MF_MT_FRAME_SIZE, ((UINT64)width << 32) | height); \
        IMFMediaType_SetUINT64((type), &MF_MT_FRAME_RATE, ((UINT64)25 << 32) | 1); \
        IMFMediaType_SetUINT32((type), &MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive); \
    } while (0)

    SET_MINIMAL_MEDIA_TYPES(out_type, codec);

    IMFMediaType_SetUINT32(out_type, &MF_MT_AVG_BITRATE, 4000000);
    if (profile > 0)
        IMFAttributes_SetUINT32(out_type, &MF_MT_MPEG2_PROFILE, profile);

    hr = IMFTransform_SetOutputType(*ppMFT, 0, out_type, 0);
    if (out_type)
        IMFMediaType_Release(out_type);

    if (FAILED(hr)) {
        test_ret = MF_TEST_CONTINUE;
        goto fail;
    }

    if (profile > 0) {
        IMFMediaType *pType = NULL;
        DWORD index = 0;
        BOOL success = FALSE;

        if (FAILED(mf_functions.MFCreateMediaType(&in_type)))
            return MF_TEST_CONTINUE;

        while (SUCCEEDED(IMFTransform_GetInputAvailableType(*ppMFT, 0, index++, &pType))) {
            GUID input_format = { 0 };

            if (SUCCEEDED(IMFAttributes_GetGUID(pType, &MF_MT_SUBTYPE, &input_format))) {
                const GUID *format = mf_map_mf_profile_to_mf_format_guid(codec, profile);

                if (!format || !IsEqualGUID(&input_format, format)) {
                    if (pType) {
                        IMFMediaType_Release(pType);
                        pType = NULL;
                    }
                    continue;
                }

                SET_MINIMAL_MEDIA_TYPES(in_type, &input_format);

                hr = IMFTransform_SetInputType(*ppMFT, 0, in_type, 0);
                if (FAILED(hr)) {
                    if (pType)
                        IMFMediaType_Release(pType);
                    if (in_type)
                        IMFMediaType_Release(in_type);
                    test_ret = MF_TEST_CONTINUE;
                    goto fail;
                }
                success = TRUE;
                break;
            }
            if (pType) {
                IMFMediaType_Release(pType);
                pType = NULL;
            }
        }
        if (in_type)
            IMFMediaType_Release(in_type);
        if (success != TRUE)
            return MF_TEST_CONTINUE;
    }
#   undef SET_MINIMAL_MEDIA_TYPES

    return MF_TEST_SUCCESS;

fail:
    if (*ppMFT) {
        IMFTransform_ProcessMessage(*ppMFT, MFT_MESSAGE_SET_D3D_MANAGER,
                                    (ULONG_PTR)NULL);
        IMFTransform_Release(*ppMFT);
        *ppMFT = NULL;
    }
    return test_ret;
}
#endif

int print_mf_encoder_info_from_d3d11va(AVTextFormatContext *tfc, uint64_t dxgi_luid)
{
#if (CONFIG_MEDIAFOUNDATION && CONFIG_D3D11VA)
    IMFAttributes* pEnumAttrs = NULL;
    LUID mft_enum_luid = {
        .LowPart  = (DWORD)(dxgi_luid & 0xFFFFFFFF),
        .HighPart = (LONG)(dxgi_luid >> 32)
    };
    HRESULT hr;
    int header_printed = 0;
    int i, j;

#   if !HAVE_UWP
    if (!mf_library)
        return AVERROR(ENOSYS);
#   endif
    if (!mf_dxgi_manager)
        return AVERROR(ENOSYS);

    hr = mf_functions.MFCreateAttributes(&pEnumAttrs, 1);
    if (FAILED(hr))
        return AVERROR_UNKNOWN;

    hr = IMFAttributes_SetBlob(pEnumAttrs, &ff_MFT_ENUM_ADAPTER_LUID,
                               (const UINT8*)&mft_enum_luid, sizeof(mft_enum_luid));
    if (FAILED(hr)) {
        IMFAttributes_Release(pEnumAttrs);
        return AVERROR_UNKNOWN;
    }

    for (i = 0; mfenc_modes[i].name && mfenc_modes[i].guid; i++) {
        IMFActivate** ppActivates = NULL;
        UINT32 activate_count = 0;
        UINT32 activate_idx = 0;
        IMFTransform* pMFT = NULL;
        const MfEncMode* mode = &mfenc_modes[i];
        MFT_REGISTER_TYPE_INFO output_type_info = {
            .guidMajorType = MFMediaType_Video,
            .guidSubtype   = *(mode->guid)
        };
        unsigned min_width = 0, min_height = 0;
        unsigned max_width = 0, max_height = 0;
        char mft_desc[128*3+1] = { 0 };
        int mft_has_desc = 0;
        int header2_printed = 0;
        int header3_printed = 0;
        enum MfTestType test_ret = MF_TEST_SUCCESS;

        if (!mode->formats)
            continue;

        hr = mf_functions.MFTEnum2(MFT_CATEGORY_VIDEO_ENCODER,
                                   (MFT_ENUM_FLAG_SORTANDFILTER | MFT_ENUM_FLAG_HARDWARE),
                                   NULL, &output_type_info, pEnumAttrs,
                                   &ppActivates, &activate_count);
        if (FAILED(hr) || !ppActivates || !activate_count) {
            if (ppActivates)
                CoTaskMemFree(ppActivates);
            continue;
        }

        /* Check min res */
        for (const HwRes *r = &hw_res_ascend[0]; r->name; r++) {
            if (mode->legacy && (r->width > 4096 || r->height > 4096))
                break;

            test_ret = mf_test_params(ppActivates[0], &pMFT, mode->guid, r->width, r->height, 0);
            if (test_ret != MF_TEST_SUCCESS) {
                if (test_ret == MF_TEST_BREAK) break;
                else continue;
            }
            min_width  = r->width;
            min_height = r->height;
            break;
        }
        if (!min_width || !min_height)
            goto next;

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

            test_ret = mf_test_params(ppActivates[0], &pMFT, mode->guid, r->width, r->height, 0);
            if (test_ret != MF_TEST_SUCCESS) {
                if (test_ret == MF_TEST_BREAK) break;
                else continue;
            }
            max_width  = r->width;
            max_height = r->height;
            break;
        }
        if (!max_width || !max_height)
            goto next;

        if (!pMFT)
            goto next;

        {
            WCHAR *mft_desc_w = NULL;
            UINT32 mft_desc_w_len = 0;

            hr = IMFActivate_GetAllocatedString(ppActivates[0],
                                                &MFT_FRIENDLY_NAME_Attribute,
                                                &mft_desc_w, &mft_desc_w_len);
            if (SUCCEEDED(hr) && mft_desc_w && mft_desc_w_len) {
                int written_bytes = WideCharToMultiByte(
                    CP_UTF8, 0, mft_desc_w, -1,
                    mft_desc, FF_ARRAY_ELEMS(mft_desc),
                    NULL, NULL
                );
                mft_has_desc = written_bytes > 0;
                mft_desc[FF_ARRAY_ELEMS(mft_desc) - 1] = '\0';
            }
            CoTaskMemFree(mft_desc_w);
        }

        if (!header_printed) {
            mark_section_show_entries(SECTION_ID_DEVICE_ENCODERS_MF, 1, NULL);
            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_ENCODERS_MF);
            header_printed = 1;
        }

        mark_section_show_entries(SECTION_ID_DEVICE_ENCODER, 1, NULL);
        avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_ENCODER);
        print_str("codec_name", avcodec_get_name(mode->codec));
        print_int("codec_id", mode->codec);
        print_str("codec_desc", mode->name);
        if (mft_has_desc)
            print_str("mft_desc", mft_desc);

        print_int("min_width",  min_width);
        print_int("min_height", min_height);
        print_int("max_width",  max_width);
        print_int("max_height", max_height);

        /* Formats */
        for (j = 0; mode->formats[j] != AV_PIX_FMT_NONE; j++) {
            const GUID *format = mf_map_av_to_mf_format_guid(mode->formats[j]);
            IMFMediaType *pType = NULL;
            DWORD index = 0;

            if (!format)
                continue;

            while (SUCCEEDED(IMFTransform_GetInputAvailableType(pMFT, 0, index++, &pType))) {
                GUID input_format = { 0 };

                if (SUCCEEDED(IMFAttributes_GetGUID(pType, &MF_MT_SUBTYPE, &input_format))) {
                    if (IsEqualGUID(&input_format, format)) {
                        if (!header2_printed) {
                            mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMTS, 1, NULL);
                            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMTS);
                            header2_printed = 1;
                        }
                        mark_section_show_entries(SECTION_ID_DEVICE_PIX_FMT, 1, NULL);
                        avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PIX_FMT);
                        print_str("format_name", av_get_pix_fmt_name(mode->formats[j]));
                        print_int("format_id", mode->formats[j]);
                        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMT

                        if (pType) {
                            IMFMediaType_Release(pType);
                            pType = NULL;
                        }
                        break;
                    }
                }
                if (pType) {
                    IMFMediaType_Release(pType);
                    pType = NULL;
                }
            }
        }
        if (header2_printed)
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PIX_FMTS

        /* Profiles */
        for (j = 0; mode->profiles[j] != AV_PROFILE_UNKNOWN; j++) {
            const UINT32 profile = mf_map_av_to_default_mf_profile(mode->codec, mode->profiles[j]);

            if (!profile)
                continue;

            test_ret = mf_test_params(ppActivates[0], &pMFT, mode->guid, min_width, min_height, profile);
            if (test_ret != MF_TEST_SUCCESS) {
                if (test_ret == MF_TEST_BREAK) break;
                else continue;
            }
            if (!header3_printed) {
                mark_section_show_entries(SECTION_ID_DEVICE_PROFILES, 1, NULL);
                avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PROFILES);
                header3_printed = 1;
            }
            mark_section_show_entries(SECTION_ID_DEVICE_PROFILE, 1, NULL);
            avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_PROFILE);
            print_str("profile_name", avcodec_profile_name(mode->codec, mode->profiles[j]));
            print_int("profile_id", mode->profiles[j]);
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PROFILE
        }
        if (header3_printed)
            avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_PROFILES

        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_ENCODER

next:
        if (pMFT)
            IMFTransform_Release(pMFT);

        for (activate_idx = 0; activate_idx < activate_count; activate_idx++) {
            if (ppActivates[activate_idx])
                IMFActivate_Release(ppActivates[activate_idx]);
        }
        CoTaskMemFree(ppActivates);
    }
    if (header_printed)
        avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_ENCODERS_MF

    if (pEnumAttrs)
        IMFAttributes_Release(pEnumAttrs);
#endif
    return 0;
}
