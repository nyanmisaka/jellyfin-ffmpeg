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

#if CONFIG_AMF
#    include <AMF/core/Factory.h>
#    include <AMF/components/VideoEncoderVCE.h>
#    include <AMF/components/VideoEncoderHEVC.h>
#    include <AMF/components/VideoEncoderAV1.h>
#endif

#if CONFIG_D3D11VA
#   define COBJMACROS
#   include <windows.h>
#   include <initguid.h>
#   include <d3d11.h>
#   include <dxgi1_2.h>
#   include "libavutil/hwcontext_d3d11va.h"
#   include "compat/w32dlfcn.h"
#endif

#if CONFIG_AMF
amf_handle         amf_lib = NULL;
AMFInit_Fn         amf_init_fn = NULL;
AMFQueryVersion_Fn amf_ver_fn = NULL;

amf_uint64 amf_ver = 0;
AMFFactory *amf_factory = NULL;
AMFContext *amf_ctx = NULL;
#endif

int init_amf_functions(void)
{
    int ret = 0;
#if (CONFIG_AMF && CONFIG_D3D11VA)
    AMF_RESULT res = AMF_OK;

    if (!amf_lib) {
        amf_lib = dlopen(AMF_DLL_NAMEA, RTLD_NOW | RTLD_LOCAL);
        if (!amf_lib)
            return AVERROR(ENOSYS);
    }

    if (!amf_ver_fn) {
        amf_ver_fn = (AMFQueryVersion_Fn)dlsym(amf_lib, AMF_QUERY_VERSION_FUNCTION_NAME);
        if (!amf_ver_fn) {
            ret = AVERROR(ENOSYS);
            goto exit;
        }
        res = amf_ver_fn(&amf_ver);
        if (res != AMF_OK) {
            ret = AVERROR(ENOSYS);
            goto exit;
        }
    }

    if (!amf_init_fn) {
        amf_init_fn = (AMFInit_Fn)dlsym(amf_lib, AMF_INIT_FUNCTION_NAME);
        if (!amf_init_fn) {
            ret = AVERROR(ENOSYS);
            goto exit;
        }
    }

    if (!amf_factory) {
        res = amf_init_fn(AMF_FULL_VERSION, &amf_factory);
        if (res != AMF_OK) {
            ret = AVERROR(ENOSYS);
            goto exit;
        }
    }

    if (amf_ctx) {
        amf_ctx->pVtbl->Terminate(amf_ctx);
        amf_ctx->pVtbl->Release(amf_ctx);
        amf_ctx = NULL;
    }

    res = amf_factory->pVtbl->CreateContext(amf_factory, &amf_ctx);
    if (res != AMF_OK) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    return 0;
exit:
    uninit_amf_functions();
#endif
    return ret;
}

void uninit_amf_functions(void)
{
#if (CONFIG_AMF && CONFIG_D3D11VA)
    if (amf_ctx) {
        amf_ctx->pVtbl->Terminate(amf_ctx);
        amf_ctx->pVtbl->Release(amf_ctx);
        amf_ctx = NULL;
    }
    if (amf_lib) {
        dlclose(amf_lib);
        amf_lib = NULL;
        amf_init_fn = NULL;
        amf_ver_fn = NULL;
    }
    amf_ver = 0;
    amf_factory = NULL;
#endif
}

int create_derive_amf_device_from_d3d11va(AVBufferRef *d3d11va_ref)
{
    int ret = 0;
#if (CONFIG_AMF && CONFIG_D3D11VA)
    AVHWDeviceContext    *dev_ctx = NULL;
    AVD3D11VADeviceContext *hwctx = NULL;
    AMF_RESULT res;

    if (!d3d11va_ref)
        return AVERROR(EINVAL);
    if ((ret = init_amf_functions()) < 0)
        return AVERROR(ENOSYS);

    dev_ctx = (AVHWDeviceContext*)d3d11va_ref->data;
    hwctx = dev_ctx->hwctx;

    res = amf_ctx->pVtbl->InitDX11(amf_ctx, hwctx->device, AMF_DX11_1);
    if (res != AMF_OK) {
        ret = AVERROR(ENOSYS);
        goto exit;
    }
    return 0;
exit:
#endif
    return ret;
}

int print_amf_device_info_from_d3d11va(WriterContext *wctx)
{
#if (CONFIG_AMF && CONFIG_D3D11VA)
    amf_uint64 api_ver = AMF_FULL_VERSION;
    if (!wctx || !amf_ctx)
        return AVERROR(EINVAL);

    mark_section_show_entries(SECTION_ID_DEVICE_INFO_AMF, 1, NULL);
    writer_print_section_header(wctx, SECTION_ID_DEVICE_INFO_AMF);

    print_int("AmfImplVersionMajor", AMF_GET_MAJOR_VERSION(amf_ver));
    print_int("AmfImplVersionMinor", AMF_GET_MINOR_VERSION(amf_ver));
    print_int("AmfImplVersionSubMinor", AMF_GET_SUBMINOR_VERSION(amf_ver));
    print_int("AmfImplVersionBuild", AMF_GET_BUILD_VERSION(amf_ver));
    print_int("AmfApiVersionMajor", AMF_GET_MAJOR_VERSION(api_ver));
    print_int("AmfApiVersionMinor", AMF_GET_MINOR_VERSION(api_ver));
    print_int("AmfApiVersionSubMinor", AMF_GET_SUBMINOR_VERSION(api_ver));
    print_int("AmfApiVersionBuild", AMF_GET_BUILD_VERSION(api_ver));

    writer_print_section_footer(wctx);
#endif
    return 0;
}

int print_amf_encoder_info_from_d3d11va(WriterContext *wctx)
{
#if (CONFIG_AMF && CONFIG_D3D11VA)
    if (!wctx || !amf_ctx)
        return AVERROR(EINVAL);
#endif
    return 0;
}
