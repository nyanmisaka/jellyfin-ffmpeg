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

/* D3D11VA */
#if CONFIG_D3D11VA
typedef HRESULT(WINAPI *PFN_CREATE_DXGI_FACTORY)(REFIID riid, void **ppFactory);
static PFN_CREATE_DXGI_FACTORY mCreateDXGIFactory = NULL;
#endif

int create_d3d11va_devices(HwDeviceRefs *refs)
{
#if CONFIG_D3D11VA
    int i, j, ret = 0;
    char ibuf[4];
    HRESULT hr;
    DXGI_ADAPTER_DESC desc;
    IDXGIFactory2 *pDXGIFactory = NULL;
    IDXGIAdapter *pAdapter = NULL;
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

    for (i = 0, j = 0; i < MAX_HW_DEVICE_NUM && refs; i++) {
        hr = IDXGIFactory2_EnumAdapters(pDXGIFactory, i, &pAdapter);
        if (FAILED(hr))
            continue;

        /* Filter out 'Microsoft Basic Render Driver' */
        hr = IDXGIAdapter2_GetDesc(pAdapter, &desc);
        if (pAdapter)
            IDXGIAdapter_Release(pAdapter);
        if (SUCCEEDED(hr) && desc.VendorId == 0x1414)
            continue;

        snprintf(ibuf, sizeof(ibuf), "%d", i);
        ret = av_hwdevice_ctx_create(&refs[j].d3d11va_ref, AV_HWDEVICE_TYPE_D3D11VA,
                                     ibuf, NULL, 0);
        if (ret < 0)
            continue;

        refs[j++].device_index = i;
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
    for (int i = 0; i < MAX_HW_DEVICE_NUM && refs && refs[i].d3d11va_ref; i++)
        av_hwdevice_ctx_create_derived(&refs[i].qsv_ref, AV_HWDEVICE_TYPE_QSV,
                                       refs[i].d3d11va_ref, 0);
}

/* D3D11VA -> OPENCL */
void create_derive_opencl_devices_from_d3d11va(HwDeviceRefs *refs)
{
    for (int i = 0; i < MAX_HW_DEVICE_NUM && refs && refs[i].d3d11va_ref; i++)
        av_hwdevice_ctx_create_derived(&refs[i].opencl_ref, AV_HWDEVICE_TYPE_OPENCL,
                                       refs[i].d3d11va_ref, 0);
}
