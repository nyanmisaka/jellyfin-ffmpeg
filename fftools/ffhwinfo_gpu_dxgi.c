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

    for (i = 0, j = 0; i < MAX_HW_DEVICE_NUM && refs; i++) {
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
    for (unsigned i = 0; i < MAX_HW_DEVICE_NUM && refs && refs[i].d3d11va_ref; i++) {
        if (refs[i].device_vendor_id != HWINFO_VENDOR_ID_INTEL)
            continue;
        av_hwdevice_ctx_create_derived(&refs[i].qsv_ref, AV_HWDEVICE_TYPE_QSV,
                                       refs[i].d3d11va_ref, 0);
    }
}

/* D3D11VA -> OPENCL */
void create_derive_opencl_devices_from_d3d11va(HwDeviceRefs *refs)
{
    for (unsigned i = 0; i < MAX_HW_DEVICE_NUM && refs && refs[i].d3d11va_ref; i++) {
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
    for (unsigned i = 0; i < MAX_HW_DEVICE_NUM && refs && refs[i].d3d11va_ref; i++) {
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

static int print_d3d11va_device_info(WriterContext *wctx, AVBufferRef *d3d11va_ref)
#if CONFIG_D3D11VA
{
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

static int print_d3d11va_decoder_info(WriterContext *wctx, AVBufferRef *d3d11va_ref)
{
#if CONFIG_D3D11VA
    return 0;
#else
    return 0;
#endif
}

int print_dxgi_based_all(WriterContext *wctx, HwDeviceRefs *refs, int accel_flags)
{
    unsigned i, j;

    if (!wctx || !refs)
        return AVERROR(EINVAL);

    for (j = 0; j < MAX_HW_DEVICE_NUM && refs[j].d3d11va_ref; j++);
    if (j == 0)
        return 0;

    mark_section_show_entries(SECTION_ID_ROOT, 1, NULL);
    mark_section_show_entries(SECTION_ID_DEVICES, 1, NULL);
    mark_section_show_entries(SECTION_ID_DEVICE, 1, NULL);
    writer_print_section_header(wctx, SECTION_ID_ROOT);
    writer_print_section_header(wctx, SECTION_ID_DEVICES);

    for (i = 0; i < j; i++) {
        writer_print_section_header(wctx, SECTION_ID_DEVICE);

        /* DXGI/D3D11VA based device index */
        print_int("DeviceIndexD3D11VA", refs[i].device_index_dxgi);

        /* D3D11VA device info */
        if ((accel_flags & HWINFO_FLAG_PRINT_DEV) &&
            (accel_flags & HWINFO_FLAG_PRINT_OS_VA))
            print_d3d11va_device_info(wctx, refs[i].d3d11va_ref);

        /* D3D11VA decoder info */
        if ((accel_flags & HWINFO_FLAG_PRINT_DEC) &&
            (accel_flags & HWINFO_FLAG_PRINT_OS_VA))
            print_d3d11va_decoder_info(wctx, refs[i].d3d11va_ref);

        /* QSV device info */
        // if (accel_flags & HWINFO_FLAG_PRINT_DEV)
        //     print_qsv_device_info(wctx, refs[i].qsv_ref);

        /* QSV decoder info */
        // if (accel_flags & HWINFO_FLAG_PRINT_DEC)
        //     print_qsv_decoder_info(wctx, refs[i].qsv_ref);

        /* QSV encoder info */
        // if (accel_flags & HWINFO_FLAG_PRINT_ENC)
        //     print_qsv_encoder_info(wctx, refs[i].qsv_ref);

        /* QSV filter info */
        // if (accel_flags & HWINFO_FLAG_PRINT_VPP)
        //     print_qsv_filter_info(wctx, refs[i].qsv_ref);

        /* AMF encoder info */
        // if (accel_flags & HWINFO_FLAG_PRINT_ENC)
        //     print_amf_encoder_info(wctx, refs[i].d3d11va_ref);

        /* OPENCL device info */
        // if (accel_flags & HWINFO_FLAG_PRINT_COMPUTE_OPENCL)
        //     print_opencl_device_info(wctx, refs[i].opencl_ref);

        writer_print_section_footer(wctx);
    }

    writer_print_section_footer(wctx);
    writer_print_section_footer(wctx);

    return 0;
}
