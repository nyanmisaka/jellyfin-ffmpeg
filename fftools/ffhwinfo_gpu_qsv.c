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

#if CONFIG_LIBMFX
#   include "libavutil/hwcontext_qsv.h"
#endif

/* QSV */
int print_qsv_device_info(WriterContext *wctx, AVBufferRef *qsv_ref)
{
    AVHWDeviceContext *dev_ctx = NULL;
    AVQSVDeviceContext *hwctx = NULL;
    mfxStatus sts;
    mfxIMPL impl;
    mfxVersion ver = {0};
    mfxPlatform platform = {0};

    if (!wctx || !qsv_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)qsv_ref->data;
    hwctx = dev_ctx->hwctx;

    sts = MFXQueryIMPL(hwctx->session, &impl);
    if (sts != MFX_ERR_NONE)
        return AVERROR(ENOSYS);

    sts = MFXQueryVersion(hwctx->session, &ver);
    if (sts != MFX_ERR_NONE)
        return AVERROR(ENOSYS);

    mark_section_show_entries(SECTION_ID_DEVICE_INFO_QSV, 1, NULL);
    writer_print_section_header(wctx, SECTION_ID_DEVICE_INFO_QSV);

    print_int("MfxImpl", impl);
    print_int("MfxImplVersionMajor", ver.Major);
    print_int("MfxImplVersionMinor", ver.Minor);
    print_int("MfxApiVersionMajor", MFX_VERSION_MAJOR);
    print_int("MfxApiVersionMinor", MFX_VERSION_MINOR);

    sts = MFXVideoCORE_QueryPlatform(hwctx->session, &platform);
    if (sts == MFX_ERR_NONE) {
#define MFX_DEPRECATED_OFF
        print_int("MfxPlatfromCodeName", platform.CodeName);
        print_int("MfxPlatfromDeviceId", platform.DeviceId);
        print_int("MfxPlatfromMediaAdapterType", platform.MediaAdapterType);
    }

    writer_print_section_footer(wctx);

    return 0;
}

// see also https://github.com/oneapi-src/oneVPL/commit/6e9f56aacbcb3b4ad1800cba091aaf9ec32135f9

int print_qsv_decoder_info(WriterContext *wctx, AVBufferRef *qsv_ref)
{
    if (!wctx || !qsv_ref)
        return AVERROR(EINVAL);

    return 0;
}

int print_qsv_encoder_info(WriterContext *wctx, AVBufferRef *qsv_ref)
{
    if (!wctx || !qsv_ref)
        return AVERROR(EINVAL);

    return 0;
}

int print_qsv_vpp_info(WriterContext *wctx, AVBufferRef *qsv_ref)
{
    if (!wctx || !qsv_ref)
        return AVERROR(EINVAL);

    return 0;
}
