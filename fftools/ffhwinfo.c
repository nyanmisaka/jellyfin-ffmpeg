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

#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "ffhwinfo_gpu.h"
#include "ffhwinfo_utils.h"

#include "cmdutils.h"
#include "opt_common.h"
#include "libavutil/opt.h"
#include "libavutil/avassert.h"
#include "libavutil/error.h"
#include "libavcodec/avcodec.h"

const char program_name[] = "ffhwinfo";
const int program_birth_year = 2023;
const char *output_filename = NULL;

static char *accel_type = NULL;
static int accel_flags = HWINFO_DEFAULT_PRINT_FLAGS;
static char *print_format = NULL;

static const char *const accel_type_names[] = {
    [HWINFO_ACCEL_TYPE_VAAPI] = "vaapi",
    [HWINFO_ACCEL_TYPE_QSV]   = "qsv",
    [HWINFO_ACCEL_TYPE_CUDA]  = "cuda",
    [HWINFO_ACCEL_TYPE_AMF]   = "amf",
};

static enum HWInfoAccelType find_accel_type_by_name(const char *name)
{
    for (unsigned type = 0; type < FF_ARRAY_ELEMS(accel_type_names); type++) {
        if (accel_type_names[type] && !strcmp(accel_type_names[type], name))
            return type;
    }
    return HWINFO_ACCEL_TYPE_NONE;
}

static int opt_accel_flags(void* optctx, const char *opt, const char *arg)
{
    static const AVOption opts[] = {
        { "accelflags", NULL, 0, AV_OPT_TYPE_FLAGS, { .i64 = 0 }, INT64_MIN, INT64_MAX,   .unit = "flags" },
        { "all",        NULL, 0, AV_OPT_TYPE_CONST, { .i64 = HWINFO_DEFAULT_PRINT_FLAGS }, .unit = "flags" },
        { "dev",        NULL, 0, AV_OPT_TYPE_CONST, { .i64 = HWINFO_FLAG_PRINT_DEV }, .unit = "flags" },
        { "dec",        NULL, 0, AV_OPT_TYPE_CONST, { .i64 = HWINFO_FLAG_PRINT_DEC }, .unit = "flags" },
        { "enc",        NULL, 0, AV_OPT_TYPE_CONST, { .i64 = HWINFO_FLAG_PRINT_ENC }, .unit = "flags" },
        { "vpp",        NULL, 0, AV_OPT_TYPE_CONST, { .i64 = HWINFO_FLAG_PRINT_VPP }, .unit = "flags" },
        { "ocl",        NULL, 0, AV_OPT_TYPE_CONST, { .i64 = HWINFO_FLAG_PRINT_OPT_OPENCL }, .unit = "flags" },
        { "vk",         NULL, 0, AV_OPT_TYPE_CONST, { .i64 = HWINFO_FLAG_PRINT_OPT_VULKAN }, .unit = "flags" },
        { "dx11",       NULL, 0, AV_OPT_TYPE_CONST, { .i64 = HWINFO_FLAG_PRINT_OPT_D3D11VA }, .unit = "flags" },
        { "osva",       NULL, 0, AV_OPT_TYPE_CONST, { .i64 = HWINFO_FLAG_PRINT_OS_VA }, .unit = "flags" },
    };
    static const AVClass class = {
        .class_name = "",
        .item_name  = av_default_item_name,
        .option     = opts,
        .version    = LIBAVUTIL_VERSION_INT,
    };
    const AVClass *pclass = &class;

    int ret = av_opt_eval_flags(&pclass, &opts[0], arg, &accel_flags);
    if (ret < 0)
        return ret;

    if (!(accel_flags & HWINFO_FLAG_PRINT_DEV) &&
        !(accel_flags & HWINFO_FLAG_PRINT_DEC) &&
        !(accel_flags & HWINFO_FLAG_PRINT_ENC) &&
        !(accel_flags & HWINFO_FLAG_PRINT_VPP)) {
        accel_flags = HWINFO_DEFAULT_PRINT_FLAGS;
    }

    return ret;
}

static void opt_output_file(void *optctx, const char *arg)
{
    if (output_filename) {
        av_log(NULL, AV_LOG_ERROR,
                "Argument '%s' provided as output filename, but '%s' was already specified.\n",
                arg, output_filename);
        exit_program(1);
    }
    if (!strcmp(arg, "-"))
        arg = "fd:";
    output_filename = arg;
}

static int opt_output_file_o(void *optctx, const char *opt, const char *arg)
{
    opt_output_file(optctx, arg);
    return 0;
}

static const OptionDef options[] = {
    { "h",            OPT_EXIT,              { .func_arg = show_help },    "show help", "topic" },
    { "?",            OPT_EXIT,              { .func_arg = show_help },    "show help", "topic" },
    { "help",         OPT_EXIT,              { .func_arg = show_help },    "show help", "topic" },
    { "-help",        OPT_EXIT,              { .func_arg = show_help },    "show help", "topic" },
    { "loglevel",     HAS_ARG,               { .func_arg = opt_loglevel }, "set logging level", "loglevel" },
    { "v",            HAS_ARG,               { .func_arg = opt_loglevel }, "set logging level", "loglevel" },
    { "hide_banner",  OPT_BOOL | OPT_EXPERT, { &hide_banner }, "do not show program banner", "hide_banner" },
    { "acceltype",    OPT_STRING | HAS_ARG,  { &accel_type },
      "set the acceleration type (available types are: vaapi, qsv, cuda, amf)", "type" },
    { "accelflags",   HAS_ARG,               { .func_arg = opt_accel_flags },
      "set the acceleration flag (available flags are: all, dev, dec, enc, vpp, ocl, vk, dx11, osva)", "flags" },
    { "print_format", OPT_STRING | HAS_ARG,  { &print_format },
      "set the output printing format (available formats are: default, json)", "format" },
    { "of",           OPT_STRING | HAS_ARG,  { &print_format }, "alias for -print_format", "format" },
    { "o",            HAS_ARG,               { .func_arg = opt_output_file_o }, "write to specified output", "output_file"},
    { NULL, },
};

static void ffhwinfo_cleanup(int ret)
{
    writer_cleanup();
}

static void show_usage(void)
{
    av_log(NULL, AV_LOG_INFO, "Simple hardware acceleration devices info analyzer\n");
    av_log(NULL, AV_LOG_INFO, "usage: %s [options]\n", program_name);
    av_log(NULL, AV_LOG_INFO, "example: %s -acceltype qsv -accelflags dev+dec+enc+vpp\n", program_name);
    av_log(NULL, AV_LOG_INFO, "\n");
}

void show_help_default(const char *opt, const char *arg)
{
    show_usage();
    show_help_options(options, "Main options:", 0, 0, 0);
    printf("\n");
}

int main (int argc, char **argv)
{
    const Writer *w;
    WriterContext *wctx;
    char *buf;
    char *w_name = NULL, *w_args = NULL;
    enum HWInfoAccelType type;
    int ret = 0;

    /* Configure fftool internals */
    register_exit(ffhwinfo_cleanup);
    parse_loglevel(argc, argv, options);
    show_banner(argc, argv, options);
    parse_options(NULL, argc, argv, options, NULL);

    /* Prepare writers */
    writer_register_all();

    if (!print_format)
        print_format = av_strdup("default");
    if (!print_format) {
        ret = AVERROR(ENOMEM);
        goto end;
    }
    w_name = av_strtok(print_format, "=", &buf);
    if (!w_name) {
        av_log(NULL, AV_LOG_ERROR,
               "No name specified for the output format\n");
        ret = AVERROR(EINVAL);
        goto end;
    }
    w_args = buf;

    w = writer_get_by_name(w_name);
    if (!w) {
        av_log(NULL, AV_LOG_ERROR, "Unknown output format with name '%s'\n", w_name);
        ret = AVERROR(EINVAL);
        goto end;
    }

    /* Prepare accel type and flags */
    if (!accel_type) {
        show_usage();
        av_log(NULL, AV_LOG_ERROR, "You have to specify one acceleration type.\n");
        av_log(NULL, AV_LOG_ERROR, "Use '%s -h' to get full help.\n", program_name);
        ret = AVERROR(EINVAL);
        goto end;
    }

    type = find_accel_type_by_name(accel_type);
    if (type == HWINFO_ACCEL_TYPE_NONE) {
        av_log(NULL, AV_LOG_ERROR, "Acceleration type '%s' is not supported!\n", accel_type);
        av_log(NULL, AV_LOG_ERROR, "Available types are: vaapi, qsv, cuda, amf\n");
        ret = AVERROR(EINVAL);
        goto end;
    }

    av_log(NULL, AV_LOG_DEBUG, "Acceleration flags: %d!\n", accel_flags);

    /* Check GPUs and write output */
    if ((ret = writer_open(&wctx, w, w_args, output_filename)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Failed to open the writer: %s\n", av_err2str(ret));
        goto end;
    }

    show_accel_device_info(wctx, type, accel_flags);

    if ((ret = writer_close(&wctx)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Writing output failed: %s\n", av_err2str(ret));
        goto end;
    }

end:
    writer_cleanup();
    return ret < 0;
}
