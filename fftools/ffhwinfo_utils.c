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

#include "ffhwinfo_utils.h"

#define SHOW_OPTIONAL_FIELDS_AUTO       -1
#define SHOW_OPTIONAL_FIELDS_NEVER       0
#define SHOW_OPTIONAL_FIELDS_ALWAYS      1
static int show_optional_fields = SHOW_OPTIONAL_FIELDS_AUTO;

static struct section sections[] = {

    [SECTION_ID_ROOT] =           { SECTION_ID_ROOT, "Root", SECTION_FLAG_IS_WRAPPER,
                                    { SECTION_ID_DEVICES, SECTION_ID_PROGRAM_VERSION, -1 } },

    [SECTION_ID_DEVICES] =        { SECTION_ID_DEVICES,       "Devices", SECTION_FLAG_IS_ARRAY, { SECTION_ID_DEVICE, -1 } },
    [SECTION_ID_DEVICE] =         { SECTION_ID_DEVICE,        "Device", 0, { -1 } },

    [SECTION_ID_DEVICE_PATH_DRM] =      { SECTION_ID_DEVICE_PATH_DRM,      "DevicePathDRM",      0, { -1 } },
    [SECTION_ID_DEVICE_INDEX_D3D11VA] = { SECTION_ID_DEVICE_INDEX_D3D11VA, "DeviceIndexD3D11VA", 0, { -1 } },
    [SECTION_ID_DEVICE_INDEX_CUDA] =    { SECTION_ID_DEVICE_INDEX_CUDA,    "DeviceIndexCUDA",    0, { -1 } },

    [SECTION_ID_DEVICE_INFO_DRM] =      { SECTION_ID_DEVICE_INFO_DRM,      "DeviceInfoDRM",      0, { -1 } },
    [SECTION_ID_DEVICE_INFO_VAAPI] =    { SECTION_ID_DEVICE_INFO_VAAPI,    "DeviceInfoVAAPI",    0, { -1 } },
    [SECTION_ID_DEVICE_INFO_D3D11VA] =  { SECTION_ID_DEVICE_INFO_D3D11VA,  "DeviceInfoD3D11VA",  0, { -1 } },
    [SECTION_ID_DEVICE_INFO_QSV] =      { SECTION_ID_DEVICE_INFO_QSV,      "DeviceInfoQSV",      0, { -1 } },
    [SECTION_ID_DEVICE_INFO_OPENCL] =   { SECTION_ID_DEVICE_INFO_OPENCL,   "DeviceInfoOPENCL",   0, { -1 } },
    [SECTION_ID_DEVICE_INFO_VULKAN] =   { SECTION_ID_DEVICE_INFO_VULKAN,   "DeviceInfoVULKAN",   0, { -1 } },
    [SECTION_ID_DEVICE_INFO_CUDA] =     { SECTION_ID_DEVICE_INFO_CUDA,     "DeviceInfoCUDA",     0, { -1 } },
    [SECTION_ID_DEVICE_INFO_AMF] =      { SECTION_ID_DEVICE_INFO_AMF,      "DeviceInfoAMF",      0, { -1 } },

    [SECTION_ID_DECODERS_VAAPI] =   { SECTION_ID_DECODERS_VAAPI,   "DecodersVAAPI",   SECTION_FLAG_IS_ARRAY, { SECTION_ID_DECODER, -1 } },
    [SECTION_ID_DECODERS_D3D11VA] = { SECTION_ID_DECODERS_D3D11VA, "DecodersD3D11VA", SECTION_FLAG_IS_ARRAY, { SECTION_ID_DECODER, -1 } },
    [SECTION_ID_DECODERS_QSV] =     { SECTION_ID_DECODERS_QSV,     "DecodersQSV",     SECTION_FLAG_IS_ARRAY, { SECTION_ID_DECODER, -1 } },
    [SECTION_ID_DECODERS_CUDA] =    { SECTION_ID_DECODERS_CUDA,    "DecodersCUDA",    SECTION_FLAG_IS_ARRAY, { SECTION_ID_DECODER, -1 } },
    [SECTION_ID_DECODER] =          { SECTION_ID_DECODER,          "Decoder", 0, { -1 } },

    [SECTION_ID_ENCODERS_VAAPI] =   { SECTION_ID_ENCODERS_VAAPI,   "EncodersVAAPI",   SECTION_FLAG_IS_ARRAY, { SECTION_ID_ENCODER, -1 } },
    [SECTION_ID_ENCODERS_QSV] =     { SECTION_ID_ENCODERS_QSV,     "EncodersQSV",     SECTION_FLAG_IS_ARRAY, { SECTION_ID_ENCODER, -1 } },
    [SECTION_ID_ENCODERS_CUDA] =    { SECTION_ID_ENCODERS_CUDA,    "EncodersCUDA",    SECTION_FLAG_IS_ARRAY, { SECTION_ID_ENCODER, -1 } },
    [SECTION_ID_ENCODERS_AMF] =     { SECTION_ID_ENCODERS_AMF,     "EncodersAMF",     SECTION_FLAG_IS_ARRAY, { SECTION_ID_ENCODER, -1 } },
    [SECTION_ID_ENCODER] =          { SECTION_ID_ENCODER,          "Encoder", 0, { -1 } },

    [SECTION_ID_PROFILES] =       { SECTION_ID_PROFILES,      "Profiles", SECTION_FLAG_IS_ARRAY, { SECTION_ID_PROFILE, -1 } },
    [SECTION_ID_PROFILE] =        { SECTION_ID_PROFILE,       "Profile", 0, { -1 } },
    [SECTION_ID_PIXEL_FORMATS] =  { SECTION_ID_PIXEL_FORMATS, "PixelFormats", SECTION_FLAG_IS_ARRAY, { SECTION_ID_PIXEL_FORMAT, -1 } },
    [SECTION_ID_PIXEL_FORMAT] =   { SECTION_ID_PIXEL_FORMAT,  "PixelFormat", 0, { -1 } },
    [SECTION_ID_PRESETS] =        { SECTION_ID_PRESETS,       "Presets", SECTION_FLAG_IS_ARRAY, { SECTION_ID_PRESET, -1 } },
    [SECTION_ID_PRESET] =         { SECTION_ID_PRESET,        "Preset", 0, { -1 } },

    [SECTION_ID_PROGRAM_VERSION] =  { SECTION_ID_PROGRAM_VERSION, "ProgramVersion", 0, { -1 } },
};

static const char *writer_get_name(void *p)
{
    WriterContext *wctx = p;
    return wctx->writer->name;
}

#define OFFSET(x) offsetof(WriterContext, x)

static const AVOption writer_options[] = {
    { "string_validation", "set string validation mode",
      OFFSET(string_validation), AV_OPT_TYPE_INT, {.i64=WRITER_STRING_VALIDATION_REPLACE}, 0, WRITER_STRING_VALIDATION_NB-1, .unit = "sv" },
    { "sv", "set string validation mode",
      OFFSET(string_validation), AV_OPT_TYPE_INT, {.i64=WRITER_STRING_VALIDATION_REPLACE}, 0, WRITER_STRING_VALIDATION_NB-1, .unit = "sv" },
    { "ignore",  NULL, 0, AV_OPT_TYPE_CONST, {.i64 = WRITER_STRING_VALIDATION_IGNORE},  .unit = "sv" },
    { "replace", NULL, 0, AV_OPT_TYPE_CONST, {.i64 = WRITER_STRING_VALIDATION_REPLACE}, .unit = "sv" },
    { "fail",    NULL, 0, AV_OPT_TYPE_CONST, {.i64 = WRITER_STRING_VALIDATION_FAIL},    .unit = "sv" },
    { "string_validation_replacement", "set string validation replacement string", OFFSET(string_validation_replacement), AV_OPT_TYPE_STRING, {.str=""}},
    { "svr", "set string validation replacement string", OFFSET(string_validation_replacement), AV_OPT_TYPE_STRING, {.str="\xEF\xBF\xBD"}},
    { NULL }
};

static void *writer_child_next(void *obj, void *prev)
{
    WriterContext *ctx = obj;
    if (!prev && ctx->writer && ctx->writer->priv_class && ctx->priv)
        return ctx->priv;
    return NULL;
}

static const AVClass writer_class = {
    .class_name = "Writer",
    .item_name  = writer_get_name,
    .option     = writer_options,
    .version    = LIBAVUTIL_VERSION_INT,
    .child_next = writer_child_next,
};

int writer_close(WriterContext **wctx)
{
    int i;
    int ret = 0;

    if (!*wctx)
        return -1;

    if ((*wctx)->writer->uninit)
        (*wctx)->writer->uninit(*wctx);
    for (i = 0; i < SECTION_MAX_NB_LEVELS; i++)
        av_bprint_finalize(&(*wctx)->section_pbuf[i], NULL);
    if ((*wctx)->writer->priv_class)
        av_opt_free((*wctx)->priv);
    av_freep(&((*wctx)->priv));
    av_opt_free(*wctx);
    if ((*wctx)->avio) {
        avio_flush((*wctx)->avio);
        ret = avio_close((*wctx)->avio);
    }
    av_freep(wctx);
    return ret;
}

static void bprint_bytes(AVBPrint *bp, const uint8_t *ubuf, size_t ubuf_size)
{
    int i;
    av_bprintf(bp, "0X");
    for (i = 0; i < ubuf_size; i++)
        av_bprintf(bp, "%02X", ubuf[i]);
}

static inline void writer_w8_avio(WriterContext *wctx, int b)
{
    avio_w8(wctx->avio, b);
}

static inline void writer_put_str_avio(WriterContext *wctx, const char *str)
{
    avio_write(wctx->avio, str, strlen(str));
}

static inline void writer_printf_avio(WriterContext *wctx, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    avio_vprintf(wctx->avio, fmt, ap);
    va_end(ap);
}

static inline void writer_w8_printf(WriterContext *wctx, int b)
{
    printf("%c", b);
}

static inline void writer_put_str_printf(WriterContext *wctx, const char *str)
{
    printf("%s", str);
}

static inline void writer_printf_printf(WriterContext *wctx, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

int writer_open(WriterContext **wctx, const Writer *writer,
                const char *args, const char *output)
{
    int i, ret = 0;

    if (!(*wctx = av_mallocz(sizeof(WriterContext)))) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    if (!((*wctx)->priv = av_mallocz(writer->priv_size))) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    (*wctx)->class = &writer_class;
    (*wctx)->writer = writer;
    (*wctx)->level = -1;
    (*wctx)->sections = sections;
    (*wctx)->nb_sections = FF_ARRAY_ELEMS(sections);

    av_opt_set_defaults(*wctx);

    if (writer->priv_class) {
        void *priv_ctx = (*wctx)->priv;
        *((const AVClass **)priv_ctx) = writer->priv_class;
        av_opt_set_defaults(priv_ctx);
    }

    /* convert options to dictionary */
    if (args) {
        AVDictionary *opts = NULL;
        const AVDictionaryEntry *opt = NULL;

        if ((ret = av_dict_parse_string(&opts, args, "=", ":", 0)) < 0) {
            av_log(*wctx, AV_LOG_ERROR, "Failed to parse option string '%s' provided to writer context\n", args);
            av_dict_free(&opts);
            goto fail;
        }

        while ((opt = av_dict_iterate(opts, opt))) {
            if ((ret = av_opt_set(*wctx, opt->key, opt->value, AV_OPT_SEARCH_CHILDREN)) < 0) {
                av_log(*wctx, AV_LOG_ERROR, "Failed to set option '%s' with value '%s' provided to writer context\n",
                       opt->key, opt->value);
                av_dict_free(&opts);
                goto fail;
            }
        }

        av_dict_free(&opts);
    }

    /* validate replace string */
    {
        const uint8_t *p = (*wctx)->string_validation_replacement;
        const uint8_t *endp = p + strlen(p);
        while (*p) {
            const uint8_t *p0 = p;
            int32_t code;
            ret = av_utf8_decode(&code, &p, endp, (*wctx)->string_validation_utf8_flags);
            if (ret < 0) {
                AVBPrint bp;
                av_bprint_init(&bp, 0, AV_BPRINT_SIZE_AUTOMATIC);
                bprint_bytes(&bp, p0, p-p0),
                    av_log(wctx, AV_LOG_ERROR,
                           "Invalid UTF8 sequence %s found in string validation replace '%s'\n",
                           bp.str, (*wctx)->string_validation_replacement);
                return ret;
            }
        }
    }

    if (!output_filename) {
        (*wctx)->writer_w8 = writer_w8_printf;
        (*wctx)->writer_put_str = writer_put_str_printf;
        (*wctx)->writer_printf = writer_printf_printf;
    } else {
        if ((ret = avio_open(&(*wctx)->avio, output, AVIO_FLAG_WRITE)) < 0) {
            av_log(*wctx, AV_LOG_ERROR,
                   "Failed to open output '%s' with error: %s\n", output, av_err2str(ret));
            goto fail;
        }
        (*wctx)->writer_w8 = writer_w8_avio;
        (*wctx)->writer_put_str = writer_put_str_avio;
        (*wctx)->writer_printf = writer_printf_avio;
    }

    for (i = 0; i < SECTION_MAX_NB_LEVELS; i++)
        av_bprint_init(&(*wctx)->section_pbuf[i], 1, AV_BPRINT_SIZE_UNLIMITED);

    if ((*wctx)->writer->init)
        ret = (*wctx)->writer->init(*wctx);
    if (ret < 0)
        goto fail;

    return 0;

fail:
    writer_close(wctx);
    return ret;
}

inline void writer_print_section_header(WriterContext *wctx,
                                        int section_id)
{
    int parent_section_id;
    wctx->level++;
    av_assert0(wctx->level < SECTION_MAX_NB_LEVELS);
    parent_section_id = wctx->level ?
        (wctx->section[wctx->level-1])->id : SECTION_ID_NONE;

    wctx->nb_item[wctx->level] = 0;
    wctx->section[wctx->level] = &wctx->sections[section_id];

    if (wctx->writer->print_section_header)
        wctx->writer->print_section_header(wctx);
}

inline void writer_print_section_footer(WriterContext *wctx)
{
    int section_id = wctx->section[wctx->level]->id;
    int parent_section_id = wctx->level ?
        wctx->section[wctx->level-1]->id : SECTION_ID_NONE;

    if (parent_section_id != SECTION_ID_NONE)
        wctx->nb_item[wctx->level-1]++;
    if (wctx->writer->print_section_footer)
        wctx->writer->print_section_footer(wctx);
    wctx->level--;
}

inline void writer_print_integer(WriterContext *wctx,
                                 const char *key, long long int val)
{
    const struct section *section = wctx->section[wctx->level];

    if (section->show_all_entries || av_dict_get(section->entries_to_show, key, NULL, 0)) {
        wctx->writer->print_integer(wctx, key, val);
        wctx->nb_item[wctx->level]++;
    }
}

static inline int validate_string(WriterContext *wctx, char **dstp, const char *src)
{
    const uint8_t *p, *endp;
    AVBPrint dstbuf;
    int invalid_chars_nb = 0, ret = 0;

    av_bprint_init(&dstbuf, 0, AV_BPRINT_SIZE_UNLIMITED);

    endp = src + strlen(src);
    for (p = (uint8_t *)src; *p;) {
        uint32_t code;
        int invalid = 0;
        const uint8_t *p0 = p;

        if (av_utf8_decode(&code, &p, endp, wctx->string_validation_utf8_flags) < 0) {
            AVBPrint bp;
            av_bprint_init(&bp, 0, AV_BPRINT_SIZE_AUTOMATIC);
            bprint_bytes(&bp, p0, p-p0);
            av_log(wctx, AV_LOG_DEBUG,
                   "Invalid UTF-8 sequence %s found in string '%s'\n", bp.str, src);
            invalid = 1;
        }

        if (invalid) {
            invalid_chars_nb++;

            switch (wctx->string_validation) {
            case WRITER_STRING_VALIDATION_FAIL:
                av_log(wctx, AV_LOG_ERROR,
                       "Invalid UTF-8 sequence found in string '%s'\n", src);
                ret = AVERROR_INVALIDDATA;
                goto end;
                break;

            case WRITER_STRING_VALIDATION_REPLACE:
                av_bprintf(&dstbuf, "%s", wctx->string_validation_replacement);
                break;
            }
        }

        if (!invalid || wctx->string_validation == WRITER_STRING_VALIDATION_IGNORE)
            av_bprint_append_data(&dstbuf, p0, p-p0);
    }

    if (invalid_chars_nb && wctx->string_validation == WRITER_STRING_VALIDATION_REPLACE) {
        av_log(wctx, AV_LOG_WARNING,
               "%d invalid UTF-8 sequence(s) found in string '%s', replaced with '%s'\n",
               invalid_chars_nb, src, wctx->string_validation_replacement);
    }

end:
    av_bprint_finalize(&dstbuf, dstp);
    return ret;
}

#define PRINT_STRING_OPT      1
#define PRINT_STRING_VALIDATE 2

inline int writer_print_string(WriterContext *wctx,
                               const char *key, const char *val, int flags)
{
    const struct section *section = wctx->section[wctx->level];
    int ret = 0;

    if (show_optional_fields == SHOW_OPTIONAL_FIELDS_NEVER ||
        (show_optional_fields == SHOW_OPTIONAL_FIELDS_AUTO
        && (flags & PRINT_STRING_OPT)
        && !(wctx->writer->flags & WRITER_FLAG_DISPLAY_OPTIONAL_FIELDS)))
        return 0;

    if (section->show_all_entries || av_dict_get(section->entries_to_show, key, NULL, 0)) {
        if (flags & PRINT_STRING_VALIDATE) {
            char *key1 = NULL, *val1 = NULL;
            ret = validate_string(wctx, &key1, key);
            if (ret < 0) goto end;
            ret = validate_string(wctx, &val1, val);
            if (ret < 0) goto end;
            wctx->writer->print_string(wctx, key1, val1);
        end:
            if (ret < 0) {
                av_log(wctx, AV_LOG_ERROR,
                       "Invalid key=value string combination %s=%s in section %s\n",
                       key, val, section->unique_name);
            }
            av_free(key1);
            av_free(val1);
        } else {
            wctx->writer->print_string(wctx, key, val);
        }

        wctx->nb_item[wctx->level]++;
    }

    return ret;
}

void writer_print_integers(WriterContext *wctx, const char *name,
                           uint8_t *data, int size, const char *format,
                           int columns, int bytes, int offset_add)
{
    AVBPrint bp;
    int offset = 0, l, i;

    av_bprint_init(&bp, 0, AV_BPRINT_SIZE_UNLIMITED);
    av_bprintf(&bp, "\n");
    while (size) {
        av_bprintf(&bp, "%08x: ", offset);
        l = FFMIN(size, columns);
        for (i = 0; i < l; i++) {
            if      (bytes == 1) av_bprintf(&bp, format, *data);
            else if (bytes == 2) av_bprintf(&bp, format, AV_RN16(data));
            else if (bytes == 4) av_bprintf(&bp, format, AV_RN32(data));
            data += bytes;
            size --;
        }
        av_bprintf(&bp, "\n");
        offset += offset_add;
    }
    writer_print_string(wctx, name, bp.str, 0);
    av_bprint_finalize(&bp, NULL);
}

#define writer_w8(wctx_, b_) (wctx_)->writer_w8(wctx_, b_)
#define writer_put_str(wctx_, str_) (wctx_)->writer_put_str(wctx_, str_)
#define writer_printf(wctx_, fmt_, ...) (wctx_)->writer_printf(wctx_, fmt_, __VA_ARGS__)

#define MAX_REGISTERED_WRITERS_NB 64

static const Writer *registered_writers[MAX_REGISTERED_WRITERS_NB + 1];

static int writer_register(const Writer *writer)
{
    static int next_registered_writer_idx = 0;

    if (next_registered_writer_idx == MAX_REGISTERED_WRITERS_NB)
        return AVERROR(ENOMEM);

    registered_writers[next_registered_writer_idx++] = writer;
    return 0;
}

const Writer *writer_get_by_name(const char *name)
{
    int i;

    for (i = 0; registered_writers[i]; i++)
        if (!strcmp(registered_writers[i]->name, name))
            return registered_writers[i];

    return NULL;
}

/* WRITERS */

#define DEFINE_WRITER_CLASS(name)                   \
static const char *name##_get_name(void *ctx)       \
{                                                   \
    return #name ;                                  \
}                                                   \
static const AVClass name##_class = {               \
    .class_name = #name,                            \
    .item_name  = name##_get_name,                  \
    .option     = name##_options                    \
}

/* Default output */

typedef struct DefaultContext {
    const AVClass *class;
    int nokey;
    int noprint_wrappers;
    int nested_section[SECTION_MAX_NB_LEVELS];
} DefaultContext;

#undef OFFSET
#define OFFSET(x) offsetof(DefaultContext, x)

static const AVOption default_options[] = {
    { "noprint_wrappers", "do not print headers and footers", OFFSET(noprint_wrappers), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1 },
    { "nw",               "do not print headers and footers", OFFSET(noprint_wrappers), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1 },
    { "nokey",          "force no key printing",     OFFSET(nokey),          AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1 },
    { "nk",             "force no key printing",     OFFSET(nokey),          AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1 },
    {NULL},
};

DEFINE_WRITER_CLASS(default);

/* lame uppercasing routine, assumes the string is lower case ASCII */
static inline char *upcase_string(char *dst, size_t dst_size, const char *src)
{
    int i;
    for (i = 0; src[i] && i < dst_size-1; i++)
        dst[i] = av_toupper(src[i]);
    dst[i] = 0;
    return dst;
}

static void default_print_section_header(WriterContext *wctx)
{
    DefaultContext *def = wctx->priv;
    char buf[32];
    const struct section *section = wctx->section[wctx->level];
    const struct section *parent_section = wctx->level ?
        wctx->section[wctx->level-1] : NULL;

    av_bprint_clear(&wctx->section_pbuf[wctx->level]);
    if (parent_section &&
        !(parent_section->flags & (SECTION_FLAG_IS_WRAPPER|SECTION_FLAG_IS_ARRAY))) {
        def->nested_section[wctx->level] = 1;
        av_bprintf(&wctx->section_pbuf[wctx->level], "%s%s:",
                   wctx->section_pbuf[wctx->level-1].str,
                   upcase_string(buf, sizeof(buf),
                                 av_x_if_null(section->element_name, section->name)));
    }

    if (def->noprint_wrappers || def->nested_section[wctx->level])
        return;

    if (!(section->flags & (SECTION_FLAG_IS_WRAPPER|SECTION_FLAG_IS_ARRAY)))
        writer_printf(wctx, "[%s]\n", upcase_string(buf, sizeof(buf), section->name));
}

static void default_print_section_footer(WriterContext *wctx)
{
    DefaultContext *def = wctx->priv;
    const struct section *section = wctx->section[wctx->level];
    char buf[32];

    if (def->noprint_wrappers || def->nested_section[wctx->level])
        return;

    if (!(section->flags & (SECTION_FLAG_IS_WRAPPER|SECTION_FLAG_IS_ARRAY)))
        writer_printf(wctx, "[/%s]\n", upcase_string(buf, sizeof(buf), section->name));
}

static void default_print_str(WriterContext *wctx, const char *key, const char *value)
{
    DefaultContext *def = wctx->priv;

    if (!def->nokey)
        writer_printf(wctx, "%s%s=", wctx->section_pbuf[wctx->level].str, key);
    writer_printf(wctx, "%s\n", value);
}

static void default_print_int(WriterContext *wctx, const char *key, long long int value)
{
    DefaultContext *def = wctx->priv;

    if (!def->nokey)
        writer_printf(wctx, "%s%s=", wctx->section_pbuf[wctx->level].str, key);
    writer_printf(wctx, "%lld\n", value);
}

static const Writer default_writer = {
    .name                  = "default",
    .priv_size             = sizeof(DefaultContext),
    .print_section_header  = default_print_section_header,
    .print_section_footer  = default_print_section_footer,
    .print_integer         = default_print_int,
    .print_string          = default_print_str,
    .flags = WRITER_FLAG_DISPLAY_OPTIONAL_FIELDS,
    .priv_class            = &default_class,
};

/* JSON output */

typedef struct JSONContext {
    const AVClass *class;
    int indent_level;
    int compact;
    const char *item_sep, *item_start_end;
} JSONContext;

#undef OFFSET
#define OFFSET(x) offsetof(JSONContext, x)

static const AVOption json_options[]= {
    { "compact", "enable compact output", OFFSET(compact), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1 },
    { "c",       "enable compact output", OFFSET(compact), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1 },
    { NULL }
};

DEFINE_WRITER_CLASS(json);

static av_cold int json_init(WriterContext *wctx)
{
    JSONContext *json = wctx->priv;

    json->item_sep       = json->compact ? ", " : ",\n";
    json->item_start_end = json->compact ? " "  : "\n";

    return 0;
}

static const char *json_escape_str(AVBPrint *dst, const char *src, void *log_ctx)
{
    static const char json_escape[] = {'"', '\\', '\b', '\f', '\n', '\r', '\t', 0};
    static const char json_subst[]  = {'"', '\\',  'b',  'f',  'n',  'r',  't', 0};
    const char *p;

    for (p = src; *p; p++) {
        char *s = strchr(json_escape, *p);
        if (s) {
            av_bprint_chars(dst, '\\', 1);
            av_bprint_chars(dst, json_subst[s - json_escape], 1);
        } else if ((unsigned char)*p < 32) {
            av_bprintf(dst, "\\u00%02x", *p & 0xff);
        } else {
            av_bprint_chars(dst, *p, 1);
        }
    }
    return dst->str;
}

#define JSON_INDENT() writer_printf(wctx, "%*c", json->indent_level * 4, ' ')

static void json_print_section_header(WriterContext *wctx)
{
    JSONContext *json = wctx->priv;
    AVBPrint buf;
    const struct section *section = wctx->section[wctx->level];
    const struct section *parent_section = wctx->level ?
        wctx->section[wctx->level-1] : NULL;

    if (wctx->level && wctx->nb_item[wctx->level-1])
        writer_put_str(wctx, ",\n");

    if (section->flags & SECTION_FLAG_IS_WRAPPER) {
        writer_put_str(wctx, "{\n");
        json->indent_level++;
    } else {
        av_bprint_init(&buf, 1, AV_BPRINT_SIZE_UNLIMITED);
        json_escape_str(&buf, section->name, wctx);
        JSON_INDENT();

        json->indent_level++;
        if (section->flags & SECTION_FLAG_IS_ARRAY) {
            writer_printf(wctx, "\"%s\": [\n", buf.str);
        } else if (parent_section && !(parent_section->flags & SECTION_FLAG_IS_ARRAY)) {
            writer_printf(wctx, "\"%s\": {%s", buf.str, json->item_start_end);
        } else {
            writer_printf(wctx, "{%s", json->item_start_end);
        }
        av_bprint_finalize(&buf, NULL);
    }
}

static void json_print_section_footer(WriterContext *wctx)
{
    JSONContext *json = wctx->priv;
    const struct section *section = wctx->section[wctx->level];

    if (wctx->level == 0) {
        json->indent_level--;
        writer_put_str(wctx, "\n}\n");
    } else if (section->flags & SECTION_FLAG_IS_ARRAY) {
        writer_w8(wctx, '\n');
        json->indent_level--;
        JSON_INDENT();
        writer_w8(wctx, ']');
    } else {
        writer_put_str(wctx, json->item_start_end);
        json->indent_level--;
        if (!json->compact)
            JSON_INDENT();
        writer_w8(wctx, '}');
    }
}

static inline void json_print_item_str(WriterContext *wctx,
                                       const char *key, const char *value)
{
    AVBPrint buf;

    av_bprint_init(&buf, 1, AV_BPRINT_SIZE_UNLIMITED);
    writer_printf(wctx, "\"%s\":", json_escape_str(&buf, key,   wctx));
    av_bprint_clear(&buf);
    writer_printf(wctx, " \"%s\"", json_escape_str(&buf, value, wctx));
    av_bprint_finalize(&buf, NULL);
}

static void json_print_str(WriterContext *wctx, const char *key, const char *value)
{
    JSONContext *json = wctx->priv;
    const struct section *parent_section = wctx->level ?
        wctx->section[wctx->level-1] : NULL;

    if (wctx->nb_item[wctx->level])
        writer_put_str(wctx, json->item_sep);
    if (!json->compact)
        JSON_INDENT();
    json_print_item_str(wctx, key, value);
}

static void json_print_int(WriterContext *wctx, const char *key, long long int value)
{
    JSONContext *json = wctx->priv;
    const struct section *parent_section = wctx->level ?
        wctx->section[wctx->level-1] : NULL;
    AVBPrint buf;

    if (wctx->nb_item[wctx->level])
        writer_put_str(wctx, json->item_sep);
    if (!json->compact)
        JSON_INDENT();

    av_bprint_init(&buf, 1, AV_BPRINT_SIZE_UNLIMITED);
    writer_printf(wctx, "\"%s\": %lld", json_escape_str(&buf, key, wctx), value);
    av_bprint_finalize(&buf, NULL);
}

static const Writer json_writer = {
    .name                 = "json",
    .priv_size            = sizeof(JSONContext),
    .init                 = json_init,
    .print_section_header = json_print_section_header,
    .print_section_footer = json_print_section_footer,
    .print_integer        = json_print_int,
    .print_string         = json_print_str,
    .flags = WRITER_FLAG_PUT_PACKETS_AND_FRAMES_IN_SAME_CHAPTER,
    .priv_class           = &json_class,
};

void writer_register_all(void)
{
    static int initialized;

    if (initialized)
        return;
    initialized = 1;

    writer_register(&default_writer);
    writer_register(&json_writer);
}

void writer_cleanup(void)
{
    int i;
    for (i = 0; i < FF_ARRAY_ELEMS(sections); i++)
        av_dict_free(&(sections[i].entries_to_show));
}

inline void mark_section_show_entries(SectionID section_id,
                                      int show_all_entries, AVDictionary *entries)
{
    struct section *section = &sections[section_id];

    section->show_all_entries = show_all_entries;
    if (show_all_entries) {
        SectionID *id;
        for (id = section->children_ids; *id != -1; id++)
            mark_section_show_entries(*id, show_all_entries, entries);
    } else {
        av_dict_copy(&section->entries_to_show, entries, 0);
    }
}
