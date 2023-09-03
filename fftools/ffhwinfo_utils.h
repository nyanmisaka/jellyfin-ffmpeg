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

#ifndef FFTOOLS_FFHWINFO_UTILS_H
#define FFTOOLS_FFHWINFO_UTILS_H

#include "config.h"
#include "libavutil/ffversion.h"

#include <string.h>
#include <math.h>

#include "libavformat/avformat.h"
#include "libavformat/version.h"
#include "libavcodec/avcodec.h"
#include "libavcodec/version.h"
#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/bprint.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/dict.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/libm.h"
#include "libavutil/parseutils.h"
#include "libavdevice/version.h"
#include "libswscale/version.h"
#include "libpostproc/version.h"
#include "libavfilter/version.h"
#include "cmdutils.h"
#include "opt_common.h"

extern const char *output_filename;

/* section structure definition */

#define SECTION_MAX_NB_CHILDREN 10

struct section {
    int id;             ///< unique id identifying a section
    const char *name;

#define SECTION_FLAG_IS_WRAPPER      1 ///< the section only contains other sections, but has no data at its own level
#define SECTION_FLAG_IS_ARRAY        2 ///< the section contains an array of elements of the same type
#define SECTION_FLAG_HAS_VARIABLE_FIELDS 4 ///< the section may contain a variable number of fields with variable keys.
                                           ///  For these sections the element_name field is mandatory.
    int flags;
    int children_ids[SECTION_MAX_NB_CHILDREN+1]; ///< list of children section IDS, terminated by -1
    const char *element_name; ///< name of the contained element, if provided
    const char *unique_name;  ///< unique section name, in case the name is ambiguous
    AVDictionary *entries_to_show;
    int show_all_entries;
};

typedef enum {
    SECTION_ID_NONE = -1,
    SECTION_ID_ROOT,
    SECTION_ID_PROGRAM_VERSION,
    SECTION_ID_DEVICES,
    SECTION_ID_DEVICE,

    SECTION_ID_DEVICE_PATH_DRM,
    SECTION_ID_DEVICE_INDEX_D3D11VA,
    SECTION_ID_DEVICE_INDEX_CUDA,

    SECTION_ID_DEVICE_INFO_DRM,
    SECTION_ID_DEVICE_INFO_VAAPI,
    SECTION_ID_DEVICE_INFO_D3D11VA,
    SECTION_ID_DEVICE_INFO_QSV,
    SECTION_ID_DEVICE_INFO_OPENCL,
    SECTION_ID_DEVICE_INFO_VULKAN,
    SECTION_ID_DEVICE_INFO_CUDA,
    SECTION_ID_DEVICE_INFO_AMF,

    SECTION_ID_DECODERS_VAAPI,
    SECTION_ID_DECODERS_D3D11VA,
    SECTION_ID_DECODERS_QSV,
    SECTION_ID_DECODERS_CUDA,
    SECTION_ID_DECODER,

    SECTION_ID_ENCODERS_VAAPI,
    SECTION_ID_ENCODERS_QSV,
    SECTION_ID_ENCODERS_CUDA,
    SECTION_ID_ENCODERS_AMF,
    SECTION_ID_ENCODER,

    SECTION_ID_PROFILES,
    SECTION_ID_PROFILE,
    SECTION_ID_PIXEL_FORMATS,
    SECTION_ID_PIXEL_FORMAT,
    SECTION_ID_PRESETS,
    SECTION_ID_PRESET,
} SectionID;

/* WRITERS API */

typedef struct WriterContext WriterContext;

#define WRITER_FLAG_DISPLAY_OPTIONAL_FIELDS 1
#define WRITER_FLAG_PUT_PACKETS_AND_FRAMES_IN_SAME_CHAPTER 2

typedef enum {
    WRITER_STRING_VALIDATION_FAIL,
    WRITER_STRING_VALIDATION_REPLACE,
    WRITER_STRING_VALIDATION_IGNORE,
    WRITER_STRING_VALIDATION_NB
} StringValidation;

typedef struct Writer {
    const AVClass *priv_class;      ///< private class of the writer, if any
    int priv_size;                  ///< private size for the writer context
    const char *name;

    int  (*init)  (WriterContext *wctx);
    void (*uninit)(WriterContext *wctx);

    void (*print_section_header)(WriterContext *wctx);
    void (*print_section_footer)(WriterContext *wctx);
    void (*print_integer)       (WriterContext *wctx, const char *, long long int);
    void (*print_rational)      (WriterContext *wctx, AVRational *q, char *sep);
    void (*print_string)        (WriterContext *wctx, const char *, const char *);
    int flags;                  ///< a combination or WRITER_FLAG_*
} Writer;

#define SECTION_MAX_NB_LEVELS 10

struct WriterContext {
    const AVClass *class;           ///< class of the writer
    const Writer *writer;           ///< the Writer of which this is an instance
    AVIOContext *avio;              ///< the I/O context used to write

    void (* writer_w8)(WriterContext *wctx, int b);
    void (* writer_put_str)(WriterContext *wctx, const char *str);
    void (* writer_printf)(WriterContext *wctx, const char *fmt, ...);

    char *name;                     ///< name of this writer instance
    void *priv;                     ///< private data for use by the filter

    const struct section *sections; ///< array containing all sections
    int nb_sections;                ///< number of sections

    int level;                      ///< current level, starting from 0

    /** number of the item printed in the given section, starting from 0 */
    unsigned int nb_item[SECTION_MAX_NB_LEVELS];

    /** section per each level */
    const struct section *section[SECTION_MAX_NB_LEVELS];
    AVBPrint section_pbuf[SECTION_MAX_NB_LEVELS]; ///< generic print buffer dedicated to each section,
                                                  ///  used by various writers

    unsigned int nb_section_packet; ///< number of the packet section in case we are in "packets_and_frames" section
    unsigned int nb_section_frame;  ///< number of the frame  section in case we are in "packets_and_frames" section
    unsigned int nb_section_packet_frame; ///< nb_section_packet or nb_section_frame according if is_packets_and_frames

    int string_validation;
    char *string_validation_replacement;
    unsigned int string_validation_utf8_flags;
};

int writer_close(WriterContext **wctx);
int writer_open(WriterContext **wctx, const Writer *writer,
                const char *args, const char *output);
void writer_print_section_header(WriterContext *wctx,
                                 int section_id);
void writer_print_section_footer(WriterContext *wctx);
void writer_print_integer(WriterContext *wctx,
                          const char *key, long long int val);
int writer_print_string(WriterContext *wctx,
                        const char *key, const char *val, int flags);
void writer_print_integers(WriterContext *wctx, const char *name,
                           uint8_t *data, int size, const char *format,
                           int columns, int bytes, int offset_add);
const Writer *writer_get_by_name(const char *name);
void writer_register_all(void);
void writer_cleanup(void);
void mark_section_show_entries(SectionID section_id,
                               int show_all_entries, AVDictionary *entries);

#endif /* FFTOOLS_FFHWINFO_UTILS_H */
