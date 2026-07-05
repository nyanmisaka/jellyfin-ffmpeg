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

#if CONFIG_OPENCL
#   include "libavutil/hwcontext_opencl.h"
#endif

#if CONFIG_OPENCL
typedef enum {
    TYPE_STR,
    TYPE_UINT,
    TYPE_ULONG,
    TYPE_SIZE_T
} OpenclAttrType;

typedef struct {
    const cl_uint        attr_val;
    const char          *attr_str;
    const OpenclAttrType attr_type;
} OpenclAttr;

static const OpenclAttr opencl_platform_attrs[] = {
    { CL_PLATFORM_NAME,                                 "platform_name",                                 TYPE_STR    },
    { CL_PLATFORM_VENDOR,                               "platform_vendor",                               TYPE_STR    },
    { CL_PLATFORM_VERSION,                              "platform_version",                              TYPE_STR    },
    { CL_PLATFORM_PROFILE,                              "platform_profile",                              TYPE_STR    },
    { CL_PLATFORM_EXTENSIONS,                           "platform_extensions",                           TYPE_STR    },
};
static const OpenclAttr opencl_device_attrs[] = {
    { CL_DEVICE_NAME,                                   "device_name",                                   TYPE_STR    },
    { CL_DEVICE_VENDOR,                                 "device_vendor",                                 TYPE_STR    },
    { CL_DEVICE_VERSION,                                "device_version",                                TYPE_STR    },
    { CL_DEVICE_OPENCL_C_VERSION,                       "device_opencl_c_version",                       TYPE_STR    },
    { CL_DEVICE_VENDOR_ID,                              "device_vendor_id",                              TYPE_UINT   },
    { CL_DEVICE_TYPE,                                   "device_type",                                   TYPE_ULONG  },
    { CL_DEVICE_PROFILE,                                "device_profile",                                TYPE_STR    },
    { CL_DRIVER_VERSION,                                "driver_version",                                TYPE_STR    },
    { CL_DEVICE_MAX_COMPUTE_UNITS,                      "device_max_compute_units",                      TYPE_UINT   },
    { CL_DEVICE_MAX_WORK_GROUP_SIZE,                    "device_max_work_group_size",                    TYPE_SIZE_T },
    { CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,               "device_max_work_item_dimensions",               TYPE_UINT   },
    { CL_DEVICE_MAX_CLOCK_FREQUENCY,                    "device_max_clock_frequency",                    TYPE_UINT   },
#   ifdef CL_VERSION_2_1
    { CL_DEVICE_MAX_NUM_SUB_GROUPS,                     "device_max_num_sub_groups",                     TYPE_UINT   },
    { CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS, "device_sub_group_independent_forward_progress", TYPE_UINT   },
#   endif
    { CL_DEVICE_SINGLE_FP_CONFIG,                       "device_single_fp_config",                       TYPE_ULONG  },
    { CL_DEVICE_DOUBLE_FP_CONFIG,                       "device_double_fp_config",                       TYPE_ULONG  },
    { CL_DEVICE_GLOBAL_MEM_SIZE,                        "device_global_mem_size",                        TYPE_ULONG  },
    { CL_DEVICE_MAX_MEM_ALLOC_SIZE,                     "device_max_mem_alloc_size",                     TYPE_ULONG  },
    { CL_DEVICE_LOCAL_MEM_SIZE,                         "device_local_mem_size",                         TYPE_ULONG  },
    { CL_DEVICE_LOCAL_MEM_TYPE,                         "device_local_mem_type",                         TYPE_UINT   },
    { CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,               "device_max_constant_buffer_size",               TYPE_ULONG  },
    { CL_DEVICE_MAX_CONSTANT_ARGS,                      "device_max_constant_args",                      TYPE_UINT   },
    { CL_DEVICE_MAX_PARAMETER_SIZE,                     "device_max_parameter_size",                     TYPE_SIZE_T },
    { CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,              "device_global_mem_cacheline_size",              TYPE_UINT   },
    { CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,                  "device_global_mem_cache_size",                  TYPE_ULONG  },
    { CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,                  "device_global_mem_cache_type",                  TYPE_UINT   },
    { CL_DEVICE_IMAGE_SUPPORT,                          "device_image_support",                          TYPE_UINT   },
    { CL_DEVICE_IMAGE2D_MAX_WIDTH,                      "device_image2d_max_width",                      TYPE_SIZE_T },
    { CL_DEVICE_IMAGE2D_MAX_HEIGHT,                     "device_image2d_max_height",                     TYPE_SIZE_T },
    { CL_DEVICE_IMAGE3D_MAX_WIDTH,                      "device_image3d_max_width",                      TYPE_SIZE_T },
    { CL_DEVICE_IMAGE3D_MAX_HEIGHT,                     "device_image3d_max_height",                     TYPE_SIZE_T },
    { CL_DEVICE_IMAGE3D_MAX_DEPTH,                      "device_image3d_max_depth",                      TYPE_SIZE_T },
    { CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,                  "device_image_max_buffer_size",                  TYPE_SIZE_T },
    { CL_DEVICE_IMAGE_MAX_ARRAY_SIZE,                   "device_image_max_array_size",                   TYPE_SIZE_T },
    { CL_DEVICE_MAX_READ_IMAGE_ARGS,                    "device_max_read_image_args",                    TYPE_UINT   },
    { CL_DEVICE_MAX_WRITE_IMAGE_ARGS,                   "device_max_write_image_args",                   TYPE_UINT   },
    { CL_DEVICE_MAX_SAMPLERS,                           "device_max_samplers",                           TYPE_UINT   },
#   ifdef CL_VERSION_2_0
    { CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS,              "device_max_read_write_image_args",              TYPE_UINT   },
#   endif
    { CL_DEVICE_MEM_BASE_ADDR_ALIGN,                    "device_mem_base_addr_align",                    TYPE_UINT   },
    { CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,               "device_min_data_type_align_size",               TYPE_UINT   },
    { CL_DEVICE_IMAGE_PITCH_ALIGNMENT,                  "device_image_pitch_alignment",                  TYPE_UINT   },
    { CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT,           "device_image_base_address_alignment",           TYPE_UINT   },
    { CL_DEVICE_EXECUTION_CAPABILITIES,                 "device_execution_capabilities",                 TYPE_ULONG  },
    { CL_DEVICE_QUEUE_PROPERTIES,                       "device_queue_properties",                       TYPE_ULONG  },
    { CL_DEVICE_PROFILING_TIMER_RESOLUTION,             "device_profiling_timer_resolution",             TYPE_SIZE_T },
    { CL_DEVICE_ENDIAN_LITTLE,                          "device_endian_little",                          TYPE_UINT   },
    { CL_DEVICE_AVAILABLE,                              "device_available",                              TYPE_UINT   },
    { CL_DEVICE_COMPILER_AVAILABLE,                     "device_compiler_available",                     TYPE_UINT   },
    { CL_DEVICE_PRINTF_BUFFER_SIZE,                     "device_printf_buffer_size",                     TYPE_SIZE_T },
#   ifdef CL_VERSION_2_1
    { CL_DEVICE_IL_VERSION,                             "device_il_version",                             TYPE_STR    },
#   endif
#   ifdef CL_VERSION_2_0
    { CL_DEVICE_SVM_CAPABILITIES,                       "device_svm_capabilities",                       TYPE_ULONG  },
    { CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE,               "device_max_global_variable_size",               TYPE_SIZE_T },
    { CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE,   "device_global_variable_preferred_total_size",   TYPE_SIZE_T },
#   endif
#   ifdef CL_VERSION_2_0
    { CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES,             "device_queue_on_device_properties",             TYPE_ULONG  },
    { CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE,         "device_queue_on_device_preferred_size",         TYPE_UINT   },
    { CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE,               "device_queue_on_device_max_size",               TYPE_UINT   },
    { CL_DEVICE_MAX_ON_DEVICE_QUEUES,                   "device_max_on_device_queues",                   TYPE_UINT   },
    { CL_DEVICE_MAX_ON_DEVICE_EVENTS,                   "device_max_on_device_events",                   TYPE_UINT   },
    { CL_DEVICE_MAX_PIPE_ARGS,                          "device_max_pipe_args",                          TYPE_UINT   },
    { CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS,           "device_pipe_max_active_reservations",           TYPE_UINT   },
    { CL_DEVICE_PIPE_MAX_PACKET_SIZE,                   "device_pipe_max_packet_size",                   TYPE_UINT   },
#   endif
#   ifdef CL_VERSION_2_0
    { CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT,    "device_preferred_platform_atomic_alignment",    TYPE_UINT   },
    { CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT,      "device_preferred_global_atomic_alignment",      TYPE_UINT   },
    { CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT,       "device_preferred_local_atomic_alignment",       TYPE_UINT   },
#   endif
    { CL_DEVICE_EXTENSIONS,                             "device_extensions",                             TYPE_STR    },
};
#endif

int print_opencl_device_info(AVTextFormatContext *tfc, AVBufferRef *opencl_ref)
{
#if CONFIG_OPENCL
    AVHWDeviceContext *dev_ctx = NULL;
    AVOpenCLDeviceContext *hwctx = NULL;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_int cle;
    int ret = 0;

    if (!tfc || !opencl_ref)
        return AVERROR(EINVAL);

    dev_ctx = (AVHWDeviceContext*)opencl_ref->data;
    hwctx = dev_ctx->hwctx;
    device_id = hwctx->device_id;

    cle = clGetDeviceInfo(device_id, CL_DEVICE_PLATFORM,
                          sizeof(platform_id), &platform_id, NULL);
    if (cle != CL_SUCCESS)
        return AVERROR(ENOSYS);

    mark_section_show_entries(SECTION_ID_DEVICE_INFO_OPENCL, 1, NULL);
    avtext_print_section_header(tfc, NULL, SECTION_ID_DEVICE_INFO_OPENCL);

    /* Platform */
    for (unsigned i = 0; i < FF_ARRAY_ELEMS(opencl_platform_attrs); i++) {
        void *str = NULL;
        size_t size = 0;

        cle = clGetPlatformInfo(platform_id, opencl_platform_attrs[i].attr_val, 0, NULL, &size);
        if (cle != CL_SUCCESS)
            continue;

        str = av_mallocz(size);
        if (!str)
            continue;

        cle = clGetPlatformInfo(platform_id, opencl_platform_attrs[i].attr_val, size, str, &size);
        if (cle == CL_SUCCESS) {
            if (opencl_platform_attrs[i].attr_type == TYPE_STR) {
                if (strlen((char*)str) + 1 != size) continue;
                print_str(opencl_platform_attrs[i].attr_str, (char*)str);
            }
        }
        av_free(str);
    }

    /* Device */
    for (unsigned i = 0; i < FF_ARRAY_ELEMS(opencl_device_attrs); i++) {
        void *str = NULL;
        size_t size = 0;
        int64_t val = 0;

        cle = clGetDeviceInfo(device_id, opencl_device_attrs[i].attr_val, 0, NULL, &size);
        if (cle != CL_SUCCESS)
            continue;

        str = av_mallocz(size);
        if (!str)
            continue;

        cle = clGetDeviceInfo(device_id, opencl_device_attrs[i].attr_val, size, str, &size);
        if (cle == CL_SUCCESS) {
            switch (opencl_device_attrs[i].attr_type) {
            case TYPE_STR:
                if (strlen((char*)str) + 1 != size) continue;
                print_str(opencl_device_attrs[i].attr_str, (char*)str);
                break;
            case TYPE_UINT:
                val = (int64_t)(*(cl_uint*)str);
                print_int(opencl_device_attrs[i].attr_str, val);
                break;
            case TYPE_SIZE_T:
                val = (int64_t)(*(size_t*)str);
                print_int(opencl_device_attrs[i].attr_str, val);
                break;
            case TYPE_ULONG:
                val = (int64_t)(*(cl_ulong*)str);
                print_int(opencl_device_attrs[i].attr_str, val);
                break;
            default:
                break;
            }
        }
        av_free(str);
    }

    avtext_print_section_footer(tfc); // SECTION_ID_DEVICE_INFO_OPENCL

    return ret;
#else
    return 0;
#endif
}
