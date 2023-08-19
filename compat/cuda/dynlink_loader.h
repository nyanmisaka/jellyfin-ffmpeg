/*
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

#ifndef COMPAT_CUDA_DYNLINK_LOADER_H
#define COMPAT_CUDA_DYNLINK_LOADER_H

#include "libavutil/log.h"
#include "compat/w32dlfcn.h"

#define FFNV_LOAD_FUNC(path) dlopen((path), RTLD_LAZY)
#define FFNV_SYM_FUNC(lib, sym) dlsym((lib), (sym))
#define FFNV_FREE_FUNC(lib) dlclose(lib)
#define FFNV_LOG_FUNC(logctx, msg, ...) av_log(logctx, AV_LOG_ERROR, msg,  __VA_ARGS__)
#define FFNV_DEBUG_LOG_FUNC(logctx, msg, ...) av_log(logctx, AV_LOG_DEBUG, msg,  __VA_ARGS__)

#include <ffnvcodec/dynlink_loader.h>

#if defined(_WIN32) || defined(__CYGWIN__)
# define CUDA_LIBNAME "nvcuda.dll"
# define NVML_LIBNAME "nvml.dll"
# define NVML_LIBNAME2 "%ProgramW6432%\\NVIDIA Corporation\\NVSMI\\nvml.dll"
#else
# define CUDA_LIBNAME "libcuda.so.1"
# define NVML_LIBNAME "libnvidia-ml.so.1"
# define NVML_LIBNAME2 NULL
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
# define NVML_API_CALL __stdcall
#else
# define NVML_API_CALL
#endif

#define LOAD_LIBRARY2(l, path1, path2)                               \
    do {                                                             \
        char *path2_ = (char *)(path2);                              \
        if (!((l) = FFNV_LOAD_FUNC(path1))) {                        \
            FFNV_LOG_FUNC(logctx, "Cannot load %s\n", path1);        \
            if (!(path2_)) {                                         \
                ret = -1;                                            \
                goto error;                                          \
            }                                                        \
            if (!((l) = FFNV_LOAD_FUNC(path2_))) {                   \
                FFNV_LOG_FUNC(logctx, "Cannot load %s\n", path2_);   \
                ret = -1;                                            \
                goto error;                                          \
            }                                                        \
            FFNV_DEBUG_LOG_FUNC(logctx, "Loaded lib: %s\n", path2_); \
        }                                                            \
        FFNV_DEBUG_LOG_FUNC(logctx, "Loaded lib: %s\n", path1);      \
    } while (0)

#define LOAD_SYMBOL(fun, tp, symbol)                             \
    do {                                                         \
        if (!((f->fun) = (tp*)FFNV_SYM_FUNC(f->lib, symbol))) {  \
            FFNV_LOG_FUNC(logctx, "Cannot load %s\n", symbol);   \
            ret = -1;                                            \
            goto error;                                          \
        }                                                        \
        FFNV_DEBUG_LOG_FUNC(logctx, "Loaded sym: %s\n", symbol); \
    } while (0)

#define LOAD_SYMBOL_OPT(fun, tp, symbol)                                      \
    do {                                                                      \
        if (!((f->fun) = (tp*)FFNV_SYM_FUNC(f->lib, symbol))) {               \
            FFNV_DEBUG_LOG_FUNC(logctx, "Cannot load optional %s\n", symbol); \
        } else {                                                              \
            FFNV_DEBUG_LOG_FUNC(logctx, "Loaded sym: %s\n", symbol);          \
        }                                                                     \
    } while (0)

#define GENERIC_LOAD_FUNC_PREAMBLE(T, n, N, NN) \
    T *f;                                       \
    int ret;                                    \
                                                \
    n##_free_functions(functions);              \
                                                \
    f = *functions = (T*)calloc(1, sizeof(*f)); \
    if (!f)                                     \
        return -1;                              \
                                                \
    LOAD_LIBRARY2(f->lib, N, NN);

#define GENERIC_LOAD_FUNC_FINALE(n) \
    return 0;                       \
error:                              \
    n##_free_functions(functions);  \
    return ret;

#define GENERIC_FREE_FUNC()                \
    if (!functions)                        \
        return;                            \
    if (*functions && (*functions)->lib)   \
        FFNV_FREE_FUNC((*functions)->lib); \
    free(*functions);                      \
    *functions = NULL;

#ifdef FFNV_DYNLINK_CUDA_H
typedef CUresult CUDAAPI _tcuDriverGetVersion(int *driverVersion);
typedef CUresult CUDAAPI _tcuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev);
typedef CUresult CUDAAPI _tcuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev);
typedef CUresult CUDAAPI _tcuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId);
typedef CUresult CUDAAPI _tcuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev);

typedef struct CudaFunctionsExt {
    _tcuDriverGetVersion *cuDriverGetVersion;
    _tcuDeviceGetUuid_v2 *cuDeviceGetUuid_v2;
    _tcuDeviceGetLuid *cuDeviceGetLuid;
    _tcuDeviceGetByPCIBusId *cuDeviceGetByPCIBusId;
    _tcuDeviceGetPCIBusId *cuDeviceGetPCIBusId;

    FFNV_LIB_HANDLE lib;
} CudaFunctionsExt;

static inline void cuda_ext_free_functions(CudaFunctionsExt **functions)
{
    GENERIC_FREE_FUNC();
}

static inline int cuda_ext_load_functions(CudaFunctionsExt **functions, void *logctx)
{
    GENERIC_LOAD_FUNC_PREAMBLE(CudaFunctionsExt, cuda_ext, CUDA_LIBNAME, NULL);

    LOAD_SYMBOL(cuDriverGetVersion, _tcuDriverGetVersion, "cuDriverGetVersion");
    LOAD_SYMBOL_OPT(cuDeviceGetUuid_v2, _tcuDeviceGetUuid_v2, "cuDeviceGetUuid_v2");
    LOAD_SYMBOL_OPT(cuDeviceGetLuid, _tcuDeviceGetLuid, "cuDeviceGetLuid");
    LOAD_SYMBOL_OPT(cuDeviceGetByPCIBusId, _tcuDeviceGetByPCIBusId, "cuDeviceGetByPCIBusId");
    LOAD_SYMBOL_OPT(cuDeviceGetPCIBusId, _tcuDeviceGetPCIBusId, "cuDeviceGetPCIBusId");

    GENERIC_LOAD_FUNC_FINALE(cuda_ext);
}

#if !defined(NVML_API_VERSION) || defined(NVML_API_VERSION) && (NVML_API_VERSION < 11)
typedef enum nvmlReturn_enum {
    NVML_SUCCESS = 0
} nvmlReturn_t;

typedef struct nvmlDevice_st* nvmlDevice_t;
typedef struct nvmlPciInfo_st
{
    char busIdLegacy[16];
    unsigned int domain;
    unsigned int bus;
    unsigned int device;
    unsigned int pciDeviceId;
    unsigned int pciSubSystemId;
    char busId[32];
} nvmlPciInfo_t;

typedef unsigned int nvmlDeviceArchitecture_t;

#define NVML_INIT_FLAG_NO_GPUS                              1  //!< Don't fail nvmlInit() when no GPUs are found
#define NVML_INIT_FLAG_NO_ATTACH                            2  //!< Don't attach GPUs
#define NVML_DEVICE_ARCH_KEPLER                             2  //!< Devices based on the NVIDIA Kepler architecture
#define NVML_DEVICE_ARCH_MAXWELL                            3  //!< Devices based on the NVIDIA Maxwell architecture
#define NVML_DEVICE_ARCH_PASCAL                             4  //!< Devices based on the NVIDIA Pascal architecture
#define NVML_DEVICE_ARCH_VOLTA                              5  //!< Devices based on the NVIDIA Volta architecture
#define NVML_DEVICE_ARCH_TURING                             6  //!< Devices based on the NVIDIA Turing architecture
#define NVML_DEVICE_ARCH_AMPERE                             7  //!< Devices based on the NVIDIA Ampere architecture
#define NVML_DEVICE_ARCH_ADA                                8  //!< Devices based on the NVIDIA Ada architecture
#define NVML_DEVICE_ARCH_HOPPER                             9  //!< Devices based on the NVIDIA Hopper architecture
#define NVML_DEVICE_ARCH_UNKNOWN                   0xffffffff  //!< Anything else, presumably something newer
#define NVML_DEVICE_PCI_BUS_ID_LEGACY_FMT  "%04X:%02X:%02X.0"  //!< PCI format string for ::busIdLegacy
#define NVML_DEVICE_PCI_BUS_ID_FMT         "%08X:%02X:%02X.0"  //!< PCI format string for ::busId
#define NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE             80  //!< Buffer size guaranteed to be large enough for nvmlSystemGetDriverVersion
#define NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE               80  //!< Buffer size guaranteed to be large enough for nvmlSystemGetNVMLVersion
#endif

typedef nvmlReturn_t NVML_API_CALL _tnvmlInit(void);
typedef nvmlReturn_t NVML_API_CALL _tnvmlInitWithFlags(unsigned int flags);
typedef nvmlReturn_t NVML_API_CALL _tnvmlShutdown(void);
typedef nvmlReturn_t NVML_API_CALL _tnvmlSystemGetCudaDriverVersion(int* cudaDriverVersion);
typedef nvmlReturn_t NVML_API_CALL _tnvmlSystemGetDriverVersion(char* version, unsigned int length);
typedef nvmlReturn_t NVML_API_CALL _tnvmlSystemGetNVMLVersion(char* version, unsigned int length);
typedef nvmlReturn_t NVML_API_CALL _tnvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device);
typedef nvmlReturn_t NVML_API_CALL _tnvmlDeviceGetHandleByUUID(const char* uuid, nvmlDevice_t* device);
typedef nvmlReturn_t NVML_API_CALL _tnvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device);
typedef nvmlReturn_t NVML_API_CALL _tnvmlDeviceGetCount(unsigned int* deviceCount);
typedef nvmlReturn_t NVML_API_CALL _tnvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length);
typedef nvmlReturn_t NVML_API_CALL _tnvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index);
typedef nvmlReturn_t NVML_API_CALL _tnvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length);
typedef nvmlReturn_t NVML_API_CALL _tnvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t* pci);
typedef nvmlReturn_t NVML_API_CALL _tnvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t* arch);
typedef nvmlReturn_t NVML_API_CALL _tnvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int* numCores);
typedef nvmlReturn_t NVML_API_CALL _tnvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber);
typedef const char*  NVML_API_CALL _tnvmlErrorString(nvmlReturn_t result);

typedef struct NvmlFunctionsExt {
    _tnvmlInit *nvmlInit;
    _tnvmlInitWithFlags *nvmlInitWithFlags;
    _tnvmlShutdown *nvmlShutdown;
    _tnvmlSystemGetCudaDriverVersion *nvmlSystemGetCudaDriverVersion;
    _tnvmlSystemGetDriverVersion *nvmlSystemGetDriverVersion;
    _tnvmlSystemGetNVMLVersion *nvmlSystemGetNVMLVersion;
    _tnvmlDeviceGetHandleByIndex *nvmlDeviceGetHandleByIndex;
    _tnvmlDeviceGetHandleByUUID *nvmlDeviceGetHandleByUUID;
    _tnvmlDeviceGetHandleByPciBusId *nvmlDeviceGetHandleByPciBusId;
    _tnvmlDeviceGetCount *nvmlDeviceGetCount;
    _tnvmlDeviceGetName *nvmlDeviceGetName;
    _tnvmlDeviceGetIndex *nvmlDeviceGetIndex;
    _tnvmlDeviceGetUUID *nvmlDeviceGetUUID;
    _tnvmlDeviceGetPciInfo *nvmlDeviceGetPciInfo;
    _tnvmlDeviceGetArchitecture *nvmlDeviceGetArchitecture;
    _tnvmlDeviceGetNumGpuCores *nvmlDeviceGetNumGpuCores;
    _tnvmlDeviceGetMinorNumber *nvmlDeviceGetMinorNumber;
    _tnvmlErrorString *nvmlErrorString;

    FFNV_LIB_HANDLE lib;
} NvmlFunctionsExt;

static inline void nvml_ext_free_functions(NvmlFunctionsExt **functions)
{
    GENERIC_FREE_FUNC();
}

static inline int nvml_ext_load_functions(NvmlFunctionsExt **functions, void *logctx)
{
#if defined(_WIN32) || defined(__CYGWIN__)
    char nvml_libname_2[512];
    DWORD len = ExpandEnvironmentStringsA(NVML_LIBNAME2, nvml_libname_2, sizeof(nvml_libname_2));

    GENERIC_LOAD_FUNC_PREAMBLE(NvmlFunctionsExt, nvml_ext, NVML_LIBNAME, (len ? nvml_libname_2 : NULL));
#else
    GENERIC_LOAD_FUNC_PREAMBLE(NvmlFunctionsExt, nvml_ext, NVML_LIBNAME, NVML_LIBNAME2);
#endif
    LOAD_SYMBOL(nvmlInit, _tnvmlInit, "nvmlInit");
    LOAD_SYMBOL(nvmlInitWithFlags, _tnvmlInitWithFlags, "nvmlInitWithFlags");
    LOAD_SYMBOL(nvmlShutdown, _tnvmlShutdown, "nvmlShutdown");
    LOAD_SYMBOL(nvmlSystemGetCudaDriverVersion, _tnvmlSystemGetCudaDriverVersion, "nvmlSystemGetCudaDriverVersion");
    LOAD_SYMBOL(nvmlSystemGetDriverVersion, _tnvmlSystemGetDriverVersion, "nvmlSystemGetDriverVersion");
    LOAD_SYMBOL(nvmlSystemGetNVMLVersion, _tnvmlSystemGetNVMLVersion, "nvmlSystemGetNVMLVersion");
    LOAD_SYMBOL(nvmlDeviceGetHandleByIndex, _tnvmlDeviceGetHandleByIndex, "nvmlDeviceGetHandleByIndex");
    LOAD_SYMBOL(nvmlDeviceGetHandleByUUID, _tnvmlDeviceGetHandleByUUID, "nvmlDeviceGetHandleByUUID");
    LOAD_SYMBOL(nvmlDeviceGetHandleByPciBusId, _tnvmlDeviceGetHandleByPciBusId, "nvmlDeviceGetHandleByPciBusId");
    LOAD_SYMBOL(nvmlDeviceGetCount, _tnvmlDeviceGetCount, "nvmlDeviceGetCount");
    LOAD_SYMBOL(nvmlDeviceGetName, _tnvmlDeviceGetName, "nvmlDeviceGetName");
    LOAD_SYMBOL(nvmlDeviceGetIndex, _tnvmlDeviceGetIndex, "nvmlDeviceGetIndex");
    LOAD_SYMBOL(nvmlDeviceGetUUID, _tnvmlDeviceGetUUID, "nvmlDeviceGetUUID");
    LOAD_SYMBOL(nvmlDeviceGetPciInfo, _tnvmlDeviceGetPciInfo, "nvmlDeviceGetPciInfo");
    LOAD_SYMBOL(nvmlDeviceGetArchitecture, _tnvmlDeviceGetArchitecture, "nvmlDeviceGetArchitecture");
    LOAD_SYMBOL(nvmlDeviceGetNumGpuCores, _tnvmlDeviceGetNumGpuCores, "nvmlDeviceGetNumGpuCores");
    LOAD_SYMBOL_OPT(nvmlDeviceGetMinorNumber, _tnvmlDeviceGetMinorNumber, "nvmlDeviceGetMinorNumber");
    LOAD_SYMBOL(nvmlErrorString, _tnvmlErrorString, "nvmlErrorString");

    GENERIC_LOAD_FUNC_FINALE(nvml_ext);
}
#endif

#undef GENERIC_LOAD_FUNC_PREAMBLE
#undef LOAD_LIBRARY2
#undef LOAD_SYMBOL
#undef LOAD_SYMBOL_OPT
#undef GENERIC_LOAD_FUNC_FINALE
#undef GENERIC_FREE_FUNC
#undef CUDA_LIBNAME
#undef NVML_LIBNAME
#undef NVML_LIBNAME2

#endif /* COMPAT_CUDA_DYNLINK_LOADER_H */
