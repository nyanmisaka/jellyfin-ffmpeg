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
#   define CUDA_LIBNAME "nvcuda.dll"
#   define NVML_LIBNAME "nvml.dll"
#   define NVML_LIBNAME2 "%ProgramW6432%\\NVIDIA Corporation\\NVSMI\\nvml.dll"
#else
#   define CUDA_LIBNAME "libcuda.so.1"
#   define NVML_LIBNAME "libnvidia-ml.so.1"
#   define NVML_LIBNAME2 NULL
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#   define NVML_API_CALL __stdcall
#else
#   define NVML_API_CALL
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
#   ifndef CU_UUID_HAS_BEEN_DEFINED
typedef struct CUuuid_st {
    char bytes[16];
} CUuuid;
#   endif

typedef CUresult CUDAAPI ff_tcuInit(unsigned int Flags);
typedef CUresult CUDAAPI ff_tcuDriverGetVersion(int *driverVersion);
typedef CUresult CUDAAPI ff_tcuDeviceGetCount(int *count);
typedef CUresult CUDAAPI ff_tcuDeviceTotalMem(size_t *bytes, CUdevice dev);
typedef CUresult CUDAAPI ff_tcuGetErrorName(CUresult error, const char** pstr);
typedef CUresult CUDAAPI ff_tcuGetErrorString(CUresult error, const char** pstr);
typedef CUresult CUDAAPI ff_tcuDeviceGetUuid(CUuuid *uuid, CUdevice dev);
typedef CUresult CUDAAPI ff_tcuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev);
typedef CUresult CUDAAPI ff_tcuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev);
typedef CUresult CUDAAPI ff_tcuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId);
typedef CUresult CUDAAPI ff_tcuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev);

typedef struct CudaFunctionsExt {
    ff_tcuInit *cuInit;
    ff_tcuDriverGetVersion *cuDriverGetVersion;
    ff_tcuDeviceGetCount *cuDeviceGetCount;
    ff_tcuDeviceTotalMem *cuDeviceTotalMem;
    ff_tcuGetErrorName *cuGetErrorName;
    ff_tcuGetErrorString *cuGetErrorString;
    ff_tcuDeviceGetUuid *cuDeviceGetUuid;
    ff_tcuDeviceGetUuid_v2 *cuDeviceGetUuid_v2;
    ff_tcuDeviceGetLuid *cuDeviceGetLuid;
    ff_tcuDeviceGetByPCIBusId *cuDeviceGetByPCIBusId;
    ff_tcuDeviceGetPCIBusId *cuDeviceGetPCIBusId;

    FFNV_LIB_HANDLE lib;
} CudaFunctionsExt;

static inline void cuda_ext_free_functions(CudaFunctionsExt **functions)
{
    GENERIC_FREE_FUNC();
}

static inline int cuda_ext_load_functions(CudaFunctionsExt **functions, void *logctx)
{
    GENERIC_LOAD_FUNC_PREAMBLE(CudaFunctionsExt, cuda_ext, CUDA_LIBNAME, NULL);

    LOAD_SYMBOL(cuInit, ff_tcuInit, "cuInit");
    LOAD_SYMBOL(cuDriverGetVersion, ff_tcuDriverGetVersion, "cuDriverGetVersion");
    LOAD_SYMBOL(cuDeviceGetCount, ff_tcuDeviceGetCount, "cuDeviceGetCount");
    LOAD_SYMBOL(cuDeviceTotalMem, ff_tcuDeviceTotalMem, "cuDeviceTotalMem_v2");
    LOAD_SYMBOL(cuGetErrorName, ff_tcuGetErrorName, "cuGetErrorName");
    LOAD_SYMBOL(cuGetErrorString, ff_tcuGetErrorString, "cuGetErrorString");
    LOAD_SYMBOL_OPT(cuDeviceGetUuid, ff_tcuDeviceGetUuid, "cuDeviceGetUuid");
    LOAD_SYMBOL_OPT(cuDeviceGetUuid_v2, ff_tcuDeviceGetUuid_v2, "cuDeviceGetUuid_v2");
    LOAD_SYMBOL_OPT(cuDeviceGetLuid, ff_tcuDeviceGetLuid, "cuDeviceGetLuid");
    LOAD_SYMBOL_OPT(cuDeviceGetByPCIBusId, ff_tcuDeviceGetByPCIBusId, "cuDeviceGetByPCIBusId");
    LOAD_SYMBOL_OPT(cuDeviceGetPCIBusId, ff_tcuDeviceGetPCIBusId, "cuDeviceGetPCIBusId");

    GENERIC_LOAD_FUNC_FINALE(cuda_ext);
}

#if !defined(NVML_API_VERSION) || defined(NVML_API_VERSION) && (NVML_API_VERSION < 11)
#   ifndef NVML_API_VERSION
typedef enum nvmlReturn_enum {
    NVML_SUCCESS = 0
} nvmlReturn_t;

typedef struct nvmlDevice_st* nvmlDevice_t;

typedef struct nvmlUtilization_st {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

typedef struct nvmlMemory_st
{
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

typedef unsigned int nvmlPcieUtilCounter_t;
#   endif

typedef unsigned int nvmlDeviceArchitecture_t;

#   define NVML_INIT_FLAG_NO_GPUS                              1  //!< Don't fail nvmlInit() when no GPUs are found
#   define NVML_INIT_FLAG_NO_ATTACH                            2  //!< Don't attach GPUs
#   define NVML_DEVICE_ARCH_KEPLER                             2  //!< Devices based on the NVIDIA Kepler architecture
#   define NVML_DEVICE_ARCH_MAXWELL                            3  //!< Devices based on the NVIDIA Maxwell architecture
#   define NVML_DEVICE_ARCH_PASCAL                             4  //!< Devices based on the NVIDIA Pascal architecture
#   define NVML_DEVICE_ARCH_VOLTA                              5  //!< Devices based on the NVIDIA Volta architecture
#   define NVML_DEVICE_ARCH_TURING                             6  //!< Devices based on the NVIDIA Turing architecture
#   define NVML_DEVICE_ARCH_AMPERE                             7  //!< Devices based on the NVIDIA Ampere architecture
#   define NVML_DEVICE_ARCH_ADA                                8  //!< Devices based on the NVIDIA Ada architecture
#   define NVML_DEVICE_ARCH_HOPPER                             9  //!< Devices based on the NVIDIA Hopper architecture
#   define NVML_DEVICE_ARCH_BLACKWELL                         10  //!< Devices based on the NVIDIA Blackwell architecture
#   define NVML_DEVICE_ARCH_UNKNOWN                   0xffffffff  //!< Anything else, presumably something newer
#   define NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE             80  //!< Buffer size guaranteed to be large enough for nvmlSystemGetDriverVersion
#   define NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE               80  //!< Buffer size guaranteed to be large enough for nvmlSystemGetNVMLVersion
#   endif

typedef nvmlReturn_t NVML_API_CALL ff_tnvmlInit(void);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlInitWithFlags(unsigned int flags);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlShutdown(void);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlSystemGetCudaDriverVersion(int* cudaDriverVersion);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlSystemGetDriverVersion(char* version, unsigned int length);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlSystemGetNVMLVersion(char* version, unsigned int length);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetHandleByUUID(const char* uuid, nvmlDevice_t* device);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetCount(unsigned int* deviceCount);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t* arch);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int* numCores);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency);
typedef nvmlReturn_t NVML_API_CALL ff_tnvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int* value);
typedef const char*  NVML_API_CALL ff_tnvmlErrorString(nvmlReturn_t result);

typedef struct NvmlFunctionsExt {
    ff_tnvmlInit *nvmlInit;
    ff_tnvmlInitWithFlags *nvmlInitWithFlags;
    ff_tnvmlShutdown *nvmlShutdown;
    ff_tnvmlSystemGetCudaDriverVersion *nvmlSystemGetCudaDriverVersion;
    ff_tnvmlSystemGetDriverVersion *nvmlSystemGetDriverVersion;
    ff_tnvmlSystemGetNVMLVersion *nvmlSystemGetNVMLVersion;
    ff_tnvmlDeviceGetHandleByIndex *nvmlDeviceGetHandleByIndex;
    ff_tnvmlDeviceGetHandleByUUID *nvmlDeviceGetHandleByUUID;
    ff_tnvmlDeviceGetHandleByPciBusId *nvmlDeviceGetHandleByPciBusId;
    ff_tnvmlDeviceGetHandleByPciBusId *nvmlDeviceGetHandleByPciBusId_v2;
    ff_tnvmlDeviceGetCount *nvmlDeviceGetCount;
    ff_tnvmlDeviceGetName *nvmlDeviceGetName;
    ff_tnvmlDeviceGetIndex *nvmlDeviceGetIndex;
    ff_tnvmlDeviceGetUUID *nvmlDeviceGetUUID;
    ff_tnvmlDeviceGetArchitecture *nvmlDeviceGetArchitecture;
    ff_tnvmlDeviceGetNumGpuCores *nvmlDeviceGetNumGpuCores;
    ff_tnvmlDeviceGetMinorNumber *nvmlDeviceGetMinorNumber;
    ff_tnvmlDeviceGetMemoryInfo *nvmlDeviceGetMemoryInfo;
    ff_tnvmlDeviceGetUtilizationRates *nvmlDeviceGetUtilizationRates;
    ff_tnvmlDeviceGetDecoderUtilization *nvmlDeviceGetDecoderUtilization;
    ff_tnvmlDeviceGetEncoderUtilization *nvmlDeviceGetEncoderUtilization;
    ff_tnvmlDeviceGetEncoderStats *nvmlDeviceGetEncoderStats;
    ff_tnvmlDeviceGetPcieThroughput *nvmlDeviceGetPcieThroughput;
    ff_tnvmlErrorString *nvmlErrorString;

    FFNV_LIB_HANDLE lib;
} NvmlFunctionsExt;

static inline void nvml_ext_free_functions(NvmlFunctionsExt **functions)
{
    GENERIC_FREE_FUNC();
}

static inline int nvml_ext_load_functions(NvmlFunctionsExt **functions, void *logctx)
{
#   if defined(_WIN32) || defined(__CYGWIN__)
    char nvml_libname_2[512];
    DWORD len = ExpandEnvironmentStringsA(NVML_LIBNAME2, nvml_libname_2, sizeof(nvml_libname_2));

    GENERIC_LOAD_FUNC_PREAMBLE(NvmlFunctionsExt, nvml_ext, NVML_LIBNAME, (len ? nvml_libname_2 : NULL));
#   else
    GENERIC_LOAD_FUNC_PREAMBLE(NvmlFunctionsExt, nvml_ext, NVML_LIBNAME, NVML_LIBNAME2);
#   endif
    LOAD_SYMBOL(nvmlInit, ff_tnvmlInit, "nvmlInit_v2");
    LOAD_SYMBOL_OPT(nvmlInitWithFlags, ff_tnvmlInitWithFlags, "nvmlInitWithFlags");
    LOAD_SYMBOL(nvmlShutdown, ff_tnvmlShutdown, "nvmlShutdown");
    LOAD_SYMBOL_OPT(nvmlSystemGetCudaDriverVersion, ff_tnvmlSystemGetCudaDriverVersion, "nvmlSystemGetCudaDriverVersion");
    LOAD_SYMBOL(nvmlSystemGetDriverVersion, ff_tnvmlSystemGetDriverVersion, "nvmlSystemGetDriverVersion");
    LOAD_SYMBOL(nvmlSystemGetNVMLVersion, ff_tnvmlSystemGetNVMLVersion, "nvmlSystemGetNVMLVersion");
    LOAD_SYMBOL(nvmlDeviceGetHandleByIndex, ff_tnvmlDeviceGetHandleByIndex, "nvmlDeviceGetHandleByIndex");
    LOAD_SYMBOL(nvmlDeviceGetHandleByUUID, ff_tnvmlDeviceGetHandleByUUID, "nvmlDeviceGetHandleByUUID");
    LOAD_SYMBOL(nvmlDeviceGetHandleByPciBusId, ff_tnvmlDeviceGetHandleByPciBusId, "nvmlDeviceGetHandleByPciBusId");
    LOAD_SYMBOL_OPT(nvmlDeviceGetHandleByPciBusId_v2, ff_tnvmlDeviceGetHandleByPciBusId, "nvmlDeviceGetHandleByPciBusId_v2");
    LOAD_SYMBOL(nvmlDeviceGetCount, ff_tnvmlDeviceGetCount, "nvmlDeviceGetCount");
    LOAD_SYMBOL(nvmlDeviceGetName, ff_tnvmlDeviceGetName, "nvmlDeviceGetName");
    LOAD_SYMBOL(nvmlDeviceGetIndex, ff_tnvmlDeviceGetIndex, "nvmlDeviceGetIndex");
    LOAD_SYMBOL(nvmlDeviceGetUUID, ff_tnvmlDeviceGetUUID, "nvmlDeviceGetUUID");
    LOAD_SYMBOL_OPT(nvmlDeviceGetArchitecture, ff_tnvmlDeviceGetArchitecture, "nvmlDeviceGetArchitecture");
    LOAD_SYMBOL_OPT(nvmlDeviceGetNumGpuCores, ff_tnvmlDeviceGetNumGpuCores, "nvmlDeviceGetNumGpuCores");
    LOAD_SYMBOL_OPT(nvmlDeviceGetMinorNumber, ff_tnvmlDeviceGetMinorNumber, "nvmlDeviceGetMinorNumber");
    LOAD_SYMBOL_OPT(nvmlDeviceGetMemoryInfo, ff_tnvmlDeviceGetMemoryInfo, "nvmlDeviceGetMemoryInfo");
    LOAD_SYMBOL_OPT(nvmlDeviceGetUtilizationRates, ff_tnvmlDeviceGetUtilizationRates, "nvmlDeviceGetUtilizationRates");
    LOAD_SYMBOL_OPT(nvmlDeviceGetDecoderUtilization, ff_tnvmlDeviceGetDecoderUtilization, "nvmlDeviceGetDecoderUtilization");
    LOAD_SYMBOL_OPT(nvmlDeviceGetEncoderUtilization, ff_tnvmlDeviceGetEncoderUtilization, "nvmlDeviceGetEncoderUtilization");
    LOAD_SYMBOL_OPT(nvmlDeviceGetEncoderStats, ff_tnvmlDeviceGetEncoderStats, "nvmlDeviceGetEncoderStats");
    LOAD_SYMBOL_OPT(nvmlDeviceGetPcieThroughput, ff_tnvmlDeviceGetPcieThroughput, "nvmlDeviceGetPcieThroughput");
    LOAD_SYMBOL(nvmlErrorString, ff_tnvmlErrorString, "nvmlErrorString");

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
