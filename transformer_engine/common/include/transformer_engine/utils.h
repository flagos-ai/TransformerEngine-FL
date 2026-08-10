/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file utils.h
<<<<<<< HEAD
 *  \brief Utility functions (e.g. host-to-device pointer copies).
=======
 *  \brief Utility functions (e.g. host-to-device value stores).
>>>>>>> dev
 */

#ifndef TRANSFORMER_ENGINE_UTILS_H_
#define TRANSFORMER_ENGINE_UTILS_H_

#include <cuda_runtime.h>
<<<<<<< HEAD
=======
#include <stddef.h>
>>>>>>> dev
#include <stdint.h>
#include <transformer_engine/transformer_engine.h>

#ifdef __cplusplus
extern "C" {
#endif

<<<<<<< HEAD
/*! \brief Copy an array of device pointers (held on host) into a device tensor.
 *
 *  \param[in]     host_ptrs    Host array of device pointer values cast to uint64_t.
 *  \param[out]    output       NVTETensor whose rowwise data buffer receives the pointer values.
 *  \param[in]     count        Number of pointers.
 *  \param[in]     stream       CUDA stream used for the operation.
=======
/*! \brief Copy a small host buffer into device memory via kernel arguments.
 *
 *  The host buffer may be modified or freed after this call returns.
 *  This is compatible with CUDA Graphs.
 *
 *  \param[in]     host_ptr     Source in host memory.
 *  \param[out]    device_ptr   Destination in device memory.
 *  \param[in]     num_bytes    Size of the value in bytes.
 *  \param[in]     stream       CUDA stream for the operation.
 */
void nvte_copy_host_to_device_via_kernel(const void *host_ptr, void *device_ptr, size_t num_bytes,
                                         cudaStream_t stream);

/*! \deprecated Use nvte_copy_host_to_device_via_kernel instead.
 *
 *  \brief Copy an array of device pointers (held on host) into a device tensor.
>>>>>>> dev
 */
void nvte_convert_pointers_to_tensor(const uint64_t *host_ptrs, NVTETensor output, int64_t count,
                                     cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_UTILS_H_
