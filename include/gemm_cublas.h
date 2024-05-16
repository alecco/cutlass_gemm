// cutlass GEMM
// Copyright (C) 2024  Ologan Ltd
// SPDX-License-Identifier: AGPL-3.0
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

// cuBLASS GEMM (NT)

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <cublas_check.h>
#include <cutlass/util/cublas_wrappers.hpp>


// cuBLAS GEMM NT with column-major layout
template<typename MType, typename Alpha, typename Beta>
inline cublasStatus_t
cublas_gemm_nt(cublasHandle_t handle,
               thrust::device_vector<MType>& a, 
               thrust::device_vector<MType>& b, 
               thrust::device_vector<MType>& c, 
               Alpha alpha, Beta beta) {
    
    return blam::cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                              m, n, k,
                              &alpha,
                              d_A.data().get(), m,
                              d_B.data().get(), n,
                              &beta,
                              d_C.data().get(), m);
}
