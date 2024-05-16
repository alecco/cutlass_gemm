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

//
// Benchmarking helper class for cuBLASLt
//

#pragma once

#include <vector>
#include <bench.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <util/cuda_check.h>
#include <util/cublas_check.h>

namespace gemm {

template <typename InType, typename OutType = InType, typename ComputeType = OutType>
class BenchCublas : public Bench {
public:
    BenchCublas(int m, int n, int k,
                ComputeType alpha    = ComputeType{0.0f},
                ComputeType beta     = ComputeType{0.0f},
                size_t workspaceSize = 1024 * 1024 * 4,
                ComputeType Ascale   = ComputeType{2.0f},
                ComputeType Bscale   = ComputeType{0.5f},
                ComputeType Cscale   = ComputeType{1.0f},
                ComputeType Dscale   = ComputeType{1.0f}) :
            m(m), n(n), k(k), alpha(alpha), beta(beta),
            workspaceSize(workspaceSize),
            Ahost(m * k), Bhost(n * k), Chost(m * n), biasHost(m),
            AscaleHost(Ascale), BscaleHost(Bscale), CscaleHost(Cscale), DscaleHost(Dscale) {

        cublas_check(cublasLtCreate(&ltHandle));
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&Adev), m * k * sizeof(InType)));
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&Bdev), n * k * sizeof(InType)));
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&Cdev), m * n * sizeof(OutType)));
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&biasDev),  m * sizeof(OutType)));
        cuda_check(cudaMalloc(&workspace, workspaceSize));
        cuda_check(cudaStreamCreate(&stream));

        perTensorScalingEnabled = std::is_same<InType, __nv_fp8_e4m3>::value || std::is_same<InType, __nv_fp8_e5m2>::value;

        if (perTensorScalingEnabled) {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&AscaleDev), sizeof(*AscaleDev)));
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&BscaleDev), sizeof(*BscaleDev)));
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&CscaleDev), sizeof(*CscaleDev)));
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&DscaleDev), sizeof(*DscaleDev)));
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&DamaxDev),  sizeof(*DamaxDev)));
        }

        fillData();
    }

    virtual ~BenchCublas() {
        cublas_check(cublasLtDestroy(ltHandle));
        cuda_check(cudaFree(Adev));
        cuda_check(cudaFree(Bdev));
        cuda_check(cudaFree(Cdev));
        cuda_check(cudaFree(biasDev));
        cuda_check(cudaFree(workspace));
        if (perTensorScalingEnabled) {
            cuda_check(cudaFree(AscaleDev));
            cuda_check(cudaFree(BscaleDev));
            cuda_check(cudaFree(CscaleDev));
            cuda_check(cudaFree(DscaleDev));
            cuda_check(cudaFree(DamaxDev));
        }
        cuda_check(cudaStreamDestroy(stream));
    }

    // XXX specialize by MType
    void run() override;

private:
    void fillData() {
        for (int i = 0; i < m * k; i++) Ahost[i]    = InType(i);
        for (int i = 0; i < n * k; i++) Bhost[i]    = InType(i);
        for (int i = 0; i < m; i++)     biasHost[i] = InType(i + 1);
    }

    void copyDataToDevice() {
        cuda_check(cudaMemcpyAsync(Adev, Ahost.data(), Ahost.size() * sizeof(Ahost[0]), cudaMemcpyHostToDevice, stream));
        cuda_check(cudaMemcpyAsync(Bdev, Bhost.data(), Bhost.size() * sizeof(Bhost[0]), cudaMemcpyHostToDevice, stream));
        cuda_check(cudaMemcpyAsync(biasDev, biasHost.data(), biasHost.size() * sizeof(biasHost[0]), cudaMemcpyHostToDevice));
        if (perTensorScalingEnabled) {
            cuda_check(cudaMemcpyAsync(AscaleDev, &AscaleHost, sizeof(AscaleHost), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpyAsync(BscaleDev, &BscaleHost, sizeof(BscaleHost), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpyAsync(CscaleDev, &CscaleHost, sizeof(CscaleHost), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpyAsync(DscaleDev, &DscaleHost, sizeof(DscaleHost), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpyAsync(DamaxDev,  &DamaxHost,  sizeof(DamaxHost),  cudaMemcpyHostToDevice));
        }
    }

    void copyDataFromDevice() {
        cuda_check(cudaMemcpyAsync(Chost.data(), Cdev, Chost.size() * sizeof(Chost[0]), cudaMemcpyDeviceToHost, stream));
    }

    void streamSynchronize() {
        cuda_check(cudaStreamSynchronize(stream));
    }

    bool perTensorScalingEnabled;
    int m, n, k;
    ComputeType alpha, beta;
    size_t workspaceSize;
    std::vector<InType>  Ahost, Bhost;
    std::vector<OutType> Chost, biasHost;
    void *workspace;
    InType  *Adev, *Bdev;
    OutType *Cdev, *biasDev;
    cudaStream_t stream;
    cublasLtHandle_t ltHandle;
    ComputeType AscaleHost, BscaleHost, CscaleHost, DscaleHost, DamaxHost;
    ComputeType *AscaleDev, *BscaleDev, *CscaleDev, *DscaleDev, *DamaxDev;
};

template <>
inline void BenchCublas<__half, __half, float>::fillData() {
    for (int i = 0; i < m * k; i++) Ahost[i]    = __float2half_rn(i);
    for (int i = 0; i < n * k; i++) Bhost[i]    = __float2half_rn(i);
    for (int i = 0; i < m;     i++) biasHost[i] = __float2half_rn(i + 1);
}

template <>
inline void BenchCublas<__half, __half, cuComplex>::fillData() {
    for (int i = 0; i < m * k; i++) Ahost[i]    = __float2half_rn(i/100.);
    for (int i = 0; i < n * k; i++) Bhost[i]    = __float2half_rn(i/100.);
    for (int i = 0; i < m;     i++) biasHost[i] = __float2half_rn(i + 1);
}

template <>
inline void
BenchCublas<float, float, float>::run() {
#if 0
    copyDataToDevice();   // XXX not in run

    // XXX here pick proper function
    LtFp8Matmul(ltHandle, m, n, k,
                &alpha, AscaleDev, Adev, k, BscaleDev, Bdev, k,
                CscaleDev, Cdev, m, DscaleDev, DamaxDev,
                workspace, workspaceSize);

    // copyDataFromDevice();
#endif
    streamSynchronize();
}

} // namespace gemm
