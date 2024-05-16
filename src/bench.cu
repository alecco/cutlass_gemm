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
// Benchmark GEMM implementations using CUTLASS vs. cuBLAS (if available)
//

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <format>
#include <getopt.h>
#include <limits>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <thrust/host_vector.h>        // TODO change to cuTe HostVector and fill etc
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>

#include <bench.h>
#include <util/cuda_check.h>
#include <util/format_helpers.h>
#include <gemm_01_naive.h>
#ifndef ENABLE_CUBLAS
    #define ENABLE_CUBLAS
#endif
#ifdef ENABLE_CUBLAS
#include <bench_cublas.h>
#endif

using namespace gemm;

constexpr int MIN_DIMENSION =  1024;
constexpr int MAX_DIMENSION = 32768;
constexpr int DEFAULT_M     =  5120;
constexpr int DEFAULT_N     =  5120;
constexpr int DEFAULT_K     =  4096;
constexpr int DEFAULT_REPS  =    20;
constexpr int MIN_REPS      =     0;
constexpr int MAX_REPS      = 10000;

bool verbose = false;

enum Precision {
    FP32,
    FP16,
    BF16,
    FP8
};

template <>
struct std::formatter<Precision> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(Precision p, format_context& ctx) const {
        switch (p) {
            case FP32: return std::format_to(ctx.out(), "fp32");
            case FP16: return std::format_to(ctx.out(), "fp16");
            case BF16: return std::format_to(ctx.out(), "bf16");
            case FP8:  return std::format_to(ctx.out(), "fp8");
        }
        return ctx.out();
    }
};

// TODO: use cuTe tensor helpers and fill (and print seed for repro)
// TODO: pass random seed
template<typename MType>
struct Gemm_Data {
    const int m;
    const int n;
    const int k;
    MType alpha;
    MType beta;
    const double gflops;
    thrust::host_vector<MType>   h_A;
    thrust::host_vector<MType>   h_B;
    thrust::host_vector<MType>   h_C;
    thrust::device_vector<MType> d_A;
    thrust::device_vector<MType> d_B;
    thrust::device_vector<MType> d_C;
    Gemm_Data(const int m, const int n, const int k, const float alpha, const float beta) :
            m(m), n(n), k(k), alpha(alpha), beta(beta), gflops((2.0 * m * n * k) * 1e-9),
            h_A(m*k), h_B(n*k), h_C(m*n), d_A(m*k), d_B(n*k), d_C(m*n) {
        for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<MType>( 2*(rand() / double(RAND_MAX)) - 1 );
        for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<MType>( 2*(rand() / double(RAND_MAX)) - 1 );
        for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<MType>(-1);
    }
    void reset() {
        d_C = h_C;
    }
};

// Benchmark a function and return the minimum time
template<typename MType>
float
bench_run_min_ms(Gemm_Data<MType>& gd, const auto& func, const int reps, cudaStream_t stream) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float minElapsedTime = std::numeric_limits<float>::max();

    for (int r = 0; r < reps; r++) {
        gd.reset();                              // init
        float run_ms;
        cudaEventRecord(start, stream);
        func();                                  // RUN
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        cudaEventElapsedTime(&run_ms, start, stop);
        minElapsedTime = std::min(run_ms, minElapsedTime);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return minElapsedTime;
}

#if 0
template<typename MType>
void
bench_cublas_nt(Gemm_Data<MType>& gd, const int reps, std::vector<cublasComputeType_t>& cts) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t handle;
    gemm::cublas_check(cublasCreate(&handle), "error creating handle");
    auto f_cublas_gen = [&](cublasComputeType_t compute_type) {
        auto f_cublas = cublas_func(handle, CUBLAS_OP_N, CUBLAS_OP_T, gd.m, gd.n, gd.k,
                                    &gd.alpha, gd.d_A.data().get(), gd.m, gd.d_B.data().get(), gd.n,
                                    &gd.beta, gd.d_C.data().get(), gd.m,
                                    compute_type);
        // benchmark this compute type
        float ms = bench_run_min_ms<MType>(gd, f_cublas, reps, stream);
        std::cout << std::format("{:<10}: ({:<8})   {: 6.2f} TFlop/s  {: 6.2f} ms\n", "cuBLAS",
                                 cublasComputeTypeStr(compute_type), gd.gflops / ((ms/1000)*1000), ms);
    };

    for (auto compute_type: cts) {
        f_cublas_gen(compute_type);
    }
}
#endif  // ENABLE_CUBLAS

#if 0
template<typename MType>
void
bench_cuda_nt(Gemm_Data<MType>& gd, const int reps) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // cuTe naive
    auto f_cute = [&]() {
        gemm_naive(gd.m, gd.n, gd.k,
             gd.alpha,
             gd.d_A.data().get(), gd.m,
             gd.d_B.data().get(), gd.n,
             gd.beta,
             gd.d_C.data().get(), gd.m);
    };
    float ms = bench_run_min_ms<MType>(gd, f_cute, reps, stream);
    std::cout << std::format("{:<10}:              {: 6.2f} TFlop/s  {: 6.2f} ms\n", "cuTe naive",
                             gd.gflops / ((ms/1000)*1000), ms);

    cudaStreamDestroy(stream);
}
#endif

template<typename MType>
void
bench(const int m, const int n, const int k, const float alpha, const float beta, const int reps) {
    Gemm_Data<MType> gd(m, n, k, alpha, beta);
#ifdef ENABLE_CUBLAS
    // bench_cublas_nt<MType>(gd, reps, cts);
#endif
    // std::vector<cublasComputeType_t> cts = {CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_FAST_TF32,
    //                                     CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_COMPUTE_32F_FAST_16F};

    // , {CUBLAS_COMPUTE_16F}
#if 0
    // XXX __half fails epilogue bench_cuda_nt<__half>(gd_fp16, r);
    bench_cuda_nt<MType>(gd, reps);
#endif
}

void
device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << std::format("GPU {:2}: {}\n", device, deviceProp.name);
        std::cout << std::format("        Compute: {}.{}  MB: {} MPs: {} thr/blk: {} regs/blk: {}k\n",
            deviceProp.totalGlobalMem / (1024 * 1024), deviceProp.major, deviceProp.minor,
            deviceProp.multiProcessorCount,
            deviceProp.maxThreadsPerBlock, deviceProp.regsPerBlock / 1024
            );

        std::cout << std::format("        max shared m/blk: {} KB shared m/MP: {} KB, max warps/MP: {}\n",
            deviceProp.sharedMemPerBlock / 1024, deviceProp.sharedMemPerMultiprocessor / 1024,
            deviceProp.maxThreadsPerMultiProcessor / 32);
    }
}

int
main(int argc, char* argv[]) {
    int m = DEFAULT_M;
    int n = DEFAULT_N;
    int k = DEFAULT_K;
    int r = DEFAULT_REPS;

    // Define the long options
    static struct option long_options[] = {
        {"v",         required_argument, nullptr, 'v'},
        {"m",         required_argument, nullptr, 'm'},
        {"n",         required_argument, nullptr, 'n'},
        {"k",         required_argument, nullptr, 'k'},
        {"r",         required_argument, nullptr, 'r'},
        {nullptr, 0, nullptr, 0}
    };

    // Parse the command-line arguments
    int option_index = 0;
    int opt;
    while ((opt = getopt_long(argc, argv, "vm:n:k:r:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'v':
                verbose = true;
                break;
            case 'm':
                m = std::stoi(optarg);
                break;
            case 'n':
                n = std::stoi(optarg);
                break;
            case 'k':
                k = std::stoi(optarg);
                break;
            case 'r':
                r = std::stoi(optarg);
                break;
            default:
                std::cout << std::format("Usage: {} -m <m> -n <n> -k <k> -r <reps>\n", argv[0]);
                return 1;
        }
    }

    if (verbose) {
        device_info();
    }

    // Check valid matrix size
    if (m != std::clamp(m, MIN_DIMENSION, MAX_DIMENSION) ||
            n != std::clamp(n, MIN_DIMENSION, MAX_DIMENSION) ||
            k != std::clamp(k, MIN_DIMENSION, MAX_DIMENSION) ||
            r != std::clamp(r, MIN_REPS, MAX_REPS)) {

        std::cerr << std::format("Invalid arguments (m = {}, n = {}, k = {}, r = {})."
                                 "Matrix size within [{}, {}], repetitions [{}, {}].\n",
                                 m, n, k, r, MIN_DIMENSION, MAX_DIMENSION, MIN_REPS, MAX_REPS);
        std::exit(EXIT_FAILURE);
    }

    std::cout << std::format("Benchmarking M = {}, N = {}, K = {}, precision {}, repetitions {}\n",
                             m, n, k, FP32, r);

    bench<float>(m, n, k, 1.0, 0.0, r);

    return 0;
}
