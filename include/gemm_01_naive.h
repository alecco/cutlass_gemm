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

// Naive implementation of GEMM in cutlass
// from sgemm_nt_1.cu

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <util/cuda_check.h>

namespace gemm {

template <class MShape, class NShape, class KShape,
          class TA, class AStride, class ABlockLayout, class AThreadLayout,
          class TB, class BStride, class BBlockLayout, class BThreadLayout,
          class TC, class CStride, class CBlockLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(MShape M, NShape N, KShape K,
            TA const* A, AStride dA, ABlockLayout blockA, AThreadLayout tA,
            TB const* B, BStride dB, BBlockLayout blockB, BThreadLayout tB,
            TC      * C, CStride dC, CBlockLayout       , CThreadLayout tC,
            Alpha alpha, Beta beta)
{
  using namespace cute;
  using X = Underscore;

  // Preconditions
  CUTE_STATIC_ASSERT(is_static<ABlockLayout>::value);
  CUTE_STATIC_ASSERT(is_static<BBlockLayout>::value);
  CUTE_STATIC_ASSERT(is_static<CBlockLayout>::value);

  CUTE_STATIC_ASSERT(is_static<AThreadLayout>::value);
  CUTE_STATIC_ASSERT(is_static<BThreadLayout>::value);
  CUTE_STATIC_ASSERT(is_static<CThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(size(tA) == size(tC));
  CUTE_STATIC_ASSERT_V(size(tB) == size(tC));

  //CUTE_STATIC_ASSERT_V(shape<0>(blockA) == shape<0>(blockC));      // BLK_M
  //CUTE_STATIC_ASSERT_V(shape<0>(blockB) == shape<1>(blockC));      // BLK_N
  CUTE_STATIC_ASSERT_V(shape<1>(blockA) == shape<1>(blockB));        // BLK_K

  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ABlockLayout>];
  __shared__ TB smemB[cosize_v<BBlockLayout>];
  auto sA = make_tensor(make_smem_ptr(smemA), blockA);               // (BLK_M,BLK_K)
  auto sB = make_tensor(make_smem_ptr(smemB), blockB);               // (BLK_N,BLK_K)

  // Represent the full tensors
  auto mA = make_tensor(make_gmem_ptr(A), make_shape(M,K), dA);      // (M,K)
  auto mB = make_tensor(make_gmem_ptr(B), make_shape(N,K), dB);      // (N,K)
  auto mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

  // Get the appropriate blocks for this thread block --
  // potential for thread block locality
  auto blk_shape = make_shape(size<0>(sA), size<0>(sB), size<1>(sB));// (BLK_M,BLK_N,BLK_K)
  auto blk_coord = make_coord(blockIdx.x, blockIdx.y, _);            // (m,n,k)

  auto gA = local_tile(mA, blk_shape, blk_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  auto gB = local_tile(mB, blk_shape, blk_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  auto gC = local_tile(mC, blk_shape, blk_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  //
  // Partition the copying of A and B tiles across the threads
  //

  // TUTORIAL: Example of simple partitioning of A|B tiles over tA|tB
  //   Default is a raked partition, but can be changed with Step<X,Y> parameter

  auto tAgA = local_partition(gA, tA, threadIdx.x);                  // (THR_M,THR_K,k)
  auto tAsA = local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K)

  auto tBgB = local_partition(gB, tB, threadIdx.x);                  // (THR_N,THR_K,k)
  auto tBsB = local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)

  //
  // Define C accumulators and A/B partitioning
  //

  // TUTORIAL: Example of partitioning via projections of tC

  // Partition sA (M,K) by the rows of tC
  auto tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K)
  // Partition sB (N,K) by the cols of tC
  auto tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K)
  // Partition gC (M,N) by the tile of tC
  auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

  // Allocate the accumulators -- same size as the projected data
  auto tCrC = make_fragment_like(tCgC);                              // (THR_M,THR_N)

  // Clear the accumulators
  clear(tCrC);

  // TUTORIAL: Example of a very simple compute loop
  //   Data is read from global to shared memory via the tA|tB partitioning
  //   gemm(.) operates on the shared memory directly via the tC partitioning

  auto k_max = size<2>(tAgA);

  for (int k = 0; k < k_max; ++k)
  {
    // Copy gmem to smem
    copy(tAgA(_,_,k), tAsA);
    copy(tBgB(_,_,k), tBsB);

    // In case copy uses cp.async, make sure that the cp.async
    // instructions are ordered with respect to other cp.async
    // instructions (fence), then wait on all the outstanding copy
    // operations (wait<0>()).  __syncthreads() alone does not do
    // this.
    //
    // NOTE: cp_async_wait<0>() currently issues cp.async.wait_all.
    // This is equivalent to cp.async.commit_group followed by
    // cp.async_wait_group 0.  This should make the first
    // cp_async_fence() (which also issues cp.async.commit_group)
    // redundant.  The tutorial works as-is, so we'll leave the
    // redundant fence in for now and study its removal later.
    cp_async_fence();
    cp_async_wait<0>();

    __syncthreads();

    // Compute gemm on smem
    cute::gemm(tCsA, tCsB, tCrC);

    __syncthreads();
  }

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);
}

template <typename TA, typename TB, typename TC,
          typename Alpha, typename Beta>
void
gemm_naive(int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta,
     TC      * C, int ldC,
     cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);

  // Define strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);

  // Define block sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};

  // Define the block layouts (static)
  auto sA = make_layout(make_shape(bM,bK));
  auto sB = make_layout(make_shape(bN,bK));
  auto sC = make_layout(make_shape(bM,bN));

  // Define the thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}));
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

  dim3 dimBlock(size(tC));
  dim3 dimGrid(ceil_div(size(M), size(bM)),
               ceil_div(size(N), size(bN)));
  gemm_device
      <<< dimGrid, dimBlock, 0, stream >>>
      (M,  N,  K,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC,
       alpha, beta);
}

} // namespace gemm
