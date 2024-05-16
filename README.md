# Fast CUTLASS GEMM from scratch

Step-by-step optimization of matrix multiplication, implemented with the Nvidia CUTLASS C++ template library.

In style of https://siboehm.com/articles/22/CUDA-MMM.

## Building

```
git submodule update --init --recursive
make
```

## Running

### Benchmark

```
make bench
```

### Test

```
make test
```

## Notes

* Matrices are A MxK and B NxK with result C MxN
* All matrices are stored in column major layout
* The matrix B is stored as transposed  (NT) (for vectorized memory access)
* GEMM as C = α * A × Bᵗ + β * C
* Tensor Core implementations
* Precision used is BF16 with FP32 accumulate (Ampere+ required)

## Other

The cuBLAS library is only needed to compile its benchmark implementation.
But if it is not present, the code still compiles and runs.

Changing NN to NT in cuBLAS gives ~27% speedup in the original CUDA-MMM code.

Using half precision gives up to 2x memory bandwidth and compute.
