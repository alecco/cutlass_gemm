add_executable(test_gemm test_gemm.cu)

if(TARGET CUDA::cublas)
    add_executable(test_cublas_gemm test_cublas_gemm.cu)
    target_link_libraries(test_cublas_gemm PRIVATE CUDA::cublas)
else()
    message(STATUS "cuBLAS not found, not testing with it.")
endif()
