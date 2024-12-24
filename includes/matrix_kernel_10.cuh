#ifndef MATRIX_KERNEL_10_CUH
#define MATRIX_KERNEL_10_CUH
#include "matrix_kernel.cuh"

template<typename T>
class MatrixKernel10;

template<>
class MatrixKernel10<__half> {
  public:
    MatrixKernel10() = default;
    void matmulFP16Kernel(const __half* A, const __half* B, __half* C, int M, int N, int K);
};

__global__ void matmulFP16ByKernel(const __half* A, const __half* B, __half* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    __half value = __float2half(0.0f);
    for (int k = 0; k < K; ++k) {
      value = __hadd(value, __hmul(A[row * K + k], B[k * N + col]));
    }
    C[row * N + col] = value;
  }
}

void MatrixKernel10<__half>::matmulFP16Kernel(const __half* A, const __half* B, __half* C, int M, int N, int K) {
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  matmulFP16ByKernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
  CHECK(cudaDeviceSynchronize());
}


#endif // MATRIX_KERNEL_10_CUH