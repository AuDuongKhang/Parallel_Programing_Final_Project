#ifndef MATRIX_KERNEL_5_CUH
#define MATRIX_KERNEL_5_CUH
#include "matrix_kernel.cuh"

template<typename T>
class MatrixKernel5 : public MatrixKernel<T> {
  public:
    MatrixKernel5() = default;
    void matmulWithRestrictUnRolling(const T* A, const T* B, T* C, int M, int N, int K);
};

template<typename T>
__global__ void matmulKernelRestrictUnRoll(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int M, int N, int K) {
  __shared__ T tileA[TILE_SIZE][TILE_SIZE];
  __shared__ T tileB[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  T value = 0.0;

  // Loop over tiles of A and B
  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    if (row < M && t * TILE_SIZE + threadIdx.x < K) {
      tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
    } 
    else {
      tileA[threadIdx.y][threadIdx.x] = 0.0;
    }

    // Load tileB
    if (col < N && t * TILE_SIZE + threadIdx.y < K) {
      tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
    } 
    else {
      tileB[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    // Multiply and accumulate using loop unrolling
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
      value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }

    __syncthreads();
  }

  // Write result to C
  if (row < M && col < N) {
    C[row * N + col] = value;
  }
}

template<typename T>
void MatrixKernel5<T>::matmulWithRestrictUnRolling(const T* A, const T* B, T* C, int M, int N, int K) {
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  matmulKernelRestrictUnRoll<<<gridDim, blockDim>>>(A, B, C, M, N, K);
  CHECK(cudaDeviceSynchronize());
}
#endif // MATRIX_KERNEL_5_CUH