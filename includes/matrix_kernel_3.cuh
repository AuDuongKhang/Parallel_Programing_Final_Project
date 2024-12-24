#ifndef MATRIX_KERNEL_3_CUH
#define MATRIX_KERNEL_3_CUH
#include "matrix_kernel.cuh"

template<typename T>
class MatrixKernel3 : public MatrixKernel<T> {
  public:
    MatrixKernel3() = default;
    void fusedMatrixMultiplyKernel(const T* A, const T* B, T* C, int M, int N, int K);
};

template<typename T>
__global__ void fusedMatrixMultiplyKernelImpl(const T* A, const T* B, T* C, int M, int N, int K) {
  // Kernel fusion logic: unroll and multiply in one step
  __shared__ T tileA[TILE_SIZE][TILE_SIZE];
  __shared__ T tileB[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  T sum = 0;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Load tiles into shared memory
    tileA[threadIdx.y][threadIdx.x] = (row < M && (t * TILE_SIZE + threadIdx.x) < K) ? A[row * K + t * TILE_SIZE + threadIdx.x] : 0;
    tileB[threadIdx.y][threadIdx.x] = (col < N && (t * TILE_SIZE + threadIdx.y) < K) ? B[(t * TILE_SIZE + threadIdx.y) * N + col] : 0;

    __syncthreads();

    // Unrolling loop for better performance
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

template<typename T>
void MatrixKernel3<T>::fusedMatrixMultiplyKernel(const T* A, const T* B, T* C, int M, int N, int K) {
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  // Launch fused kernel
  fusedMatrixMultiplyKernelImpl<<<gridDim, blockDim>>>(A, B, C, M, N, K);
  CHECK(cudaDeviceSynchronize());
}
#endif // MATRIX_KERNEL_3_CUH