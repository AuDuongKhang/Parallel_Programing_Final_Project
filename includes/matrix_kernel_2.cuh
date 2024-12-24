#ifndef MATRIX_KERNEL_2_CUH
#define MATRIX_KERNEL_2_CUH
#include "matrix_kernel.cuh"

template<typename T>
class MatrixKernel2 : public MatrixKernel<T> {
  public:
    MatrixKernel2() = default;
    void matmulSharedMemKernel(const T* A, const T* B, T* C, int M, int N, int K);
};

template<typename T>
__global__ void matmulSharedKernel(const T* A, const T* B, T* C, int M, int N, int K) {
  __shared__ T tileA[TILE_SIZE][TILE_SIZE];
  __shared__ T tileB[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  T sum = 0;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Load tiles into shared memory
    if (row < M && t * TILE_SIZE + threadIdx.x < K) {
      tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
    } 
    else {
      tileA[threadIdx.y][threadIdx.x] = 0;
    }

    if (col < N && t * TILE_SIZE + threadIdx.y < K) {
      tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
    } 
    else {
      tileB[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    // Multiply tiles
    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }

    __syncthreads();
  }

  // Write result to output matrix
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

template<typename T>
void MatrixKernel2<T>::matmulSharedMemKernel(const T* A, const T* B, T* C, int M, int N, int K) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    matmulSharedKernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    CHECK(cudaDeviceSynchronize()); // Ensure kernel execution is completed
}
#endif // MATRIX_KERNEL_2_CUH