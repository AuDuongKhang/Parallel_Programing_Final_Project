#ifndef MATRIX_KERNEL_7_CUH
#define MATRIX_KERNEL_7_CUH
#include "matrix_kernel.cuh"

template<typename T>
class MatrixKernel7 : public MatrixKernel<T> {
  public:
    MatrixKernel7() = default;
    void matmulWithConstantAtomic(const T* input, T* output, int M, int N, int K) const;
};

template<typename T>
__global__ void matmulWithConstantKernelAtomic(const T* input, T* output, int M, int N, int K) {
  __shared__ T tileA[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Load input tiles into shared memory
    if (row < M && t * TILE_SIZE + threadIdx.x < K) {
      tileA[threadIdx.y][threadIdx.x] = input[row * K + t * TILE_SIZE + threadIdx.x];
    } 
    else {
      tileA[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    // Multiply and write result atomically
    if (row < M && col < N) {
      for (int k = 0; k < TILE_SIZE; ++k) {
        T value = tileA[threadIdx.y][k] * d_constant_weights[(t * TILE_SIZE + k) * N + col];
        atomicAdd(&output[row * N + col], value);
      }
    }

    __syncthreads();
  }
}

template<typename T>
void MatrixKernel7<T>::matmulWithConstantAtomic(const T* input, T* output, int M, int N, int K) const {
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  matmulWithConstantKernelAtomic<<<gridDim, blockDim>>>(input, output, M, N, K);
  CHECK(cudaDeviceSynchronize());
}
#endif // MATRIX_KERNEL_7_CUH