#ifndef MATRIX_KERNEL_4_CUH
#define MATRIX_KERNEL_4_CUH
#include "matrix_kernel.cuh"

template<typename T>
class MatrixKernel4 : public MatrixKernel<T> {
  public:
    MatrixKernel4() = default;
    void matmulWithConstant(const T* input, T* output, int M, int N, int K) const;
};

template<typename T>
__global__ void matmulWithConstantKernel(const T* input, T* output, int M, int N, int K) {
  __shared__ T tileA[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  T value = 0.0;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Load input tiles into shared memory
    if (row < M && t * TILE_SIZE + threadIdx.x < K) {
      tileA[threadIdx.y][threadIdx.x] = input[row * K + t * TILE_SIZE + threadIdx.x];
    } 
    else {
      tileA[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    // Multiply with constant weights
    if (col < N) {
      for (int k = 0; k < TILE_SIZE; ++k) {
        int weightIdx = (t * TILE_SIZE + k) * N + col;
        if (weightIdx < MAX_CONSTANT_WEIGHTS) {
          value += tileA[threadIdx.y][k] * d_constant_weights[weightIdx];
        }
      }
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    output[row * N + col] = value;
  }
}

template<typename T>
void MatrixKernel4<T>::matmulWithConstant(const T* input, T* output, int M, int N, int K) const {
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  matmulWithConstantKernel<<<gridDim, blockDim>>>(input, output, M, N, K);
  CHECK(cudaDeviceSynchronize());
}
#endif // MATRIX_KERNEL_4_CUH