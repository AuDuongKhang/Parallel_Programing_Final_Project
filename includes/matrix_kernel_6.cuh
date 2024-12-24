#ifndef MATRIX_KERNEL_6_CUH
#define MATRIX_KERNEL_6_CUH
#include "matrix_kernel.cuh"

template<typename T>
class MatrixKernel6 : public MatrixKernel<T> {
  public:
    MatrixKernel6() = default;
    void matmulWithConstantTree(const T* input, T* output, int M, int N, int K) const;
};

template<typename T>
__global__ void matmulWithConstantKernelTree(const T* input, T* output, int M, int N, int K) {
  __shared__ T tileA[TILE_SIZE][TILE_SIZE];
  __shared__ T partialSum[TILE_SIZE];

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

    // Multiply input with constant weights and store partial sum in shared memory
    if (col < N) {
      for (int k = 0; k < TILE_SIZE; ++k) {
        partialSum[threadIdx.x] = tileA[threadIdx.y][k] * d_constant_weights[(t * TILE_SIZE + k) * N + col];
      }

      // Perform tree reduction on partial sums
      for (int offset = TILE_SIZE / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
          partialSum[threadIdx.x] += partialSum[threadIdx.x + offset];
        }
        __syncthreads();
      }

      // Final sum for this thread
      if (threadIdx.x == 0) {
        value += partialSum[0];
      }
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    output[row * N + col] = value;
  }
}

template<typename T>
void MatrixKernel6<T>::matmulWithConstantTree(const T* input, T* output, int M, int N, int K) const {
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  matmulWithConstantKernelTree<<<gridDim, blockDim>>>(input, output, M, N, K);
  CHECK(cudaDeviceSynchronize());
}
#endif // MATRIX_KERNEL_6_CUH