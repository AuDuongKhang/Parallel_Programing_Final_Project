#ifndef MATRIX_KERNEL_CUH
#define MATRIX_KERNEL_CUH
#include "constant.cuh"

#define TILE_SIZE 16 
#define BLOCK_SIZE 256 
  
template<typename T>
class MatrixKernel {
public:
  MatrixKernel() = default;
  void matmulKernel(const T* A, const T* B, T* C, int M, int N, int K);
  void addKernel(const T* A, const T* B, T* C, int M, int N);    
  void multiplyScalarKernel(const T* A, T* B, const T scalar, int M, int N);
  void multiplyElementwiseKernel(const T* A, const T* B, T* C, int numElements);
  template<typename Func>
  void applyFunctionKernel(const T* d_input, T* d_output, int numElements, Func func);
  void transposeKernel(const T* d_input, T* d_output, int rows, int cols);
  void convertToFP16(const T* input, __half* output, size_t size);
  void convertFromFP16(const __half* input, T* output, size_t size);
  void addBiasKernel(T* output, const T* bias, int M, int N);
};

template<typename T>
__global__ void matmulByKernel(const T* A, const T* B, T* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    T value = 0;

    for (int k = 0; k < K; ++k) {
      value += A[row * K + k] * B[k * N + col];
    }

    // Store the result in C
    C[row * N + col] = value;
  }
}


template<typename T>
void MatrixKernel<T>::matmulKernel(const T* A, const T* B, T* C, int M, int N, int K) {
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  // Launch kernel
  matmulByKernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
  CHECK(cudaDeviceSynchronize());
}

template<typename T>
__global__ void matrixAddKernel(const T* A, const T* B, T* C, int rows, int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
  }
}

template<typename T>
void MatrixKernel<T>::addKernel(const T* A, const T* B, T* C, int M, int N) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((M * N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  matrixAddKernel<<<gridDim, blockDim>>>(A, B, C, M, N);
  CHECK(cudaDeviceSynchronize());
}


template<typename T>
__global__ void matrixMultiplyScalarKernel(const T* input, T* output, const T scalar, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < M * N) {
    output[idx] = input[idx] * scalar;
  }
}

template<typename T>
void MatrixKernel<T>::multiplyScalarKernel(const T* A, T* B, const T scalar, int M, int N) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((M * N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  matrixMultiplyScalarKernel<<<gridDim, blockDim>>>(A, B, scalar, M, N);
  CHECK(cudaDeviceSynchronize());
}

template<typename T>
__global__ void elementwiseMultiplyKernel(const T* input1, const T* input2, T* output, int numElements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
  if (idx < numElements) {
    output[idx] = input1[idx] * input2[idx];
  }
}

template<typename T>
void MatrixKernel<T>::multiplyElementwiseKernel(const T* A, const T* B, T* C, int numElements) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((numElements + BLOCK_SIZE - 1) / BLOCK_SIZE);

  elementwiseMultiplyKernel<<<gridDim, blockDim>>>(A, B, C, numElements);
  CHECK(cudaDeviceSynchronize());
}

template<typename T, typename Func>
__global__ void applyFunctionByKernel(const T* input, T* output, int numElements, Func func) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numElements) {
    output[idx] = func(input[idx]);
  }
}

template<typename T>
template<typename Func>
void MatrixKernel<T>::applyFunctionKernel(const T* d_input, T* d_output, int numElements, Func func) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((numElements + BLOCK_SIZE - 1) / BLOCK_SIZE);

  applyFunctionByKernel<<<gridDim, blockDim>>>(d_input, d_output, numElements, func);
  CHECK(cudaDeviceSynchronize());
}

template<typename T>
__global__ void transposeByKernel(const T* input, T* output, int rows, int cols) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < cols && y < rows) {
    output[x * rows + y] = input[y * cols + x];
  }
}

template<typename T>
void MatrixKernel<T>::transposeKernel(const T* d_input, T* d_output, int rows, int cols) {
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

  transposeByKernel<<<gridDim, blockDim>>>(d_input, d_output, rows, cols);
  CHECK(cudaDeviceSynchronize());
}

template<typename T>
__global__ void matrixSubKernel(const T* A, const T* B, T* C, int rows, int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    C[row * cols + col] = A[row * cols + col] - B[row * cols + col];
  }
}

template<typename T>
void MatrixKernel<T>::convertToFP16(const T* input, __half* output, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    output[i] = __float2half(static_cast<float>(input[i]));
  }
}

template<typename T>
void MatrixKernel<T>::convertFromFP16(const __half* input, T* output, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    output[i] = __half2float(input[i]);
  }
}

template<typename T>
__global__ void addBiasByKernel(T* output, const T* bias, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < M * N) {
    int col = idx % N; // Find column index
    output[idx] += bias[col];
  }
}

template<typename T> 
void MatrixKernel<T>::addBiasKernel(T* output, const T* bias, int M, int N) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  
  addBiasByKernel<<<gridDim, blockDim>>>(output, bias, M, N);
  CHECK(cudaDeviceSynchronize());
}

#endif // MATRIX_KERNEL_CUH