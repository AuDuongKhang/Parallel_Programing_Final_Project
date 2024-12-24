#ifndef MATRIX_KERNEL_8_CUH
#define MATRIX_KERNEL_8_CUH
#include "matrix_kernel.cuh"

template<typename T>
class MatrixKernel8 : public MatrixKernel<T> {
  public:
    MatrixKernel8() = default;
    void strassenMatrixMultiply(T* d_A, T* d_B, T* d_C, int M, int K, int N, int depth = 0);
};

template<typename T>
__global__ void splitMatrixKernel(const T* input, 
                                  T* A11, 
                                  T* A12, 
                                  T* A21, 
                                  T* A22, 
                                  int rows, 
                                  int cols, 
                                  int halfRows, 
                                  int halfCols) 
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < halfRows && col < halfCols) {
    // Top-left quadrant (A11)
    A11[row * halfCols + col] = input[row * cols + col];
  } 
  else if (row < halfRows && col >= halfCols && col < cols) {
    // Top-right quadrant (A12)
    A12[row * (cols - halfCols) + (col - halfCols)] = input[row * cols + col];
  } 
  else if (row >= halfRows && row < rows && col < halfCols) {
    // Bottom-left quadrant (A21)
    A21[(row - halfRows) * halfCols + col] = input[row * cols + col];
  } 
  else if (row >= halfRows && row < rows && col >= halfCols && col < cols) {
    // Bottom-right quadrant (A22)
    A22[(row - halfRows) * (cols - halfCols) + (col - halfCols)] = input[row * cols + col];
  }
}

template<typename T>
__global__ void combineSubMatricesKernel(const T* M1, 
                                         const T* M2, 
                                         const T* M3, 
                                         const T* M4,
                                         const T* M5, 
                                         const T* M6, 
                                         const T* M7, 
                                         T* C,
                                         int M, 
                                         int N, 
                                         int halfM, 
                                         int halfN)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    int halfRow = row % halfM;
    int halfCol = col % halfN;

    int blockRow = row / halfM;
    int blockCol = col / halfN;

    T value = 0;

    if (blockRow == 0 && blockCol == 0) {
      // C11 = M1 + M4 - M5 + M7
      value = M1[halfRow * halfN + halfCol] + 
              M4[halfRow * halfN + halfCol] - 
              M5[halfRow * halfN + halfCol] + 
              M7[halfRow * halfN + halfCol];
    } 
    else if (blockRow == 0 && blockCol == 1) {
      // C12 = M3 + M5
      value = M3[halfRow * halfN + halfCol] + 
              M5[halfRow * halfN + halfCol];
    }
    else if (blockRow == 1 && blockCol == 0) {
      // C21 = M2 + M4
      value = M2[halfRow * halfN + halfCol] + 
              M4[halfRow * halfN + halfCol];
    } 
    else if (blockRow == 1 && blockCol == 1) {
      // C22 = M1 + M3 - M2 + M6
      value = M1[halfRow * halfN + halfCol] + 
              M3[halfRow * halfN + halfCol] - 
              M2[halfRow * halfN + halfCol] + 
              M6[halfRow * halfN + halfCol];
    }

    C[row * N + col] = value;
  }
}

template<typename T>
void MatrixKernel8<T>::strassenMatrixMultiply(T* d_A, T* d_B, T* d_C, int M, int K, int N, int depth) {
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  if (M <= TILE_SIZE || K <= TILE_SIZE || N <= TILE_SIZE || depth >= 2) {
    // Fallback to standard matrix multiplication for small matrices or depth limit
    matmulSharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK(cudaDeviceSynchronize());
    return;
  }
  
  int halfM = (M + 1) / 2, halfK = (K + 1) / 2, halfN = (N + 1) / 2;

  // Allocate memory for submatrices
  T *A11, *A12, *A21, *A22, *B11, *B12, *B21, *B22;
  T *M1, *M2, *M3, *M4, *M5, *M6, *M7;
  CHECK(cudaMalloc(&A11, halfM * halfK * sizeof(T)));
  CHECK(cudaMalloc(&A12, halfM * halfK * sizeof(T)));
  CHECK(cudaMalloc(&A21, halfM * halfK * sizeof(T)));
  CHECK(cudaMalloc(&A22, halfM * halfK * sizeof(T)));
  CHECK(cudaMalloc(&B11, halfK * halfN * sizeof(T)));
  CHECK(cudaMalloc(&B12, halfK * halfN * sizeof(T)));
  CHECK(cudaMalloc(&B21, halfK * halfN * sizeof(T)));
  CHECK(cudaMalloc(&B22, halfK * halfN * sizeof(T)));
  CHECK(cudaMalloc(&M1, halfM * halfN * sizeof(T)));
  CHECK(cudaMalloc(&M2, halfM * halfN * sizeof(T)));
  CHECK(cudaMalloc(&M3, halfM * halfN * sizeof(T)));
  CHECK(cudaMalloc(&M4, halfM * halfN * sizeof(T)));
  CHECK(cudaMalloc(&M5, halfM * halfN * sizeof(T)));
  CHECK(cudaMalloc(&M6, halfM * halfN * sizeof(T)));
  CHECK(cudaMalloc(&M7, halfM * halfN * sizeof(T)));

  // Launch kernels to split matrices into submatrices
  splitMatrixKernel<<<gridDim, blockDim>>>(d_A, A11, A12, A21, A22, M, K, halfM, halfK);
  splitMatrixKernel<<<gridDim, blockDim>>>(d_B, B11, B12, B21, B22, K, N, halfK, halfN);

  // Temporary matrices for intermediate calculations
  T *tempA, *tempB;
  CHECK(cudaMalloc(&tempA, halfM * halfK * sizeof(T)));
  CHECK(cudaMalloc(&tempB, halfK * halfN * sizeof(T)));

  // Compute M1 to M7
  matrixAddKernel<<<gridDim, blockDim>>>(A11, A22, tempA, halfM, halfK);
  matrixAddKernel<<<gridDim, blockDim>>>(B11, B22, tempB, halfK, halfN);
  strassenMatrixMultiply(tempA, tempB, M1, halfM, halfK, halfN, depth + 1);

  matrixAddKernel<<<gridDim, blockDim>>>(A21, A22, tempA, halfM, halfK);
  strassenMatrixMultiply(tempA, B11, M2, halfM, halfK, halfN, depth + 1);

  matrixSubKernel<<<gridDim, blockDim>>>(B12, B22, tempB, halfK, halfN);
  strassenMatrixMultiply(A11, tempB, M3, halfM, halfK, halfN, depth + 1);

  matrixSubKernel<<<gridDim, blockDim>>>(B21, B11, tempB, halfK, halfN);
  strassenMatrixMultiply(A22, tempB, M4, halfM, halfK, halfN, depth + 1);

  matrixAddKernel<<<gridDim, blockDim>>>(A11, A12, tempA, halfM, halfK);
  strassenMatrixMultiply(tempA, B22, M5, halfM, halfK, halfN, depth + 1);

  matrixSubKernel<<<gridDim, blockDim>>>(A21, A11, tempA, halfM, halfK);
  matrixAddKernel<<<gridDim, blockDim>>>(B11, B12, tempB, halfK, halfN);
  strassenMatrixMultiply(tempA, tempB, M6, halfM, halfK, halfN, depth + 1);

  matrixSubKernel<<<gridDim, blockDim>>>(A12, A22, tempA, halfM, halfK);
  matrixAddKernel<<<gridDim, blockDim>>>(B21, B22, tempB, halfK, halfN);
  strassenMatrixMultiply(tempA, tempB, M7, halfM, halfK, halfN, depth + 1);

  // Combine submatrices into result matrix
  combineSubMatricesKernel<<<gridDim, blockDim>>>(M1, M2, M3, M4, M5, M6, M7, d_C, M, N, halfM, halfN);

  // Free temporary device memory
  CHECK(cudaFree(tempA));
  CHECK(cudaFree(tempB));
  CHECK(cudaFree(A11)); 
  CHECK(cudaFree(A12)); 
  CHECK(cudaFree(A21)); 
  CHECK(cudaFree(A22));
  CHECK(cudaFree(B11)); 
  CHECK(cudaFree(B12)); 
  CHECK(cudaFree(B21)); 
  CHECK(cudaFree(B22));
  CHECK(cudaFree(M1)); 
  CHECK(cudaFree(M2)); 
  CHECK(cudaFree(M3)); 
  CHECK(cudaFree(M4)); 
  CHECK(cudaFree(M5)); 
  CHECK(cudaFree(M6)); 
  CHECK(cudaFree(M7));
}

#endif // MATRIX_KERNEL_8_CUH