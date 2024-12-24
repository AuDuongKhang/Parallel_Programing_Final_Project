#ifndef MATRIX_KERNEL_9_CUH
#define MATRIX_KERNEL_9_CUH
#include "matrix_kernel.cuh"

template<typename T>
class MatrixKernel9 : public MatrixKernel<T> {
  public:
    MatrixKernel9() = default;
    void matmul_dense(const T* input, const T* weights, T* output, int M, int N, int K);
};

template<typename T>
__global__ void smallLayerKernel(const T* input, const T* weights, T* output, int M, int N, int K) {
    __shared__ T tileInput[32][32];
    __shared__ T tileWeights[32][32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = 0;

    for (int t = 0; t < (K + 31) / 32; ++t) {
        if (row < M && t * 32 + threadIdx.x < K)
            tileInput[threadIdx.y][threadIdx.x] = input[row * K + t * 32 + threadIdx.x];
        else
            tileInput[threadIdx.y][threadIdx.x] = 0;

        if (col < N && t * 32 + threadIdx.y < K)
            tileWeights[threadIdx.y][threadIdx.x] = weights[(t * 32 + threadIdx.y) * N + col];
        else
            tileWeights[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < 32; ++k)
            sum += tileInput[threadIdx.y][k] * tileWeights[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        output[row * N + col] = sum;
}
template<typename T>
__global__ void denseLayerKernel(const T* input, const T* weights, T* output, int M, int N, int K) {
    __shared__ T tileInput[16][16];
    __shared__ T tileWeights[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = 0;

    for (int t = 0; t < (K + 15) / 16; ++t) {
        if (row < M && t * 16 + threadIdx.x < K)
            tileInput[threadIdx.y][threadIdx.x] = input[row * K + t * 16 + threadIdx.x];
        else
            tileInput[threadIdx.y][threadIdx.x] = 0;

        if (col < N && t * 16 + threadIdx.y < K)
            tileWeights[threadIdx.y][threadIdx.x] = weights[(t * 16 + threadIdx.y) * N + col];
        else
            tileWeights[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 16; ++k)
            sum += tileInput[threadIdx.y][k] * tileWeights[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        output[row * N + col] = sum;
}
template<typename T>
__global__ void outputLayerKernel(const T* input, const T* weights, T* output, int M, int N, int K) {
    __shared__ T tileInput[16][10];
    __shared__ T tileWeights[10][10];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = 0;

    for (int t = 0; t < (K + 15) / 16; ++t) {
        if (row < M && t * 16 + threadIdx.x < K)
            tileInput[threadIdx.y][threadIdx.x] = input[row * K + t * 16 + threadIdx.x];
        else
            tileInput[threadIdx.y][threadIdx.x] = 0;

        if (col < N && t * 10 + threadIdx.y < K)
            tileWeights[threadIdx.y][threadIdx.x] = weights[(t * 10 + threadIdx.y) * N + col];
        else
            tileWeights[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < 10; ++k)
            sum += tileInput[threadIdx.y][k] * tileWeights[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        output[row * N + col] = sum;
}

template<typename T>
void MatrixKernel9<T>::matmul_dense(const T* input, const T* weights, T* output, int M, int N, int K) {
    if (N == 128 && K == 784) {
        dim3 blockDim(32, 32);
        dim3 gridDim((128 + 31) / 32, (784 + 31) / 32);
        smallLayerKernel<<<gridDim, blockDim>>>(input, weights, output, M, N, K);
    } else if (N == 128 && K == 128) {
        dim3 blockDim(16, 16);
        dim3 gridDim((128 + 15) / 16, (128 + 15) / 16);
        denseLayerKernel<<<gridDim, blockDim>>>(input, weights, output, M, N, K);
    } else if (N == 10 && K == 128) {
        dim3 blockDim(16, 10);
        dim3 gridDim((10 + 15) / 16, (128 + 15) / 16);
        outputLayerKernel<<<gridDim, blockDim>>>(input, weights, output, M, N, K);
    }
    CHECK(cudaDeviceSynchronize());
}

#endif // MATRIX_KERNEL_9_CUH