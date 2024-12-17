#include "../includes/activation.cuh"
#include "../includes/gpu-support.cuh"

GPU_Support gpuSupport;

__global__ void ReLuKernel(float* input,
                           const int size);

__global__ void SoftmaxKernel(const int output_size,
                                 const int batch_size,
                                 float* values);

void ReLU::operator()(const int output_size,
                      const int batch_size,
                      float* d_value) {
    const int total_size = batch_size * output_size;
    const int threadsPerBlock = 256;
    const int numBlocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    ReLuKernel<<<numBlocks, threadsPerBlock>>>(d_value, total_size);
    gpuSupport.checkLastCudaError();
}

void SoftMax::operator()(const int batch_size,
                            const int output_size,
                            float* d_value) {
    const int total_size = batch_size * output_size;
    const int threadsPerBlock = 256;
    const int numBlocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    //__global__ void LogSoftmaxBatkh(const float* input, float* output, const int outputSize, const int batchSize) {
    SoftmaxKernel<<<numBlocks, threadsPerBlock>>>(output_size, batch_size, d_value);
    gpuSupport.checkLastCudaError();
}

__global__ void ReLuKernel(float* input,
                           const int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size) {
        input[idx] = fmaxf(0, input[idx]);
    }
}

__global__ void SoftmaxKernel(const int output_size,
                              const int batch_size,
                              float* values) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batch_size)
        return;

    float maxInput = -INFINITY;
    for (int j = 0; j < output_size; ++j) {
        maxInput = fmaxf(maxInput, values[idx * output_size + j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < output_size; ++j) {
        sum += expf(values[idx * output_size + j] - maxInput);
    }

    for (int j = 0; j < output_size; ++j) {
        values[idx * output_size + j] = expf(values[idx * output_size + j] - maxInput) / sum;
    }
}
