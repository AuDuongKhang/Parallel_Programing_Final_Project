#include "../includes/dense_layer.cuh"
#include "../includes/gpu-support.cuh"
#include "../includes/activation.cuh"
#include <cuda_runtime.h>
#include <cmath>


void initWeightsAndBias(float* d_weights, float* d_bias, int input_size, int output_size) {
  // Allocate memory on the host
    std::vector<float> weights(input_size * output_size);
    std::vector<float> biases(output_size);

    // Initialize weights and biases on the host
    float stddev = sqrtf(2.0f / input_size); // He initialization
    for (int i = 0; i < input_size * output_size; i++) {
        weights[i] = stddev * (rand() / (RAND_MAX + 1.0f));
    }

    for (int i = 0; i < output_size; i++) {
        biases[i] = 0.0f;
    }

    // Copy initialized weights and biases to the device
    CHECK(cudaMemcpy(d_weights, weights.data(), input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias, biases.data(), output_size * sizeof(float), cudaMemcpyHostToDevice));
}

__global__ void DenseForwardKernel(const float* input, const float* weights, const float* biases, 
                                   float* output, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float sum = biases[idx];
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights[i * output_size + idx];
        }
        output[idx] = fmaxf(0, sum); 
    }
}

DenseLayer::DenseLayer(int input_size, int output_size, bool is_output_layer) 
    : input_size(input_size), output_size(output_size), is_output_layer(is_output_layer) {
    CHECK(cudaMalloc(&weights, input_size * output_size * sizeof(float)));
    CHECK(cudaMalloc(&biases, output_size * sizeof(float)));
    CHECK(cudaMalloc(&output, output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_weights, input_size * output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_biases, output_size * sizeof(float)));

    float stddev = sqrtf(2.0f / input_size); 
    std::vector<float> host_weights(input_size * output_size);
    std::vector<float> host_biases(output_size, 0.0f);

    for (int i = 0; i < input_size * output_size; i++) {
        host_weights[i] = stddev * ((float)rand() / RAND_MAX - 0.5f);
    }

    CHECK(cudaMemcpy(weights, host_weights.data(), input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(biases, host_biases.data(), output_size * sizeof(float), cudaMemcpyHostToDevice));
}

const float* DenseLayer::Forward(const float* input) {
    int threadsPerBlock = 256;
    int numBlocks = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    DenseForwardKernel<<<numBlocks, threadsPerBlock>>>(input, weights, biases, output, input_size, output_size);
    CHECK(cudaDeviceSynchronize());

    if (is_output_layer) {
        SoftmaxKernel<<<1, output_size>>>(output_size, 1, output);
        CHECK(cudaDeviceSynchronize());
    }

    return output;
}


// Destructor
DenseLayer::~DenseLayer() {
    CHECK(cudaFree(weights));
    CHECK(cudaFree(biases));
    CHECK(cudaFree(output));
    CHECK(cudaFree(d_weights));
    CHECK(cudaFree(d_biases));
}
