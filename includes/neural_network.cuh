#ifndef NEURAL_NETWORK_CPU_CUH
#define NEURAL_NETWORK_CPU_CUH
#include "constant.cuh"
#include "matrix.cuh"

#define BLOCK_SIZE 256 

using namespace matrix;

namespace nn {
  template<typename T>
  struct ReluFunctor {
    __device__ __host__ T operator()(T x) const {
      return x > 0 ? x : 0;
    }
  };

  template<typename T>
  matrix::Matrix<T> relu(const matrix::Matrix<T>& input, bool useDevice = false) {
    if (useDevice) {
      ReluFunctor<T> reluFunc;
      return input.applyFunctionKernel(reluFunc);
    }
    else {
      return input.applyFunction([](T x) {
        return x > 0 ? x : 0;
      });
    }
  }

  template<typename T>
  struct DReluFunctor {
    __device__ __host__ T operator()(T x) const {
      return x > 0 ? 1 : 0;
    }
  };

  template<typename T>
  matrix::Matrix<T> d_relu(const matrix::Matrix<T>& input, bool useDevice = false) {
    if (useDevice) {
      DReluFunctor<T> d_reluFunc;
      return input.applyFunctionKernel(d_reluFunc);
    }
    else {
      return input.applyFunction([](T x) {
          return x > 0 ? 1 : 0;
      });
    }
  }

  template<typename T>
  struct expKernelFunctor {
    __device__ __host__ T operator()(T x) const {
      return std::exp(x);
    }
  };
  
  template <typename T>
  __global__ void softmaxSumKernel(const T* input, T* partialSums, size_t size) {
    __shared__ T sharedData[256];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sharedData[tid] = (idx < size) ? input[idx] : T(0);
    __syncthreads();

    // Perform reduction within the block
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write block's partial sum to the output
    if (tid == 0) {
        partialSums[blockIdx.x] = sharedData[0];
    }
}

  template <typename T>
  matrix::Matrix<T> softmax(const matrix::Matrix<T>& input, bool useDevice = false) {
    if (useDevice) {
      size_t numElements = input.numel;

      // Step 1: Compute exp(x) for each element
      expKernelFunctor<T> expFunctor;
      auto exp_values = input.applyFunctionKernel(expFunctor);

      // Step 2: Compute the sum of exp(x) using a reduction kernel
      T *d_partialSums, *d_sum, *d_expValues;
      size_t numBlocks = (numElements + 255) / 256;

      CHECK(cudaMalloc(&d_partialSums, numBlocks * sizeof(T)));
      CHECK(cudaMalloc(&d_sum, sizeof(T)));
      CHECK(cudaMalloc(&d_expValues, numElements * sizeof(T)));

      CHECK(cudaMemcpy(d_expValues, exp_values.getData().data(), numElements * sizeof(T), cudaMemcpyHostToDevice));

      // Launch kernel to compute partial sums
      softmaxSumKernel<<<numBlocks, 256>>>(d_expValues, d_partialSums, numElements);
      CHECK(cudaDeviceSynchronize());

      // Compute the total sum from partial sums
      T h_partialSums[numBlocks];
      CHECK(cudaMemcpy(h_partialSums, d_partialSums, numBlocks * sizeof(T), cudaMemcpyDeviceToHost));

      T totalSum = std::accumulate(h_partialSums, h_partialSums + numBlocks, T(0));
      CHECK(cudaMemcpy(d_sum, &totalSum, sizeof(T), cudaMemcpyHostToDevice));

      // Step 3: Divide each exp(x) by the total sum
      auto divideFunc = [=] __device__(T x) -> T {
        return x / totalSum;
      };

      auto result = exp_values.applyFunctionKernel(divideFunc);

      CHECK(cudaFree(d_partialSums));
      CHECK(cudaFree(d_sum));
      CHECK(cudaFree(d_expValues));

      return result;
    } 
    else {
      // CPU implementation
      auto exp_values = input.applyFunction([](const T& x) {
        return std::exp(x);
      });

      T sum_exp = std::accumulate(exp_values.getData().begin(), exp_values.getData().end(), T(0));

      return exp_values.applyFunction([sum_exp](const T& x) {
        return x / sum_exp;
      });
    }
  }


  template<typename T>
  class MLP {
  public:
    std::vector<size_t> units_per_layer;
    std::vector<matrix::Matrix<T>> bias_vectors;
    std::vector<matrix::Matrix<T>> weight_matrices;
    std::vector<matrix::Matrix<T>> activations;
    float lr;

    MLP(std::vector<size_t> units_per_layer, float lr);
    void initializeWeights();
    int uploadWeightsToConstant(size_t layer);
    matrix::Matrix<T> forward(matrix::Matrix<T> x, bool useDevice, int mode);
    matrix::Matrix<T> operator()(matrix::Matrix<T> x, bool useDevice, int mode);
    void backprop(matrix::Matrix<T> target, bool useDevice, int mode);
    void computeLayer(const matrix::Matrix<T>& input, matrix::Matrix<T>& weights, 
                      matrix::Matrix<T>& biases, matrix::Matrix<T>& output, bool useDevice);
  };


/* template<typename T>
void MLP<T>::computeLayer(const matrix::Matrix<T>& input, matrix::Matrix<T>& weights, 
                            matrix::Matrix<T>& biases, matrix::Matrix<T>& output, bool useDevice) {
  if (!useDevice) {
    // CPU fallback
    output = input.matmul(weights).add(biases);
  } 
  else {
    // GPU computation
    MatrixKernel9<T> kernel9;

    const T* d_input = input.getDevicePointer();
    const T* d_weights = weights.getDevicePointer();
    T* d_output = output.getDevicePointer();

    int M = input.getDims()[0];
    int K = input.getDims()[1];
    int N = weights.getDims()[1];

    kernel9.matmul_dense(d_input, d_weights, d_output, M, N, K);
    // Add bias
    MatrixKernel<T> matrix_kernel;
    matrix_kernel.addBiasKernel(d_output, biases.getDevicePointer(), M, N);
  }
}*/


  template<typename T>
  void MLP<T>::initializeWeights() {
    for (auto& weight : weight_matrices) {
      float stddev = std::sqrt(2.0 / weight.getDims()[0]);
      weight.fillRandom(-stddev, stddev);
    }
  }

  template<typename T>
  MLP<T>::MLP(std::vector<size_t> units_per_layer, float lr) : 
                                  units_per_layer(units_per_layer), 
                                  lr(lr) 
  {
    std::cout << "Initializing MLP...\n";
    for (size_t i = 0; i < units_per_layer.size() - 1; ++i) {
        size_t rows = units_per_layer[i + 1];
        size_t cols = units_per_layer[i];
        
        weight_matrices.emplace_back(Matrix<T>({rows, cols}));
        initializeWeights();
        
        bias_vectors.emplace_back(Matrix<T>({rows, 1}));
        bias_vectors.back().fill_(0.0);
    }
    std::cout << "MLP initialized successfully.\n";
  }

  template<typename T>
  int MLP<T>::uploadWeightsToConstant(size_t layer) {
    if (layer >= weight_matrices.size()) {
        throw std::runtime_error("Invalid layer index.");
    }

    const auto& weights = weight_matrices[layer];
    size_t weight_size = weights.getDims()[0] * weights.getDims()[1];

    if (weight_size * sizeof(T) <= MAX_CONSTANT_WEIGHTS) {
      CHECK(cudaMemcpyToSymbol(d_constant_weights, weights.getData().data(), weight_size * sizeof(T), 0, cudaMemcpyHostToDevice));
      return 1;
    }
    else {
      return 0;
    }
  }


  template<typename T>
  matrix::Matrix<T> MLP<T>::forward(matrix::Matrix<T> x, bool useDevice, int mode) {
    activations.clear();
    activations.push_back(x);

    for (size_t i = 0; i < weight_matrices.size(); ++i) {
      // Use for mode 9
      // matrix::Matrix<T> output({x.getDims()[0], weight_matrices[i].getDims()[1]});
      
      if (!useDevice) {
        x = weight_matrices[i].matmul(x).add(bias_vectors[i].repeat(x.getDims()[1]));
      }
      else {
        // Matmul using GPU (mode = 1)
        if (mode == 1) {
          x = weight_matrices[i].matmulVer1(x).addKernel(bias_vectors[i].repeat(x.getDims()[1]));
        }
        // Matmul with shared memory (mode = 2)
        else if (mode == 2) {
          x = weight_matrices[i].matmulVer2(x).addKernel(bias_vectors[i].repeat(x.getDims()[1]));
        }
        // Kernel fusion for unrolling and matrix-multiplication (mode = 3)
        else if (mode == 3) {
          x = weight_matrices[i].matmulVer3(x).addKernel(bias_vectors[i].repeat(x.getDims()[1]));
        }
        // Matmul with shared memory and constant memory (mode = 4)
        else if (mode == 4) {
          if (uploadWeightsToConstant(i) == 1) {
            x = weight_matrices[i].matmulVer4(x).addKernel(bias_vectors[i].repeat(x.getDims()[1]));
          }
          else {
            x = weight_matrices[i].matmulVer1(x).addKernel(bias_vectors[i].repeat(x.getDims()[1]));
          }
        }
        // Tuning with restrict and loop unrolling (mode = 5)
        else if (mode == 5) {
          x = weight_matrices[i].matmulVer5(x).addKernel(bias_vectors[i].repeat(x.getDims()[1]));
        }
        // Input channel reduction: tree (mode = 6)
        else if (mode == 6) {
          if (uploadWeightsToConstant(i) == 1) {
            x = weight_matrices[i].matmulVer6(x).addKernel(bias_vectors[i].repeat(x.getDims()[1]));
          }
          else {
            x = weight_matrices[i].matmulVer1(x).addKernel(bias_vectors[i].repeat(x.getDims()[1]));
          }        
        } 
        // Input channel reduction: atomic (mode = 7)
        else if (mode == 7) {
          if (uploadWeightsToConstant(i) == 1) {
            x = weight_matrices[i].matmulVer7(x).addKernel(bias_vectors[i].repeat(x.getDims()[1]));
          }
          else {
            x = weight_matrices[i].matmulVer1(x).addKernel(bias_vectors[i].repeat(x.getDims()[1]));
          }        
        }       
        // Strassen matmul (mode = 8)     
        else if (mode == 8) {
          x = weight_matrices[i].matmulVer8(x).addKernel(bias_vectors[i].repeat(x.getDims()[1]));
        }
        // Multiple kernel for different layer sizes
        /* else if (mode == 9) {
          computeLayer(x, weight_matrices[i], bias_vectors[i], output, useDevice);
        }*/
        // FP16 matmul (mode = 10)
        else if (mode == 10) {
          x = weight_matrices[i].matmulVer10(x).addKernel(bias_vectors[i].repeat(x.getDims()[1]));
        }
      }
      
      if (i < weight_matrices.size() - 1) {
        if (mode == 9) {
          // x = relu(output, useDevice);
        }
        else {
          x = relu(x, useDevice);
        }
      } 
      else {
        if (mode == 9) {
          // x = softmax(output, useDevice);
        }
        else {
          x = softmax(x, useDevice);
        }
      }

      activations.push_back(x);
    }

    return activations.back();
  }

  template<typename T>
  matrix::Matrix<T> MLP<T>::operator()(matrix::Matrix<T> x, bool useDevice, int mode) {
    return forward(x, useDevice, mode);
  }

  template<typename T>
  void MLP<T>::backprop(matrix::Matrix<T> target, bool useDevice, int mode) {
    assert(activations.size() > 1);
    size_t batch_size = target.getDims()[1];

    auto output = activations.back();
    auto delta = output - target; // Delta: {output_units, batch_size}

    // Gradient of bias and weight
    std::vector<matrix::Matrix<T>> d_weights(weight_matrices.size());
    std::vector<matrix::Matrix<T>> d_biases(bias_vectors.size());

    for (int i = weight_matrices.size() - 1; i >= 0; --i) {
      // d_weights: Delta * (activations[i]^T) / batch_size
      if (!useDevice) {
        d_biases[i] = delta.sum(1).multiplyScalar(1.0 / batch_size);
        d_weights[i] = delta.matmul(activations[i].transpose()).multiplyScalar(1.0 / batch_size);
      }
      else {
        d_biases[i] = delta.sum(1).multiplyScalarKernel(1.0 / batch_size);
        if (mode == 2) {
          d_weights[i] = delta.matmulVer2(activations[i].transposeKernel()).multiplyScalarKernel(1.0 / batch_size);
        }
        else if (mode == 3) {
          d_weights[i] = delta.matmulVer3(activations[i].transposeKernel()).multiplyScalarKernel(1.0 / batch_size);
        }
        else if (mode == 5) {
          d_weights[i] = delta.matmulVer5(activations[i].transposeKernel()).multiplyScalarKernel(1.0 / batch_size);
        }
        else if (mode == 8) {
          d_weights[i] = delta.matmulVer8(activations[i].transposeKernel()).multiplyScalarKernel(1.0 / batch_size);
        }
        // Multiple kernel implementations for different layer sizes matmul (mode = 9)
        /* else if (mode == 9) {
          //d_weights[i] = delta.matmulVer9(activations[i].transposeKernel()).multiplyScalarKernel(1.0 / batch_size);
          d_weights[i] = delta.matmulVer1(activations[i].transposeKernel()).multiplyScalarKernel(1.0 / batch_size);        
        }*/
        else if (mode == 10) {
          d_weights[i] = delta.matmulVer10(activations[i].transposeKernel()).multiplyScalarKernel(1.0 / batch_size);
        }
        else {
          d_weights[i] = delta.matmulVer1(activations[i].transposeKernel()).multiplyScalarKernel(1.0 / batch_size);
        }
      }
      
      if (i > 0) {
        if (!useDevice) {
          delta = weight_matrices[i].transpose().matmul(delta)
                    .multiplyElementwise(d_relu(activations[i]));
        }
        else {
          // Matmul using GPU (mode = 1)
          if (mode == 1) {
            delta = weight_matrices[i].transposeKernel().matmulVer1(delta).multiplyElementwiseKernel(d_relu(activations[i], useDevice));
          }
          // Matmul with shared memory (mode = 2)
          else if (mode == 2) {
            delta = weight_matrices[i].transposeKernel().matmulVer2(delta).multiplyElementwiseKernel(d_relu(activations[i], useDevice));
          }
          // Kernel fusion for unrolling and matrix-multiplication (mode = 3)
          else if (mode == 3) {
            delta = weight_matrices[i].transposeKernel().matmulVer3(delta).multiplyElementwiseKernel(d_relu(activations[i], useDevice));
          }
          // Matmul with shared memory and constant memory (mode = 4)
          else if (mode == 4) {
            if (uploadWeightsToConstant(i) == 1) {
              delta = weight_matrices[i].transposeKernel().matmulVer4(delta).multiplyElementwiseKernel(d_relu(activations[i], useDevice));
            }
            else {
              delta = weight_matrices[i].transposeKernel().matmulVer1(delta).multiplyElementwiseKernel(d_relu(activations[i], useDevice));
            }
          }
          // Tuning with restrict and loop unrolling (mode = 5)
          else if (mode == 5) {
            delta = weight_matrices[i].transposeKernel().matmulVer5(delta).multiplyElementwiseKernel(d_relu(activations[i], useDevice));
          }
          // Input channel reduction: tree (mode = 6)
          else if (mode == 6) {
            if (uploadWeightsToConstant(i) == 1) {
              delta = weight_matrices[i].transposeKernel().matmulVer6(delta).multiplyElementwiseKernel(d_relu(activations[i], useDevice));
            }
            else {
              delta = weight_matrices[i].transposeKernel().matmulVer1(delta).multiplyElementwiseKernel(d_relu(activations[i], useDevice));
            }       
          } 
          // Input channel reduction: atomic (mode = 7)
          else if (mode == 7) {
            if (uploadWeightsToConstant(i) == 1) {
              delta = weight_matrices[i].transposeKernel().matmulVer7(delta).multiplyElementwiseKernel(d_relu(activations[i], useDevice));
            }
            else {
              delta = weight_matrices[i].transposeKernel().matmulVer1(delta).multiplyElementwiseKernel(d_relu(activations[i], useDevice));
            }        
          }       
          // Strassen matmul (mode = 8)     
          else if (mode == 8) {
            delta = weight_matrices[i].transposeKernel().matmulVer8(delta).multiplyElementwiseKernel(d_relu(activations[i], useDevice));
          }
          // Multiple kernel implementations for different layer sizes matmul (mode = 9)
          /* else if (mode == 9) {
            // delta = weight_matrices[i].transposeKernel().matmulVer9(delta).multiplyElementwiseKernel(d_relu(activations[i], useDevice));
            delta = weight_matrices[i].transposeKernel().matmulVer1(delta).multiplyElementwiseKernel(d_relu(activations[i], useDevice));
          }*/
          // FP16 matmul (mode = 10) 
          else if (mode == 10) {
            delta = weight_matrices[i].transposeKernel().matmulVer10(delta).multiplyElementwiseKernel(d_relu(activations[i], useDevice));
          }
        }
      }
    }

    // Update bias and weight
    for (size_t i = 0; i < weight_matrices.size(); ++i) {
      weight_matrices[i] = weight_matrices[i] - d_weights[i].multiplyScalar(lr);
      bias_vectors[i] = bias_vectors[i] - d_biases[i].multiplyScalar(lr);
    }
  }
}
#endif // NEURAL_NETWORK_CPU_CUH