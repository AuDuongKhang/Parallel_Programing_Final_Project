#ifndef MATRIX_CUH
#define MATRIX_CUH
#include "matrix_kernel.cuh"
#include "matrix_kernel_2.cuh"
#include "matrix_kernel_3.cuh"
#include "matrix_kernel_4.cuh"
#include "matrix_kernel_5.cuh"
#include "matrix_kernel_6.cuh"
#include "matrix_kernel_7.cuh"
#include "matrix_kernel_8.cuh"
#include "matrix_kernel_9.cuh"
#include "matrix_kernel_10.cuh"

std::vector<size_t> unravel_index(size_t index, const std::vector<size_t>& dims) {
  std::vector<size_t> indices(dims.size());
  for (int i = dims.size() - 1; i >= 0; --i) {
    indices[i] = index % dims[i];
    index /= dims[i];
  }
  return indices;
}

namespace matrix {
  template<typename Type>
  class Matrix {
    private:
      std::vector<size_t> dimensions;
      std::vector<Type> data;
      
      Type* d_data;
      bool device_data_allocated;
      void allocateDeviceMemory();
      void freeDeviceMemory();
      void copyToDevice();
      void copyToHost();
    public:
      size_t numel; 
      Type* getDevicePointer();
      Type* getDevicePointer() const;
      // Constructor
      Matrix<Type>(const std::vector<size_t>& dims);
      Matrix<Type>();
      std::vector<Type>& getData();
      const std::vector<Type>& getData() const;
      std::vector<size_t> getDims() const;
      std::vector<Type> getColumn(size_t col_index) const;
      Matrix<Type> getColumn(size_t start_idx, size_t end_idx) const;
      void setData(const std::vector<Type>& new_data);
      // Print matrix
      void printShape();
      void print();
      // Access matrix elements
      Type& operator()(const std::vector<size_t>& indices);
      const Type& operator()(const std::vector<size_t>& indices) const;
      // Matrix Multiplication
      Matrix<Type> matmul(const Matrix &target) const;
      // Matrix Multiplication using GPU
      Matrix<Type> matmulVer1(const Matrix &target) const;
      // Matmul using shared memory
      Matrix<Type> matmulVer2(const Matrix &target) const;
      // Kernel fusion for unrolling and matrix-multiplication
      Matrix<Type> matmulVer3(const Matrix &target) const;
      // Matmul with shared memory and constant memory      
      Matrix<Type> matmulVer4(const Matrix &target) const;
      // Tuning with restrict and loop unrolling
      Matrix<Type> matmulVer5(const Matrix &target) const;
      // Input channel reduction: tree
      Matrix<Type> matmulVer6(const Matrix &target) const;
      // Input channel reduction: atomic      
      Matrix<Type> matmulVer7(const Matrix &target) const;
      // Strassen matmul
      Matrix<Type> matmulVer8(const Matrix &target) const;
      // Multiple kernel implementations for different layer sizes matmul
      Matrix<Type> matmulVer9(const Matrix &target) const;
      // FP16 matmul
      Matrix<Type> matmulVer10(const Matrix &target) const;
      // Matmmul Elementwise
      Matrix<Type> multiplyElementwiseKernel(const Matrix &target) const;
      Matrix<Type> multiplyElementwise(const Matrix &target) const;
      // Matmul scalar
      Matrix<Type> multiplyScalarKernel(Type scalar);
      Matrix<Type> multiplyScalar(Type scalar);
      // Matrix addition
      Matrix<Type> add(const Matrix &target) const;
      // Matrix addition using GPU
      Matrix<Type> addKernel(const Matrix &target) const;
      Matrix<Type> operator+(Matrix &target);
      // Matrix subtraction
      Matrix operator-() const;
      Matrix<Type> sub(const Matrix &target) const;
      Matrix<Type> operator-(const Matrix &target) const;
      // Matrix transpose
      Matrix<Type> transpose();
      Matrix<Type> transposeKernel();
      Matrix<Type> T();
      // Element-wise function application    
      Matrix<Type> applyFunction(const std::function<Type(const Type &)> &function) const;
      template<typename Func>
      Matrix<Type> applyFunctionKernel(Func function) const;
      // In-place fill with single value
      void fill_(Type val); 
      
      bool all();
      Matrix<Type> sum();
      Matrix<Type> sum(size_t dim);

      void fillRandom(std::function<Type()> generator);
      void fillRandom(Type min, Type max);
      Matrix<Type> repeat(size_t repeats) const;
      // Destructor
      ~Matrix<Type>();
  };


  template<typename Type>
  Matrix<Type>::Matrix(const std::vector<size_t>& dims) : dimensions(dims), 
                                                          d_data(nullptr), 
                                                          device_data_allocated(false) 
  {
    numel = 1;
    for (const auto& dim : dimensions) {
      numel *= dim;
    }
    data.resize(numel, Type());  // init empty vector for data
  }

  template<typename Type>
  Matrix<Type>::Matrix() : numel(0) { 
    dimensions = {};
    data = {};
  }

  template<typename Type>
  std::vector<Type>& Matrix<Type>::getData() {
    return data;
  }

  template<typename Type>
  const std::vector<Type>& Matrix<Type>::getData() const {
    return data;
  }

  template<typename Type>
  std::vector<size_t> Matrix<Type>::getDims() const {
    return dimensions;
  }

  template <typename Type>
  std::vector<Type> Matrix<Type>::getColumn(size_t col_index) const {
    assert(dimensions.size() == 2 && "getColumn only works on 2D matrices.");
    assert(col_index < dimensions[1] && "Column index out of range.");

    std::vector<Type> column(dimensions[0]);
    for (size_t row = 0; row < dimensions[0]; ++row) {
        column[row] = (*this)({row, col_index});
    }
    return column;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::getColumn(size_t start_idx, size_t end_idx) const {
    assert(dimensions.size() == 2);
    assert(start_idx < end_idx && end_idx <= dimensions[1]);

    size_t rows = dimensions[0];
    size_t cols = end_idx - start_idx;
    Matrix<Type> result({rows, cols});

    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = start_idx; j < end_idx; ++j) {
        result({i, j - start_idx}) = (*this)({i, j});
      }
    }
    return result;
  }


  template <typename Type>
  void Matrix<Type>::setData(const std::vector<Type>& new_data) {
    assert(new_data.size() == numel && "Input data size must match the matrix size.");
    data = new_data;
  }

  template<typename Type>
  void Matrix<Type>::printShape() {
    std::cout << "Matrix Size([";
    for (size_t i = 0; i < dimensions.size(); ++i) {
        std::cout << dimensions[i];
        if (i != dimensions.size() - 1) std::cout << ", ";
    }
    std::cout << "])" << std::endl;
  }

  template<typename Type>
  void Matrix<Type>::print() {
    if (dimensions.size() == 2) {
      for (size_t i = 0; i < dimensions[0]; ++i) {
        std::cout << "[";
        for (size_t j = 0; j < dimensions[1]; ++j) {
          std::cout << (*this)({i, j});
          if (j != dimensions[1] - 1) {
            std::cout << ", ";
          }
        }
        std::cout << "]" << std::endl;
      }
    }
    else {
      for (size_t i = 0; i < numel; ++i) {
        std::vector<size_t> indices = unravel_index(i, dimensions);
        for (size_t dim = 0; dim < indices.size(); ++dim) {
          std::cout << (dim == 0 ? "[" : "") << indices[dim] << (dim == indices.size() - 1 ? "]" : ",");
        }
        std::cout << ": " << data[i] << std::endl;
      }
      std::cout << std::endl; 
    }
  }

  template<typename Type>
  Type& Matrix<Type>::operator()(const std::vector<size_t>& indices) {
    assert(indices.size() == dimensions.size() && "Index dimensions do not match");
    size_t flat_index = 0;
    size_t multiplier = 1;
    for (int i = dimensions.size() - 1; i >= 0; --i) {
        assert(indices[i] < dimensions[i] && "Index out of range");
        flat_index += indices[i] * multiplier;
        multiplier *= dimensions[i];
    }
    return data[flat_index];
  }


  template<typename Type>
  const Type& Matrix<Type>::operator()(const std::vector<size_t>& indices) const {
    assert(indices.size() == dimensions.size() && "Index dimensions do not match");
    size_t flat_index = 0;
    size_t multiplier = 1;
    for (int i = dimensions.size() - 1; i >= 0; --i) {
        assert(indices[i] < dimensions[i] && "Index out of range");
        flat_index += indices[i] * multiplier;
        multiplier *= dimensions[i];
    }
    return data[flat_index];
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::matmul(const Matrix<Type> &target) const {
    assert(dimensions.size() == 2 && target.dimensions.size() == 2);
    assert(dimensions[1] == target.dimensions[0] && "Matrix dimensions do not match for multiplication"); // cols(A) == rows(B)

    std::vector<size_t> result_dims = {dimensions[0], target.dimensions[1]};
    Matrix<Type> result(result_dims);

    for (size_t i = 0; i < result_dims[0]; ++i) {
        for (size_t j = 0; j < result_dims[1]; ++j) {
            Type sum = 0;
            for (size_t k = 0; k < dimensions[1]; ++k) {
                sum += (*this)({i, k}) * target({k, j});
            }
            result({i, j}) = sum;
        }
    }
    return result;
  }

  // Matrix Multiplication using GPU
  template<typename Type>
  Matrix<Type> Matrix<Type>::matmulVer1(const Matrix<Type>& target) const {
    assert(dimensions.size() == 2 && target.dimensions.size() == 2);
    assert(dimensions[1] == target.dimensions[0] && "Matrix dimensions do not match for multiplication");

    size_t  M = dimensions[0];
    size_t  K = dimensions[1];
    size_t  N = target.dimensions[1];

    Matrix<Type> result({M, N});

    // Allocate memory on GPU
    Type *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, M * K * sizeof(Type)));
    CHECK(cudaMalloc(&d_B, K * N * sizeof(Type)));
    CHECK(cudaMalloc(&d_C, M * N * sizeof(Type)));

    // Copy data to GPU
    CHECK(cudaMemcpy(d_A, data.data(), M * K * sizeof(Type), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, target.data.data(), K * N * sizeof(Type), cudaMemcpyHostToDevice));

    // Call the kernel
    MatrixKernel<Type> matrix_kernel;
    matrix_kernel.matmulKernel(d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    CHECK(cudaMemcpy(result.data.data(), d_C, M * N * sizeof(Type), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return result;
  } 

  // Matmul using shared memory
  template<typename Type>
  Matrix<Type> Matrix<Type>::matmulVer2(const Matrix<Type>& target) const {
    assert(dimensions.size() == 2 && target.dimensions.size() == 2);
    assert(dimensions[1] == target.dimensions[0] && "Matrix dimensions do not match for multiplication");

    size_t  M = dimensions[0];
    size_t  K = dimensions[1];
    size_t  N = target.dimensions[1];

    Matrix<Type> result({M, N});

    // Allocate memory on GPU
    Type *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, M * K * sizeof(Type)));
    CHECK(cudaMalloc(&d_B, K * N * sizeof(Type)));
    CHECK(cudaMalloc(&d_C, M * N * sizeof(Type)));

    // Copy data to GPU
    CHECK(cudaMemcpy(d_A, data.data(), M * K * sizeof(Type), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, target.data.data(), K * N * sizeof(Type), cudaMemcpyHostToDevice));

    // Call the kernel
    MatrixKernel2<Type> matrix_kernel;
    matrix_kernel.matmulSharedMemKernel(d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    CHECK(cudaMemcpy(result.data.data(), d_C, M * N * sizeof(Type), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return result;
  } 

  // Kernel fusion for unrolling and matrix-multiplication
  template<typename Type>
  Matrix<Type> Matrix<Type>::matmulVer3(const Matrix<Type>& target) const {
    assert(dimensions.size() == 2 && target.dimensions.size() == 2);
    assert(dimensions[1] == target.dimensions[0] && "Matrix dimensions do not match for multiplication");

    size_t  M = dimensions[0];
    size_t  K = dimensions[1];
    size_t  N = target.dimensions[1];

    Matrix<Type> result({M, N});

    // Allocate memory on GPU
    Type *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, M * K * sizeof(Type)));
    CHECK(cudaMalloc(&d_B, K * N * sizeof(Type)));
    CHECK(cudaMalloc(&d_C, M * N * sizeof(Type)));

    // Copy data to GPU
    CHECK(cudaMemcpy(d_A, data.data(), M * K * sizeof(Type), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, target.data.data(), K * N * sizeof(Type), cudaMemcpyHostToDevice));

    // Call the kernel
    MatrixKernel3<Type> matrix_kernel;
    matrix_kernel.fusedMatrixMultiplyKernel(d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    CHECK(cudaMemcpy(result.data.data(), d_C, M * N * sizeof(Type), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return result;
  }

  // Matmul with shared memory and constant memory      
  template<typename Type>
  Matrix<Type> Matrix<Type>::matmulVer4(const Matrix<Type>& target) const {
    assert(dimensions.size() == 2 && target.dimensions.size() == 2);
    assert(dimensions[1] == target.dimensions[0] && "Matrix dimensions do not match for multiplication");

    size_t  M = dimensions[0];
    size_t  K = dimensions[1];
    size_t  N = target.dimensions[1];

    Matrix<Type> result({M, N});

    // Allocate memory on GPU
    Type *d_B, *d_C;
    CHECK(cudaMalloc(&d_B, K * N * sizeof(Type)));
    CHECK(cudaMalloc(&d_C, M * N * sizeof(Type)));

    // Copy data to GPU
    CHECK(cudaMemcpy(d_B, target.data.data(), K * N * sizeof(Type), cudaMemcpyHostToDevice));

    // Call the kernel
    MatrixKernel4<Type> matrix_kernel;
    matrix_kernel.matmulWithConstant(d_B, d_C, M, N, K);

    // Copy result back to host
    CHECK(cudaMemcpy(result.data.data(), d_C, M * N * sizeof(Type), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return result;
  }

  // Tuning with restrict and loop unrolling
  template<typename Type>
  Matrix<Type> Matrix<Type>::matmulVer5(const Matrix<Type>& target) const {
    assert(dimensions.size() == 2 && target.dimensions.size() == 2);
    assert(dimensions[1] == target.dimensions[0] && "Matrix dimensions do not match for multiplication");

    size_t  M = dimensions[0];
    size_t  K = dimensions[1];
    size_t  N = target.dimensions[1];

    Matrix<Type> result({M, N});

    // Allocate memory on GPU
    Type *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, M * K * sizeof(Type)));
    CHECK(cudaMalloc(&d_B, K * N * sizeof(Type)));
    CHECK(cudaMalloc(&d_C, M * N * sizeof(Type)));

    // Copy data to GPU
    CHECK(cudaMemcpy(d_A, data.data(), M * K * sizeof(Type), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, target.data.data(), K * N * sizeof(Type), cudaMemcpyHostToDevice));

    // Call the kernel
    MatrixKernel5<Type> matrix_kernel;
    matrix_kernel.matmulWithRestrictUnRolling(d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    CHECK(cudaMemcpy(result.data.data(), d_C, M * N * sizeof(Type), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return result;
  } 
  
  // Input channel reduction: tree
  template<typename Type>
  Matrix<Type> Matrix<Type>::matmulVer6(const Matrix<Type>& target) const {
    assert(dimensions.size() == 2 && target.dimensions.size() == 2);
    assert(dimensions[1] == target.dimensions[0] && "Matrix dimensions do not match for multiplication");

    size_t  M = dimensions[0];
    size_t  K = dimensions[1];
    size_t  N = target.dimensions[1];

    Matrix<Type> result({M, N});

    // Allocate memory on GPU
    Type *d_B, *d_C;
    CHECK(cudaMalloc(&d_B, K * N * sizeof(Type)));
    CHECK(cudaMalloc(&d_C, M * N * sizeof(Type)));

    // Copy data to GPU
    CHECK(cudaMemcpy(d_B, target.data.data(), K * N * sizeof(Type), cudaMemcpyHostToDevice));

    // Call the kernel
    MatrixKernel6<Type> matrix_kernel;
    matrix_kernel.matmulWithConstantTree(d_B, d_C, M, N, K);

    // Copy result back to host
    CHECK(cudaMemcpy(result.data.data(), d_C, M * N * sizeof(Type), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return result;
  }
  
  // Input channel reduction: atomic
  template<typename Type>
  Matrix<Type> Matrix<Type>::matmulVer7(const Matrix<Type>& target) const {
    assert(dimensions.size() == 2 && target.dimensions.size() == 2);
    assert(dimensions[1] == target.dimensions[0] && "Matrix dimensions do not match for multiplication");

    size_t  M = dimensions[0];
    size_t  K = dimensions[1];
    size_t  N = target.dimensions[1];

    Matrix<Type> result({M, N});

    // Allocate memory on GPU
    Type *d_B, *d_C;
    CHECK(cudaMalloc(&d_B, K * N * sizeof(Type)));
    CHECK(cudaMalloc(&d_C, M * N * sizeof(Type)));

    // Copy data to GPU
    CHECK(cudaMemcpy(d_B, target.data.data(), K * N * sizeof(Type), cudaMemcpyHostToDevice));

    // Call the kernel
    MatrixKernel7<Type> matrix_kernel;
    matrix_kernel.matmulWithConstantAtomic(d_B, d_C, M, N, K);

    // Copy result back to host
    CHECK(cudaMemcpy(result.data.data(), d_C, M * N * sizeof(Type), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return result;
  }

  // Strassen matmul
  template<typename Type>
  Matrix<Type> Matrix<Type>::matmulVer8(const Matrix<Type>& target) const {
    assert(dimensions.size() == 2 && target.dimensions.size() == 2);
    assert(dimensions[1] == target.dimensions[0] && "Matrix dimensions do not match for multiplication");

    size_t  M = dimensions[0];
    size_t  K = dimensions[1];
    size_t  N = target.dimensions[1];

    Matrix<Type> result({M, N});

    // Allocate memory on GPU
    Type *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, M * K * sizeof(Type)));
    CHECK(cudaMalloc(&d_B, K * N * sizeof(Type)));
    CHECK(cudaMalloc(&d_C, M * N * sizeof(Type)));

    // Copy data to GPU
    CHECK(cudaMemcpy(d_A, data.data(), M * K * sizeof(Type), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, target.data.data(), K * N * sizeof(Type), cudaMemcpyHostToDevice));

    // Call the kernel
    MatrixKernel8<Type> matrix_kernel;
    matrix_kernel.strassenMatrixMultiply(d_A, d_B, d_C, M, K, N);

    // Copy result back to host
    CHECK(cudaMemcpy(result.data.data(), d_C, M * N * sizeof(Type), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return result;
  }

  // Multiple kernel implementations for different layer sizes matmul
  template<typename Type>
  Matrix<Type> Matrix<Type>::matmulVer9(const Matrix<Type>& target) const {
    assert(dimensions.size() == 2 && target.dimensions.size() == 2);
    assert(dimensions[1] == target.dimensions[0] && "Matrix dimensions do not match for multiplication");

    size_t M = dimensions[0];
    size_t K = dimensions[1];
    size_t N = target.dimensions[1];

    Matrix<Type> result({M, N});

    Type* d_input = getDevicePointer();
    Type* d_weights = target.getDevicePointer();
    Type* d_output;
    CHECK(cudaMalloc(&d_output, M * N * sizeof(Type)));

    MatrixKernel9<Type> kernel9;
    kernel9.matmul_dense(d_input, d_weights, d_output, M, N, K);

    CHECK(cudaMemcpy(result.data.data(), d_output, M * N * sizeof(Type), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_output));

    return result;
  }

  // FP16 matmul
  template<typename Type>
  Matrix<Type> Matrix<Type>::matmulVer10(const Matrix<Type>& target) const {
    assert(dimensions.size() == 2 && target.dimensions.size() == 2);
    assert(dimensions[1] == target.dimensions[0] && "Matrix dimensions do not match for multiplication");

    size_t  M = dimensions[0];
    size_t  K = dimensions[1];
    size_t  N = target.dimensions[1];

    Matrix<Type> result({M, N});

    // Allocate memory on GPU
    __half* d_A, *d_B, *d_C;;
    CHECK(cudaMalloc(&d_A, M * K * sizeof(__half)));
    CHECK(cudaMalloc(&d_B, K * N * sizeof(__half)));
    CHECK(cudaMalloc(&d_C, M * N * sizeof(__half)));

    MatrixKernel<Type> matrix_kernel;
    __half* temp_A = (__half*)malloc(M * K * sizeof(__half));
    __half* temp_B = (__half*)malloc(K * N * sizeof(__half));
    __half* temp_C = (__half*)malloc(M * N * sizeof(__half));

    matrix_kernel.convertToFP16(data.data(), temp_A, M * K);
    matrix_kernel.convertToFP16(target.getData().data(), temp_B, K * N);

    // Copy data to GPU
    CHECK(cudaMemcpy(d_A, temp_A, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, temp_B, K * N * sizeof(__half), cudaMemcpyHostToDevice));

    // Call the kernel
    MatrixKernel10<__half> matrix_kernel_10;
    matrix_kernel_10.matmulFP16Kernel(d_A, d_B, d_C, M, K, N);

    // Copy result back to host
    CHECK(cudaMemcpy(temp_C, d_C, M * N * sizeof(__half), cudaMemcpyDeviceToHost));
    matrix_kernel.convertFromFP16(temp_C, result.data.data(), M * N);

    // Free GPU memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    free(temp_A);
    free(temp_B);
    free(temp_C);
    
    return result;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::multiplyElementwise(const Matrix<Type> &target) const {
    assert(dimensions == target.dimensions);
    Matrix<Type> result(dimensions);
    for (size_t i = 0; i < numel; ++i) {
        result.data[i] = data[i] * target.data[i];
    }
    return result;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::multiplyElementwiseKernel(const Matrix<Type> &target) const {
    assert(dimensions == target.dimensions);
    Matrix<Type> result(dimensions);

    size_t numElements = numel;
    Type *d_A, *d_B, *d_C;

    CHECK(cudaMalloc(&d_A, numElements * sizeof(Type)));
    CHECK(cudaMalloc(&d_B, numElements * sizeof(Type)));
    CHECK(cudaMalloc(&d_C, numElements * sizeof(Type)));
    
    // Copy data to GPU
    CHECK(cudaMemcpy(d_A, data.data(), numElements * sizeof(Type), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, target.data.data(), numElements * sizeof(Type), cudaMemcpyHostToDevice));

    // Call the kernel
    MatrixKernel<Type> matrix_kernel;
    matrix_kernel.multiplyElementwiseKernel(d_A, d_B, d_C, numElements);
    
    // Copy result back to host
    CHECK(cudaMemcpy(result.data.data(), d_C, numElements * sizeof(Type), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return result;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::multiplyScalarKernel(Type scalar) {
    Matrix<Type> result(dimensions);
    
    size_t  M = dimensions[0];
    size_t  N = dimensions[1];
    size_t numElements = numel;
    
    Type *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, numElements * sizeof(Type)));
    CHECK(cudaMalloc(&d_output, numElements * sizeof(Type)));

    CHECK(cudaMemcpy(d_input, data.data(), numElements * sizeof(Type), cudaMemcpyHostToDevice));
    
    // Call the kernel
    MatrixKernel<Type> matrix_kernel;
    matrix_kernel.multiplyScalarKernel(d_input, d_output, scalar, M, N);

    CHECK(cudaMemcpy(result.data.data(), d_output, numElements * sizeof(Type), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));

    return result;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::multiplyScalar(Type scalar) {
    Matrix<Type> result(dimensions);
    for (size_t i = 0; i < numel; ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::add(const Matrix<Type> &target) const {
    assert(dimensions == target.dimensions);
    Matrix<Type> result(dimensions);
    for (size_t i = 0; i < numel; ++i) {
        result.data[i] = data[i] + target.data[i];
    }
    return result;
  }

  // Matrix addition using GPU
  template<typename Type>
  Matrix<Type> Matrix<Type>::addKernel(const Matrix<Type> &target) const {
    assert(dimensions == target.dimensions);
    
    size_t  M = dimensions[0];
    size_t  N = dimensions[1];
    size_t numElements = numel;
    Matrix<Type> result(dimensions);
    Type *d_A, *d_B, *d_C;
    
    cudaMalloc(&d_A, numElements * sizeof(Type));
    cudaMalloc(&d_B, numElements * sizeof(Type));
    cudaMalloc(&d_C, numElements * sizeof(Type));
    
    CHECK(cudaMemcpy(d_A, data.data(), numElements * sizeof(Type), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, target.data.data(), numElements * sizeof(Type), cudaMemcpyHostToDevice));
    
    // Call the kernel
    MatrixKernel<Type> matrix_kernel;
    matrix_kernel.addKernel(d_A, d_B, d_C, M, N);

    // Copy result back to host
    CHECK(cudaMemcpy(result.data.data(), d_C, numElements * sizeof(Type), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return result;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::operator+(Matrix<Type> &target) {
    return add(target);
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::operator-() const {
    Matrix<Type> result(dimensions);
    for (size_t i = 0; i < numel; ++i) {
        result.data[i] = -data[i];
    }
    return result;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::sub(const Matrix<Type> &target) const {
    Matrix<Type> neg_target = -target;
    return add(neg_target);
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::operator-(const Matrix<Type> &target) const {
    return sub(target);
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::transpose() {
    assert(dimensions.size() == 2);
    std::vector<size_t> transposed_dims = {dimensions[1], dimensions[0]};
    Matrix<Type> result(transposed_dims);

    for (size_t i = 0; i < dimensions[0]; ++i) {
        for (size_t j = 0; j < dimensions[1]; ++j) {
            result({j, i}) = (*this)({i, j});
        }
    }
    return result;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::transposeKernel() {
    assert(dimensions.size() == 2);
    
    size_t rows = dimensions[0];
    size_t cols = dimensions[1];
    std::vector<size_t> transposed_dims = {cols, rows};
    
    Matrix<Type> result(transposed_dims);
    
    Type *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, numel * sizeof(Type)));
    CHECK(cudaMalloc(&d_output, numel * sizeof(Type)));

    CHECK(cudaMemcpy(d_input, data.data(), numel * sizeof(Type), cudaMemcpyHostToDevice));
    
    MatrixKernel<Type> matrix_kernel;
    matrix_kernel.transposeKernel(d_input, d_output, rows, cols);

    CHECK(cudaMemcpy(result.data.data(), d_output, numel * sizeof(Type), cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));

    return result;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::T(){ // Similar to numpy
    return transpose(); 
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::applyFunction(const std::function<Type(const Type &)> &function) const {
    Matrix<Type> result(dimensions);
    for (size_t i = 0; i < numel; ++i) {
        result.data[i] = function(data[i]);
    }
    return result;
  }

  template<typename Type>
  template<typename Func>
  Matrix<Type> Matrix<Type>::applyFunctionKernel(Func function) const {
    Matrix<Type> result(dimensions);

    size_t numElements = numel;
    Type *d_input, *d_output;

    CHECK(cudaMalloc(&d_input, numElements * sizeof(Type)));
    CHECK(cudaMalloc(&d_output, numElements * sizeof(Type)));

    CHECK(cudaMemcpy(d_input, data.data(), numElements * sizeof(Type), cudaMemcpyHostToDevice));

    MatrixKernel<Type> matrix_kernel;
    matrix_kernel.applyFunctionKernel(d_input, d_output, numElements, function);
    
    CHECK(cudaMemcpy(result.data.data(), d_output, numElements * sizeof(Type), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));

    return result;
  }

  template<typename Type>
  void Matrix<Type>::fill_(Type val) {
    std::fill(data.begin(), data.end(), val);
  }

  template<typename Type>
  bool Matrix<Type>::all() {
    return std::all_of(data.begin(), data.end(), [](Type x) { return x != 0; });
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::sum() {
    Type total = std::accumulate(data.begin(), data.end(), Type(0));
    Matrix<Type> result({1});
    result.data[0] = total;
    return result;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::sum(size_t dim) {
    assert(dim < dimensions.size());

    std::vector<size_t> new_dims = dimensions;
    new_dims[dim] = 1;
    Matrix<Type> result(new_dims);

    for (size_t i = 0; i < numel; ++i) {
        std::vector<size_t> indices = unravel_index(i, dimensions);
        indices[dim] = 0;
        result(indices) += data[i];
    }
    return result;
  }

  template<typename Type>
  void Matrix<Type>::fillRandom(std::function<Type()> generator) {
    for (auto& elem : data) {
      elem = generator();
    }
  }

  template<typename Type>
  void Matrix<Type>::fillRandom(Type min, Type max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Type> dist(min, max);

    for (auto& elem : data) {
        elem = dist(gen); 
    }
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::repeat(size_t repeats) const {
    size_t rows = dimensions[0];
    size_t cols = dimensions[1];

    assert(cols == 1 && "Repeat only works on matrices with a single column.");

    std::vector<size_t> new_dims = {rows, repeats};
    Matrix<Type> result(new_dims);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < repeats; ++j) {
            result({i, j}) = data[i];
        }
    }
    return result;
  }

  template<typename Type>
  void Matrix<Type>::allocateDeviceMemory() {
    if (!device_data_allocated) {
      CHECK(cudaMalloc(&d_data, numel * sizeof(Type)));
      device_data_allocated = true;
    }
  }

  template<typename Type>
  void Matrix<Type>::freeDeviceMemory() {
    if (device_data_allocated) {
        CHECK(cudaFree(d_data));
        device_data_allocated = false;
        d_data = nullptr;
    }
  }

  template<typename Type>
  void Matrix<Type>::copyToDevice() {
    allocateDeviceMemory();
    CHECK(cudaMemcpy(d_data, data.data(), numel * sizeof(Type), cudaMemcpyHostToDevice));
  }

  template<typename Type>
  void Matrix<Type>::copyToHost() {
    if (device_data_allocated) {
      CHECK(cudaMemcpy(data.data(), d_data, numel * sizeof(Type), cudaMemcpyDeviceToHost));
    }
  }

  template<typename Type>
  Type* Matrix<Type>::getDevicePointer() {
    if (!device_data_allocated) {
      copyToDevice();
    }
    return d_data;
  }

  template<typename Type>
  Type* Matrix<Type>::getDevicePointer() const {
    if (!device_data_allocated) {
      const_cast<Matrix<Type>*>(this)->copyToDevice();
    }
    
    return d_data;
  }

  template <typename Type>
  Matrix<Type>::~Matrix() {
    freeDeviceMemory();
  }


  template<typename T>
  struct mtx {
    static Matrix<T> zeros(const std::vector<size_t>& dims) {
        Matrix<T> M{dims};
        M.fill_(T(0));
        return M;
    }

    static Matrix<T> ones(const std::vector<size_t>& dims) {
        Matrix<T> M{dims};
        M.fill_(T(1));
        return M;
    }

    static Matrix<T> randn(const std::vector<size_t>& dims) {
        Matrix<T> M{dims};

        std::random_device rd{};
        std::mt19937 gen{rd()};
        T n(M.numel);
        T stdev{1 / sqrt(n)};
        std::normal_distribution<T> d{0, stdev};

        for (size_t i = 0; i < M.numel; ++i) {
            M.getData()[i] = d(gen);
        }
        return M;
    }

    static Matrix<T> rand(const std::vector<size_t>& dims) {
        Matrix<T> M{dims};

        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<T> d{0, 1};

        for (size_t i = 0; i < M.numel; ++i) {
            M.getData()[i] = d(gen);
        }
        return M;
    }
  };
}
#endif // MATRIX_CUH