#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <tuple>
#include <functional>
#include <random>

namespace matrix {
  template<typename Type>
  class Matrix {
    private:
      size_t cols;
      size_t rows;
      std::vector<Type> data;
    public:
      std::tuple<size_t, size_t> shape;
      int numel;
      // Constructor
      Matrix<Type>(size_t rows, size_t cols);
      Matrix<Type>();
      // Print matrix
      void printShape();
      void print();
      // Access matrix elements
      Type& operator()(size_t row, size_t col);
      // Matrix Multiplication
      Matrix<Type> matmul(Matrix &target);
      Matrix<Type> multiplyElementwise(Matrix &target);
      Matrix<Type> square();
      Matrix<Type> multiplyScalar(Type scalar);
      // Matrix addition
      Matrix<Type> add(Matrix &target);
      Matrix<Type> operator+(Matrix &target);
      // Matrix subtraction
      Matrix operator-();
      Matrix<Type> sub(Matrix &target);
      Matrix<Type> operator-(Matrix &target);
      // Matrix transpose
      Matrix<Type> transpose();
      Matrix<Type> T();
      // Element-wise function application    
      Matrix<Type> applyFunction(const std::function<Type(const Type &)> &function);

      // In-place fill with single value
      void fill_(Type val); 
      Matrix<ushort> operator==(Matrix &target);
      Matrix<ushort> operator!=(Matrix &target);
      bool all();
      Matrix<Type> sum();
      Matrix<Type> sum(size_t dim);
      Matrix<Type> mean();
      Matrix<Type> mean(size_t dim);
      Matrix<Type> cat(Matrix target, size_t dim);
      Matrix<Type> diag();
      // Destructor
      ~Matrix<Type>();
  };


  template<typename Type>
  Matrix<Type>::Matrix(size_t rows, size_t cols) : cols(cols), rows(rows), numel(rows * cols) {
    data.resize(rows * cols, Type());  // init empty vector for data
    shape = {rows, cols};
  }

  template<typename Type>
  Matrix<Type>::Matrix() : cols(0), rows(0), numel(0), data({}) { 
    shape = {rows, cols};
  }

  template<typename Type>
  void Matrix<Type>::printShape() {
    std::cout << "Matrix Size([" << rows << ", " << cols << "])" << std::endl;
  }

  template<typename Type>
  void Matrix<Type>::print() {
    for (size_t r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        std::cout << (*this)(r, c) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  template<typename Type>
  Type& Matrix<Type>::operator()(size_t row, size_t col) {
      return data[row * cols + col];
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::matmul(Matrix<Type> &target) {
    assert(cols == target.rows);
    Matrix<Type> output(rows, target.cols);

    for (size_t r = 0; r < output.rows; ++r) {
      for (size_t c = 0; c < output.cols; ++c) {
        for (size_t k = 0; k < target.rows; ++k)
          output(r, c) += (*this)(r, k) * target(k, c);
      }
    }

    return output;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::multiplyElementwise(Matrix<Type> &target){
    assert(shape == target.shape);
    Matrix<Type> output((*this));
    for (size_t r = 0; r < output.rows; ++r) {
      for (size_t c = 0; c < output.cols; ++c) {
        output(r, c) = target(r,c) * (*this)(r, c);
      }
    }
    return output;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::square() { 
    Matrix<Type> output((*this));
    output = multiplyElementwise(output);
    return output;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::multiplyScalar(Type scalar) {
    Matrix<Type> output((*this));
    for (size_t r = 0; r < output.rows; ++r) {
      for (size_t c = 0; c < output.cols; ++c) {
        output(r, c) = scalar * (*this)(r, c);
      }
    }
    return output;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::add(Matrix<Type> &target) {
    assert(shape == target.shape);
    Matrix<Type> output(rows, target.cols);

    for (size_t r = 0; r < output.rows; ++r) {
      for (size_t c = 0; c < output.cols; ++c) {
        output(r, c) = (*this)(r, c) + target(r, c);
      }
    }
    return output;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::operator+(Matrix<Type> &target) {
    return add(target);
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::operator-() {
    Matrix<Type> output(rows, cols);
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        output(r, c) = -(*this)(r, c);
      }
    }
    return output;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::sub(Matrix<Type> &target) {
    Matrix<Type> neg_target = -target;
    return add(neg_target);
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::operator-(Matrix<Type> &target) {
    return sub(target);
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::transpose() {
    size_t new_rows{cols}, new_cols{rows};
    Matrix<Type> transposed(new_rows, new_cols);

    for (size_t r = 0; r < new_rows; ++r) {
      for (size_t c = 0; c < new_cols; ++c) {
        transposed(r, c) = (*this)(c, r);  // swap row and col
      }
    }
    return transposed;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::T(){ // Similar to numpy
    return transpose(); 
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::applyFunction(const std::function<Type(const Type &)> &function) {
    Matrix<Type> output((*this));
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        output(r, c) = function((*this)(r, c));
      }
    }
    return output;
  }

  template<typename Type>
  void Matrix<Type>::fill_(Type val) {
    for (size_t r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        (*this)(r, c) = val;
      }
    }
  }

  template<typename Type>
  Matrix<ushort> Matrix<Type>::operator==(Matrix<Type> &target) {
    assert(shape == target.shape);
    Matrix<ushort> output(rows, cols);

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        if ((*this)(r, c) - target(r, c) == 0.)
          output(r, c) = 1;
        else
          output(r, c) = 0;
      }
    }
    return output;
  }

  template<typename Type>
  Matrix<ushort> Matrix<Type>::operator!=(Matrix<Type> &target){
    return !(*this) == target;
  }

  template<typename Type>
  bool Matrix<Type>::all() {
    int counter{0};
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        if ((*this)(r, c))
          counter++;
      }
    }
    return (counter == numel);
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::sum() {
    Matrix<Type> output{1, 1};
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        output(0, 0) += (*this)(r, c);
      }
    }
    return output;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::sum(size_t dim) {
    assert (dim >= 0 && dim < 2);
    auto output = (dim == 0) ? Matrix<Type>{1, cols} : Matrix<Type>{rows, 1};

    if (dim == 0) {
      for (size_t c = 0; c < cols; ++c)
        for (size_t r = 0; r < rows; ++r)
          output(0, c) += (*this)(r, c);
    }
    else {
      for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
          output(r, 0) += (*this)(r, c);
    }
    return output;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::mean() {
    auto n = Type(numel);
    return sum().multiply_scalar(1 / n);
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::mean(size_t dim) {
    auto n = (dim == 0) ? Type(rows) : Type(cols);
    return sum().multiply_scalar(1 / n);
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::cat(Matrix<Type> target, size_t dim) {
    (dim == 0) ? assert(rows == target.rows) : assert(cols == target.cols);
    auto output = (dim == 0) ? Matrix<Type>{rows + target.rows, cols} : Matrix<Type>{rows, cols + target.cols};

    // copy self
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < cols; ++c)
        output(r, c) = (*this)(r, c);

    // copy target
    if (dim == 0) {
      for (size_t r = 0; r < target.rows; ++r)
        for (size_t c = 0; c < cols; ++c)
          output(r + rows, c) = target(r, c);
    } 
    else {
      for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < target.cols; ++c)
          output(r, c + cols) = target(r, c);
    }
    return output;
  }

  template<typename Type>
  Matrix<Type> Matrix<Type>::diag() {
    assert((rows == 1 || cols == 1) || (rows == cols));
    if (rows == 1 || cols == 1) {
      Matrix<Type> output{std::max(rows, cols), std::max(rows, cols)};
      for (size_t i = 0; i < rows; ++i)
        output(i, i) = (*this)(i, 0);
      return output;
    } 
    else {
      assert(rows == cols);
      Matrix<Type> output{rows, 1};
      for (size_t i = 0; i < rows; ++i)
        output(i, 0) = (*this)(i, i);
      return output;
    }
  }

  template <typename Type>
  Matrix<Type>::~Matrix() {}


  template<typename T>
  struct mtx {
    static Matrix<T> zeros(size_t rows, size_t cols) {
      Matrix<T> M{rows, cols};
      M.fill_(T(0));
      return M;
    }

    static Matrix<T> ones(size_t rows, size_t cols) {
      Matrix<T> M{rows, cols};
      M.fill_(T(1));
      return M;
    }

    static Matrix<T> randn(size_t rows, size_t cols) {
      Matrix<T> M{rows, cols};

      std::random_device rd{};
      std::mt19937 gen{rd()};
      T n(M.numel);
      T stdev{1 / sqrt(n)};
      std::normal_distribution<T> d{0, stdev};

      for (size_t r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          M(r, c) = d(gen);
        }
      }
      return M;
    }

    static Matrix<T> rand(size_t rows, size_t cols) {
      Matrix<T> M{rows, cols};

      std::random_device rd{};
      std::mt19937 gen{rd()};
      std::uniform_real_distribution<T> d{0, 1};

      for (size_t r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          M(r, c) = d(gen);
        }
      }
      return M;
    }
  };
}
