
cmake_minimum_required(VERSION 3.10)
project(MyCudaProject)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set CUDA flags (optional optimization flags)
set(CUDA_NVCC_FLAGS "-O3 -gencode arch=compute_75,code=sm_75")  # Set sm_75 for Tesla T4 or similar GPUs

# Include CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})

# Add the source files and handle CUDA source files with nvcc
cuda_add_executable(Project src/main.cu src/activation.cu src/load_mnist.cu src/dense_layer.cu src/kernel.cu src/gpu-support.cu)

# Link CUDA libraries
target_link_libraries(Project ${CUDA_LIBRARIES})
