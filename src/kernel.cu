#include <iostream>
#include "../src/custom/gpu-support.h"

__global__ void hello_cuda_kernel() {
    printf("Hello from Device!\n");
}

extern "C" void hello_cuda() {
    hello_cuda_kernel<<<1, 1>>>();
    CHECK(cudaGetLastError());
}
