#include <iostream>
#include "../src/custom/gpu-support.h"

extern "C" {
  void hello_cuda();  
}

int main() {
    GPU_Support gpu_support;
    gpu_support.printDeviceInfo(); 
    
    printf("Hello from Host!\n");

    // Launch the CUDA kernel
    hello_cuda();

    // Synchronize device
    CHECK(cudaDeviceSynchronize());

    return 0;
}
