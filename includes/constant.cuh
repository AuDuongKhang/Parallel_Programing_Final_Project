#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include <utility>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <tuple>
#include <functional>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 

#include "gpu_support.cuh"

#define MAX_CONSTANT_WEIGHTS 16384 
__constant__ float d_constant_weights[MAX_CONSTANT_WEIGHTS];
 
#endif // CONSTANTS_CUH
