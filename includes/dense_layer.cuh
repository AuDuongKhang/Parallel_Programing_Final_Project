#include <cstdint>
#include <cuda_runtime.h>
#include <vector>


class DenseLayer {
public:
    DenseLayer(int input_size, int output_size, bool is_output_layer);
    ~DenseLayer();
    const float* Forward(const float* input);

private:
    int input_size;
    int output_size;
    bool is_output_layer;
    float *weights;
    float *biases;
    float *d_weights;
    float *d_biases;
    float *output;
};