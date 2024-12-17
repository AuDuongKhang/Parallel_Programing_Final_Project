struct Activation {
    virtual ~Activation() = default;
    virtual void operator()(const int batch_size,
                            const int output_size,
                            float* d_value) = 0;
};

struct ReLU : public Activation {
    void operator()(const int batch_size,
                    const int output_size,
                    float* d_value);
};

struct SoftMax : public Activation {
    void operator()(const int batch_size,
                    const int output_size,
                    float* d_value);
};

__global__ void ReLuKernel(float* input, const int output_size);
__global__ void SoftmaxKernel(const int output_size, const int batch_size, float* values);