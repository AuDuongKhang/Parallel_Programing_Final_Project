#include "../includes/load_mnist.cuh"
#include "../includes/dense_layer.cuh"
#include "../includes/gpu-support.cuh"
#include <iostream>
#include <cmath>

int ComputeAccuracy(const float* predictions, const std::vector<uint8_t>& labels, int num_samples, int num_classes) {
    int correct = 0;
    for (int i = 0; i < num_samples; ++i) {
        int predicted_label = 0;
        float max_value = predictions[i * num_classes];

        for (int j = 1; j < num_classes; ++j) {
            if (predictions[i * num_classes + j] > max_value) {
                max_value = predictions[i * num_classes + j];
                predicted_label = j;
            }
        }

        if (predicted_label == labels[i]) {
            correct++;
        }
    }
    return correct;
}

int main() {
    MNISTDataSet train_data, test_data;
    train_data.ReadMNISTData("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte");
    test_data.ReadMNISTData("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte");

    const auto& train_images = train_data.GetImages();
    const auto& train_labels = train_data.GetLabels();
    const auto& test_images = test_data.GetImages();
    const auto& test_labels = test_data.GetLabels();

    const int input_size = 784;
    const int hidden_size = 128;
    const int output_size = 10;
    const int batch_size = 64;

    DenseLayer dense1(input_size, hidden_size, false);  // Hidden Layer 1 (ReLU)
    DenseLayer dense2(hidden_size, hidden_size, false); // Hidden Layer 2 (ReLU)
    DenseLayer dense3(hidden_size, output_size, true);  // Output Layer (Softmax)

    float* d_input;
    CHECK(cudaMalloc(&d_input, batch_size * input_size * sizeof(float)));

    std::vector<float> predictions(batch_size * output_size);
    for (size_t i = 0; i < train_images.size(); i += batch_size) {
        size_t current_batch_size = std::min(batch_size, (int)(train_images.size() - i));

        // Copy batch input lên GPU
        CHECK(cudaMemcpy(d_input, train_images[i].data(), current_batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));

        // Forward propagation qua các lớp
        const float* output1 = dense1.Forward(d_input);
        const float* output2 = dense2.Forward(output1);
        const float* output3 = dense3.Forward(output2); // Softmax

        // Copy output về host để tính toán
        CHECK(cudaMemcpy(predictions.data(), output3, current_batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Đánh giá accuracy cho batch hiện tại
        int correct = ComputeAccuracy(predictions.data(), train_labels, current_batch_size, output_size);
        std::cout << "Train Accuracy (Batch " << i / batch_size + 1 << "): " 
                  << (float)correct / current_batch_size * 100.0f << "%" << std::endl;
    }

    std::cout << "Evaluating on Test Set..." << std::endl;
    float* d_test_input;
    cudaMalloc(&d_test_input, batch_size * input_size * sizeof(float));
    int total_correct = 0;
    int total_samples = test_images.size();

    for (size_t i = 0; i < test_images.size(); i += batch_size) {
        size_t current_batch_size = std::min(batch_size, (int)(test_images.size() - i));

        // Copy batch test input lên GPU
        CHECK(cudaMemcpy(d_test_input, test_images[i].data(), current_batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));

        // Forward propagation
        const float* output1 = dense1.Forward(d_test_input);
        const float* output2 = dense2.Forward(output1);
        const float* output3 = dense3.Forward(output2); // Softmax

        // Copy output về host
        CHECK(cudaMemcpy(predictions.data(), output3, current_batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Tính accuracy
        total_correct += ComputeAccuracy(predictions.data(), test_labels, current_batch_size, output_size);
    }

    std::cout << "Final Test Accuracy: " 
              << (float)total_correct / total_samples * 100.0f << "%" << std::endl;

    // Giải phóng bộ nhớ
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_test_input));

    return 0;
}
