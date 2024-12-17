#include "../includes/load_mnist.cuh"
#include <iostream>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: ./load_mnist <path_to_mnist_data>" << std::endl;
        return -1;
    }

    auto data_path = std::filesystem::path(argv[1]);

    MNISTDataSet mnist_training_data;

    try {
        // Load training data
        mnist_training_data.ReadMNISTData(data_path / "train-images-idx3-ubyte", data_path / "train-labels-idx1-ubyte");
        std::cout << "Training data loaded successfully!" << std::endl;

        // Print the number of images and labels
        std::cout << "Number of training images: " << mnist_training_data.GetImages().size() << std::endl;
        std::cout << "Number of training labels: " << mnist_training_data.GetLabels().size() << std::endl;

        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
