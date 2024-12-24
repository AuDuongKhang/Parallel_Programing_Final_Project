#include "../includes/neural_network.cuh"
#include "../includes/load_mnist.cuh"
#include "../includes/gpu_support.cuh"
#include <cmath>
#include <algorithm>
#include <random>

using namespace matrix;
using namespace nn;

void test_matrix() {
  std::vector<size_t> dims = {2, 2}; 
  auto M = matrix::mtx<float>::randn(dims); 

  M.printShape();
  M.print(); // print the OG matrix

  (M-M).print();  // print M minus itself

  (M+M).print();  // print its sum
  (M.multiplyScalar(2.f)).print();  // print 2x itself

  (M.multiplyElementwise(M)).print(); // mult M w itself

  auto MT = M.T(); // transpose the matrix
  MT.print();
  (MT.matmul(M)).print();  // form symm. pos. def. matrix

  (M.applyFunction([](auto x){return x-x;} )).print(); // apply fun
}

float crossEntropyLoss(const matrix::Matrix<float>& predictions, const matrix::Matrix<float>& targets) {
  const auto& pred_data = predictions.getData();
  const auto& target_data = targets.getData();
  float loss = 0.0f;

  for (size_t i = 0; i < pred_data.size(); ++i) {
    loss -= target_data[i] * std::log(pred_data[i] + 1e-7);
  }
  return loss / predictions.getDims()[1];
}

matrix::Matrix<float> normalizeData(const std::vector<std::vector<float>>& data, size_t rows, size_t cols) {
  matrix::Matrix<float> normalized({rows, cols});
  std::vector<float> flattened_data;
  
  for (size_t i = 0; i < cols; ++i) {
    flattened_data.insert(flattened_data.end(), data[i].begin(), data[i].end());
  }

  assert(flattened_data.size() == rows * cols && "Input data size must match the matrix size.");
  std::transform(flattened_data.begin(), flattened_data.end(), flattened_data.begin(),
                   [](float x) { return x / 255.0f; });
  normalized.setData(flattened_data);
  return normalized;
}

matrix::Matrix<float> oneHotEncode(const std::vector<uint8_t>& labels, size_t num_classes) {
    matrix::Matrix<float> one_hot({num_classes, labels.size()});
    for (size_t i = 0; i < labels.size(); ++i) {
        one_hot({labels[i], i}) = 1.0f;
    }
    return one_hot;
}

float calculateAccuracy(const matrix::Matrix<float>& predictions, const matrix::Matrix<float>& targets) {
  const auto& pred_data = predictions.getData();
  const auto& target_data = targets.getData();

  size_t num_samples = predictions.getDims()[1];
  size_t num_classes = predictions.getDims()[0];
  size_t correct_predictions = 0;

  for (size_t sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
    // Find index has highest probability in predictions
    auto pred_start = pred_data.begin() + sample_idx * num_classes;
    auto pred_end = pred_start + num_classes;
    size_t pred_class = std::distance(pred_start, std::max_element(pred_start, pred_end));

    // Find correct index in target
    auto target_start = target_data.begin() + sample_idx * num_classes;
    auto target_end = target_start + num_classes;
    size_t target_class = std::distance(target_start, std::max_element(target_start, target_end));

    if (pred_class == target_class) {
      ++correct_predictions;
    }
  }

  return static_cast<float>(correct_predictions) / num_samples;
}

// Shuffle function
void shuffleData(matrix::Matrix<float>& input, matrix::Matrix<float>& target) {
    size_t num_samples = input.getDims()[1]; // Số lượng cột
    std::vector<size_t> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0); // Tạo dãy [0, 1, ..., num_samples - 1]

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    matrix::Matrix<float> shuffled_input(input.getDims());
    matrix::Matrix<float> shuffled_target(target.getDims());

    for (size_t i = 0; i < num_samples; ++i) {
        shuffled_input.setColumn(i, input.getColumn(indices[i]));
        shuffled_target.setColumn(i, target.getColumn(indices[i]));
    }

    input = shuffled_input;
    target = shuffled_target;
}


void test_mlp(std::vector<std::vector<float>> trainingData,
              matrix::Matrix<float> train_input, 
              matrix::Matrix<float> train_target, 
              matrix::Matrix<float> val_input, 
              matrix::Matrix<float> val_target,
              size_t num_epochs,
              size_t batch_size,
              float lr,
              bool useDevice,
              int mode) 
{
  GpuTimer timer;

  nn::MLP<float> mlp({784, 128, 128, 10}, lr);
  size_t num_samples = train_input.getDims()[1];
  size_t num_batches = (num_samples + batch_size - 1) / batch_size;

  timer.Start();
  for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
    shuffleData(train_input, train_target);
	float epoch_loss = 0.0f;
    size_t correct_train = 0;

    std::cout << "Epoch " << epoch << std::endl;

    for (size_t batch = 0; batch < num_batches; ++batch) {
      size_t start_idx = batch * batch_size;
      size_t end_idx = std::min(start_idx + batch_size, num_samples);

      matrix::Matrix<float> batch_input = train_input.getColumn(start_idx, end_idx);
      matrix::Matrix<float> batch_target = train_target.getColumn(start_idx, end_idx);

      // Forward pass
      auto output = mlp(batch_input, useDevice, mode);

      // Compute loss
      epoch_loss += crossEntropyLoss(output, batch_target);

      // Compute accuracy
      correct_train += calculateAccuracy(output, batch_target) * batch_size;

      // Backpropagation
      mlp.backprop(batch_target, useDevice, mode);
    }

    // Validation
    auto val_output = mlp(val_input, useDevice, mode);
    float val_accuracy = calculateAccuracy(val_output, val_target) * 100;

    std::cout << "Epoch Loss: " << epoch_loss / num_batches << std::endl;
    std::cout << "Training Accuracy: " << (float(correct_train) / trainingData.size()) * 100 << "%" << std::endl;
    std::cout << "Validation Accuracy: " << val_accuracy << "%" << std::endl;
  }

  timer.Stop();

  float time = timer.Elapsed() * 0.001;
	printf("Processing time (%s): %f s\n\n", 
			useDevice == true? "use device" : "use host", time);
}

int main(int argc, char ** argv){
  //test_matrix();
  GPU_Support gpu_support;
  gpu_support.printDeviceInfo();
  
  // Initial hyperparameter
  size_t num_epochs = 20;
  size_t batch_size = 64;
  float lr = 0.01;
  int mode = 0; // cpu
  
  if (argc == 2) {
    mode = atoi(argv[1]);
  }

  if (argc == 3) {
    mode = atoi(argv[1]);
    num_epochs = atoi(argv[2]);
  }

  if (argc == 4) {
    mode = atoi(argv[1]);
    num_epochs = atoi(argv[2]);
    batch_size = atoi(argv[3]);
  }

  if (argc == 5) {
    mode = atoi(argv[1]);
    num_epochs = atoi(argv[2]);
    batch_size = atoi(argv[3]);
    lr = atoi(argv[4]);
  }

  // Training data
  MNISTDataSet mnist_trainind_data;
  mnist_trainind_data.ReadMNISTData("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte");
  const std::vector<std::vector<float>>& trainingData = mnist_trainind_data.GetImages();
  const std::vector<uint8_t>& trainingLabels = mnist_trainind_data.GetLabels();
  
  // Validation data
  MNISTDataSet mnist_validation_data;
  mnist_validation_data.ReadMNISTData("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte");
  const std::vector<std::vector<float>>& validationData = mnist_validation_data.GetImages();
  const std::vector<uint8_t>& validationLabels = mnist_validation_data.GetLabels();
  
  // Normalize data
  matrix::Matrix<float> train_input = normalizeData(trainingData, 784, batch_size);
  matrix::Matrix<float> train_target = oneHotEncode(trainingLabels, 10);
  matrix::Matrix<float> val_input = normalizeData(validationData, 784, batch_size);
  matrix::Matrix<float> val_target = oneHotEncode(validationLabels, 10);
  
  if (mode == 9 || mode < 0 || mode > 10) {
    std::cout<< "Feature is not available! Try another mode" << std::endl;
    return;
  }
  // Use CPU
  if (mode == 0) {
    test_mlp(trainingData,
            train_input, 
            train_target, 
            val_input, 
            val_target, 
            num_epochs, 
            batch_size, 
            lr,
            false,
            mode);
  }
  // Use GPU
  else {
    test_mlp(trainingData,
            train_input, 
            train_target, 
            val_input, 
            val_target, 
            num_epochs, 
            batch_size, 
            lr, 
            true,
            mode);
  }
  
  return 0;
}