#include "../includes/matrix.cuh"
#include "../includes/neural_network_cpu.cuh"

auto make_model(size_t in_channels,
                          size_t out_channels,
                          size_t hidden_units_per_layer,
                          int hidden_layers,
                          float lr) {
  std::vector<size_t> units_per_layer;

  units_per_layer.push_back(in_channels);

  for (int i = 0; i < hidden_layers; ++i)
    units_per_layer.push_back(hidden_units_per_layer);

  units_per_layer.push_back(out_channels);

  nn::MLP<float> model(units_per_layer, 0.01f);
  return model;
}

void test_matrix() {
  auto M = mtx<float>::randn(2, 2); // init randn matrix

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

int main(){
  test_matrix();
  auto model = make_model(
      in_channels=1,
      out_channels=1,
      hidden_units_per_layer=8,
      hidden_layers=3,
      lr=.5f);
  return 0;
}