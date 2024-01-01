#include <iostream>
#include <lexocraft/neural_network/neural_network.hpp>

int main() {
    std::vector<std::size_t> layer_sizes {3, 4, 4, 3};
    std::cout << "layer_sizes.size(): " << layer_sizes.size() << '\n';
    lc::NeuralNetwork neural_network (layer_sizes);

    std::cout << "Neural network: " << neural_network << '\n';
}
