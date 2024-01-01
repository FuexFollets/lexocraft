#include <iostream>
#include <lexocraft/neural_network/neural_network.hpp>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    // Test neural network serialization

    const std::string filename {*std::next(argv, 1)};

    if (argc > 2) {
        // save at filename
        std::vector<std::size_t> layer_sizes {10, 7, 7, 10};
        lc::NeuralNetwork network(layer_sizes);

        network.dump_file(filename);

        std::cout << "Neural network saved\n";
    }

    else {
        // load from filename

        lc::NeuralNetwork::load_file(filename);

        std::cout << "Neural network loaded\n";
    }
}
