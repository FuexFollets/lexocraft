#include <iostream>
#include <lexocraft/neural_network/neural_network.hpp>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    // Test neural network serialization

    std::vector<std::size_t> layer_sizes {10, 7, 7, 10};
    lc::NeuralNetwork network(layer_sizes);

    std::cout << "Neural network created:\n" << network << "\n";
    std::cout << "Serialized and deserialized neural network:\n";

    /*
    const std::string filename {*std::next(argv, 1)};

    std::cout << "Filename: " << filename << '\n';

    if (argc > 2) {
        // save at filename
        std::vector<std::size_t> layer_sizes {10, 7, 7, 10};
        lc::NeuralNetwork network(layer_sizes);

        std::cout << "Neural network created:\n" << network << "\n";

        network.dump_file(filename);

        std::cout << "Neural network saved\n";
    }

    else {
        // load from filename

        lc::NeuralNetwork network = lc::NeuralNetwork::load_file(filename);

        std::cout << "Neural network loaded\n";

        std::cout << "Neural network:\n" << network << "\n";
    }
    */
}
