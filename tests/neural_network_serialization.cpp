#include <filesystem>
#include <iostream>
#include <lexocraft/neural_network/neural_network.hpp>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    // Test neural network serialization

    // print arguments

    std::cout << "argc: " << argc << "\n";
    for (int index = 0; index < argc; ++index) {
        std::cout << "argv[" << index << "]: " << argv [index] << "\n";
    }

    std::filesystem::path path = *std::next(argv, 1);

    if (argc > 2) {
        std::vector<std::size_t> layer_sizes {10, 7, 7, 10};
        lc::NeuralNetwork network(layer_sizes);

        std::cout << "Neural network created:\n" << network << "\n";
        std::cout << "Serialized and deserialized neural network:\n";
        network.save_file(path);

        std::cout << "Network saved to file: " << path << "\n";
    }
    else {
        lc::NeuralNetwork loaded_network = lc::NeuralNetwork::load_file(path);
        std::cout << "Loaded file from path: " << path << "\n";
        std::cout << "Loaded neural network:\n" << loaded_network << "\n";
    }
}
