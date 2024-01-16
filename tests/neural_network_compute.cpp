#include <chrono>
#include <iostream>
#include <vector>

#include <Eigen/Eigen>

#include <lexocraft/neural_network/neural_network.hpp>

int main(int argc, char** argv) {
    std::cout << "argc: " << argc << "\n";
    for (int index = 0; index < argc; ++index) {
        std::cout << "argv[" << index << "]: " << argv [index] << "\n";
    }

    // get layer sizes from args

    std::vector<std::size_t> layer_sizes(argc - 1);

    for (int index = 1; index < argc; ++index) {
        layer_sizes [index - 1] = std::stoi(argv [index]);
    }

    lc::NeuralNetwork neural_network(layer_sizes);

    Eigen::VectorXf input = Eigen::VectorXf::Random(layer_sizes [0]);

    // std::cout << "input: " << input << "\n";
    // print the first 10 elements. If there are more, print "..."

    std::cout << "input: [";
    for (int index = 0; index < 10; ++index) {
        std::cout << input [index] << ", ";
    }
    if (input.size() > 10) {
        std::cout << "...";
    }
    std::cout << "]\n";

    std::chrono::high_resolution_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();

    Eigen::VectorXf output = neural_network.compute(input);

    std::chrono::high_resolution_clock::time_point end_time =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end_time - start_time;

    // std::cout << "output: " << output << "\n";
    // print the first 10 elements. If there are more, print "..."
    std::cout << "output: [";
    for (int index = 0; index < 10; ++index) {
        std::cout << output [index] << ", ";
    }
    if (output.size() > 10) {
        std::cout << "...";
    }
    std::cout << "]\n";

    std::cout << "duration: " << duration.count() << " seconds\n";

    // print nano seconds

    std::cout << "duration: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count()
              << " nanoseconds\n";

    std::cout << "output.size(): " << output.size() << "\n";
}
