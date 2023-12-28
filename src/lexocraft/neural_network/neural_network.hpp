#ifndef LEXICRAFT_NEURAL_NETWORK_HPP
#define LEXICRAFT_NEURAL_NETWORK_HPP

#include <cstddef>
#include <vector>

#include <eigen3/Eigen/Core>

namespace lc {

    class NeuralNetwork {
        private:

        std::size_t _iterations;
        std::vector<std::size_t> _layer_sizes;

        std::vector<Eigen::MatrixXf> _weights;
        std::vector<Eigen::VectorXf> _biases;

        public:

        explicit NeuralNetwork(std::vector<std::size_t> layer_sizes, bool randomize = true);
        explicit NeuralNetwork(std::vector<std::uint8_t> alpaca_bytes);

        [[nodiscard]] std::vector<std::uint8_t> serialize() const;
    };

} // namespace lc

#endif // LEXICRAFT_NEURAL_NETWORK_HPP
