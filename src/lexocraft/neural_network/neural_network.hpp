#ifndef LEXICRAFT_NEURAL_NETWORK_HPP
#define LEXICRAFT_NEURAL_NETWORK_HPP

#include <cstddef>
#include <eigen3/Eigen/Core>

namespace lc {

    class NeuralNetwork {
        private:

        std::size_t _iterations;
        std::vector<std::size_t> _layer_sizes;

        std::vector<Eigen::MatrixXd> _weights;
        std::vector<Eigen::VectorXd> _biases;

        public:

        explicit NeuralNetwork(std::vector<std::size_t> layer_sizes, bool randomize = true);
        explicit NeuralNetwork(std::vector<std::uint8_t> alpaca_bytes);

        [[nodiscard]] std::vector<std::uint8_t> to_bytes() const;
    };

} // namespace lc


#endif // LEXICRAFT_NEURAL_NETWORK_HPP
