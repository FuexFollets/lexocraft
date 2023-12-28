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
        [[nodiscard]] Eigen::VectorXf compute(Eigen::VectorXf input) const;

        class NeuralNetworkDiff {
            std::vector<Eigen::MatrixXf> _weight_diffs;
            std::vector<Eigen::VectorXf> _bias_diffs;
            std::vector<std::size_t> _layer_sizes;

            public:

            explicit NeuralNetworkDiff(std::vector<std::size_t> layer_sizes);

            NeuralNetworkDiff operator+(const NeuralNetworkDiff& other) const;
            NeuralNetworkDiff operator-(const NeuralNetworkDiff& other) const;
            NeuralNetworkDiff operator*(float scalar) const;
            NeuralNetworkDiff operator/(float scalar) const;
            NeuralNetworkDiff& operator+=(const NeuralNetworkDiff& other);
            NeuralNetworkDiff& operator-=(const NeuralNetworkDiff& other);
            NeuralNetworkDiff& operator*=(float scalar);
            NeuralNetworkDiff& operator/=(float scalar);

            friend class NeuralNetwork;
        };

        void modify(NeuralNetworkDiff diff);

        static float sigmoid(float value);
    };

} // namespace lc

#endif // LEXICRAFT_NEURAL_NETWORK_HPP
