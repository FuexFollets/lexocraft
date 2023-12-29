#ifndef LEXICRAFT_NEURAL_NETWORK_HPP
#define LEXICRAFT_NEURAL_NETWORK_HPP

#include <cstddef>
#include <vector>

#include <eigen3/Eigen/Core>

namespace lc {

    class NeuralNetwork {
        public:

        class NeuralNetworkDiff {
            std::vector<Eigen::MatrixXf> _weight_diffs;
            std::vector<Eigen::VectorXf> _bias_diffs;
            std::vector<std::size_t> _layer_sizes;

            public:

            explicit NeuralNetworkDiff(const std::vector<std::size_t>& layer_sizes);

            NeuralNetworkDiff& operator+=(const NeuralNetworkDiff& other);
            NeuralNetworkDiff& operator-=(const NeuralNetworkDiff& other);
            NeuralNetworkDiff& operator*=(float scalar);
            NeuralNetworkDiff& operator/=(float scalar);

            NeuralNetworkDiff operator+(const NeuralNetworkDiff& other) const;
            NeuralNetworkDiff operator-(const NeuralNetworkDiff& other) const;
            NeuralNetworkDiff operator*(float scalar) const;
            NeuralNetworkDiff operator/(float scalar) const;

            friend class NeuralNetwork;
        };

        private:

        std::size_t _iterations;
        std::vector<std::size_t> _layer_sizes;

        std::vector<Eigen::MatrixXf> _weights;
        std::vector<Eigen::VectorXf> _biases;

        NeuralNetworkDiff _most_recent_diff;

        public:

        explicit NeuralNetwork(std::vector<std::size_t> layer_sizes, bool randomize = true);
        explicit NeuralNetwork(std::vector<std::uint8_t> alpaca_bytes);

        void modify(NeuralNetworkDiff diff, bool apply_biases = true, bool apply_weights = true);

        void train(float cost);

        [[nodiscard]] std::vector<std::uint8_t> serialize() const;
        [[nodiscard]] Eigen::VectorXf compute(Eigen::VectorXf input) const;

        [[nodiscard]] std::size_t iterations() const;
        [[nodiscard]] std::size_t layer_count() const;
        [[nodiscard]] std::size_t layer_size(std::size_t layer) const;

        [[nodiscard]] const Eigen::MatrixXf& weights(std::size_t layer) const;
        [[nodiscard]] const Eigen::VectorXf& biases(std::size_t layer) const;

        [[nodiscard]] const NeuralNetworkDiff& most_recent_diff() const;

        static float sigmoid_abs(float value);
    };
} // namespace lc

#endif // LEXICRAFT_NEURAL_NETWORK_HPP
