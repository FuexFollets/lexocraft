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

            NeuralNetworkDiff() = default;
            NeuralNetworkDiff(NeuralNetworkDiff&& other) noexcept = default;
            NeuralNetworkDiff(const NeuralNetworkDiff& other) noexcept = default;
            explicit NeuralNetworkDiff(const std::vector<std::size_t>& layer_sizes);

            NeuralNetworkDiff& operator=(NeuralNetworkDiff&& other) noexcept = default;
            NeuralNetworkDiff& operator=(const NeuralNetworkDiff& other) noexcept = default;

            NeuralNetworkDiff& operator+=(const NeuralNetworkDiff& other) noexcept;
            NeuralNetworkDiff& operator-=(const NeuralNetworkDiff& other) noexcept;
            NeuralNetworkDiff& operator*=(float scalar) noexcept;
            NeuralNetworkDiff& operator/=(float scalar);

            NeuralNetworkDiff operator+(const NeuralNetworkDiff& other) const noexcept;
            NeuralNetworkDiff operator-(const NeuralNetworkDiff& other) const noexcept;
            NeuralNetworkDiff operator*(float scalar) const noexcept;
            NeuralNetworkDiff operator/(float scalar) const;

            void invert();

            [[nodiscard]] NeuralNetworkDiff inverted() const;

            [[nodiscard]] std::size_t layer_count() const;
            [[nodiscard]] std::size_t layer_size(std::size_t layer) const;

            [[nodiscard]] const Eigen::MatrixXf& weight_diff(std::size_t layer) const;
            [[nodiscard]] const Eigen::VectorXf& bias_diff(std::size_t layer) const;

            [[nodiscard]] Eigen::MatrixXf& weight_diff(std::size_t layer);
            [[nodiscard]] Eigen::VectorXf& bias_diff(std::size_t layer);

            friend class NeuralNetwork;
        };

        private:

        constexpr static float GOOD_COST {0.1F};

        std::size_t _iterations;
        std::vector<std::size_t> _layer_sizes;

        std::vector<Eigen::MatrixXf> _weights;
        std::vector<Eigen::VectorXf> _biases;

        NeuralNetworkDiff _most_recent_diff;
        float _most_recent_cost;
        std::size_t _diff_improvement_streak;

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
        [[nodiscard]] float most_recent_cost() const;
        [[nodiscard]] std::size_t diff_improvement_streak() const;

        static float sigmoid_abs(float value);
    };
} // namespace lc

#endif // LEXICRAFT_NEURAL_NETWORK_HPP
