#include <lexocraft/neural_network/neural_network.hpp>
#include <vector>

namespace lc {
    // Give each random float values between -1 and 1.
    NeuralNetwork::NeuralNetworkDiff::NeuralNetworkDiff(
        const std::vector<std::size_t>& layer_sizes) :
        _weight_diffs(layer_sizes.size() - 1),
        _bias_diffs(layer_sizes.size() - 1), _layer_sizes(layer_sizes) {
        for (std::size_t index {0}; index < _layer_sizes.size() - 1; index++) {
            _weight_diffs [index] =
                Eigen::MatrixXf::Random(_layer_sizes [index + 1], _layer_sizes [index]);
            _bias_diffs [index] = Eigen::VectorXf::Random(_layer_sizes [index + 1]);
        }
    }

    NeuralNetwork::NeuralNetworkDiff& NeuralNetwork::NeuralNetworkDiff::operator+=(
        const NeuralNetwork::NeuralNetworkDiff& other) noexcept {
        for (std::size_t index {0}; index < _layer_sizes.size() - 1; index++) {
            _weight_diffs [index] += other._weight_diffs [index];
            _bias_diffs [index] += other._bias_diffs [index];
        }
        return *this;
    }

    NeuralNetwork::NeuralNetworkDiff& NeuralNetwork::NeuralNetworkDiff::operator-=(
        const NeuralNetwork::NeuralNetworkDiff& other) noexcept {
        for (std::size_t index {0}; index < _layer_sizes.size() - 1; index++) {
            _weight_diffs [index] -= other._weight_diffs [index];
            _bias_diffs [index] -= other._bias_diffs [index];
        }
        return *this;
    }

    NeuralNetwork::NeuralNetworkDiff&
        NeuralNetwork::NeuralNetworkDiff::operator*=(float scalar) noexcept {
        for (std::size_t index {0}; index < _layer_sizes.size() - 1; index++) {
            _weight_diffs [index] *= scalar;
            _bias_diffs [index] *= scalar;
        }
        return *this;
    }

    NeuralNetwork::NeuralNetworkDiff& NeuralNetwork::NeuralNetworkDiff::operator/=(float scalar) {
        for (std::size_t index {0}; index < _layer_sizes.size() - 1; index++) {
            _weight_diffs [index] /= scalar;
            _bias_diffs [index] /= scalar;
        }
        return *this;
    }

    NeuralNetwork::NeuralNetworkDiff NeuralNetwork::NeuralNetworkDiff::operator+(
        const NeuralNetwork::NeuralNetworkDiff& other) const noexcept {
        NeuralNetwork::NeuralNetworkDiff copy_of_this {*this};

        copy_of_this += other;

        return copy_of_this;
    }

    NeuralNetwork::NeuralNetworkDiff NeuralNetwork::NeuralNetworkDiff::operator-(
        const NeuralNetwork::NeuralNetworkDiff& other) const noexcept {
        NeuralNetwork::NeuralNetworkDiff copy_of_this {*this};

        copy_of_this -= other;

        return copy_of_this;
    }

    NeuralNetwork::NeuralNetworkDiff
        NeuralNetwork::NeuralNetworkDiff::operator*(float scalar) const noexcept {
        NeuralNetwork::NeuralNetworkDiff copy_of_this {*this};

        copy_of_this *= scalar;

        return copy_of_this;
    }

    NeuralNetwork::NeuralNetworkDiff
        NeuralNetwork::NeuralNetworkDiff::operator/(float scalar) const {
        NeuralNetwork::NeuralNetworkDiff copy_of_this {*this};

        copy_of_this /= scalar;

        return copy_of_this;
    }

    /* Implement:
        [[nodiscard]] std::size_t layer_count() const;
        [[nodiscard]] std::size_t layer_size(std::size_t layer) const;

        [[nodiscard]] const Eigen::MatrixXf& weight_diff(std::size_t layer) const;
        [[nodiscard]] const Eigen::VectorXf& bias_diff(std::size_t layer) const;

        [[nodiscard]] Eigen::MatrixXf& weight_diff(std::size_t layer);
        [[nodiscard]] Eigen::VectorXf& bias_diff(std::size_t layer);
    */

    std::size_t NeuralNetwork::NeuralNetworkDiff::layer_count() const {
        return _layer_sizes.size() - 1;
    }

    std::size_t NeuralNetwork::NeuralNetworkDiff::layer_size(std::size_t layer) const {
        return _layer_sizes [layer];
    }

    const Eigen::MatrixXf& NeuralNetwork::NeuralNetworkDiff::weight_diff(std::size_t layer) const {
        return _weight_diffs [layer];
    }

    const Eigen::VectorXf& NeuralNetwork::NeuralNetworkDiff::bias_diff(std::size_t layer) const {
        return _bias_diffs [layer];
    }

    Eigen::MatrixXf& NeuralNetwork::NeuralNetworkDiff::weight_diff(std::size_t layer) {
        return _weight_diffs [layer];
    }

    Eigen::VectorXf& NeuralNetwork::NeuralNetworkDiff::bias_diff(std::size_t layer) {
        return _bias_diffs [layer];
    }
} // namespace lc
