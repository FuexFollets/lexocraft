#include <lexocraft/neural_network/neural_network.hpp>
#include <vector>

namespace lc {
    // Give each random float values between -1 and 1.
    NeuralNetwork::NeuralNetworkDiff::NeuralNetworkDiff(
        const std::vector<std::size_t>& layer_sizes) :
        weight_diffs(layer_sizes.size() - 1),
        bias_diffs(layer_sizes.size() - 1), layer_sizes(layer_sizes) {
        for (std::size_t index {0}; index < layer_sizes.size() - 1; index++) {
            weight_diffs [index] =
                Eigen::MatrixXf::Random(layer_sizes [index + 1], layer_sizes [index]);
            bias_diffs [index] = Eigen::VectorXf::Random(layer_sizes [index + 1]);
        }
    }

    NeuralNetwork::NeuralNetworkDiff& NeuralNetwork::NeuralNetworkDiff::operator+=(
        const NeuralNetwork::NeuralNetworkDiff& other) noexcept {
        for (std::size_t index {0}; index < layer_sizes.size() - 1; index++) {
            weight_diffs [index] += other.weight_diffs [index];
            bias_diffs [index] += other.bias_diffs [index];
        }
        return *this;
    }

    NeuralNetwork::NeuralNetworkDiff& NeuralNetwork::NeuralNetworkDiff::operator-=(
        const NeuralNetwork::NeuralNetworkDiff& other) noexcept {
        for (std::size_t index {0}; index < layer_sizes.size() - 1; index++) {
            weight_diffs [index] -= other.weight_diffs [index];
            bias_diffs [index] -= other.bias_diffs [index];
        }
        return *this;
    }

    NeuralNetwork::NeuralNetworkDiff&
        NeuralNetwork::NeuralNetworkDiff::operator*=(float scalar) noexcept {
        for (std::size_t index {0}; index < layer_sizes.size() - 1; index++) {
            weight_diffs [index] *= scalar;
            bias_diffs [index] *= scalar;
        }
        return *this;
    }

    NeuralNetwork::NeuralNetworkDiff& NeuralNetwork::NeuralNetworkDiff::operator/=(float scalar) {
        for (std::size_t index {0}; index < layer_sizes.size() - 1; index++) {
            weight_diffs [index] /= scalar;
            bias_diffs [index] /= scalar;
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

    void NeuralNetwork::NeuralNetworkDiff::invert() noexcept {
        *this *= -1;
    }

    NeuralNetwork::NeuralNetworkDiff NeuralNetwork::NeuralNetworkDiff::inverted() const noexcept {
        return *this * -1;
    }
} // namespace lc
