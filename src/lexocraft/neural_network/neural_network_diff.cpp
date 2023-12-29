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

    /* Implement:
        NeuralNetworkDiff& operator+=(const NeuralNetworkDiff& other);
        NeuralNetworkDiff& operator-=(const NeuralNetworkDiff& other);
        NeuralNetworkDiff& operator*=(float scalar);
        NeuralNetworkDiff& operator/=(float scalar);

        NeuralNetworkDiff operator+(const NeuralNetworkDiff& other) const;
        NeuralNetworkDiff operator-(const NeuralNetworkDiff& other) const;
        NeuralNetworkDiff operator*(float scalar) const;
        NeuralNetworkDiff operator/(float scalar) const;
    */

    NeuralNetwork::NeuralNetworkDiff& NeuralNetwork::NeuralNetworkDiff::operator+=(
        const NeuralNetwork::NeuralNetworkDiff& other) {
        for (std::size_t index {0}; index < _layer_sizes.size() - 1; index++) {
            _weight_diffs [index] += other._weight_diffs [index];
            _bias_diffs [index] += other._bias_diffs [index];
        }
        return *this;
    }

    NeuralNetwork::NeuralNetworkDiff& NeuralNetwork::NeuralNetworkDiff::operator-=(
        const NeuralNetwork::NeuralNetworkDiff& other) {
        for (std::size_t index {0}; index < _layer_sizes.size() - 1; index++) {
            _weight_diffs [index] -= other._weight_diffs [index];
            _bias_diffs [index] -= other._bias_diffs [index];
        }
        return *this;
    }

    NeuralNetwork::NeuralNetworkDiff& NeuralNetwork::NeuralNetworkDiff::operator*=(float scalar) {
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
        const NeuralNetwork::NeuralNetworkDiff& other) const {
        NeuralNetwork::NeuralNetworkDiff copy_of_this {*this};

        copy_of_this += other;

        return copy_of_this;
    }

    NeuralNetwork::NeuralNetworkDiff NeuralNetwork::NeuralNetworkDiff::operator-(
        const NeuralNetwork::NeuralNetworkDiff& other) const {
        NeuralNetwork::NeuralNetworkDiff copy_of_this {*this};

        copy_of_this -= other;

        return copy_of_this;
    }

    NeuralNetwork::NeuralNetworkDiff
        NeuralNetwork::NeuralNetworkDiff::operator*(float scalar) const {
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
} // namespace lc
