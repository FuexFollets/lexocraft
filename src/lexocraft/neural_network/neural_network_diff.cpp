#include <Eigen/Eigen>
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

    NeuralNetwork::NeuralNetworkDiff::SerializeMedium
        NeuralNetwork::NeuralNetworkDiff::serialize() const noexcept {
        using Medium_t = NeuralNetwork::NeuralNetworkDiff::SerializeMedium;

        Eigen::Serializer<Eigen::MatrixXf> matrix_dynamic_serializer;
        Eigen::Serializer<Eigen::VectorXf> vector_dynamic_serializer;

        Medium_t medium;

        // Serialize weight_diffs

        medium.weight_diffs_buffers.resize(weight_diffs.size());
        medium.bias_diffs_buffers.resize(bias_diffs.size());
        medium.layer_sizes = layer_sizes;

        for (std::size_t index {0}; index < weight_diffs.size(); index++) {
            const auto& weight_diff {weight_diffs [index]};
            const auto size = matrix_dynamic_serializer.size(weight_diff);
            std::vector<std::uint8_t> bytes(size);
            auto* buffer = bytes.data();
            matrix_dynamic_serializer.serialize(buffer, std::next(buffer, size), weight_diff);
            medium.weight_diffs_buffers [index] = std::move(bytes);
        }

        for (std::size_t index {0}; index < bias_diffs.size(); index++) {
            const auto& bias_diff {bias_diffs [index]};
            const auto size = vector_dynamic_serializer.size(bias_diff);
            std::vector<std::uint8_t> bytes(size);
            auto* buffer = bytes.data();
            vector_dynamic_serializer.serialize(buffer, std::next(buffer, size), bias_diff);
            medium.bias_diffs_buffers [index] = std::move(bytes);
        }

        return medium;
    }

    // demediumize
    NeuralNetwork::NeuralNetworkDiff
        NeuralNetwork::NeuralNetworkDiff::SerializeMedium::demediumize() const noexcept {
        NeuralNetwork::NeuralNetworkDiff diff;

        diff.layer_sizes = layer_sizes;

        Eigen::Serializer<Eigen::MatrixXf> matrix_dynamic_deserializer;
        Eigen::Serializer<Eigen::VectorXf> vector_dynamic_deserializer;

        for (std::size_t index {0}; index < layer_sizes.size(); index++) {
            const auto& bytes = weight_diffs_buffers [index];
            const auto size = bytes.size();
            const auto* buffer = bytes.data();
            matrix_dynamic_deserializer.deserialize(buffer, std::next(buffer, size),
                                                    diff.weight_diffs [index]);
        }

        for (std::size_t index {0}; index < layer_sizes.size(); index++) {
            const auto& bytes = bias_diffs_buffers [index];
            const auto size = bytes.size();
            const auto* buffer = bytes.data();
            vector_dynamic_deserializer.deserialize(buffer, std::next(buffer, size),
                                                    diff.bias_diffs [index]);
        }

        return diff;
    }
} // namespace lc
