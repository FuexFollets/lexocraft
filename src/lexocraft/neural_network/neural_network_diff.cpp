#include <Eigen/Eigen>
#include <iostream>
#include <lexocraft/neural_network/neural_network.hpp>
#include <vector>

namespace lc {
    // Give each random float values between -1 and 1.
    NeuralNetwork::NeuralNetworkDiff::NeuralNetworkDiff(
        const std::vector<std::size_t>& layer_sizes) :
        weight_diffs(layer_sizes.size() - 1),
        bias_diffs(layer_sizes.size() - 1), layer_sizes(layer_sizes) {
        std::cout << "Layer sizes.size(): " << layer_sizes.size() << '\n';

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
            const Eigen::MatrixXf& weight_diff {weight_diffs [index]};
            const std::size_t size {matrix_dynamic_serializer.size(weight_diff)};
            vbuffer_t bytes(size);
            std::uint8_t* buffer = bytes.data();
            matrix_dynamic_serializer.serialize(buffer, std::next(buffer, size), weight_diff);
            medium.weight_diffs_buffers [index] = bytes;

            for (std::size_t index {0}; index < 10; index++) {
                std::cout << "bytes[" << index << "]: " << static_cast<int>(bytes [index]) << "\n";
            }

            std::cout << "bytes[size - 1]: " << static_cast<int>(bytes [size - 1]) << "\n";

            // try to deserialize

            Eigen::MatrixXf deserialized_weight_diff;
            matrix_dynamic_serializer.deserialize(buffer, std::next(buffer, size),
                                                  deserialized_weight_diff);

            std::cout << "Deserialized weight diff (inside NeuralNetworkDiff::serialize()):"
                      << "\n"; // deserialized_weight_diff << "\n";
            std::cout << "Size: " << size << "\n\n";
            std::cout << "bytes.size(): " << bytes.size() << "\n";
        }

        for (std::size_t index {0}; index < bias_diffs.size(); index++) {
            const Eigen::VectorXf& bias_diff {bias_diffs [index]};
            const std::size_t size = vector_dynamic_serializer.size(bias_diff);
            vbuffer_t bytes(size);
            std::uint8_t* buffer = bytes.data();
            vector_dynamic_serializer.serialize(buffer, std::next(buffer, size), bias_diff);
            medium.bias_diffs_buffers [index] = bytes;

            for (std::size_t index {0}; index < 10; index++) {
                std::cout << "bytes[" << index << "]: " << static_cast<int>(bytes [index]) << "\n";
            }

            // try to deserialize

            Eigen::VectorXf deserialized_bias_diff;
            vector_dynamic_serializer.deserialize(buffer, std::next(buffer, size),
                                                  deserialized_bias_diff);

            std::cout << "Deserialized bias diff (inside NeuralNetworkDiff::serialize()):"
                      << "\n"; // deserialized_bias_diff << "\n";
            std::cout << "Size: " << size << "\n\n";
            std::cout << "bytes.size(): " << bytes.size() << "\n";
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

        std::cout
            << "layer_sizes.size() (inside NeuralNetworkDiff::SerializeMedium::demediumize()): "
            << layer_sizes.size() << "\n";

        for (std::size_t index {0}; index < layer_sizes.size() - 1; index++) {
            const vbuffer_t& bytes(weight_diffs_buffers [index]);
            const std::size_t size {bytes.size()};
            std::cout << "bytes.size(): " << size << "\n";

            for (std::size_t index {0}; index < 10; index++) {
                std::cout << "bytes[" << index << "]: " << static_cast<int>(bytes [index]) << "\n";
            }

            std::cout << "bytes[size - 1]: " << static_cast<int>(bytes [size - 1]) << "\n";

            const std::uint8_t* buffer {bytes.data()};
            matrix_dynamic_deserializer.deserialize(buffer, std::next(buffer, size),
                                                    diff.weight_diffs [index]);
        }

        for (std::size_t index {0}; index < layer_sizes.size() - 1; index++) {
            const auto& bytes = bias_diffs_buffers [index];
            const auto size = bytes.size();
            const auto* buffer = bytes.data();
            vector_dynamic_deserializer.deserialize(buffer, std::next(buffer, size),
                                                    diff.bias_diffs [index]);
        }

        return diff;
    }
} // namespace lc

// Print neural network diff

std::ostream& operator<<(std::ostream& stream, const lc::NeuralNetwork::NeuralNetworkDiff& diff) {
    stream << "diff:\n";
    stream << "  layer_sizes: ";

    for (const auto& layer_size: diff.layer_sizes) {
        stream << layer_size << " ";
    }

    stream << "\n  weight_diffs:\n";

    for (std::size_t weight_diff_number {0}; weight_diff_number < diff.weight_diffs.size();
         weight_diff_number++) {
        const auto& weight_diff = diff.weight_diffs [weight_diff_number];

        stream << "    weight diff number " << weight_diff_number << ":\n";
        stream << "      " << weight_diff << "\n";
    }

    stream << "  bias_diffs:\n";

    for (std::size_t bias_diff_number {0}; bias_diff_number < diff.bias_diffs.size();
         bias_diff_number++) {
        const auto& bias_diff = diff.bias_diffs [bias_diff_number];

        stream << "    bias diff number " << bias_diff_number << ":\n";
        stream << "      " << bias_diff << "\n";
    }

    return stream;
}
