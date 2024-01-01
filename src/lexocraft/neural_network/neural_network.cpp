#include <cmath>
#include <filesystem>
#include <system_error>
#include <utility>

#include <alpaca/alpaca.h>

#include <lexocraft/neural_network/neural_network.hpp>
#include <vector>

namespace lc {

    NeuralNetwork::NeuralNetwork(std::vector<std::size_t> layer_sizes, bool randomize) :
        layer_sizes(std::move(layer_sizes)), most_recent_diff(layer_sizes) {
        for (std::size_t i = 1; i < layer_sizes.size(); ++i) {
            weights.emplace_back(layer_sizes [i], layer_sizes [i - 1]);
            biases.emplace_back(layer_sizes [i]);
        }

        if (randomize) {
            for (auto& weight: weights) {
                weight = Eigen::MatrixXf::Random(weight.rows(), weight.cols());
            }

            for (auto& bias: biases) {
                bias = Eigen::VectorXf::Random(bias.rows());
            }
        }
    }

    NeuralNetwork::NeuralNetwork(vbuffer_t alpaca_bytes) {
        using Medium_t = NeuralNetwork::SerializeMedium;

        std::error_code error_code;
        auto medium = alpaca::deserialize<Medium_t>(alpaca_bytes, error_code);

        if (error_code) {
            throw std::system_error(error_code);
        }

        iterations = medium.iterations;
        layer_sizes = medium.layer_sizes;
        most_recent_diff = medium.most_recent_diff;
        most_recent_cost = medium.most_recent_cost;
        diff_improvement_streak = medium.diff_improvement_streak;

        Eigen::Serializer<Eigen::MatrixXf> matrix_dynamic_deserializer;
        Eigen::Serializer<Eigen::VectorXf> vector_dynamic_deserializer;

        for (std::size_t index {0}; index < medium.weights_buffer.size(); index++) {
            const auto& bytes = medium.weights_buffer [index];
            const auto size = bytes.size();
            const auto* buffer = bytes.data();
            matrix_dynamic_deserializer.deserialize(buffer, std::next(buffer, size),
                                                    weights [index]);
        }

        for (std::size_t index {0}; index < medium.biases_buffer.size(); index++) {
            const auto& bytes = medium.biases_buffer [index];
            const auto size = bytes.size();
            const auto* buffer = bytes.data();
            vector_dynamic_deserializer.deserialize(buffer, std::next(buffer, size),
                                                    biases [index]);
        }
    }

    NeuralNetwork::SerializeMedium NeuralNetwork::serialize_medium() const noexcept {
        using Medium_t = NeuralNetwork::SerializeMedium;

        Medium_t medium;

        medium.iterations = iterations;
        medium.layer_sizes = layer_sizes;
        medium.most_recent_diff = most_recent_diff.serialize();
        medium.most_recent_cost = most_recent_cost;
        medium.diff_improvement_streak = diff_improvement_streak;

        Eigen::Serializer<Eigen::MatrixXf> matrix_dynamic_serializer;
        Eigen::Serializer<Eigen::VectorXf> vector_dynamic_serializer;

        medium.weights_buffer.resize(weights.size());
        medium.biases_buffer.resize(biases.size());

        for (std::size_t index {0}; index < weights.size(); index++) {
            const auto& weight {weights [index]};
            const auto size = matrix_dynamic_serializer.size(weight);
            std::vector<std::uint8_t> bytes(size);
            auto* buffer = bytes.data();
            matrix_dynamic_serializer.serialize(buffer, std::next(buffer, size), weight);
            medium.weights_buffer [index] = std::move(bytes);
        }

        for (std::size_t index {0}; index < biases.size(); index++) {
            const auto& bias {biases [index]};
            const auto size = vector_dynamic_serializer.size(bias);
            std::vector<std::uint8_t> bytes(size);
            auto* buffer = bytes.data();
            vector_dynamic_serializer.serialize(buffer, std::next(buffer, size), bias);
            medium.biases_buffer [index] = std::move(bytes);
        }

        return medium;
    }

    vbuffer_t NeuralNetwork::serialize() const noexcept {
        using Medium_t = NeuralNetwork::SerializeMedium;

        Medium_t medium {serialize_medium()};

        vbuffer_t buffer;

        alpaca::serialize<Medium_t>(medium, buffer);

        return buffer;
    }

    Eigen::VectorXf NeuralNetwork::compute(Eigen::VectorXf input) const noexcept {
        for (std::size_t index {0}; index < weights.size(); ++index) {
            input = weights [index] * input + biases [index];
            input = input.unaryExpr([](float value) { return sigmoid_abs(value); });
        }

        return input;
    }

    float NeuralNetwork::sigmoid_abs(float value) {
        return 0.5F + value / (2 * (1 + std::abs(value)));
    }

    void NeuralNetwork::modify(NeuralNetwork::NeuralNetworkDiff diff, bool apply_biases,
                               bool apply_weights) {
        if (apply_biases) {
            for (std::size_t index {0}; index < biases.size(); ++index) {
                biases [index] += diff.bias_diffs [index];
            }
        }

        if (apply_weights) {
            for (std::size_t index {0}; index < weights.size(); ++index) {
                weights [index] += diff.weight_diffs [index];
            }
        }
    }

    /* Example usage:
     repeat {
         NeuralNetwork nn {...};

         auto res = nn.compute(...);

         float cost = get_cost(res, ...);

         nn.train(cost);
     }
    */

    void NeuralNetwork::train(float cost) { /* cost is between 0 and 1 */
        // If the cost is worse than GOOD_COST or the previous cost, revert the NN using
        // _most_recent_diff, reset the streak, and apply a different NeuralNetworkDiff. Otherwise,
        // continue to apply the same NeuralNetworkDiff and increment the streak. Based on the
        // streak and most recent cost, modify the NeuralNetworkDiff and apply it each time. If
        // _iterations is zero, choose a random starting NeuralNetworkDiff.

        if (iterations == 0) {
            most_recent_diff = NeuralNetworkDiff(layer_sizes);

            return;
        }

        if (cost < GOOD_COST || cost < most_recent_cost) {
            diff_improvement_streak = 0;
            most_recent_diff.invert();
            modify(most_recent_diff);
            most_recent_diff = NeuralNetworkDiff(layer_sizes);
        }

        else {
            ++diff_improvement_streak;
            most_recent_diff *= 1.0F + diff_improvement_streak / 10.0F;
        }
    }

    void NeuralNetwork::dump_file(const std::filesystem::path& filename) const {
        std::ofstream dumpfile(filename, std::ios::out | std::ios::binary);
        /*
        alpaca::serialize<NeuralNetwork, NeuralNetwork::FIELD_COUNT>(*this, dumpfile);
        */
    }

    NeuralNetwork NeuralNetwork::load_file(const std::filesystem::path& filename) {
        std::ifstream dumpfile(filename, std::ios::in | std::ios::binary);
        std::error_code error_code;
        const auto file_size = std::filesystem::file_size(filename);

        /*
        return alpaca::deserialize<NeuralNetwork, NeuralNetwork::FIELD_COUNT>(dumpfile, file_size,
                                                                              error_code);
                                                                              */
    }

} // namespace lc
