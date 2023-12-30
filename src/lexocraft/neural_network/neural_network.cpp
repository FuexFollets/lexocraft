#include <cmath>
#include <filesystem>
#include <system_error>
#include <utility>

#include <alpaca/alpaca.h>

#include <lexocraft/neural_network/neural_network.hpp>
#include <vector>

namespace lc {

    NeuralNetwork::NeuralNetwork(std::vector<std::size_t> layer_sizes, bool randomize) :
        iterations {0}, layer_sizes(std::move(layer_sizes)), most_recent_diff(layer_sizes),
        most_recent_cost {0}, diff_improvement_streak {0} {
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

    NeuralNetwork::NeuralNetwork(std::vector<std::uint8_t> alpaca_bytes) :
        most_recent_cost {0}, diff_improvement_streak {0} {
        std::error_code error_code;

        auto object = alpaca::deserialize<NeuralNetwork>(alpaca_bytes, error_code);

        if (error_code) {
            throw std::runtime_error(error_code.message());
        }

        iterations = object.iterations;
        layer_sizes = std::move(object.layer_sizes);
        weights = std::move(object.weights);
        biases = std::move(object.biases);
        most_recent_diff = std::move(object.most_recent_diff);
    }

    std::vector<std::uint8_t> NeuralNetwork::serialize() const {
        std::vector<std::uint8_t> bytes;

        alpaca::serialize(*this, bytes);

        return bytes;
    }

    Eigen::VectorXf NeuralNetwork::compute(Eigen::VectorXf input) const {
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
        alpaca::serialize(*this, dumpfile);
    }

    NeuralNetwork NeuralNetwork::load_file(const std::filesystem::path& filename) {
        std::ifstream dumpfile(filename, std::ios::in | std::ios::binary);
        std::error_code error_code;
        const auto file_size = std::filesystem::file_size(filename);

        return alpaca::deserialize<NeuralNetwork>(dumpfile, file_size, error_code);
    }

} // namespace lc
