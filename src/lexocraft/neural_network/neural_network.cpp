#include <cmath>
#include <filesystem>
#include <system_error>
#include <utility>

#include <alpaca/alpaca.h>

#include <lexocraft/neural_network/neural_network.hpp>
#include <vector>

namespace lc {

    NeuralNetwork::NeuralNetwork(std::vector<std::size_t> layer_sizes, bool randomize) :
        _iterations {0}, _layer_sizes(std::move(layer_sizes)), _most_recent_diff(_layer_sizes),
        _most_recent_cost {0}, _diff_improvement_streak {0} {
        for (std::size_t i = 1; i < _layer_sizes.size(); ++i) {
            _weights.emplace_back(_layer_sizes [i], _layer_sizes [i - 1]);
            _biases.emplace_back(_layer_sizes [i]);
        }

        if (randomize) {
            for (auto& weight: _weights) {
                weight = Eigen::MatrixXf::Random(weight.rows(), weight.cols());
            }

            for (auto& bias: _biases) {
                bias = Eigen::VectorXf::Random(bias.rows());
            }
        }
    }

    NeuralNetwork::NeuralNetwork(std::vector<std::uint8_t> alpaca_bytes) :
        _most_recent_cost {0}, _diff_improvement_streak {0} {
        std::error_code error_code;

        auto object = alpaca::deserialize<NeuralNetwork>(alpaca_bytes, error_code);

        if (error_code) {
            throw std::runtime_error(error_code.message());
        }

        _iterations = object._iterations;
        _layer_sizes = std::move(object._layer_sizes);
        _weights = std::move(object._weights);
        _biases = std::move(object._biases);
        _most_recent_diff = std::move(object._most_recent_diff);
    }

    std::vector<std::uint8_t> NeuralNetwork::serialize() const {
        std::vector<std::uint8_t> bytes;

        alpaca::serialize(*this, bytes);

        return bytes;
    }

    Eigen::VectorXf NeuralNetwork::compute(Eigen::VectorXf input) const {
        for (std::size_t index {0}; index < _weights.size(); ++index) {
            input = _weights [index] * input + _biases [index];
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
            for (std::size_t index {0}; index < _biases.size(); ++index) {
                _biases [index] += diff._bias_diffs [index];
            }
        }

        if (apply_weights) {
            for (std::size_t index {0}; index < _weights.size(); ++index) {
                _weights [index] += diff._weight_diffs [index];
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

        if (_iterations == 0) {
            _most_recent_diff = NeuralNetworkDiff(_layer_sizes);

            return;
        }

        if (cost < GOOD_COST || cost < _most_recent_cost) {
            _diff_improvement_streak = 0;
            _most_recent_diff.invert();
            modify(_most_recent_diff);
            _most_recent_diff = NeuralNetworkDiff(_layer_sizes);
        }

        else {
            ++_diff_improvement_streak;
            _most_recent_diff *= 1.0F + _diff_improvement_streak / 10.0F;
        }
    }

    std::size_t NeuralNetwork::iterations() const {
        return _iterations;
    }

    std::size_t NeuralNetwork::layer_count() const {
        return _layer_sizes.size();
    }

    std::size_t NeuralNetwork::layer_size(std::size_t layer) const {
        return _layer_sizes [layer];
    }

    const Eigen::MatrixXf& NeuralNetwork::weights(std::size_t layer) const {
        return _weights [layer];
    }

    const Eigen::VectorXf& NeuralNetwork::biases(std::size_t layer) const {
        return _biases [layer];
    }

    const NeuralNetwork::NeuralNetworkDiff& NeuralNetwork::most_recent_diff() const {
        return _most_recent_diff;
    }

    float NeuralNetwork::most_recent_cost() const {
        return _most_recent_cost;
    }

    std::size_t NeuralNetwork::diff_improvement_streak() const {
        return _diff_improvement_streak;
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
