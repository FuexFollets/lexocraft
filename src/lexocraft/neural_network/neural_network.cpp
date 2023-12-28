#include <system_error>
#include <utility>

#include <alpaca/alpaca.h>

#include <lexocraft/neural_network/neural_network.hpp>

namespace lc {

    NeuralNetwork::NeuralNetwork(std::vector<std::size_t> layer_sizes, bool randomize) :
        _iterations(0), _layer_sizes(std::move(layer_sizes)) {
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

    NeuralNetwork::NeuralNetwork(std::vector<std::uint8_t> alpaca_bytes) {
        std::error_code error_code;

        auto object = alpaca::deserialize<NeuralNetwork>(alpaca_bytes, error_code);

        if (error_code) {
            throw std::runtime_error(error_code.message());
        }

        _iterations = object._iterations;
        _layer_sizes = std::move(object._layer_sizes);
        _weights = std::move(object._weights);
        _biases = std::move(object._biases);
    }

    std::vector<std::uint8_t> NeuralNetwork::serialize() const {
        std::vector<std::uint8_t> bytes;

        alpaca::serialize(*this, bytes);

        return bytes;
    }
} // namespace lc