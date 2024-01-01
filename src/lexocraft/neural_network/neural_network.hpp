#ifndef LEXICRAFT_NEURAL_NETWORK_HPP
#define LEXICRAFT_NEURAL_NETWORK_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <Eigen/Core>

namespace lc {
    using vbuffer_t = std::vector<std::uint8_t>;

    class NeuralNetwork {
        public:

        class NeuralNetworkDiff {
            public:

            class SerializeMedium {
                public:

                std::vector<vbuffer_t> weight_diffs_buffers;
                std::vector<vbuffer_t> bias_diffs_buffers;
                std::vector<std::size_t> layer_sizes;

                [[nodiscard]] NeuralNetworkDiff demediumize() const noexcept;
            };

            constexpr static std::size_t FIELD_COUNT {3};
            std::vector<Eigen::MatrixXf> weight_diffs;
            std::vector<Eigen::VectorXf> bias_diffs;
            std::vector<std::size_t> layer_sizes;

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

            void invert() noexcept;

            [[nodiscard]] NeuralNetworkDiff inverted() const noexcept;
            [[nodiscard]] SerializeMedium serialize() const noexcept;

            friend class NeuralNetwork;
        };

        class SerializeMedium {
            public:

            std::size_t iterations {};
            std::vector<std::size_t> layer_sizes;

            std::vector<vbuffer_t> weights_buffer;
            std::vector<vbuffer_t> biases_buffer;

            NeuralNetworkDiff::SerializeMedium most_recent_diff;
            float most_recent_cost {};
            std::size_t diff_improvement_streak {};

            [[nodiscard]] NeuralNetwork demediumize() const noexcept;
        };

        constexpr static float GOOD_COST {0.1F};
        constexpr static std::size_t FIELD_COUNT {7};

        std::size_t iterations {};
        std::vector<std::size_t> layer_sizes;

        std::vector<Eigen::MatrixXf> weights;
        std::vector<Eigen::VectorXf> biases;

        NeuralNetworkDiff most_recent_diff;
        float most_recent_cost {};
        std::size_t diff_improvement_streak {};

        NeuralNetwork() = default;
        NeuralNetwork(NeuralNetwork&& other) noexcept = default;
        NeuralNetwork(const NeuralNetwork& other) noexcept = default;

        NeuralNetwork& operator=(NeuralNetwork&& other) noexcept = default;

        explicit NeuralNetwork(std::vector<std::size_t> layer_sizes, bool randomize = true);

        void modify(NeuralNetworkDiff diff, bool apply_biases = true, bool apply_weights = true);

        void train(float cost);

        [[nodiscard]] SerializeMedium serialize_medium() const noexcept;
        [[nodiscard]] vbuffer_t serialize() const noexcept;
        [[nodiscard]] Eigen::VectorXf compute(Eigen::VectorXf input) const noexcept;

        void dump_file(const std::filesystem::path& filepath) const;

        static NeuralNetwork load_file(const std::filesystem::path& filepath);
        static float sigmoid_abs(float value);
    };
} // namespace lc

std::ostream& operator<<(std::ostream& stream, const lc::NeuralNetwork& network);
std::ostream& operator<<(std::ostream& stream, const lc::NeuralNetwork::NeuralNetworkDiff& diff);

#endif // LEXICRAFT_NEURAL_NETWORK_HPP
