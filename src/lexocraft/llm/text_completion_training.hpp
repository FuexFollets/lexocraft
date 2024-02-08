#ifndef LEXOCRAFT_TEXT_COMPLETION_TRAINING_HPP
#define LEXOCRAFT_TEXT_COMPLETION_TRAINING_HPP

#include <optional>
#include <string>

#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/neural_network/neural_network.hpp>

namespace lc {
    class TextCompletionTrainer {
        public:

        struct CostWeightCoefficients {
            float incorrect_token_type {};
            float incorrect_section_termination {};
            float predicted_word_vector_euclidean_distance_magnitude {};

            constexpr float cost_weight_total() {
                return incorrect_token_type + incorrect_section_termination +
                       predicted_word_vector_euclidean_distance_magnitude;
            }
        };

        constexpr static CostWeightCoefficients DEFAULT_COST_WEIGHT_COEFFICIENTS {
            .incorrect_token_type = 2.0F,
            .incorrect_section_termination = 1.0F,
            .predicted_word_vector_euclidean_distance_magnitude = 1.0F};

        struct TrainingModification {
            std::optional<NeuralNetwork::NeuralNetworkDiff> ephemeral_memory_diff;
            std::optional<NeuralNetwork::NeuralNetworkDiff> context_builder_diff;
            std::optional<NeuralNetwork::NeuralNetworkDiff> word_vector_improviser_diff;

            float original_cost {};
            float improved_cost {};
        };

        std::optional<TextCompleter> text_completer;

        float calculate_prediction_costs(
            const std::string& training_data_section,
            const std::optional<CostWeightCoefficients>& cost_weight_coefficients);
        float calculate_prediction_costs(
            const std::vector<std::string>& training_data_sections,
            const std::optional<CostWeightCoefficients>& cost_weight_coefficients);

        TrainingModification train_neural_network(
            const std::vector<std::string>& training_data_sections,
            const std::optional<CostWeightCoefficients>& cost_weight_coefficients = std::nullopt);
        TrainingModification train_neural_network(const std::string& training_data);

        TextCompleter& apply_training_modification(const TrainingModification& modification);
    };
} // namespace lc

#endif // LEXOCRAFT_TEXT_COMPLETION_TRAINING_HPP
