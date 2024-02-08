#ifndef LEXOCRAFT_TEXT_COMPLETION_TRAINING_HPP
#define LEXOCRAFT_TEXT_COMPLETION_TRAINING_HPP

#include <optional>
#include <string>

#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/neural_network/neural_network.hpp>

namespace lc {
    class TextCompletionTrainer {
        public:

        struct TrainingModification {
            std::optional<NeuralNetwork::NeuralNetworkDiff> ephemeral_memory_diff;
            std::optional<NeuralNetwork::NeuralNetworkDiff> context_builder_diff;
            std::optional<NeuralNetwork::NeuralNetworkDiff> word_vector_improviser_diff;

            float original_cost {};
            float improved_cost {};
        };

        std::optional<TextCompleter> text_completer;

        TrainingModification
            train_neural_network(const std::vector<std::string>& training_data_sections);
        TrainingModification train_neural_network(const std::string& training_data);

        TextCompleter& apply_training_modification(const TrainingModification& modification);
    };
} // namespace lc

#endif // LEXOCRAFT_TEXT_COMPLETION_TRAINING_HPP
