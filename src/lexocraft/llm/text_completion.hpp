#ifndef TEXT_COMPLETION_HPP
#define TEXT_COMPLETION_HPP

#include <Eigen/Eigen>

#include <lexocraft/llm/vector_database.hpp>
#include <lexocraft/neural_network/neural_network.hpp>

namespace lc {
    class TextCompleter {
        public:

        Eigen::VectorXf ephemeral_memory;
        Eigen::VectorXf context_memory;

        NeuralNetwork ephemeral_memory_accmulator;
        NeuralNetwork context_builder;
        VectorDatabase* vector_database;

        struct NNFieldsInput {
            virtual ~NNFieldsInput() = default; // Abstract
            [[nodiscard]] virtual Eigen::VectorXf to_vector() const = 0;
        };

        template <typename Output>
        struct NNOutput { // Abstract
            virtual ~NNOutput() = default;
            virtual bool from_output(const Output& output) = 0;
        };

        struct EphemeralMemoryNNFields : NNFieldsInput {
            /* TODO: Vector fields for EphemeralMemoryNN */
            float sentence_length_mean;
            float sentence_length_stddev;
            float word_sophistication;  // Interval: [0, 1] - 0 = Uncommon, 1 = Most common
            float flesch_kincaid_grade; // Interval: [0, 20] - 0 = Most difficult, 20 = Easiest
            Eigen::VectorXf ephemeral_memory;
        };

        struct EphemeralMemoryNNOutput : NNOutput<EphemeralMemoryNNOutput> {
            /* TODO: Vector fields for EphemeralMemoryNN output */
        };

        struct ContextBuilderNNFields : NNFieldsInput {
            /* TODO: Vector fields for ContextBuilderNN */
        };

        struct ContextBuilderNNOutput : NNOutput<ContextBuilderNNOutput> {
            /* TODO: Vector fields for ContextBuilderNN output */
        };

        static float flesch_kincaid_level(const std::string& text);
    };

} // namespace lc

#endif // TEXT_COMPLETION_HPP
