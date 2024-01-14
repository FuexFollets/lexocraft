#ifndef TEXT_COMPLETION_HPP
#define TEXT_COMPLETION_HPP

#include "lexocraft/llm/lexer.hpp"
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
        std::shared_ptr<VectorDatabase> vector_database;

        struct NNFieldsInput {
            virtual ~NNFieldsInput() = default; // Abstract
            [[nodiscard]] virtual Eigen::VectorXf to_vector() const = 0;
        };

        template <typename Output>
        struct NNOutput { // Abstract
            virtual ~NNOutput() = default;
            virtual bool from_output(const Eigen::VectorXf& output) = 0;
        };

        struct EphemeralMemoryNNFields : NNFieldsInput {
            /* Vector fields for EphemeralMemoryNN */
            float sentence_length_mean;
            float sentence_length_stddev;
            // float word_sophistication;  // Interval: [0, 1] - 0 = Uncommon, 1 = Most common
            float flesch_kincaid_grade; // Interval: [0, 20] - 0 = Most difficult, 20 = Easiest
            WordVector word;
            Eigen::VectorXf ephemeral_memory;
            Eigen::VectorXf context_memory;

            [[nodiscard]] Eigen::VectorXf to_vector() const final;
        };

        struct EphemeralMemoryNNOutput : NNOutput<EphemeralMemoryNNOutput> {
            /* Vector fields for EphemeralMemoryNN output */

            std::size_t ephemeral_memory_size;

            Eigen::VectorXf ephemeral_memory;
            Eigen::VectorXf word_vector_value;

            bool from_output(const Eigen::VectorXf& output) final;
        };

        struct ContextBuilderNNFields : NNFieldsInput {
            /* Vector fields for ContextBuilderNN */

            float sentence_length_mean;
            float sentence_length_stddev;
            // float word_sophistication;  // Interval: [0, 1] - 0 = Uncommon, 1 = Most common
            float flesch_kincaid_grade; // Interval: [0, 20] - 0 = Most difficult, 20 = Easiest
            Eigen::VectorXf ephemeral_memory;
            Eigen::VectorXf context_memory;

            [[nodiscard]] Eigen::VectorXf to_vector() const final;
        };

        struct ContextBuilderNNOutput : NNOutput<ContextBuilderNNOutput> {
            /* Vector fields for ContextBuilderNN output */

            std::size_t ephemeral_memory_size;

            Eigen::VectorXf context_memory;

            bool from_output(const Eigen::VectorXf& output) final;
        };

        static float flesch_kincaid_level(const std::string& text);
    };

    float sentence_length_mean(const std::vector<grammar::Token>& tokens); 
    float sentence_length_stddev(const std::vector<grammar::Token>& tokens);
} // namespace lc

#endif // TEXT_COMPLETION_HPP
