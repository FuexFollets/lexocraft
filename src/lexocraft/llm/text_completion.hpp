#ifndef TEXT_COMPLETION_HPP
#define TEXT_COMPLETION_HPP

#include <memory>

#include <Eigen/Eigen>

#include <lexocraft/llm/lexer.hpp>
#include <lexocraft/llm/vector_database.hpp>
#include <lexocraft/neural_network/neural_network.hpp>

namespace lc {
    class TextCompleter {
        public:

        Eigen::VectorXf ephemeral_memory;
        Eigen::VectorXf context_memory;

        NeuralNetwork ephemeral_memory_accmulator;
        NeuralNetwork context_builder;
        NeuralNetwork word_vector_improviser;

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

        struct WordVectorImproviserNNFields : NNFieldsInput {
            /* Vector fields for WordVectorImproviserNN */

            VectorDatabase::SearchResult word_vectors_search_result;
            Eigen::VectorXf ephemeral_memory;
            Eigen::VectorXf word_vector_ephemeral_memory;

            [[nodiscard]] Eigen::VectorXf to_vector() const final;
        };

        struct WordVectorImproviserNNOutput : NNOutput<WordVectorImproviserNNOutput> {
            /* Vector fields for WordVectorImproviserNN output */

            Eigen::VectorXf word_vector;
            Eigen::VectorXf word_vector_ephemeral_memory;
        };

        static float flesch_kincaid_level(const std::string& text);

        struct SearchedWordVector {
            WordVector word_vector;
            bool improvised;
        };

        SearchedWordVector find_word_vector(const std::string& word);

        WordVector improvised_word_vector(
            const std::string& word,
            std::vector<VectorDatabase::SearchResult> word_vectors_search_result);
    };

    float sentence_length_mean(const std::vector<grammar::Token>& tokens);
    float sentence_length_stddev(const std::vector<grammar::Token>& tokens);
} // namespace lc

#endif // TEXT_COMPLETION_HPP
