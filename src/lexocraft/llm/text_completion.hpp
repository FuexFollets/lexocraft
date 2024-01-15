#ifndef TEXT_COMPLETION_HPP
#define TEXT_COMPLETION_HPP

#include <memory>

#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <Eigen/Eigen>

#include <lexocraft/cereal_eigen.hpp>
#include <lexocraft/llm/lexer.hpp>
#include <lexocraft/llm/vector_database.hpp>
#include <lexocraft/neural_network/neural_network.hpp>

namespace lc {
    class TextCompleter {
        public:

        template <class Archive>
        void serialize(Archive& archive) {
            // clang-format off
            archive(ephemeral_memory, ephemeral_memory_size, context_memory, context_memory_size,
                    ephemeral_memory_accmulator, context_builder, word_vector_improviser,
                    vector_database,

                    ephemeral_memory_fields_sizes,
                    ephemeral_memory_output_sizes,

                    context_builder_fields_sizes,
                    context_builder_output_sizes,

                    word_vector_improviser_fields_sizes,
                    word_vector_improviser_output_sizes
                    );
            // clang-format on
        }

        Eigen::VectorXf ephemeral_memory;
        std::size_t ephemeral_memory_size;

        Eigen::VectorXf context_memory;
        std::size_t context_memory_size;

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

            EphemeralMemoryNNFields(float sentence_length_mean, float sentence_length_stddev,
                                    float flesch_kincaid_grade, const WordVector& word,
                                    const Eigen::VectorXf& ephemeral_memory,
                                    const Eigen::VectorXf& context_memory);

            float sentence_length_mean;
            float sentence_length_stddev;
            // float word_sophistication;  // Interval: [0, 1] - 0 = Uncommon, 1 = Most common
            float flesch_kincaid_grade; // Interval: [0, 20] - 0 = Most difficult, 20 = Easiest
            WordVector word;
            Eigen::VectorXf ephemeral_memory;
            Eigen::VectorXf context_memory;

            [[nodiscard]] Eigen::VectorXf to_vector() const final;
        };

        struct ephemeral_memory_fields_sizes_t {
            std::size_t word_vector;
            std::size_t ephemeral_memory;
            std::size_t context_memory;

            [[nodiscard]] std::size_t total() const;

            template <class Archive>
            void serialize(Archive& archive) {
                archive(word_vector, ephemeral_memory, context_memory);
            }
        } ephemeral_memory_fields_sizes;

        struct EphemeralMemoryNNOutput : NNOutput<EphemeralMemoryNNOutput> {
            /* Vector fields for EphemeralMemoryNN output */

            Eigen::VectorXf ephemeral_memory;
            Eigen::VectorXf word_vector_value;

            bool from_output(const Eigen::VectorXf& output) final;
        };

        struct ephemeral_memory_output_sizes_t {
            std::size_t ephemeral_memory;
            std::size_t word_vector_value;

            [[nodiscard]] std::size_t total() const;

            template <class Archive>
            void serialize(Archive& archive) {
                archive(ephemeral_memory, word_vector_value);
            }
        } ephemeral_memory_output_sizes;

        struct ContextBuilderNNFields : NNFieldsInput {
            /* Vector fields for ContextBuilderNN */

            ContextBuilderNNFields(float sentence_length_mean, float sentence_length_stddev,
                                   float flesch_kincaid_grade,
                                   const Eigen::VectorXf& ephemeral_memory,
                                   const Eigen::VectorXf& context_memory);

            float sentence_length_mean;
            float sentence_length_stddev;
            // float word_sophistication;  // Interval: [0, 1] - 0 = Uncommon, 1 = Most common
            float flesch_kincaid_grade; // Interval: [0, 20] - 0 = Most difficult, 20 = Easiest
            Eigen::VectorXf ephemeral_memory;
            Eigen::VectorXf context_memory;

            [[nodiscard]] Eigen::VectorXf to_vector() const final;
        };

        struct context_builder_fields_sizes_t {
            std::size_t ephemeral_memory;
            std::size_t context_memory;

            [[nodiscard]] std::size_t total() const;

            template <class Archive>
            void serialize(Archive& archive) {
                archive(ephemeral_memory, context_memory);
            }
        } context_builder_fields_sizes;

        struct ContextBuilderNNOutput : NNOutput<ContextBuilderNNOutput> {
            /* Vector fields for ContextBuilderNN output */

            Eigen::VectorXf context_memory;

            bool from_output(const Eigen::VectorXf& output) final;
        };

        struct context_builder_output_sizes_t {
            std::size_t context_memory;

            [[nodiscard]] std::size_t total() const;

            template <class Archive>
            void serialize(Archive& archive) {
                archive(context_memory);
            }
        } context_builder_output_sizes;

        struct WordVectorImproviserNNFields : NNFieldsInput {
            /* Vector fields for WordVectorImproviserNN */

            WordVectorImproviserNNFields(const VectorDatabase::SearchResult& result,
                                         const Eigen::VectorXf& ephemeral_memory,
                                         Eigen::VectorXf& word_vector_value);

            VectorDatabase::SearchResult word_vectors_search_result;
            Eigen::VectorXf ephemeral_memory;
            Eigen::VectorXf word_vector_value;

            [[nodiscard]] Eigen::VectorXf to_vector() const final;
        };

        struct word_vector_improviser_fields_sizes_t {
            std::size_t ephemeral_memory;
            std::size_t word_vector_value;

            [[nodiscard]] std::size_t total() const;

            template <class Archive>
            void serialize(Archive& archive) {
                archive(ephemeral_memory, word_vector_value);
            }
        } word_vector_improviser_fields_sizes;

        struct WordVectorImproviserNNOutput : NNOutput<WordVectorImproviserNNOutput> {
            /* Vector fields for WordVectorImproviserNN output */

            Eigen::VectorXf word_vector_value;

            bool from_output(const Eigen::VectorXf& output) final;
        };

        struct word_vector_improviser_output_sizes_t {
            std::size_t word_vector_value;

            [[nodiscard]] std::size_t total() const;

            template <class Archive>
            void serialize(Archive& archive) {
                archive(word_vector_value);
            }
        } word_vector_improviser_output_sizes;

        static float flesch_kincaid_level(const std::string& text);

        struct SearchedWordVector {
            WordVector word_vector;
            bool improvised;
        };

        SearchedWordVector find_word_vector(const std::string& word);

        WordVector improvised_word_vector(
            const std::string& word,
            const std::vector<VectorDatabase::SearchResult>& word_vectors_search_result);

        TextCompleter();
        TextCompleter(const TextCompleter&) = delete;
        TextCompleter& operator=(const TextCompleter&) = delete;
        TextCompleter(TextCompleter&&) = default;
        TextCompleter& operator=(TextCompleter&&) = default;

        TextCompleter(
            const std::shared_ptr<VectorDatabase>& vector_database,
            const ephemeral_memory_fields_sizes_t& ephemeral_memory_fields_sizes,
            const ephemeral_memory_output_sizes_t& ephemeral_memory_output_sizes,
            const context_builder_fields_sizes_t& context_builder_fields_sizes,
            const context_builder_output_sizes_t& context_builder_output_sizes,
            const word_vector_improviser_fields_sizes_t& word_vector_improviser_fields_sizes,
            const word_vector_improviser_output_sizes_t& word_vector_improviser_output_sizes);

        TextCompleter(const std::shared_ptr<VectorDatabase>& vector_database,
                      std::size_t ephemeral_memory_size, std::size_t context_memory_size);
    };

    float sentence_length_mean(const std::vector<grammar::Token>& tokens);
    float sentence_length_stddev(const std::vector<grammar::Token>& tokens);
} // namespace lc

#endif // TEXT_COMPLETION_HPP
