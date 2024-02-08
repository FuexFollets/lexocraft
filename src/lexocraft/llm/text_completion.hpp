#ifndef TEXT_COMPLETION_HPP
#define TEXT_COMPLETION_HPP

#include <cstddef>
#include <filesystem>
#include <functional>
#include <vector>

#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <Eigen/Eigen>

#include <lexocraft/cereal_eigen.hpp>
#include <lexocraft/llm/lexer.hpp>
#include <lexocraft/llm/vector_database.hpp>
#include <lexocraft/neural_network/neural_network.hpp>

namespace lc {
    using UnaryLayerSizeVectorGenerator_t = std::function<std::size_t(std::size_t)>;
    using BinaryLayerSizeVectorGenerator_t = std::function<std::size_t(std::size_t, std::size_t)>;

    class TextCompleter {
        public:

        using DatabaseTypePairElement_t = std::tuple<VectorDatabase&, grammar::Token::Type>;

        struct SearchedWordVector {
            WordVector word_vector;
            bool is_lowercase_adjusted;
            bool improvised;
        };

        template <class Archive>
        void serialize(Archive& archive) {
            // clang-format off
            archive(ephemeral_memory, ephemeral_memory_size, context_memory, context_memory_size,
                    ephemeral_memory_accmulator, context_builder, word_vector_improviser,

                    vector_database,
                    alphanumeric_vector_subdatabase,
                    digit_vector_subdatabase,
                    homogeneous_vector_subdatabase,
                    symbol_vector_subdatabase,

                    lowercase_alphanumeric_vector_subdatabase,
                    lowercase_homogeneous_vector_subdatabase,

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

        VectorDatabase vector_database;

        VectorDatabase alphanumeric_vector_subdatabase {};
        VectorDatabase digit_vector_subdatabase {};
        VectorDatabase homogeneous_vector_subdatabase {};
        VectorDatabase symbol_vector_subdatabase {};

        VectorDatabase lowercase_alphanumeric_vector_subdatabase {};
        VectorDatabase lowercase_homogeneous_vector_subdatabase {};

        std::array<DatabaseTypePairElement_t, 4> database_type_pairs {
            {
             {alphanumeric_vector_subdatabase, grammar::Token::Type::Alphanumeric},
             {digit_vector_subdatabase, grammar::Token::Type::Digit},
             {homogeneous_vector_subdatabase, grammar::Token::Type::Homogeneous},
             {symbol_vector_subdatabase, grammar::Token::Type::Symbol},
             }
        };

        std::array<DatabaseTypePairElement_t, 2> lowercase_database_type_pairs {
            {
             {lowercase_alphanumeric_vector_subdatabase, grammar::Token::Type::Alphanumeric},
             {lowercase_homogeneous_vector_subdatabase, grammar::Token::Type::Homogeneous},
             }
        };

        struct NNFieldsInput {
            virtual ~NNFieldsInput() = default; // Abstract
            [[nodiscard]] virtual Eigen::VectorXf to_vector() const = 0;
        };

        template <typename Output>
        struct NNOutput { // Abstract
            virtual ~NNOutput() = default;
            virtual bool from_output(const Eigen::VectorXf& output) = 0;
        };

        /******************** EphemeralMemoryNNFields ********************/

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

        struct EphemeralMemoryNNFields : NNFieldsInput {
            /* Vector fields for EphemeralMemoryNN */

            EphemeralMemoryNNFields(float sentence_length_mean, float sentence_length_stddev,
                                    float flesch_kincaid_grade, float sentence_count,
                                    const SearchedWordVector& word,
                                    const Eigen::VectorXf& ephemeral_memory,
                                    const Eigen::VectorXf& context_memory,
                                    const ephemeral_memory_fields_sizes_t& size_info);

            float sentence_length_mean;
            float sentence_length_stddev;
            // float word_sophistication;  // Interval: [0, 1] - 0 = Uncommon, 1 = Most common
            float flesch_kincaid_grade; // Interval: [0, 20] - 0 = Most difficult, 20 = Easiest
            float sentence_count;
            SearchedWordVector word;
            Eigen::VectorXf ephemeral_memory;
            Eigen::VectorXf context_memory;

            ephemeral_memory_fields_sizes_t size_info;

            [[nodiscard]] Eigen::VectorXf to_vector() const final;
        };

        /******************** EphemeralMemoryNNOutput ********************/

        struct ephemeral_memory_output_sizes_t {
            std::size_t ephemeral_memory;
            std::size_t word_vector_value;

            [[nodiscard]] std::size_t total() const;

            template <class Archive>
            void serialize(Archive& archive) {
                archive(ephemeral_memory, word_vector_value);
            }
        } ephemeral_memory_output_sizes;

        struct EphemeralMemoryNNOutput : NNOutput<EphemeralMemoryNNOutput> {
            /* Vector fields for EphemeralMemoryNN output */

            explicit EphemeralMemoryNNOutput(ephemeral_memory_output_sizes_t size_info);

            float token_is_alphanumeric {};
            float token_is_digit {};
            float token_is_homogeneous {};
            float token_is_symbol {};
            float is_end {};
            Eigen::VectorXf ephemeral_memory;
            Eigen::VectorXf word_vector_value;

            ephemeral_memory_output_sizes_t size_info;

            bool from_output(const Eigen::VectorXf& output) final;
        };

        /******************** ContextBuilderNNFields ********************/

        struct context_builder_fields_sizes_t {
            std::size_t ephemeral_memory;
            std::size_t context_memory;

            [[nodiscard]] std::size_t total() const;

            template <class Archive>
            void serialize(Archive& archive) {
                archive(ephemeral_memory, context_memory);
            }
        } context_builder_fields_sizes;

        struct ContextBuilderNNFields : NNFieldsInput {
            /* Vector fields for ContextBuilderNN */

            ContextBuilderNNFields(float sentence_length_mean, float sentence_length_stddev,
                                   float flesch_kincaid_grade,
                                   const Eigen::VectorXf& ephemeral_memory,
                                   const Eigen::VectorXf& context_memory,
                                   const context_builder_fields_sizes_t& size_info);

            float sentence_length_mean;
            float sentence_length_stddev;
            // float word_sophistication;  // Interval: [0, 1] - 0 = Uncommon, 1 = Most common
            float flesch_kincaid_grade; // Interval: [0, 20] - 0 = Most difficult, 20 = Easiest
            Eigen::VectorXf ephemeral_memory;
            Eigen::VectorXf context_memory;

            context_builder_fields_sizes_t size_info;

            [[nodiscard]] Eigen::VectorXf to_vector() const final;
        };

        /******************** ContextBuilderNNOutput ********************/

        struct context_builder_output_sizes_t {
            std::size_t context_memory;

            [[nodiscard]] std::size_t total() const;

            template <class Archive>
            void serialize(Archive& archive) {
                archive(context_memory);
            }
        } context_builder_output_sizes;

        struct ContextBuilderNNOutput : NNOutput<ContextBuilderNNOutput> {
            /* Vector fields for ContextBuilderNN output */

            explicit ContextBuilderNNOutput(context_builder_output_sizes_t size_info);

            Eigen::VectorXf context_memory;

            context_builder_output_sizes_t size_info;

            bool from_output(const Eigen::VectorXf& output) final;
        };

        /******************** WordVectorImproviserNNFields ********************/

        struct word_vector_improviser_fields_sizes_t {
            std::size_t word_vector_search_result;
            std::size_t ephemeral_memory;
            std::size_t word_vector_value;

            [[nodiscard]] std::size_t total() const;

            template <class Archive>
            void serialize(Archive& archive) {
                archive(ephemeral_memory, word_vector_value);
            }
        } word_vector_improviser_fields_sizes;

        struct WordVectorImproviserNNFields : NNFieldsInput {
            /* Vector fields for WordVectorImproviserNN */

            WordVectorImproviserNNFields(const VectorDatabase::SearchResult& result,
                                         const Eigen::VectorXf& ephemeral_memory,
                                         const Eigen::VectorXf& word_vector_value,
                                         const word_vector_improviser_fields_sizes_t& size_info);

            VectorDatabase::SearchResult word_vectors_search_result;
            Eigen::VectorXf ephemeral_memory;
            Eigen::VectorXf word_vector_value;

            word_vector_improviser_fields_sizes_t size_info;

            [[nodiscard]] Eigen::VectorXf to_vector() const final;
        };

        /******************** WordVectorImproviserNNOutput ********************/

        struct word_vector_improviser_output_sizes_t {
            std::size_t word_vector_value;

            [[nodiscard]] std::size_t total() const;

            template <class Archive>
            void serialize(Archive& archive) {
                archive(word_vector_value);
            }
        } word_vector_improviser_output_sizes;

        struct WordVectorImproviserNNOutput : NNOutput<WordVectorImproviserNNOutput> {
            /* Vector fields for WordVectorImproviserNN output */

            explicit WordVectorImproviserNNOutput(word_vector_improviser_output_sizes_t size_info);

            Eigen::VectorXf word_vector_value;

            word_vector_improviser_output_sizes_t size_info;

            bool from_output(const Eigen::VectorXf& output) final;
        };

        /******************** End ********************/

        static float flesch_kincaid_level(const std::string& text);

        std::tuple<SearchedWordVector, grammar::Token::Type>
            find_word_vector(const std::string& word);

        WordVector improvised_word_vector(
            const std::string& word,
            const std::vector<VectorDatabase::SearchResult>& word_vectors_search_result);

        TextCompleter& reset_ephemeral_memory();
        TextCompleter& reset_context_memory();

        Eigen::VectorXf accumulate_context_memory(float sentence_length_mean,
                                                  float sentence_length_stddev,
                                                  float flesch_kincaid_grade);

        EphemeralMemoryNNOutput predict_next_token_value(const grammar::Token& token,
                                                         float sentence_length_mean_,
                                                         float sentence_length_stddev_,
                                                         float flesch_kincaid_grade_,
                                                         float sentence_count_);

        /*
            NeuralNetwork ephemeral_memory_accmulator;
            NeuralNetwork context_builder;
            NeuralNetwork word_vector_improviser;
        */

        TextCompleter&
            set_ephemeral_memory_accmulator_nn(const NeuralNetwork& ephemeral_memory_accmulator);
        TextCompleter&
            set_ephemeral_memory_accmulator_nn(NeuralNetwork&& ephemeral_memory_accmulator);
        TextCompleter&
            set_ephemeral_memory_accmulator_nn(const std::vector<std::size_t>& layer_sizes,
                                               bool random = false);

        TextCompleter& set_context_builder_nn(const NeuralNetwork& context_builder);
        TextCompleter& set_context_builder_nn(NeuralNetwork&& context_builder);
        TextCompleter& set_context_builder_nn(const std::vector<std::size_t>& layer_sizes,
                                              bool random = false);

        TextCompleter& set_word_vector_improviser_nn(const NeuralNetwork& word_vector_improviser);
        TextCompleter& set_word_vector_improviser_nn(NeuralNetwork&& word_vector_improviser);
        TextCompleter& set_word_vector_improviser_nn(const std::vector<std::size_t>& layer_sizes,
                                                     bool random = false);

        TextCompleter& set_vector_database(VectorDatabase&& vector_database);

        TextCompleter& create_vector_subdatabases();

        TextCompleter();
        TextCompleter(const TextCompleter&) = default;
        TextCompleter& operator=(const TextCompleter&) = default;
        TextCompleter(TextCompleter&&) = default;
        TextCompleter& operator=(TextCompleter&&) = default;

        TextCompleter(
            VectorDatabase&& vector_database,
            const ephemeral_memory_fields_sizes_t& ephemeral_memory_fields_sizes,
            const ephemeral_memory_output_sizes_t& ephemeral_memory_output_sizes,
            const context_builder_fields_sizes_t& context_builder_fields_sizes,
            const context_builder_output_sizes_t& context_builder_output_sizes,
            const word_vector_improviser_fields_sizes_t& word_vector_improviser_fields_sizes,
            const word_vector_improviser_output_sizes_t& word_vector_improviser_output_sizes);

        // ----------------------------- Primary Interface -----------------------------

        explicit TextCompleter(VectorDatabase&& vector_database,
                               std::size_t ephemeral_memory_size = 1000,
                               std::size_t context_memory_size = 500);

        explicit TextCompleter(const std::filesystem::path& filepath,
                               std::size_t ephemeral_memory_size = 1000,
                               std::size_t context_memory_size = 500);

        // Based on the previous neural network layer size, it creates the next neural network layer
        // size
        using UnaryLayerSizeVectorGenerator_t = std::function<std::size_t(std::size_t)>;

        // Based on the first and last neural network layer size, it creates the next neural network
        // layer size
        using BinaryLayerSizeVectorGenerator_t =
            std::function<std::size_t(std::size_t, std::size_t)>;

        TextCompleter& set_ephemeral_memory_accumulator_layer_sizes(std::size_t layer_count);
        TextCompleter& set_ephemeral_memory_accumulator_layer_sizes(
            const UnaryLayerSizeVectorGenerator_t& unary_layer_size_vector_generator,
            std::size_t layer_count);
        TextCompleter& set_ephemeral_memory_accumulator_layer_sizes(
            const BinaryLayerSizeVectorGenerator_t& binary_layer_size_vector_generator,
            std::size_t layer_count);

        TextCompleter& set_context_builder_layer_sizes(std::size_t layer_count);
        TextCompleter& set_context_builder_layer_sizes(
            const UnaryLayerSizeVectorGenerator_t& unary_layer_size_vector_generator,
            std::size_t layer_count);
        TextCompleter& set_context_builder_layer_sizes(
            const BinaryLayerSizeVectorGenerator_t& binary_layer_size_vector_generator,
            std::size_t layer_count);

        TextCompleter& set_word_vector_improviser_layer_sizes(std::size_t layer_count);
        TextCompleter& set_word_vector_improviser_layer_sizes(
            const UnaryLayerSizeVectorGenerator_t& unary_layer_size_vector_generator,
            std::size_t layer_count);
        TextCompleter& set_word_vector_improviser_layer_sizes(
            const BinaryLayerSizeVectorGenerator_t& binary_layer_size_vector_generator,
            std::size_t layer_count);

        TextCompleter& add_word_vector(const WordVector& added_word_vector);
        TextCompleter& add_word_vector(const std::vector<WordVector>& added_word_vectors);
        TextCompleter& add_word_vector(const std::string& word, const Eigen::VectorXf& vector);
        TextCompleter& add_word_vector(const std::string& word, bool random = false);

        TextCompleter& save_file(const std::filesystem::path& filepath);
        TextCompleter& load_file(const std::filesystem::path& filepath);

        std::vector<grammar::Token> tokenize(const std::string& text);
    };

    float sentence_length_mean(const std::vector<grammar::Token>& tokens);
    float sentence_length_stddev(const std::vector<grammar::Token>& tokens);
} // namespace lc

#endif // TEXT_COMPLETION_HPP
