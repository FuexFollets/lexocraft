#include <fstream>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>

#include <lexocraft/llm/text_completion.hpp>

namespace lc {
    // --------------------------- General ---------------------------

    TextCompleter& TextCompleter::start_new_section(float sentence_length_mean,
                                                    float sentence_length_stddev,
                                                    float flesch_kincaid_grade) {
        this->context_memory = this->accumulate_context_memory(
            sentence_length_mean, sentence_length_stddev, flesch_kincaid_grade);
        return *this;
    }

    TextCompleter& TextCompleter::add_word_vector(const WordVector& added_word_vector) {
        vector_database.add_word(added_word_vector);
        return create_vector_subdatabases();
    }

    TextCompleter&
        TextCompleter::add_word_vector(const std::vector<WordVector>& added_word_vectors) {
        for (const auto& word_vector: added_word_vectors) {
            vector_database.add_word(word_vector);
        }

        return create_vector_subdatabases();
    }

    TextCompleter& TextCompleter::add_word_vector(const std::string& word,
                                                  const Eigen::VectorXf& vector) {
        vector_database.add_word(WordVector(word, vector));

        return create_vector_subdatabases();
    }

    TextCompleter& TextCompleter::add_word_vector(const std::string& word, bool random) {
        vector_database.add_word(WordVector(word, random));

        return create_vector_subdatabases();
    }

    TextCompleter& TextCompleter::save_file(const std::filesystem::path& filepath) {
        std::ofstream file {filepath};

        cereal::BinaryOutputArchive archive {file};

        archive(*this);

        return *this;
    }

    TextCompleter& TextCompleter::load_file(const std::filesystem::path& filepath) {
        std::ifstream file {filepath};

        cereal::BinaryInputArchive archive {file};

        archive(*this);

        return *this;
    }

    std::vector<grammar::Token> TextCompleter::tokenize(const std::string& text) {
        return grammar::tokenize(text, vector_database);
    }

    // -------------------------- Ephermal Memory Accumulator --------------------------
    TextCompleter& TextCompleter::set_ephemeral_memory_accumulator_layer_sizes(
        const BinaryLayerSizeVectorGenerator_t& binary_layer_size_vector_generator,
        std::size_t layer_count) {
        const std::size_t ephemeral_memory_input_layer_size = ephemeral_memory_fields_sizes.total();
        const std::size_t ephemeral_memory_output_layer_size =
            ephemeral_memory_fields_sizes.total();

        std::size_t previous_layer_size = ephemeral_memory_input_layer_size;

        std::vector<std::size_t> layer_sizes(layer_count + 1);
        layer_sizes.insert(layer_sizes.begin(), ephemeral_memory_input_layer_size);

        for (std::size_t index = 0; index < layer_count; ++index) {
            layer_sizes [index + 1] = binary_layer_size_vector_generator(
                previous_layer_size, ephemeral_memory_output_layer_size);
            previous_layer_size = layer_sizes [index + 1];
        }

        layer_sizes.push_back(ephemeral_memory_output_layer_size);

        return *this;
    }

    TextCompleter& TextCompleter::set_ephemeral_memory_accumulator_layer_sizes(
        const UnaryLayerSizeVectorGenerator_t& unary_layer_size_vector_generator,
        std::size_t layer_count) {
        const std::size_t ephemeral_memory_input_layer_size = ephemeral_memory_fields_sizes.total();
        const std::size_t ephemeral_memory_output_layer_size =
            ephemeral_memory_fields_sizes.total();

        std::size_t previous_layer_size = ephemeral_memory_input_layer_size;

        std::vector<std::size_t> layer_sizes(layer_count + 1);
        layer_sizes.insert(layer_sizes.begin(), ephemeral_memory_input_layer_size);

        for (std::size_t index = 0; index < layer_count; ++index) {
            layer_sizes [index + 1] = unary_layer_size_vector_generator(previous_layer_size);
            previous_layer_size = layer_sizes [index + 1];
        }

        layer_sizes.push_back(ephemeral_memory_output_layer_size);

        return *this;
    }

    TextCompleter&
        TextCompleter::set_ephemeral_memory_accumulator_layer_sizes(std::size_t layer_count) {
        std::size_t layer_number = 0;

        // Linear from the output layer to the input layer
        BinaryLayerSizeVectorGenerator_t binary_layer_size_vector_generator =
            [&layer_count, &layer_number](std::size_t previous_layer_size,
                                          std::size_t output_layer_size) {
                return (output_layer_size - previous_layer_size) * (layer_number + 1) /
                           layer_count +
                       previous_layer_size;
            };

        return set_ephemeral_memory_accumulator_layer_sizes(binary_layer_size_vector_generator,
                                                            layer_count);
    }

    // -------------------------- Context Builder --------------------------

    TextCompleter& TextCompleter::set_context_builder_layer_sizes(
        const BinaryLayerSizeVectorGenerator_t& binary_layer_size_vector_generator,
        std::size_t layer_count) {
        const std::size_t context_builder_input_layer_size = context_builder_fields_sizes.total();
        const std::size_t context_builder_output_layer_size = context_builder_fields_sizes.total();

        std::size_t previous_layer_size = context_builder_input_layer_size;

        std::vector<std::size_t> layer_sizes(layer_count + 1);
        layer_sizes.insert(layer_sizes.begin(), context_builder_input_layer_size);

        for (std::size_t index = 0; index < layer_count; ++index) {
            layer_sizes [index + 1] = binary_layer_size_vector_generator(
                previous_layer_size, context_builder_output_layer_size);
            previous_layer_size = layer_sizes [index + 1];
        }

        layer_sizes.push_back(context_builder_output_layer_size);

        return *this;
    }

    TextCompleter& TextCompleter::set_context_builder_layer_sizes(
        const UnaryLayerSizeVectorGenerator_t& unary_layer_size_vector_generator,
        std::size_t layer_count) {
        const std::size_t context_builder_input_layer_size = context_builder_fields_sizes.total();
        const std::size_t context_builder_output_layer_size = context_builder_fields_sizes.total();

        std::size_t previous_layer_size = context_builder_input_layer_size;

        std::vector<std::size_t> layer_sizes(layer_count + 1);
        layer_sizes.insert(layer_sizes.begin(), context_builder_input_layer_size);

        for (std::size_t index = 0; index < layer_count; ++index) {
            layer_sizes [index + 1] = unary_layer_size_vector_generator(previous_layer_size);
            previous_layer_size = layer_sizes [index + 1];
        }

        layer_sizes.push_back(context_builder_output_layer_size);

        return *this;
    }

    TextCompleter& TextCompleter::set_context_builder_layer_sizes(std::size_t layer_count) {
        std::size_t layer_number = 0;

        // Linear from the output layer to the input layer
        BinaryLayerSizeVectorGenerator_t binary_layer_size_vector_generator =
            [&layer_count, &layer_number](std::size_t previous_layer_size,
                                          std::size_t output_layer_size) {
                return (output_layer_size - previous_layer_size) * (layer_number + 1) /
                           layer_count +
                       previous_layer_size;
            };

        return set_context_builder_layer_sizes(binary_layer_size_vector_generator, layer_count);
    }

    // -------------------------- Word Vector Improviser --------------------------

    TextCompleter& TextCompleter::set_word_vector_improviser_layer_sizes(
        const BinaryLayerSizeVectorGenerator_t& binary_layer_size_vector_generator,
        std::size_t layer_count) {
        const std::size_t word_vector_improviser_input_layer_size =
            word_vector_improviser_fields_sizes.total();
        const std::size_t word_vector_improviser_output_layer_size =
            word_vector_improviser_fields_sizes.total();

        std::size_t previous_layer_size = word_vector_improviser_input_layer_size;

        std::vector<std::size_t> layer_sizes(layer_count + 1);
        layer_sizes.insert(layer_sizes.begin(), word_vector_improviser_input_layer_size);

        for (std::size_t index = 0; index < layer_count; ++index) {
            layer_sizes [index + 1] = binary_layer_size_vector_generator(
                previous_layer_size, word_vector_improviser_output_layer_size);
            previous_layer_size = layer_sizes [index + 1];
        }

        layer_sizes.push_back(word_vector_improviser_output_layer_size);

        return *this;
    }

    TextCompleter& TextCompleter::set_word_vector_improviser_layer_sizes(
        const UnaryLayerSizeVectorGenerator_t& unary_layer_size_vector_generator,
        std::size_t layer_count) {
        const std::size_t word_vector_improviser_input_layer_size =
            word_vector_improviser_fields_sizes.total();
        const std::size_t word_vector_improviser_output_layer_size =
            word_vector_improviser_fields_sizes.total();

        std::size_t previous_layer_size = word_vector_improviser_input_layer_size;

        std::vector<std::size_t> layer_sizes(layer_count + 1);
        layer_sizes.insert(layer_sizes.begin(), word_vector_improviser_input_layer_size);

        for (std::size_t index = 0; index < layer_count; ++index) {
            layer_sizes [index + 1] = unary_layer_size_vector_generator(previous_layer_size);
            previous_layer_size = layer_sizes [index + 1];
        }

        layer_sizes.push_back(word_vector_improviser_output_layer_size);

        return *this;
    }

    TextCompleter& TextCompleter::set_word_vector_improviser_layer_sizes(std::size_t layer_count) {
        std::size_t layer_number = 0;

        // Linear from the output layer to the input layer
        BinaryLayerSizeVectorGenerator_t binary_layer_size_vector_generator =
            [&layer_count, &layer_number](std::size_t previous_layer_size,
                                          std::size_t output_layer_size) {
                return (output_layer_size - previous_layer_size) * (layer_number + 1) /
                           layer_count +
                       previous_layer_size;
            };

        return set_word_vector_improviser_layer_sizes(binary_layer_size_vector_generator,
                                                      layer_count);
    }
} // namespace lc
