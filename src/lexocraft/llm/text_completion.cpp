#include <cctype>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/llm/vector_database.hpp>

namespace lc {
    TextCompleter::TextCompleter(
        const std::shared_ptr<VectorDatabase>& vector_database,
        const ephemeral_memory_fields_sizes_t& ephemeral_memory_fields_sizes,
        const ephemeral_memory_output_sizes_t& ephemeral_memory_output_sizes,
        const context_builder_fields_sizes_t& context_builder_fields_sizes,
        const context_builder_output_sizes_t& context_builder_output_sizes,
        const word_vector_improviser_fields_sizes_t& word_vector_improviser_fields_sizes,
        const word_vector_improviser_output_sizes_t& word_vector_improviser_output_sizes) :
        vector_database {vector_database},
        ephemeral_memory_fields_sizes {ephemeral_memory_fields_sizes},
        ephemeral_memory_output_sizes {ephemeral_memory_output_sizes},
        context_builder_fields_sizes {context_builder_fields_sizes},
        context_builder_output_sizes {context_builder_output_sizes},
        word_vector_improviser_fields_sizes {word_vector_improviser_fields_sizes},
        word_vector_improviser_output_sizes {word_vector_improviser_output_sizes} {
        std::function<bool(const std::initializer_list<std::size_t>&)> all_equal_size_t =
            [this](const std::initializer_list<std::size_t>& sizes) {
                return std::all_of(sizes.begin(), sizes.end(), [this](std::size_t size) {
                    return size == ephemeral_memory_size;
                });
            };

        assert(all_equal_size_t({
            ephemeral_memory_fields_sizes.word_vector,
            ephemeral_memory_output_sizes.word_vector_value,
            word_vector_improviser_fields_sizes.word_vector_value,
            word_vector_improviser_output_sizes.word_vector_value,
        }));

        assert(all_equal_size_t({
            ephemeral_memory_fields_sizes.ephemeral_memory,
            ephemeral_memory_output_sizes.ephemeral_memory,
            word_vector_improviser_fields_sizes.ephemeral_memory,
        }));

        assert(all_equal_size_t({
            ephemeral_memory_fields_sizes.context_memory,
            context_builder_fields_sizes.context_memory,
            context_builder_output_sizes.context_memory,
        }));

        ephemeral_memory_size = ephemeral_memory_fields_sizes.ephemeral_memory;
        context_memory_size = context_builder_fields_sizes.context_memory;
    }

    /*

        ephemeral_memory_fields_sizes {ephemeral_memory_fields_sizes},
        ephemeral_memory_output_sizes {ephemeral_memory_output_sizes},
        context_builder_fields_sizes {context_builder_fields_sizes},
        context_builder_output_sizes {context_builder_output_sizes},
        word_vector_improviser_fields_sizes {word_vector_improviser_fields_sizes},
        word_vector_improviser_output_sizes {word_vector_improviser_output_sizes} {
    */

    TextCompleter::TextCompleter(const std::shared_ptr<VectorDatabase>& vector_database,
                                 std::size_t ephemeral_memory_size,
                                 std::size_t context_memory_size) :

        // clang-format off
        ephemeral_memory_size { ephemeral_memory_size },
        context_memory_size {context_memory_size}, vector_database {vector_database},
        // clang-format on

        ephemeral_memory_fields_sizes {ephemeral_memory_fields_sizes_t {
            .word_vector = WordVector::WORD_VECTOR_DIMENSIONS,
            .ephemeral_memory = ephemeral_memory_size,
            .context_memory = context_memory_size,
        }},

        ephemeral_memory_output_sizes {ephemeral_memory_output_sizes_t {
            .ephemeral_memory = ephemeral_memory_size,
            .word_vector_value = WordVector::WORD_VECTOR_DIMENSIONS,
        }},

        context_builder_fields_sizes {context_builder_fields_sizes_t {
            .ephemeral_memory = ephemeral_memory_size,
            .context_memory = context_memory_size,
        }},

        context_builder_output_sizes {context_builder_output_sizes_t {
            .context_memory = context_memory_size,
        }},

        word_vector_improviser_fields_sizes {word_vector_improviser_fields_sizes_t {
            .word_vector_search_result = WordVector::WORD_VECTOR_DIMENSIONS,
            .ephemeral_memory = ephemeral_memory_size,
            .word_vector_value = WordVector::WORD_VECTOR_DIMENSIONS,
        }},

        word_vector_improviser_output_sizes {word_vector_improviser_output_sizes_t {
            .word_vector_value = WordVector::WORD_VECTOR_DIMENSIONS,
        }} {
    }

    float TextCompleter::flesch_kincaid_level(const std::string& text) {
        std::vector<std::string> words;
        std::vector<std::string> sentences;

        // Split text into words and sentences
        size_t start = 0;

        while (true) {
            size_t end = text.find_first_of(".!?", start);

            if (end == std::string::npos) {
                sentences.push_back(text.substr(start));
                break;
            }

            sentences.push_back(text.substr(start, end - start + 1));
            start = end + 1;
        }

        for (const std::string& sentence: sentences) {
            words.emplace_back("");
            for (char letter: sentence) {
                if (std::isalnum(letter) != 0) {
                    words.back() += letter;
                }
                else {
                    if (!words.back().empty()) {
                        words.emplace_back("");
                    }
                }
            }
        }

        // Count syllables
        int syllables = 0;
        for (const std::string& word: words) {
            for (char letter: word) {
                if (std::string("aeiouyAEIOUY").find(letter) != std::string::npos &&
                    !word.ends_with("e")) {
                    syllables++;
                }
            }
        }

        // Calculate grade level
        float average_sentence_length = static_cast<float>(words.size()) / (sentences.size() - 1);
        float average_syllables_per_word = syllables / static_cast<float>(words.size());

        return 0.39 * average_sentence_length + 11.8 * average_syllables_per_word - 15.59;
    }

    /********************** EphemeralMemoryNNFields ********************/

    std::size_t TextCompleter::ephemeral_memory_fields_sizes_t::total() const {
        return word_vector + ephemeral_memory + context_memory + 3;
    }

    TextCompleter::EphemeralMemoryNNFields::EphemeralMemoryNNFields(
        float sentence_length_mean, float sentence_length_stddev, float flesch_kincaid_grade,
        const WordVector& word, const Eigen::VectorXf& ephemeral_memory,
        const Eigen::VectorXf& context_memory, const ephemeral_memory_fields_sizes_t& size_info) :
        sentence_length_mean(sentence_length_mean),
        sentence_length_stddev(sentence_length_stddev), flesch_kincaid_grade(flesch_kincaid_grade),
        word(word), ephemeral_memory(ephemeral_memory), context_memory(context_memory),
        size_info(size_info) {
        assert(static_cast<std::size_t>(word.vector.size()) == size_info.word_vector);
        assert(static_cast<std::size_t>(ephemeral_memory.size()) == size_info.ephemeral_memory);
        assert(static_cast<std::size_t>(context_memory.size()) == size_info.context_memory);
    }

    Eigen::VectorXf TextCompleter::EphemeralMemoryNNFields::to_vector() const {
        const std::size_t vector_length = size_info.total();

        Eigen::VectorXf vector(vector_length);

        /* Vector layout encoding:
         * sentence_length_mean
         * sentence_length_stddev
         * word_sophistication
         * flesch_kincaid_grade
         * word.vector
         * ephemeral_memory
         * context_memory
         **/

        std::size_t index {0};

        vector(index++) = sentence_length_mean;
        vector(index++) = sentence_length_stddev;
        // vector(index++) = word_sophistication;
        vector(index++) = flesch_kincaid_grade;

        vector.segment(index, size_info.word_vector) = word.vector;
        index += size_info.word_vector;

        vector.segment(index, size_info.ephemeral_memory) = ephemeral_memory;
        index += size_info.ephemeral_memory;

        vector.segment(index, size_info.context_memory) = context_memory;
        index += size_info.context_memory;

        return vector;
    }

    /********************** EphemeralMemoryNNOutput ********************/

    std::size_t TextCompleter::ephemeral_memory_output_sizes_t::total() const {
        return ephemeral_memory + word_vector_value;
    }

    TextCompleter::EphemeralMemoryNNOutput::EphemeralMemoryNNOutput(
        ephemeral_memory_output_sizes_t size_info) :
        size_info(size_info) {
    }

    bool TextCompleter::EphemeralMemoryNNOutput::from_output(const Eigen::VectorXf& output) {
        const std::size_t expected_size = size_info.total();

        if (static_cast<std::size_t>(output.size()) != expected_size) {
            return false;
        }

        /* Vector layout encoding:
         * ephemeral_memory
         * word_vector_value
         **/

        ephemeral_memory = output.segment(0, size_info.ephemeral_memory);
        word_vector_value = output.segment(size_info.ephemeral_memory, size_info.word_vector_value);

        return true;
    }

    /********************** ContextBuilderNNFields ********************/

    std::size_t TextCompleter::context_builder_fields_sizes_t::total() const {
        return ephemeral_memory + context_memory + 3;
    }

    TextCompleter::ContextBuilderNNFields::ContextBuilderNNFields(
        float sentence_length_mean, float sentence_length_stddev, float flesch_kincaid_grade,
        const Eigen::VectorXf& ephemeral_memory, const Eigen::VectorXf& context_memory,
        const context_builder_fields_sizes_t& size_info) :
        sentence_length_mean(sentence_length_mean),
        sentence_length_stddev(sentence_length_stddev), flesch_kincaid_grade(flesch_kincaid_grade),
        ephemeral_memory(ephemeral_memory), context_memory(context_memory), size_info(size_info) {
        assert(static_cast<std::size_t>(ephemeral_memory.size()) == size_info.ephemeral_memory);
        assert(static_cast<std::size_t>(context_memory.size()) == size_info.context_memory);
    }

    Eigen::VectorXf TextCompleter::ContextBuilderNNFields::to_vector() const {
        const std::size_t vector_length = ephemeral_memory.size() + context_memory.size();

        Eigen::VectorXf vector(vector_length);

        /* Vector layout encoding:
         * sentence_length_mean
         * sentence_length_stddev
         * flesch_kincaid_grade
         * ephemeral_memory
         * context_memory
         **/

        std::size_t index {0};

        vector(index++) = sentence_length_mean;
        vector(index++) = sentence_length_stddev;
        // vector(index++) = word_sophistication;
        vector(index++) = flesch_kincaid_grade;

        vector.segment(index, size_info.ephemeral_memory) = ephemeral_memory;
        index += size_info.ephemeral_memory;

        vector.segment(index, size_info.context_memory) = context_memory;
        index += size_info.context_memory;

        return vector;
    }

    /********************** ContextBuilderNNOutput ********************/

    std::size_t TextCompleter::context_builder_output_sizes_t::total() const {
        return context_memory;
    }

    TextCompleter::ContextBuilderNNOutput::ContextBuilderNNOutput(
        context_builder_output_sizes_t size_info) :
        size_info(size_info) {
    }

    bool TextCompleter::ContextBuilderNNOutput::from_output(const Eigen::VectorXf& output) {
        if (static_cast<std::size_t>(output.size()) != size_info.total()) {
            return false;
        }

        context_memory = output;

        return true;
    }

    /********************** WordVectorImproviserNNFields ********************/

    std::size_t TextCompleter::word_vector_improviser_fields_sizes_t::total() const {
        return ephemeral_memory + word_vector_value + word_vector_search_result + 1;
    }

    TextCompleter::WordVectorImproviserNNFields::WordVectorImproviserNNFields(
        const VectorDatabase::SearchResult& result, const Eigen::VectorXf& ephemeral_memory,
        const Eigen::VectorXf& word_vector_value,
        const word_vector_improviser_fields_sizes_t& size_info) :
        word_vectors_search_result(result),
        ephemeral_memory(ephemeral_memory), word_vector_value(word_vector_value),
        size_info(size_info) {
        assert(static_cast<std::size_t>(ephemeral_memory.size()) == size_info.ephemeral_memory);
        assert(static_cast<std::size_t>(word_vector_value.size()) == size_info.word_vector_value);
    }

    Eigen::VectorXf TextCompleter::WordVectorImproviserNNFields::to_vector() const {
        const std::size_t vector_length = size_info.total();

        Eigen::VectorXf vector(vector_length);

        /* Vector layout encoding:
         * word_vectors_search_result.similarity
         * word_vectors_search_result.word.vector
         * ephemeral_memory
         * word_vector_ephemeral_memory
         **/

        std::size_t index {0};

        vector(index++) = word_vectors_search_result.similarity;

        vector.segment(index, size_info.word_vector_search_result) =
            word_vectors_search_result.word.vector;
        index += size_info.word_vector_search_result;

        vector.segment(index, size_info.ephemeral_memory) = ephemeral_memory;
        index += size_info.ephemeral_memory;

        vector.segment(index, size_info.word_vector_value) = word_vector_value;
        index += size_info.word_vector_value;

        return vector;
    }

    /********************** WordVectorImproviserNNOutput ********************/

    std::size_t TextCompleter::word_vector_improviser_output_sizes_t::total() const {
        return word_vector_value;
    }

    TextCompleter::WordVectorImproviserNNOutput::WordVectorImproviserNNOutput(
        word_vector_improviser_output_sizes_t size_info) :
        size_info(size_info) {
    }

    bool TextCompleter::WordVectorImproviserNNOutput::from_output(const Eigen::VectorXf& output) {
        const std::size_t expected_output_vector_size = size_info.total();

        if (static_cast<std::size_t>(output.size()) != expected_output_vector_size) {
            return false;
        }

        word_vector_value = output.segment(0, size_info.word_vector_value);

        return true;
    }

    /********************** End ********************/

    float sentence_length_mean(const std::vector<grammar::Token>& tokens) {
        std::size_t token_count = 0;
        std::size_t sentence_count = 0;

        for (const grammar::Token& token: tokens) {
            token_count++;

            if (token.value == "." || token.value == "!" || token.value == "?") {
                sentence_count++;
            }
        }

        return static_cast<float>(token_count) / std::max(1UL, sentence_count);
    }

    float sentence_length_stddev(const std::vector<grammar::Token>& tokens) {
        std::vector<int> sentence_lengths;
        int current_sentence_length = 0;

        for (const grammar::Token& token: tokens) {
            if (token.value == ".") {
                sentence_lengths.push_back(current_sentence_length);
                current_sentence_length = 0;
            }
            else {
                current_sentence_length++;
            }
        }

        if (current_sentence_length > 0) {
            sentence_lengths.push_back(current_sentence_length);
        }

        if (sentence_lengths.empty()) {
            return 0.0F; // No sentences
        }

        float mean_length = sentence_length_mean(tokens);

        std::vector<float> squared_deviations;
        for (int length: sentence_lengths) {
            float deviation = length - mean_length;
            squared_deviations.push_back(deviation * deviation);
        }

        float variance =
            std::accumulate(squared_deviations.begin(), squared_deviations.end(), 0.0F) /
            sentence_lengths.size();

        return std::sqrt(variance);
    }

    TextCompleter::SearchedWordVector TextCompleter::find_word_vector(const std::string& word) {
        constexpr std::size_t TOP_N = 10;

        if (vector_database == nullptr) {
            throw std::runtime_error("vector_database is null");
        }

        if (const std::optional<WordVector> word_vector = vector_database->search_from_map(word)) {
            return {word_vector.value(), false};
        }

        for (float threshold = 0.9F; threshold >= -0.1F; threshold -= 0.1F) {
            const std::vector<VectorDatabase::SearchResult> word_vectors =
                vector_database->rapidfuzz_search_closest_n(word, TOP_N, threshold);

            if (!word_vectors.empty()) {
                return {improvised_word_vector(word, word_vectors), true};
            }
        }

        return {improvised_word_vector(word, {}), true};
    }
} // namespace lc
