#include <cctype>
#include <string>
#include <vector>

#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/llm/vector_database.hpp>

namespace lc {
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

    Eigen::VectorXf TextCompleter::EphemeralMemoryNNFields::to_vector() const {
        const std::size_t vector_length = ephemeral_memory.size() + context_memory.size() +
                                          WordVector::WORD_VECTOR_DIMENSIONS + 4;

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

        vector.segment(index, word.vector.size()) = word.vector;
        index += word.vector.size();

        vector.segment(index, ephemeral_memory.size()) = ephemeral_memory;
        index += ephemeral_memory.size();

        vector.segment(index, context_memory.size()) = context_memory;
        index += context_memory.size();

        return vector;
    }

    bool TextCompleter::EphemeralMemoryNNOutput::from_output(const Eigen::VectorXf& output) {
        const std::size_t total_size = ephemeral_memory_size + WordVector::WORD_VECTOR_DIMENSIONS;

        if (static_cast<std::size_t>(output.size()) != total_size) {
            return false;
        }

        /* Vector layout encoding:
         * ephemeral_memory
         * context_memory
         **/

        ephemeral_memory = output.segment(0, ephemeral_memory_size);
        word_vector_value =
            output.segment(ephemeral_memory_size, WordVector::WORD_VECTOR_DIMENSIONS);

        return true;
    }

    Eigen::VectorXf TextCompleter::ContextBuilderNNFields::to_vector() const {
        const std::size_t vector_length = ephemeral_memory.size() + context_memory.size();

        Eigen::VectorXf vector(vector_length);

        /* Vector layout encoding:
         * sentence_length_mean
         * sentence_length_stddev
         * word_sophistication
         * flesch_kincaid_grade
         * ephemeral_memory
         * context_memory
         **/

        std::size_t index {0};

        vector(index++) = sentence_length_mean;
        vector(index++) = sentence_length_stddev;
        // vector(index++) = word_sophistication;
        vector(index++) = flesch_kincaid_grade;

        vector.segment(index, ephemeral_memory.size()) = ephemeral_memory;
        index += ephemeral_memory.size();

        vector.segment(index, context_memory.size()) = context_memory;
        index += context_memory.size();

        return vector;
    }

    bool TextCompleter::ContextBuilderNNOutput::from_output(const Eigen::VectorXf& output) {
        if (static_cast<std::size_t>(output.size()) != ephemeral_memory_size) {
            return false;
        }

        context_memory = output;

        return true;
    }

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
} // namespace lc
