#include <cctype>
#include <string>
#include <vector>

#include <lexocraft/llm/text_completion.hpp>

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
} // namespace lc
