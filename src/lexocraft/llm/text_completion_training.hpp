#ifndef LEXOCRAFT_TEXT_COMPLETION_TRAINING_HPP
#define LEXOCRAFT_TEXT_COMPLETION_TRAINING_HPP

#include <optional>

#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/neural_network/neural_network.hpp>

namespace lc {
    class TextCompletionTraining {
        public:

        enum class Modifier {
            EphemeralMemory = 0b0001,
            ContextBuilder = 0b0010,
            word_vector_improviser = 0b0100,
        };

        std::optional<TextCompleter> text_completer;
    };
} // namespace lc

#endif // LEXOCRAFT_TEXT_COMPLETION_TRAINING_HPP
