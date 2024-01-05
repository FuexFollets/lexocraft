#include <cstddef>
#include <string>
#include <vector>

#include <lexocraft/llm/lexer.hpp>

namespace lc::grammar {
    std::vector<Token> tokkenize(const std::string& input) {
        // Delimited by spaces, newlines, or symbols. Symbols are considered to be their own tokens.

        std::vector<Token> tokens;
        std::string current_token;

        for (std::size_t index {0}; index < input.size(); index++) {

        }
    }
} // namespace lc::grammar
