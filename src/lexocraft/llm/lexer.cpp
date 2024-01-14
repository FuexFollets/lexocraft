#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

#include <lexocraft/llm/lexer.hpp>

namespace lc::grammar {
    std::vector<Token> tokenize(const std::string& input) {
        // Delimited by spaces, newlines, or symbols. Symbols are considered to be their own tokens.

        std::vector<Token> tokens;
        std::string current_token;

        for (std::size_t index {0}; index < input.size(); index++) {
            const char current_char {input.at(index)};

            if (std::isalnum(current_char) != 0) {
                current_token += current_char;

                continue;
            }

            if (current_char == ' ' || current_char == '\n' || !current_token.empty()) {
                tokens.push_back(Token {current_token});
                current_token.clear();

                continue;
            }

            if (!current_token.empty()) {
                tokens.push_back(Token {current_token});
                current_token.clear();
            }

            tokens.push_back(Token {std::string {current_char}});
        }

        return tokens;
    }
} // namespace lc::grammar
