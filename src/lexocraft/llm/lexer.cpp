#include <algorithm>
#include <cctype>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <lexocraft/llm/lexer.hpp>

namespace lc::grammar {
    std::vector<Token> tokenize(const std::string& input) {
        // remove any chars whose values are not between '!' and
        std::vector<Token> tokens;
        std::string current_token;

        for (char letter: input) {
            if (std::isspace(letter) != 0) {
                // Space delimiter: add current token as a special token
                if (!current_token.empty()) {
                    tokens.push_back({current_token});
                    current_token.clear();
                }
            }

            else if (std::isalnum(letter) != 0) {
                // Letter or digit: add to current word token
                current_token += letter;
            }

            else {
                // Symbol: create a new token
                tokens.push_back({current_token});
                current_token.clear();
                // std::cout << "symbol: " << "\"" << letter << "\" " << static_cast<int>(letter) <<
                // "\n";
                tokens.push_back({std::string(1, letter)});
            }
        }

        // Add the last token if any
        if (!current_token.empty()) {
            tokens.push_back({current_token});
        }

        // remove any token that is just spaces or empty

        tokens.erase(std::remove_if(tokens.begin(), tokens.end(),
                                    [](const Token& token) {
                                        return token.value.empty() || token.value == " ";
                                    }),
                     tokens.end());

        return tokens;
    }
} // namespace lc::grammar
