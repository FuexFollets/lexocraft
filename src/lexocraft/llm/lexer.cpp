#include <algorithm>
#include <cctype>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <lexocraft/llm/lexer.hpp>

namespace lc::grammar {
    Token::Type token_type(const std::string& value) {
        // Check for acronyms first, as they have a stricter pattern
        if (value.size() == 1 && std::string {"`!@#$%^&*()+-={}[]|\\}|;:\"<,>?"}.find(
                                     value.at(0)) != std::string::npos) {
            return Token::Type::Symbol;
        }

        if (std::all_of(value.begin(), value.end(),
                        [](char letter) { return (std::isupper(letter) != 0) || letter == '.'; })) {
            return Token::Type::Acronym;
        }

        // Check for digits
        if (std::all_of(value.begin(), value.end(), ::isdigit)) {
            return Token::Type::Digit;
        }

        // Check for homogeneous tokens
        if (std::all_of(value.begin(), value.end(), [](char letter) {
                return (std::isalnum(letter) != 0) ||
                       std::string {"~_/-'."}.find(letter) != std::string::npos;
            })) {
            return Token::Type::Homogeneous;
        }

        // Everything else is considered alphanumeric
        return Token::Type::Alphanumeric;
    }

    /*
    Token::Token(const std::string& value) : Token(value, false) {}

    Token::Token(const std::string& value, bool next_is_space) : value(value),
    next_is_space(next_is_space) { bool has_letters = false; bool has_digits = false; bool
    has_symbols = false;

        for (char letter: value) {
            if (std::isalnum(letter) != 0) {
                has_letters |= std::isalpha(letter);
                has_digits |= std::isdigit(letter);
            }

            else {
                has_symbols = true;
            }
        }

        if (has_letters && !has_digits && !has_symbols && value.size() == 1) {
            type = Type::Letter;
        }

        else if (has_digits && !has_letters && !has_symbols) {
            type = Type::Digit;
        }

        else if (has_letters && !has_digits && !has_symbols) {
            type = Type::Alphanumeric;
        }

        else if (!has_letters && !has_digits && has_symbols) {
            type = Type::Symbol;
        }

        else {
            type = Type::Homogeneous;
        }
    }

    std::vector<Token> tokenize(const std::string& input) {
        // remove any chars whose values are not between '!' and
        std::vector<Token> tokens;
        std::string current_token;

        std::size_t index = 0;
        for (char letter: input) {
            if (std::isspace(letter) != 0) {
                // Space delimiter: add current token as a special token
                if (!current_token.empty()) {
                    tokens.emplace_back(current_token, true);
                    current_token.clear();
                }
            }

            else if (std::isalnum(letter) != 0) {
                // Letter or digit: add to current word token
                current_token += letter;
            }

            else {
                // Symbol: create a new token
                tokens.emplace_back(current_token, false);
                current_token.clear();

                const bool is_next_letter_space =
                    (index + 1 < input.size()) && (std::isspace(input [index + 1]) != 0);

                tokens.emplace_back(std::string(1, letter), is_next_letter_space);
            }

            index++;
        }

        // Add the last token if any
        if (!current_token.empty()) {
            tokens.emplace_back(current_token, false);
        }

        // remove any token that is just spaces or empty

        tokens.erase(std::remove_if(tokens.begin(), tokens.end(),
                                    [](const Token& token) {
                                        return token.value.empty() || token.value == " ";
                                    }),
                     tokens.end());

        // Separate any multi-digit numbers into separate tokens
        for (std::size_t index {}; index < tokens.size() - 1; index++) {
            if (tokens [index].type == Token::Type::Digit && tokens [index].value.size() > 1) {
                std::vector<Token> sub_tokens;
                std::cout << "Split up " << tokens [index].value << std::endl;

                for (char letter: tokens [index].value) {
                    sub_tokens.emplace_back(std::string(1, letter), false);
                }

                tokens.erase(tokens.begin() + index);
                tokens.insert(tokens.begin() + index, sub_tokens.begin(), sub_tokens.end());
            }
        }

        return tokens;
    }
    */

    std::ostream& operator<<(std::ostream& output_stream, const Token& token) {
        /* Name layout:
         * (value: type, next_is_space)
         */

        output_stream << "(" << token.value << ": " << Token::TOKEN_TYPES.at(token.type).c_str()
                      << ", " << token.next_is_space << ")";

        return output_stream;
    }
} // namespace lc::grammar
