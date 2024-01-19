#include <algorithm>
#include <cctype>
#include <cstddef>
#include <functional>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include <lexocraft/llm/lexer.hpp>

namespace lc::grammar {
    Token::Token(std::string&& value, Token::Type type, bool next_is_space) :
        value {value}, type {type}, next_is_space {next_is_space} {
    }

    bool is_component_symbol(char letter) {
        return std::string {"~_/-'."}.find(letter) != std::string::npos;
    }

    bool is_terminating_symbol(const std::optional<const char>& letter) {
        if (!letter.has_value()) {
            return true;
        }

        return !is_component_symbol(letter.value()) && (std::isalpha(letter.value()) == 0);
    }

    Token::Type token_type(const std::string& value) {
        // Check for acronyms first, as they have a stricter pattern
        if (value.size() == 1 && (std::isalnum(value.at(0)) == 0)) {
            return Token::Type::Symbol;
        }

        if (std::all_of(value.begin(), value.end(), ::isalpha)) {
            return Token::Type::Alphanumeric;
        }

        char previous_char {};

        if (std::all_of(value.begin(), value.end(), [&previous_char](char letter) {
                if (previous_char == '.' && letter != '.') {
                    previous_char = letter;
                    return true;
                }

                if (previous_char != '.' && letter == '.') {
                    previous_char = letter;
                    return true;
                }

                return false;
            })) {
            return Token::Type::Acronym;
        }

        // Check for digits
        if (std::all_of(value.begin(), value.end(), ::isdigit)) {
            return Token::Type::Digit;
        }

        // Everything else is considered homogeneous
        return Token::Type::Homogeneous;
    }

    std::vector<Token> tokenize(const std::string& text, const VectorDatabase& vector_database) {
        const std::size_t longest_element_size = vector_database.longest_element();
        const std::size_t text_length = text.size();
        const std::span<const char> text_span {text};
        std::vector<Token> tokens;

        for (std::size_t index {}; index < text_length;) {
            const bool is_space = text.at(index) == ' ';

            std::optional<Token> longest_possible_token_for_this_index_from_database {};

            for (std::size_t span_index {1};
                 (span_index <= longest_element_size) && (index + span_index <= text_length);
                 span_index++) {
                const std::span<const char> sub_span = text_span.subspan(index, span_index);
                const std::size_t right_span_index = index + span_index;
                const std::optional<char> char_after_span =
                    (right_span_index < text_length) ? std::optional {text.at(right_span_index)}
                                                     : std::nullopt;

                // Can be sectioned off as a token
                const bool this_can_be_token =
                    !char_after_span.has_value() || (std::isalpha(char_after_span.value()) == 0);

                if (!this_can_be_token) {
                    continue;
                }

                if (const std::optional<WordVector> token = vector_database.search_from_map(
                        std::string {sub_span.begin(), sub_span.end()})) {
                    std::string token_value = token.value().word;
                    const bool next_is_space =
                        char_after_span.has_value() && char_after_span.value() == ' ';

                    longest_possible_token_for_this_index_from_database = Token {
                        std::move(token_value), token_type(token.value().word), next_is_space};
                }
            }

            if (longest_possible_token_for_this_index_from_database.has_value()) {
                tokens.push_back(longest_possible_token_for_this_index_from_database.value());
                index += longest_possible_token_for_this_index_from_database.value().value.size();

                continue;
            }

            if (is_space) {
                index++;

                continue;
            }

            for (std::size_t span_index {1};; span_index++) {
                const std::size_t right_span_index = index + span_index;
                const std::optional<char> char_after_span =
                    (right_span_index < text_length - 1) ? std::optional {text.at(right_span_index)}
                                                         : std::nullopt;
                const bool space_after_span =
                    char_after_span.has_value() && char_after_span.value() == ' ';

                if (span_index == 1 && is_terminating_symbol(text.at(index))) {
                    tokens.push_back(Token {std::string {text.at(index)}, Token::Type::Symbol,
                                            space_after_span});
                }

                if (span_index == 1 && (std::isdigit(text.at(index)) != 0)) {
                    tokens.push_back(
                        Token {std::string {text.at(index)}, Token::Type::Digit, space_after_span});
                }

                const std::span<const char> sub_span = text_span.subspan(index, span_index);

                if (space_after_span || is_terminating_symbol(char_after_span)) {
                    std::string token_value {sub_span.begin(), sub_span.end()};
                    tokens.emplace_back(std::move(token_value), token_type(token_value),
                                        space_after_span);

                    index += token_value.size();

                    break;
                }
            }
        }

        return tokens;
    }

    std::ostream& operator<<(std::ostream& output_stream, const Token& token) {
        /* Name layout:
         * (value: type, next_is_space)
         */

        output_stream << "(" << token.value << ": " << Token::TOKEN_TYPES.at(token.type).c_str()
                      << ", " << token.next_is_space << ")";

        return output_stream;
    }
} // namespace lc::grammar
