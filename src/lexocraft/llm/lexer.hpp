#ifndef LEXOCRAFT_LEXER_HPP
#define LEXOCRAFT_LEXER_HPP

#include <ostream>
#include <string>
#include <vector>

#include <mapbox/eternal.hpp>

namespace lc::grammar {
    class Token {
        public:

        enum class Type {
            Alphanumeric,
            Digit,
            Letter,
            Symbol,
            Homogeneous, // Mix of letters, digits, and symbols
        };

        static MAPBOX_ETERNAL_CONSTEXPR auto TOKEN_TYPES = mapbox::eternal::map<Type, mapbox::eternal::string>({
            {Type::Alphanumeric, "Alphanumeric"},
            {Type::Digit,        "Digit"       },
            {Type::Letter,       "Letter"      },
            {Type::Symbol,       "Symbol"      },
            {Type::Homogeneous,  "Homogeneous" },
        });

        std::string value;
        bool next_is_space {};
        Type type {};

        Token() = default;
        Token(const Token&) = default;
        Token(Token&&) = default;
        Token& operator=(const Token&) = default;
        Token& operator=(Token&&) = default;

        explicit Token(const std::string& value);
        Token(const std::string& value, bool next_is_space, Type type);
        Token(const std::string& value, bool next_is_space);
    };

    std::vector<Token>
        tokenize(const std::string& input); // Delimited by spaces, newlines, or symbols. Symbols
                                            // are considered to be their own tokens.

    std::ostream& operator<<(std::ostream& output_stream, const Token& token);
} // namespace lc::grammar

#endif
