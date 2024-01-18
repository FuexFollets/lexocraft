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
            Acronym,
            Digit,
            Homogeneous,
            Symbol,
        };

        static constexpr std::array<char, 10> token_component_symbols {'-', '\'', '/', '.'};

        static MAPBOX_ETERNAL_CONSTEXPR auto TOKEN_TYPES =
            mapbox::eternal::map<Type, mapbox::eternal::string>({
                {Type::Alphanumeric, "Alphanumeric"},
                {Type::Acronym,      "Acronym"     },
                {Type::Digit,        "Digit"       },
                {Type::Homogeneous,  "Homogeneous" },
                {Type::Symbol,       "Symbol"      },
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
        Token(const std::string& value, bool next_is_space);
    };

    bool is_component_symbol(char letter);

    Token::Type token_type(const std::string& value);

    std::vector<Token> tokenize(const std::string& input);

    std::ostream& operator<<(std::ostream& output_stream, const Token& token);
} // namespace lc::grammar

#endif
