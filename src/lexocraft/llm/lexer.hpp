#ifndef LEXOCRAFT_LEXER_HPP
#define LEXOCRAFT_LEXER_HPP

#include <ostream>
#include <string>
#include <vector>

#include <mapbox/eternal.hpp>

#include <lexocraft/llm/vector_database.hpp>

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

        static MAPBOX_ETERNAL_CONSTEXPR auto TOKEN_TYPES =
            mapbox::eternal::map<Type, mapbox::eternal::string>({
                {Type::Alphanumeric, "Alphanumeric"},
                {Type::Acronym,      "Acronym"     },
                {Type::Digit,        "Digit"       },
                {Type::Homogeneous,  "Homogeneous" },
                {Type::Symbol,       "Symbol"      },
        });

        std::string value;
        Type type {};
        bool next_is_space {};

        Token() = default;
        Token(const Token&) = default;
        Token(Token&&) = default;
        Token& operator=(const Token&) = default;
        Token& operator=(Token&&) = default;

        Token(std::string&& value, Type type, bool next_is_space);
    };

    bool is_component_symbol(const std::optional<const char>& letter);
    bool is_terminating_symbol(const std::optional<const char>& letter);
    // bool is_potentially_terminating_symbol(const std::optional<const char>& letter);

    Token::Type token_type(const std::string& value);

    std::vector<Token> tokenize(const std::string& text, const VectorDatabase& vector_database);

    std::ostream& operator<<(std::ostream& output_stream, const Token& token);
} // namespace lc::grammar

#endif
