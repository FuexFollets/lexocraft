#ifndef LEXOCRAFT_SURFACE_ANALYSIS_HPP
#define LEXOCRAFT_SURFACE_ANALYSIS_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <Eigen/Eigen>

namespace lc::grammar {
    // All grammar types: Container, Word, Symbol

    struct GrammarObject {};

    class Symbol : public GrammarObject {
        public:

        enum class Type {
            Exclamation = '!',
            CommercialAt = '@',
            Hashtag = '#',
            DollarSign = '$',
            Percent = '%',
            Caret = '^',
            Ampersand = '&',
            Asterisk = '*',
            OpenParenthesis = '(',
            CloseParenthesis = ')',
            Dash = '-',
            Underscore = '_',
            Plus = '+',
            Equals = '=',
            OpenBracket = '[',
            CloseBracket = ']',
            OpenBrace = '{',
            CloseBrace = '}',
            Pipe = '|',
            Backslash = '\\',
            Colon = ':',
            Semicolon = ';',
            DoubleQuote = '"',
            SingleQuote = '\'',
            LessThan = '<',
            GreaterThan = '>',
            Comma = ',',
            Period = '.',
            Question = '?',
            Slash = '/',
            Tilde = '~',
            Backtick = '`',
            Other = 0,
            None = -1,
        };

        static constexpr std::array<Type, 6> OPENING_TYPES = {
            Type::OpenParenthesis, Type::OpenBracket, Type::OpenBrace,
            Type::DoubleQuote,     Type::SingleQuote, Type::LessThan,
        };

        static constexpr std::array<std::tuple<Type, char>, 33> TYPES {
            {
             {Type::Exclamation, '!'},
             {Type::CommercialAt, '@'},
             {Type::Hashtag, '#'},
             {Type::DollarSign, '$'},
             {Type::Percent, '%'},
             {Type::Caret, '^'},
             {Type::Ampersand, '&'},
             {Type::Asterisk, '*'},
             {Type::OpenParenthesis, '('},
             {Type::CloseParenthesis, ')'},
             {Type::Dash, '-'},
             {Type::Underscore, '_'},
             {Type::Plus, '+'},
             {Type::Equals, '='},
             {Type::OpenBracket, '['},
             {Type::CloseBracket, ']'},
             {Type::OpenBrace, '{'},
             {Type::CloseBrace, '}'},
             {Type::Pipe, '|'},
             {Type::Backslash, '\\'},
             {Type::Colon, ':'},
             {Type::Semicolon, ';'},
             {Type::DoubleQuote, '"'},
             {Type::SingleQuote, '\''},
             {Type::LessThan, '<'},
             {Type::GreaterThan, '>'},
             {Type::Comma, ','},
             {Type::Period, '.'},
             {Type::Question, '?'},
             {Type::Slash, '/'},
             {Type::Tilde, '~'},
             {Type::Backtick, '`'},
             {Type::Other, 0},
             }
        };

        Type type;

        static Type type_from_symbol(char symbol) {
            for (auto [type, symbol_]: TYPES) {
                if (symbol == symbol_) {
                    return type;
                }
            }
            return Type::Other;
        }

        static char symbol_from_type(Type type) {
            for (auto [type_, symbol]: TYPES) {
                if (type == type_) {
                    return symbol;
                }
            }
            return 0;
        }
    };

    class Word : public GrammarObject {
        public:

        enum class Type {
            Number,
            Alphanumeric,
            Other,
        };

        Type type;
        std::string content; // Always lowercase
        std::optional<std::vector<bool>> capitalization_case;
    };

    class Container : public GrammarObject { // Can contain
                                             // Container(s), Word(s),
                                             // and Punctuation(s)
        public:

        enum class Type {
            Sentence,
            Paragraph,
            Quoted,
            Parenthesized,
            Bracketed,
            Braced,
            AngleBracketed,
        };

        std::vector<GrammarObject> contents;
        Symbol::Type opening_symbol;
    };

    std::size_t find_closing_symbol(const std::string& text, std::size_t opening_symbol_index);
    Container parse(const std::string& text, Symbol::Type opening_symbol = Symbol::Type::None);
} // namespace lc::grammar

#endif
