#ifndef LEXOCAD_SURFACE_ANALYSIS_HPP
#define LEXOCAD_SURFACE_ANALYSIS_HPP

#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <Eigen/Eigen>

namespace lc::grammar {
    // All grammar types: Container, Paragraph, Sentence, Word, Symbol

    struct GrammarObject {};

    class Symbol : GrammarObject {
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
        };

        static constexpr std::array<std::tuple<Type, char>, 33> types {
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
            for (auto [type, symbol_]: types) {
                if (symbol == symbol_) {
                    return type;
                }
            }
            return Type::Other;
        }

        static char symbol_from_type(Type type) {
            for (auto [type_, symbol]: types) {
                if (type == type_) {
                    return symbol;
                }
            }
            return 0;
        }
    };

    class Word : GrammarObject {
        public:

        std::string content; // Always lowercase
        std::vector<bool> capitalization_case;
    };

    class Sentence : GrammarObject { // Can contain Container(s), Word(s), and Punctuation(s)
        public:

        std::vector<GrammarObject> contents;
        Symbol ending_punctuation;
    };

    class Paragraph : GrammarObject { // Can contain Sentence(s)
        std::vector<Sentence> sentences;
    };

    class Container : GrammarObject { // Can contain Container(s), Paragraph(s),
                                      // Sentence(s), Word(s), and Punctuation(s)
        enum class ContainerType {
            Basic,
            Parenthesis,
            Quote,
        };

        std::vector<GrammarObject> contents;
    };

    Container parse(const std::string& text);
} // namespace lc::grammar

#endif
