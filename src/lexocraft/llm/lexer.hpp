#ifndef LEXOCRAFT_LEXER_HPP
#define LEXOCRAFT_LEXER_HPP

#include <string>
#include <vector>

namespace lc::grammar {
    class Token {
        public:

        std::string value;
    };

    std::vector<Token> tokenize(
        const std::string& input); // Delimited by spaces, newlines, or symbols. Symbols are
                                   // considered to be their own tokens. } // namespace lc::grammar
} // namespace lc::grammar

#endif
