#ifndef LEXOCAD_SURFACE_ANALYSIS_HPP
#define LEXOCAD_SURFACE_ANALYSIS_HPP

#include <optional>
#include <string>
#include <vector>

namespace lc::grammar {
    // All grammar types: Container, Paragraph, Sentence, Word, Symbol

    class Symbol {};
    class Word {};

    class Sentence {}; // Can contain Container(s), Word(s), and Symbol(s)
    class Paragraph {}; // Can contain Sentence(s)
    class Container {}; // Can contain Container(s), Paragraph(s), Sentence(s), Word(s), and Symbol(s)
} // namespace lc::grammar

#endif
