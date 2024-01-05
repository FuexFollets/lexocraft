#include <algorithm>
#include <any>

#include <lexocraft/llm/grammar.hpp>

namespace lc::grammar {
    Container parse(const std::string& input,
                    Symbol::Type opening_symbol) {
        Container container;

        container.opening_symbol = opening_symbol;

        if (input.empty()) {
            return {};
        }

        for (std::size_t index = 0; index < input.size(); index++) {
            const char char_at_index = input.at(index);

            if (std::any_of()
        }

        return container;
    }
} // namespace lc::grammar
