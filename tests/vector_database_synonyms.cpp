#include <iostream>
#include <string>
#include <vector>

#include <lexocraft/fancy_eigen_print.hpp>
#include <lexocraft/llm/vector_database.hpp>

int main(int argc, char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";
    for (const auto& arg: args) {
        std::cout << arg << "\n";
    }

    const std::string database_path = args.at(0);
    const std::string word = args.at(1);

    std::cout << "database_path: " << database_path << "\n";
    std::cout << "word: " << word << "\n";

    lc::VectorDatabase database {};

    std::cout << "loading database from " << database_path << "\n";

    database.load(database_path);

    std::cout << "database loaded\n";

    const std::optional<lc::WordVector> word_vector = database.search_from_map(word);

    if (!word_vector.has_value()) {
        std::cout << "word not found\n";
        exit(1);
    }

    std::cout << "word found\n";

    std::cout << "Searched vector value for \"" << word_vector.value().word
              << "\": " << word_vector.value().vector << "\n";

    const auto search_results =
        database.search_closest_vector_value_n(word_vector.value(), 10, 0.5F, false);

    std::cout << "-------------------\n\n";

    for (const auto& result: search_results) {
        std::cout << std::setprecision(2) << "Similarity: " << result.similarity << " "
                  << result.word.word << " " << lc::fancy_eigen_vector_str(result.word.vector)
                  << "\n";
    }

    std::cout << "-------------------\n\n";

    return 0;
}
