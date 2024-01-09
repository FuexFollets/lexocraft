#include <ios>
#include <iostream>
#include <string>
#include <vector>

#include <lexocraft/llm/vector_database.hpp>

int main(int argc, char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";
    for (const auto& arg: args) {
        std::cout << arg << "\n";
    }

    if (args.size() < 2) {
        std::cout
            << "Usage: <database> <searched_word> [rapidfuzz/default] OPTIONAL: <top_n> "
               "<threshold> <soundex_weight> <levenshtein_weight> <stop_when_top_n_are_found>";

        return 1;
    }

    const std::string database_path = args.at(0);
    const std::string searched_word = args.at(1);
    const bool use_rapidfuzz = args.at(2) == "rapidfuzz";

    const bool optional_args_provided = args.size() > 3;

    const int top_n = optional_args_provided ? std::stoi(args.at(3)) : 10;
    const float threshold = optional_args_provided ? std::stof(args.at(4)) : 0.2F;
    const float soundex_weight = optional_args_provided ? std::stof(args.at(5)) : 0.5F;
    const float levenshtein_weight = optional_args_provided ? std::stof(args.at(6)) : 0.5F;
    const bool stop_when_top_n_are_found =
        optional_args_provided ? static_cast<bool>(std::stoi(args.at(7))) : false;

    std::cout << std::boolalpha;

    std::cout << "database_path: " << database_path << "\n"
              << "searched_word: " << searched_word << "\n"
              << "top_n: " << top_n << "\n"
              << "threshold: " << threshold << "\n"
              << "soundex_weight: " << soundex_weight << "\n"
              << "levenshtein_weight: " << levenshtein_weight << "\n"
              << "stop_when_top_n_are_found: " << stop_when_top_n_are_found << "\n";

    lc::VectorDatabase database;

    std::cout << "Loading database from " << database_path << "...\n";

    database.load(database_path);

    std::cout << "Database loaded.\n";

    std::cout << "Searching for " << searched_word << "...\n";

    const auto results =
        use_rapidfuzz
            ? database.rapidfuzz_search_closest_n(searched_word, top_n, threshold,
                                                  stop_when_top_n_are_found)
            : database.search_closest_n(searched_word, top_n, threshold, soundex_weight,
                                        levenshtein_weight, stop_when_top_n_are_found);

    std::cout << "Search results:\n";

    for (const auto& result: results) {
        std::cout << result.word.word << " " << result.similarity << "\n";
    }
}
