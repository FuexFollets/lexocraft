#include <chrono>
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
            << "Usage: <database> <searched_word> [rapidfuzz/default/hash] <iterations> OPTIONAL: "
               "<top_n> <threshold> <soundex_weight> <levenshtein_weight> "
               "<stop_when_top_n_are_found>";

        return 1;
    }

    const std::string database_path = args.at(0);
    const std::string searched_word = args.at(1);
    const bool use_rapidfuzz = args.at(2) == "rapidfuzz";
    const bool use_hash = args.at(2) == "hash";
    const std::size_t iterations = std::stoi(args.at(3));

    const bool optional_args_provided = args.size() > 3;

    const int top_n = optional_args_provided ? std::stoi(args.at(4)) : 10;
    const float threshold = optional_args_provided ? std::stof(args.at(5)) : 0.2F;
    const float soundex_weight = optional_args_provided ? std::stof(args.at(6)) : 0.5F;
    const float levenshtein_weight = optional_args_provided ? std::stof(args.at(7)) : 0.5F;
    const bool stop_when_top_n_are_found =
        optional_args_provided ? static_cast<bool>(std::stoi(args.at(8))) : false;

    std::cout << std::boolalpha;

    if (!use_hash) {
        std::cout << "database_path: " << database_path << "\n"
                  << "searched_word: " << searched_word << "\n"
                  << "method: " << (use_rapidfuzz ? "rapidfuzz" : "default") << "\n"
                  << "iterations: " << iterations << "\n"
                  << "top_n: " << top_n << "\n"
                  << "threshold: " << threshold << "\n"
                  << "soundex_weight: " << soundex_weight << "\n"
                  << "levenshtein_weight: " << levenshtein_weight << "\n"
                  << "stop_when_top_n_are_found: " << stop_when_top_n_are_found << "\n";
    }

    lc::VectorDatabase database;

    std::cout << "Loading database from " << database_path << "...\n";

    database.load(database_path);

    std::cout << "Database loaded.\n";

    std::cout << "Searching for " << searched_word << "...\n";

    if (use_hash) {
        // measure time

        std::size_t average_duration_ns = 0;
        const auto result = database.search_from_map(searched_word);

        for (std::size_t iteration {0}; iteration < iterations; ++iteration) {
            const auto start = std::chrono::high_resolution_clock::now();
            const auto result = database.search_from_map(searched_word);
            const auto end = std::chrono::high_resolution_clock::now();
            const auto duration_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            average_duration_ns += duration_ns.count();
        }

        average_duration_ns /= iterations;

        std::cout << "Average duration: " << average_duration_ns << " nanoseconds = ("
                  << average_duration_ns / 1000000.0F << " ms)\n";

        if (result.has_value()) {
            std::cout << "Search result: " << result->word << "\n";
        }

        else {
            std::cout << "Search result: not found\n";
        }

        return 0;
    }

    std::size_t average_duration_ns = 0;

    const auto results =
        use_rapidfuzz ? database.rapidfuzz_search_closest_n(searched_word, top_n, threshold,
                                                            stop_when_top_n_are_found)
                      : database.search_closest_n(searched_word, top_n, threshold, soundex_weight,
                                                  levenshtein_weight, stop_when_top_n_are_found);

    for (std::size_t iteration {0}; iteration < iterations; ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        const auto results =
            use_rapidfuzz
                ? database.rapidfuzz_search_closest_n(searched_word, top_n, threshold,
                                                      stop_when_top_n_are_found)
                : database.search_closest_n(searched_word, top_n, threshold, soundex_weight,
                                            levenshtein_weight, stop_when_top_n_are_found);
        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        average_duration_ns += duration_ns.count();
    }

    average_duration_ns /= iterations;

    std::cout << "Average duration: " << average_duration_ns << " nanoseconds = ("
              << average_duration_ns / 1000000.0F << " ms)\n";

    std::cout << "Search results:\n";

    for (const auto& result: results) {
        std::cout << result.word.word << " " << result.similarity << "\n";
    }
}
