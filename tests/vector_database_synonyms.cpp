#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <lexocraft/fancy_eigen_print.hpp>
#include <lexocraft/llm/vector_database.hpp>

std::string remove_whitespace(std::string str) {
    str.erase(str.begin(), std::find_if_not(str.begin(), str.end(), ::isspace));
    str.erase(std::find_if_not(str.rbegin(), str.rend(), ::isspace).base(), str.end());

    return str;
}

int main(int argc, char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";
    for (const auto& arg: args) {
        std::cout << arg << "\n";
    }

    const std::string database_type = args.at(0);
    const std::string database_path = args.at(1);
    const std::string word = args.at(2);

    std::cout << "database_type: " << database_type << "\n";
    std::cout << "database_path: " << database_path << "\n";
    std::cout << "word: " << word << "\n";

    lc::VectorDatabase database {};

    // ---------------------- Building Database ----------------------

    if (database_type == "plaintext") {
        std::cout << "loading database from " << database_path << "\n";

        std::cout << "database loaded\n";

        std::ifstream file {database_path};
        std::vector<lc::WordVector> words;
        std::string line;

        std::cout << "Reading from file: " << database_path << "\n";

        while (std::getline(file, line)) {
            words.emplace_back(remove_whitespace(line));
        }

        std::cout << "Read " << words.size() << " words from file\n";

        database = lc::VectorDatabase {words};

        std::cout << "Building Annoy index\n";
        database.build_annoy_index(10);
        std::cout << "Annoy index built\n";
    }

    if (database_type == "binary") {
        std::cout << "loading database from " << database_path << "\n";

        database.load_file(database_path);

        std::cout << "database loaded\n";
    }

    // ---------------------- Searching Database ----------------------

    std::vector<float> item(lc::WordVector::WORD_VECTOR_DIMENSIONS);

    if (!database.annoy_index_is_built) {
        std::cout << "Building Annoy index\n";
        database.build_annoy_index();
    }

    database.annoy_index->get_item(100, item.data());

    std::cout << "Database word vector example: " << item [0] << " " << item [1] << " " << item [2]
              << " " << item [3] << " " << item [4] << " " << item [5] << " " << item [6] << " "
              << item [7] << " " << item [8] << " " << item [9] << " ...\n";

    const std::optional<lc::WordVector> word_vector = database.search_from_map(word);

    if (!word_vector.has_value()) {
        std::cout << "word not found\n";
        exit(1);
    }

    std::cout << "word found\n";

    const std::size_t printed_dimensions = 15;
    std::cout << "Searched vector value for \"" << word_vector.value().word << "\": "
              << lc::fancy_eigen_vector_str(word_vector.value().vector, printed_dimensions) << "\n";

    const auto search_results = database.search_closest_vector_value_n(word_vector.value(), 10);

    std::cout << "-------------------\n\n";

    for (const auto& result: search_results) {
        std::cout << std::setprecision(2) << "Similarity: " << result.similarity << " "
                  << result.word.word << " "
                  << lc::fancy_eigen_vector_str(result.word.vector, printed_dimensions) << "\n";
    }

    std::cout << "-------------------\n\n";

    return 0;
}
