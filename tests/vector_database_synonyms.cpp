#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <nanobench.h>

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
    const std::optional<std::string> maybe_temp_file =
        (args.size() == 4) ? std::optional<std::string> {args.at(3)} : std::nullopt;

    std::cout << "database_type: " << database_type << "\n";
    std::cout << "database_path: " << database_path << "\n";
    std::cout << "word: " << word << "\n";
    std::cout << "temp_file: " << maybe_temp_file.value_or("not set") << "\n";

    lc::VectorDatabase database {};

    // ---------------------- Building Database ----------------------

    if (database_type == "plaintext") {
        std::cout << "loading database from " << database_path << "\n";

        std::cout << "database loaded\n";

        std::ifstream file {database_path};
        std::vector<lc::WordVector> words;
        std::string line;

        std::cout << "Reading from file: " << database_path << "\n";

        constexpr auto MAX_WORDS_COUNT = 10000000;

        int word_count = 0;
        while (std::getline(file, line) && word_count++ < MAX_WORDS_COUNT) {
            words.emplace_back(remove_whitespace(line));
        }


        std::cout << "Read " << words.size() << " words from file\n";

        database = lc::VectorDatabase {words};

        std::cout << "Building Annoy index\n";
        ankerl::nanobench::Bench().run("build_annoy_index", [&] {
            database.build_annoy_index(10);
        });

        std::cout << "Annoy index built\n";
    }

    if (database_type == "binary") {
        std::cout << "loading database from " << database_path << "\n";

        database.load_file(database_path);

        std::cout << "database loaded\n";
    }

    // ---------------------- Serializing and Deserializing Database ----------------------

    std::cout << "Serializing and deserializing database\n";

    if (!maybe_temp_file.has_value()) {
        std::stringstream sstream;

        std::cout << "Serializing database\n";
        cereal::BinaryOutputArchive archive(sstream);
        archive(database);

        std::cout << "Serialized database\n";

        std::cout << "Deserializing database\n";

        cereal::BinaryInputArchive iarchive(sstream);

        iarchive(database);

        std::cout << "Deserialized database\n";
    }

    if (maybe_temp_file.has_value()) {
        std::cout << "Saving database to " << maybe_temp_file.value() << "\n";
        database.save_file(maybe_temp_file.value());
        std::cout << "Saved database\n";

        std::cout << "Loading database from " << maybe_temp_file.value() << "\n";
        lc::VectorDatabase loaded_database {};

        loaded_database.load_file(maybe_temp_file.value());
        database = std::move(loaded_database);

        std::cout << "Loaded database\n";
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

    std::optional<std::vector<lc::VectorDatabase::SearchResult>> search_results;

    ankerl::nanobench::Bench().run("search_closest_vector_value_n", [&] {
        ankerl::nanobench::doNotOptimizeAway(
            search_results = database.search_closest_vector_value_n(word_vector.value(), 10));
    });

    std::cout << "-------------------\n\n";

    if (!search_results.has_value()) {
        std::cout << "Error: search results not found\n";
        exit(1);
    }

    for (const auto& result: search_results.value()) {
        std::cout << std::setprecision(2) << "Similarity: " << result.similarity << " "
                  << result.word.word << " "
                  << lc::fancy_eigen_vector_str(result.word.vector, printed_dimensions) << "\n";
    }

    std::cout << "-------------------\n\n";

    return 0;
}
