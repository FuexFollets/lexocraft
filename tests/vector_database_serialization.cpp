#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>

#include <lexocraft/llm/vector_database.hpp>

int main(int argc, char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";
    for (const auto& arg: args) {
        std::cout << arg << "\n";
    }

    const std::filesystem::path tmp_path =
        (!args.empty() ? std::filesystem::path {args [0]} : std::filesystem::temp_directory_path());

    lc::VectorDatabase vector_database;

    vector_database.add_word("hello", true);
    vector_database.add_word("bob", true);
    vector_database.add_word("alice", true);
    vector_database.add_word("world", true);
    vector_database.add_word("foo", true);
    vector_database.add_word("bar", true);

    vector_database.annoy_index->build(10);
    vector_database.annoy_index->unbuild();
    vector_database.annoy_index->build(10000);

    std::stringstream sstream {};

    cereal::BinaryOutputArchive output_archive(sstream);

    output_archive(vector_database);

    cereal::BinaryInputArchive input_archive(sstream);

    lc::VectorDatabase vector_database2;

    input_archive(vector_database2);

    std::cout << "vector_database2: " << vector_database2.words [2].word << "\n";

    std::ofstream file_stream(tmp_path);
    cereal::BinaryOutputArchive file_output_archive(file_stream);

    file_output_archive(vector_database);

    file_stream.close();

    std::ifstream file_stream2(tmp_path);

    cereal::BinaryInputArchive file_input_archive(file_stream2);

    lc::VectorDatabase vector_database_file_input;

    file_input_archive(vector_database_file_input);

    std::cout << "vector_database_file_input.words [2].word: "
              << vector_database_file_input.words [2].word << "\n";

    vector_database.save_file(tmp_path);

    lc::VectorDatabase vector_database_file;

    vector_database_file.load_file(tmp_path);

    std::cout << "vector_database_file.words [2].word: " << vector_database_file.words [2].word
              << "\n";
}
