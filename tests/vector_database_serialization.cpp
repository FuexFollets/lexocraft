#include <iostream>


#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>

#include <lexocraft/llm/vector_database.hpp>
#include <sstream>

int main(int argc, char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";
    for (const auto& arg: args) {
        std::cout << arg << "\n";
    }

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

    std::cout << "vector_database2: " << vector_database2.words[2].word << "\n";
}
