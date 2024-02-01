#include <chrono>
#include <fstream>
#include <iostream>
#include <ratio>
#include <sstream>

#include <lexocraft/llm/lexer.hpp>
#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/llm/vector_database.hpp>

int main(int argc, char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";
    for (const auto& arg: args) {
        std::cout << arg << "\n";
    }

    const std::string filepath = args.at(0);
    const std::string vector_database_path = args.at(1);

    lc::VectorDatabase database {};
    std::cout << "Loading database from path: " << vector_database_path << '\n';
    database.load_file(vector_database_path);
    std::cout << "Successfully loaded vector database\n";

    std::cout << "Reading from file: " << filepath << "\n";

    const std::ifstream file {filepath};
    std::stringstream file_contents_buffer;
    file_contents_buffer << file.rdbuf();
    const std::string file_contents {file_contents_buffer.str()};

    std::cout << "file_contents: " << file_contents << "\n";

    std::cout << "Tokenizing...\n";

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    const std::vector<lc::grammar::Token> result = lc::grammar::tokenize(file_contents, database);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::nano> duration = end - start;

    std::cout << "Tokenization took " << duration.count() << " nanoseconds (milliseconds: "
              << duration.count() / 1000000.0 << ")\n";

    std::cout << "Tokens: " << result.size() << "\n";

    std::cout << "result:\n";

    std::cout << "[";

    for (const auto& token: result) {
        std::cout << token << '\n';
    }

    std::cout << "]\n";

    std::cout << "Sentence mean: " << lc::sentence_length_mean(result) << "\n";
    std::cout << "Sentence stddev: " << lc::sentence_length_stddev(result) << "\n";
}
