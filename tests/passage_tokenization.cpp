#include <fstream>
#include <iostream>
#include <sstream>

#include <lexocraft/llm/lexer.hpp>

int main(int argc, char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";
    for (const auto& arg: args) {
        std::cout << arg << "\n";
    }

    const std::string filepath = args.at(0);

    std::cout << "Reading from file: " << filepath << "\n";

    const std::ifstream file {filepath};
    std::stringstream file_contents_buffer;
    file_contents_buffer << file.rdbuf();
    const std::string file_contents {file_contents_buffer.str()};

    std::cout << "file_contents: " << file_contents << "\n";

    const auto result = lc::grammar::tokenize(file_contents);

    std::cout << "result:\n";

    std::cout << "[";

    for (const auto& token: result) {
        std::cout << "\"" << token.value << "\", ";
    }

    std::cout << "]\n";
}
