#include <iostream>
#include <vector>

#include <lexocraft/llm/vector_database.hpp>

template <typename T>
std::string vector_to_string(const std::vector<T>& vec) {
    std::string str;

    str += "[";

    for (const auto& item: vec) {
        str += std::to_string(item);
        str += ", ";
    }

    str += "]";

    return str;
}

int main(int argc, char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";
    for (const auto& arg: args) {
        std::cout << arg << "\n";
    }

    std::cout << vector_to_string(std::vector<float> {1, 2, 3}) << "\n";

    lc::WordVector word1 {std::move(args.at(0))};
    lc::WordVector word2 {std::move(args.at(1))};

    std::cout << "word1: " << word1.word << "\n";
    std::cout << "word2: " << word2.word << "\n";
}
