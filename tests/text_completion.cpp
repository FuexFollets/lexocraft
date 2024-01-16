#include <iostream>

#include <lexocraft/llm/text_completion.hpp>
#include <lexocraft/llm/vector_database.hpp>
#include <memory>

int main(int argc, char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";

    for (const auto& arg: args) {
        std::cout << "Arg: " << arg << "\n";
    }

    const std::string database_path = args.at(0);

    std::cout << "database_path: " << database_path << "\n";

    lc::VectorDatabase database {};

    database.load(database_path);

    std::cout << "Database loaded from path: " << database_path << "\n";
    std::cout << "Constructing text completer\n";

    lc::TextCompleter completer {std::make_shared<lc::VectorDatabase>(database), 500, 500};

    std::cout << "Text completer created\n";
}
