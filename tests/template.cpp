#include <iostream>
#include <string>
#include <vector>

#include <nanobench.h>
#include <icecream.hpp>

int main(const int argc, const char** argv) {
    std::vector<std::string> args {std::next(argv, 1), std::next(argv, argc)};

    std::cout << "args: " << args.size() << "\n";

    for (std::size_t index = 0; index < args.size(); ++index) {
        std::cout << "arg[" << index << "]: " << args [index] << "\n";
    }
}
