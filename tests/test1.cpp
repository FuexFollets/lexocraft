#include <iostream>
#include <lexocraft/neural_network/nn.hpp>

int main() {
    auto output {nnfunction(100)};

    std::cout << "hello world\n"
              << "Your output is: " << output << "\n";

    return 0;
}
