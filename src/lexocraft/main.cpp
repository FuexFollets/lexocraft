#include <Eigen/Dense>
#include <iostream>
#include <lexocraft/neural_network/nn.hpp>

int main() {
    Eigen::MatrixXd matrix(3, 3);
    matrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    std::cout << "Calling function " << nnfunction(100);

    // Your code using Eigen
    std::cout << matrix << '\n';

    return 0;
}
