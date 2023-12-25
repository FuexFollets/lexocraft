#include <Eigen/Dense>
#include <iostream>
#include <lexocraft/neural_network/nn.hpp>

int nnfunction(int value) {
    auto random_matrix = Eigen::MatrixXd::Random(3,3);

    std::cout << "Random matrix: " << random_matrix << '\n';
    std::cout << "Random matrix determinant: " << random_matrix.determinant() << '\n';

    return random_matrix.determinant() * value;
}
