#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXd matrix(3, 3);
    matrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    // Your code using Eigen
    std::cout << matrix << '\n';

    return 0;
}

