#include <iostream>
#include <Eigen/Dense>

int function(int value) {
    std::cout << "Your value is: " << value << '\n';
    Eigen::MatrixXd matrix (2, 2);
    matrix(0, 0) = 3;

    return value + matrix(0, 0);
}
